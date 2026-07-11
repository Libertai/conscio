"""Compatibility-preserving V3 runtime layered around the V2 executor."""

from __future__ import annotations

from typing import Any

from conscio.core.runtime import CognitiveRuntime, EpisodeResult, _TickState
from conscio.core.workspace import EntryType
from conscio.v3.contracts import (
    ActionOutcome,
    AffectiveState,
    CognitiveEvent,
    CoreCheckpoint,
)
from conscio.v3.environment import EnvironmentAdapter, TextEnvironmentAdapter
from conscio.v3.recurrent_core import HybridRecurrentCore


def _checkpoint_from_dict(data: dict[str, Any]) -> CoreCheckpoint:
    payload = dict(data)
    payload["affect"] = AffectiveState(**payload["affect"])
    payload["deterministic_state"] = tuple(payload["deterministic_state"])
    payload["stochastic_state"] = tuple(payload["stochastic_state"])
    return CoreCheckpoint(**payload)


class V3CognitiveRuntime(CognitiveRuntime):
    """Runs recurrent specialist cycles before the frozen LLM specialist.

    API consumers continue to receive ``EpisodeResult``.  The added fields are
    populated from the same append-only records available through episode
    retrieval, so in-memory and persisted observability cannot silently diverge.
    """

    def __init__(
        self,
        *args: Any,
        environment: EnvironmentAdapter | None = None,
        cognitive_cycles: int = 3,
        core_seed: int = 17,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.environment = environment or TextEnvironmentAdapter()
        self.recurrent_core = HybridRecurrentCore(seed=core_seed)
        self.cognitive_cycles = max(2, int(cognitive_cycles))
        self._episode_predictions: dict[str, list[dict[str, Any]]] = {}
        self._episode_affect: dict[str, list[dict[str, Any]]] = {}
        self._episode_proposals: dict[str, list[dict[str, Any]]] = {}

    async def initialize(self) -> None:
        await super().initialize()
        latest = await self.memory.latest_core_checkpoint()
        if latest is not None:
            self.recurrent_core.restore(_checkpoint_from_dict(latest))

    async def _prepare_episode(self, ts: _TickState) -> None:
        observation = await self.environment.observe(ts.event, episode_id=ts.episode_id)
        await self.memory.append_cognitive_event(observation)
        results = self.recurrent_core.run_cycles(
            observation,
            cycles=self.cognitive_cycles,
            memory_enabled=self.ablation.memory_retrieval,
            self_model_enabled=self.ablation.self_state_coupling,
            prediction_enabled=self.ablation.prediction,
            broadcast_enabled=self.ablation.attention_gating,
        )
        self._episode_predictions[ts.episode_id] = []
        self._episode_affect[ts.episode_id] = []
        self._episode_proposals[ts.episode_id] = []
        for cycle in results:
            if self.ablation.attention_gating:
                await self._append(
                    ts.episode_id, "broadcast", "recurrent_workspace", cycle.broadcast.to_dict(),
                    parent_event_id=observation.event_id,
                )
            for candidate in cycle.broadcast.candidates:
                entry_type = EntryType.SELF_STATE if candidate.specialist == "self_model" else EntryType.BROADCAST
                self.workspace.write(
                    candidate.content,
                    source=f"v3.{candidate.specialist}",
                    type=entry_type,
                    priority=max(1, min(9, round(candidate.salience * 10))),
                    salience=candidate.salience,
                    confidence=candidate.confidence,
                    novelty=0.55,
                    metadata={
                        "candidate_id": candidate.candidate_id,
                        "epistemic_kind": candidate.kind,
                        "broadcast_id": cycle.broadcast.broadcast_id,
                        "cycle": cycle.broadcast.cycle,
                    },
                )
            for prediction in cycle.predictions:
                data = prediction.to_dict()
                self._episode_predictions[ts.episode_id].append(data)
                await self._append(ts.episode_id, "prediction", "world_model", data)
                self.workspace.write(
                    f"P={prediction.probability:.2f}: {prediction.observable}",
                    source="v3.world_model",
                    type=EntryType.PREDICTION,
                    priority=5,
                    confidence=prediction.probability,
                    metadata={"prediction_id": prediction.prediction_id, "target": prediction.target},
                )
            affect = cycle.affect.to_dict()
            self._episode_affect[ts.episode_id].append(affect)
            await self._append(ts.episode_id, "affect", "affect", affect)
            self.workspace.write(
                f"valence={cycle.affect.valence:.3f}, arousal={cycle.affect.arousal:.3f}, "
                f"controllability={cycle.affect.controllability:.3f}",
                source="v3.affect",
                type=EntryType.AFFECT,
                priority=4,
                salience=cycle.affect.arousal,
                metadata={"need_errors": cycle.affect.need_errors},
            )
            for proposal in cycle.proposals:
                data = proposal.to_dict()
                self._episode_proposals[ts.episode_id].append(data)
                await self._append(ts.episode_id, "action_proposal", proposal.specialist, data)

    async def _append(
        self,
        episode_id: str,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        *,
        parent_event_id: str | None = None,
        model_input: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
    ) -> None:
        await self.memory.append_cognitive_event(
            CognitiveEvent(
                event_type=event_type,
                source=source,
                payload=payload,
                episode_id=episode_id,
                parent_event_id=parent_event_id,
                model_input=model_input,
                checkpoint_id=checkpoint_id,
            )
        )

    async def _finalize_episode(self, ts: _TickState, start: float) -> EpisodeResult:
        result = await super()._finalize_episode(ts, start)
        proposals = self._episode_proposals.pop(ts.episode_id, [])
        selected = next((p for p in proposals if p["action"] == result.selected_action), None)
        if selected is None and proposals:
            selected = proposals[0]
        outcome = ActionOutcome(
            proposal_id=str((selected or {}).get("proposal_id", "executor")),
            action=result.selected_action,
            succeeded=result.selected_action not in {"wait"} or bool(result.output),
            observation=result.output[:1000],
            prediction_errors={"runtime_prediction_errors": float(result.metrics.prediction_errors)},
        )
        await self._append(ts.episode_id, "action_outcome", "environment", outcome.to_dict())
        checkpoint = self.recurrent_core.checkpoint()
        await self.memory.save_core_checkpoint(checkpoint)
        await self._append(
            ts.episode_id,
            "checkpoint",
            "recurrent_core",
            {"lineage_id": checkpoint.lineage_id, "model_version": checkpoint.model_version},
            checkpoint_id=checkpoint.checkpoint_id,
            model_input={
                "dynamic_context": result.model_context,
                "calls": self.executor.model_inputs,
                "lesions": {
                    "memory": not self.ablation.memory_retrieval,
                    "self_model": not self.ablation.self_state_coupling,
                    "prediction": not self.ablation.prediction,
                    "attention": not self.ablation.attention_gating,
                },
            },
        )
        trace = await self.memory.cognitive_events(ts.episode_id)
        result.causal_trace = trace
        result.checkpoint_reference = checkpoint.checkpoint_id
        result.predictions = self._episode_predictions.pop(ts.episode_id, [])
        result.affect_trajectory = self._episode_affect.pop(ts.episode_id, [])
        result.action_outcomes = [outcome.to_dict()]
        result.exact_model_inputs = self.executor.model_inputs
        return result

    async def set_safe_affect_state(
        self,
        *,
        reason: str,
        operator: str = "operator",
        episode_id: str = "",
    ) -> AffectiveState:
        """Audited operator recovery control; never silently mutates affect."""
        before = self.recurrent_core.affect
        intervention_id = f"affect_{__import__('uuid').uuid4().hex}"
        after = AffectiveState(intervention_id=intervention_id)
        self.recurrent_core.affect = after
        await self.memory.record_affect_intervention(
            intervention_id=intervention_id,
            episode_id=episode_id,
            operator=operator,
            reason=reason,
            before_state=before.to_dict(),
            after_state=after.to_dict(),
        )
        return after

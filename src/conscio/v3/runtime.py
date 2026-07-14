"""Compatibility-preserving V3 runtime layered around the V2 executor."""

from __future__ import annotations

from typing import Any

from conscio.core.runtime import CognitiveRuntime, EpisodeResult, _TickState
from conscio.core.tool_loop import ToolRequest
from conscio.core.workspace import EntryType
from conscio.v3.contracts import (
    ActionOutcome,
    ActionProposal,
    AffectiveState,
    CognitiveEvent,
    CoreCheckpoint,
)
from conscio.v3.environment import EnvironmentAdapter, TextEnvironmentAdapter
from conscio.v3.learning import AdapterState
from conscio.v3.recurrent_core import MODEL_VERSION, HybridRecurrentCore


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
        max_action_risk: float = 0.35,
        affect_min_valence: float = -0.85,
        affect_max_arousal: float = 0.90,
        affect_exposure_cycles: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.environment = environment or TextEnvironmentAdapter()
        self.recurrent_core = HybridRecurrentCore(seed=core_seed)
        self.cognitive_cycles = max(2, int(cognitive_cycles))
        self.max_action_risk = max(0.0, min(1.0, float(max_action_risk)))
        self.affect_min_valence = max(-1.0, min(0.0, float(affect_min_valence)))
        self.affect_max_arousal = max(0.0, min(1.0, float(affect_max_arousal)))
        self.affect_exposure_cycles = max(1, int(affect_exposure_cycles))
        self._unsafe_affect_cycles = 0
        self._episode_predictions: dict[str, list[dict[str, Any]]] = {}
        self._episode_affect: dict[str, list[dict[str, Any]]] = {}
        self._episode_proposals: dict[str, list[dict[str, Any]]] = {}
        self._episode_selected_proposal: dict[str, dict[str, Any]] = {}
        self._episode_initial_uncertainty: dict[str, float] = {}
        self.prediction_adapter = AdapterState(base_model_version=MODEL_VERSION)
        self.executor.authorize_tool = self._authorize_tool_proposal

    async def initialize(self) -> None:
        await super().initialize()
        latest = await self.memory.latest_core_checkpoint()
        if latest is not None:
            self.recurrent_core.restore(_checkpoint_from_dict(latest))
        adapter = await self.memory.latest_prediction_adapter()
        if adapter is not None:
            self.activate_prediction_adapter(AdapterState.from_dict(adapter["state"]))

    def activate_prediction_adapter(self, state: AdapterState) -> None:
        """Activate only an explicitly promoted adapter for this exact base model."""
        if state.base_model_version != MODEL_VERSION:
            raise ValueError(
                f"adapter base model {state.base_model_version!r} does not match {MODEL_VERSION!r}"
            )
        self.prediction_adapter = state

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
        self._episode_initial_uncertainty[ts.episode_id] = self.self_state.uncertainty
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
                data["raw_probability"] = data["probability"]
                data["probability"] = self.prediction_adapter.calibrate(prediction.probability)
                data["adapter_digest"] = self.prediction_adapter.digest()
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
            unsafe_affect = (
                cycle.affect.valence <= self.affect_min_valence
                or cycle.affect.arousal >= self.affect_max_arousal
            )
            self._unsafe_affect_cycles = self._unsafe_affect_cycles + 1 if unsafe_affect else 0
            for proposal in cycle.proposals:
                data = proposal.to_dict()
                self._episode_proposals[ts.episode_id].append(data)
                await self._append(ts.episode_id, "action_proposal", proposal.specialist, data)
        selected = self._select_proposal(self._episode_proposals[ts.episode_id])
        self._episode_selected_proposal[ts.episode_id] = selected
        await self._append(ts.episode_id, "intention_selected", "action_competition", selected)
        self.workspace.write(
            f"Selected {selected['action']}: {selected['rationale']}",
            source="v3.action_competition",
            type=EntryType.INTENTION,
            priority=7,
            confidence=float(selected.get("confidence", 0.0)),
            metadata={"proposal_id": selected["proposal_id"], "risk": selected["risk"]},
        )
        if selected["action"] == "wait":
            ts.executable = False
        if self._unsafe_affect_cycles >= self.affect_exposure_cycles:
            await self.set_safe_affect_state(
                reason="automatic recovery after sustained affect exposure limit",
                operator="runtime_safety",
                episode_id=ts.episode_id,
            )
            self._unsafe_affect_cycles = 0

    def _select_proposal(self, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [
            proposal
            for proposal in proposals
            if proposal.get("constraints_satisfied", False)
            and float(proposal.get("risk", 1.0)) <= self.max_action_risk
        ]
        if eligible:
            return max(
                eligible,
                key=lambda proposal: (
                    float(proposal.get("utility", 0.0))
                    + 0.2 * float(proposal.get("confidence", 0.0))
                    - float(proposal.get("risk", 1.0))
                ),
            )
        return {
            "proposal_id": "safety_wait",
            "specialist": "safety",
            "action": "wait",
            "rationale": "No proposal satisfied the active constraints and risk bound.",
            "expected_outcomes": [],
            "confidence": 1.0,
            "utility": 0.0,
            "risk": 0.0,
            "constraints_satisfied": True,
        }

    async def _authorize_tool_proposal(self, request: ToolRequest) -> dict[str, Any]:
        """Treat an LLM tool call as a proposal; policy and risk arbitration execute it."""
        episode_id = self.workspace.current_episode
        capabilities = set(self.executor.tools.tool_capabilities(request.name))
        capability_risk = {
            "memory_read": 0.05,
            "memory_write": 0.10,
            "self_management": 0.15,
            "external_content": 0.15,
            "local_read": 0.10,
            "local_write": 0.20,
            "self_modification": 0.25,
            "network_read": 0.20,
            "network_write": 0.30,
            "delegation": 0.25,
        }
        risk = max((capability_risk.get(item, 0.20) for item in capabilities), default=0.05)
        selected = self._episode_selected_proposal.get(episode_id, {})
        allowed = selected.get("action") != "wait" and risk <= self.max_action_risk
        reason = (
            "approved by cognitive action competition and runtime tool policy"
            if allowed
            else "proposal exceeds the active cognitive risk bound or the selected intention is wait"
        )
        typed_proposal = ActionProposal(
            specialist="llm_specialist",
            action=f"tool:{request.name}",
            rationale="Language specialist requested a tool; execution requires independent authorization.",
            expected_outcomes=(f"observable {request.name} result",),
            confidence=max(0.0, 1.0 - risk),
            utility=0.5,
            risk=risk,
            constraints_satisfied=allowed,
        )
        proposal = {
            "proposal": typed_proposal.to_dict(),
            "tool": request.name,
            "args": request.args,
            "capabilities": sorted(capabilities),
            "risk": risk,
            "allowed": allowed,
            "reason": reason,
            "selected_intention_id": selected.get("proposal_id", ""),
        }
        await self._append(episode_id, "tool_proposal", "llm_specialist", proposal)
        await self._append(episode_id, "tool_authorization", "action_competition", proposal)
        self.workspace.write(
            f"Tool proposal {request.name}: {'approved' if allowed else 'rejected'} ({reason})",
            source="v3.action_competition",
            type=EntryType.INTENTION,
            priority=8,
            confidence=1.0 - risk,
            metadata=proposal,
        )
        return {"allowed": allowed, "reason": reason}

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
        self._episode_proposals.pop(ts.episode_id, None)
        selected = self._episode_selected_proposal.pop(ts.episode_id, None)
        resolutions = await self._resolve_predictions(ts.episode_id, result)
        selected_action = str((selected or {}).get("action", ""))
        tool_failed = any(bool(item.get("error")) for item in result.tool_results)
        outcome = ActionOutcome(
            proposal_id=str((selected or {}).get("proposal_id", "executor")),
            action=result.selected_action,
            succeeded=(
                not selected_action
                or result.selected_action == selected_action
                or (selected_action == "act" and bool(result.tool_results))
            )
            and not tool_failed,
            observation=result.output[:1000],
            prediction_errors={
                **{item["prediction_id"]: item["error"] for item in resolutions if item["error"] is not None},
                "runtime_prediction_errors": float(result.metrics.prediction_errors),
            },
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
                "prediction_adapter_digest": self.prediction_adapter.digest(),
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

    async def _resolve_predictions(
        self, episode_id: str, result: EpisodeResult
    ) -> list[dict[str, Any]]:
        initial_uncertainty = self._episode_initial_uncertainty.pop(episode_id, 0.0)
        resolutions: list[dict[str, Any]] = []
        for prediction in self._episode_predictions.get(episode_id, []):
            target = prediction["target"]
            observed: bool | None
            if target == "next_observation":
                observed = bool(result.output.strip() or result.tool_results)
            elif target == "future_uncertainty" and self.ablation.self_state_coupling:
                observed = self.self_state.uncertainty <= initial_uncertainty
            else:
                observed = None
            error = (
                (float(prediction["probability"]) - float(observed)) ** 2
                if observed is not None
                else None
            )
            resolution = {
                "prediction_id": prediction["prediction_id"],
                "target": target,
                "observed": observed,
                "error": error,
                "scoring_rule": "brier",
            }
            resolutions.append(resolution)
            await self._append(episode_id, "prediction_resolution", "action_evaluation", resolution)
            prediction["resolved"] = observed is not None
            prediction["error"] = error
        return resolutions

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
        if episode_id:
            await self._append(
                episode_id,
                "affect_intervention",
                operator,
                {
                    "intervention_id": intervention_id,
                    "reason": reason,
                    "before": before.to_dict(),
                    "after": after.to_dict(),
                },
            )
        return after

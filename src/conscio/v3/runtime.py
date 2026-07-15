"""Compatibility-preserving V3 runtime layered around the V2 executor."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from conscio.core.runtime import INTERNAL_OBSERVATION_MESSAGE, CognitiveRuntime, EpisodeResult, _TickState
from conscio.core.tool_loop import ToolRequest
from conscio.core.workspace import EntryType
from conscio.tools.registry import ScopedToolRegistry
from conscio.v3.action_competition import (
    SCORER_VERSION,
    ActionCandidate,
    AffectSnapshot,
    CompetitionContext,
    ConstraintDisposition,
    LesionMask,
    NeedSnapshot,
    PredictionSignal,
    UpstreamIntention,
    canonical_digest,
    compete,
)
from conscio.v3.contracts import (
    CORE_CHECKPOINT_SCHEMA_VERSION,
    ActionOutcome,
    AffectiveState,
    CognitiveEvent,
    CoreCheckpoint,
    ExecutionIntent,
    ExecutionOutcome,
    ExecutionReconciliation,
    ExecutionRecovery,
)
from conscio.v3.environment import EnvironmentAdapter, TextEnvironmentAdapter
from conscio.v3.language_bridge import LanguageSpecialistToolLoopBridge, trace_to_dict
from conscio.v3.language_specialist import LanguageCallTrace
from conscio.v3.learning import AdapterState
from conscio.v3.recurrent_core import (
    LEGACY_SPECIALIST_ARCHITECTURE_ID,
    MODEL_VERSION,
    CoreWeightBundle,
    HybridRecurrentCore,
    migrate_legacy_specialist_checkpoint,
)


@dataclass(frozen=True, slots=True)
class _PreActionCompetitionSnapshot:
    final_cycle: int
    final_broadcast_id: str
    runtime_identity: str
    adapter_digest: str
    language_manifest_digest: str
    predictions: tuple[PredictionSignal, ...]
    prediction_channel_available: bool
    affect: AffectSnapshot
    lesions: LesionMask
    upstream_intention: UpstreamIntention
    response_constraints: tuple[ConstraintDisposition, ...]
    action_constraints: tuple[ConstraintDisposition, ...]
    response_risk: float
    risk_limit: float

    def context(self, language_response_digest: str) -> CompetitionContext:
        return CompetitionContext(
            final_cycle=self.final_cycle,
            final_broadcast_id=self.final_broadcast_id,
            runtime_identity=self.runtime_identity,
            adapter_digest=self.adapter_digest,
            language_manifest_digest=self.language_manifest_digest,
            language_response_digest=language_response_digest,
            predictions=self.predictions,
            prediction_channel_available=self.prediction_channel_available,
            affect=self.affect,
            lesions=self.lesions,
            upstream_intention=self.upstream_intention,
            response_constraints=self.response_constraints,
            response_risk=self.response_risk,
            risk_limit=self.risk_limit,
        )


def _checkpoint_from_dict(data: dict[str, Any]) -> CoreCheckpoint:
    payload = dict(data)
    schema_version = payload.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("checkpoint schema_version must be an integer")
    if schema_version == 1:
        architecture = payload.get("specialist_architecture_id")
        if architecture not in (None, LEGACY_SPECIALIST_ARCHITECTURE_ID):
            raise ValueError("legacy checkpoint declares an incompatible architecture")
        payload["specialist_architecture_id"] = LEGACY_SPECIALIST_ARCHITECTURE_ID
    elif schema_version == CORE_CHECKPOINT_SCHEMA_VERSION:
        if not payload.get("specialist_architecture_id"):
            raise ValueError("current checkpoint is missing specialist architecture identity")
    else:
        raise ValueError(f"unsupported checkpoint schema version: {schema_version}")
    payload["affect"] = AffectiveState(**payload["affect"])
    payload["deterministic_state"] = tuple(payload["deterministic_state"])
    payload["stochastic_state"] = tuple(payload["stochastic_state"])
    return CoreCheckpoint(**payload)


def _control_manifest_digest(action_kind: str, name: str) -> str:
    """Content identity for the two non-registry control contracts."""

    required_argument = "question" if action_kind == "ask" else "reason"
    return canonical_digest(
        {
            "action_kind": action_kind,
            "additional_properties": False,
            "contract_version": "conscio.v3.control-action.v1",
            "name": name,
            "required_argument": required_argument,
        }
    )


def _control_argument_error(action_kind: str, arguments: dict[str, Any]) -> str | None:
    required_argument = "question" if action_kind == "ask" else "reason"
    if set(arguments) != {required_argument}:
        return f"control action requires only {required_argument!r}"
    value = arguments.get(required_argument)
    if not isinstance(value, str) or not value.strip():
        return f"control action {required_argument!r} must be a non-empty string"
    return None


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
        recurrent_weights: CoreWeightBundle | None = None,
        restore_checkpoint_id: str | None = None,
        max_action_risk: float = 0.35,
        affect_min_valence: float = -0.85,
        affect_max_arousal: float = 0.90,
        affect_exposure_cycles: int = 8,
        strict_recurrent_workspace: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.strict_recurrent_workspace = bool(strict_recurrent_workspace)
        if self.strict_recurrent_workspace:
            # In strict research mode the legacy V2 modules and direct prompt
            # retrieval surfaces cannot independently expose memory/self state.
            # The language specialist sees those signals only when selected
            # into the recurrent workspace broadcast.
            self.modules = []
            self.prompt_assembler.memory_enabled = False
            self.prompt_assembler.self_state_enabled = False
            self.chat_strategy.memory_enabled = False
            self.autonomous_strategy.memory_enabled = False
            # Prompt retrieval is not the only memory surface: registered
            # memory tools otherwise remain in the language model's function
            # schemas.  Scope the executor itself so both schema advertisement
            # and delegated calls exclude direct reads and writes, while the
            # recurrent specialists retain their private memory access.
            self.executor.tools = ScopedToolRegistry(
                self.tools,
                denied_names=frozenset(),
                denied_capabilities=frozenset({"memory_read", "memory_write"}),
            )
        self.environment = environment or TextEnvironmentAdapter()
        self.recurrent_core = HybridRecurrentCore(seed=core_seed, weights=recurrent_weights)
        self.restore_checkpoint_id = restore_checkpoint_id
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
        self._episode_initial_need_pressure: dict[str, float] = {}
        self._episode_action_context: dict[str, _PreActionCompetitionSnapshot] = {}
        self._episode_competition_sequence: dict[str, int] = {}
        self._episode_action_competitions: dict[str, list[dict[str, Any]]] = {}
        self._episode_language_calls: dict[str, list[dict[str, Any]]] = {}
        self._execution_intents: dict[str, ExecutionIntent] = {}
        self._execution_intent_episodes: dict[str, str] = {}
        self._restart_orphan_execution_ids: set[str] = set()
        self._execution_journal_failed = False
        self._language_manifests: dict[str, dict[str, Any]] = {}
        self._language_boundaries: list[LanguageSpecialistToolLoopBridge] = []
        self.prediction_adapter = AdapterState(base_model_version=self.recurrent_core.runtime_identity)
        self.executor.authorize_tools = self._authorize_tool_proposals
        self.executor.on_execution_intent = self._record_execution_intent
        self.executor.on_execution_outcome = self._record_execution_outcome
        for boundary in (self.chat_strategy.llm, self.autonomous_strategy.llm):
            if isinstance(boundary, LanguageSpecialistToolLoopBridge):
                self.attach_language_specialist(boundary)

    def attach_language_specialist(self, boundary: LanguageSpecialistToolLoopBridge) -> None:
        """Attach a typed language boundary and persist each call in sequence."""
        if boundary not in self._language_boundaries:
            self._language_boundaries.append(boundary)
        self._language_manifests[boundary.manifest_digest] = boundary.manifest
        boundary.set_trace_observer(self._record_language_trace)

    @property
    def language_manifest_digests(self) -> tuple[str, ...]:
        return tuple(sorted(self._language_manifests))

    @property
    def execution_safe_mode(self) -> bool:
        """Whether an execution may have occurred without a durable outcome."""

        return self._execution_journal_failed or bool(self._execution_intents)

    @property
    def unresolved_execution_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._execution_intents))

    async def _record_language_trace(self, trace: LanguageCallTrace) -> None:
        episode_id = self.workspace.current_episode
        if not episode_id:
            raise RuntimeError("language specialist call occurred outside a cognitive episode")
        structured = trace_to_dict(trace)
        self._episode_language_calls.setdefault(episode_id, []).append(structured)
        await self._append(
            episode_id,
            "language_specialist_call",
            "language_specialist",
            structured,
        )

    async def initialize(self) -> None:
        await super().initialize()
        await self._recover_unresolved_executions()
        latest = (
            await self.memory.get_core_checkpoint(self.restore_checkpoint_id)
            if self.restore_checkpoint_id
            else await self.memory.latest_core_checkpoint()
        )
        if self.restore_checkpoint_id and latest is None:
            raise ValueError(f"required migrated checkpoint not found: {self.restore_checkpoint_id}")
        if latest is not None:
            checkpoint = _checkpoint_from_dict(latest)
            checkpoint = await self._resolve_specialist_checkpoint(checkpoint)
            self.recurrent_core.restore(checkpoint)
        adapter = await self.memory.latest_prediction_adapter(base_model_version=self.recurrent_core.runtime_identity)
        if adapter is not None:
            self.activate_prediction_adapter(AdapterState.from_dict(adapter["state"]))

    async def _recover_unresolved_executions(self) -> None:
        """Fail closed on intents whose dispatch result is unknown after restart."""

        await self._resync_execution_state(mark_as_restart=True)
        for execution_id, intent in tuple(self._execution_intents.items()):
            episode_id = self._execution_intent_episodes[execution_id]
            recovery = ExecutionRecovery(
                execution_id=intent.execution_id,
                intent_digest=intent.intent_digest,
            )
            await self._append(
                episode_id,
                "execution_recovery",
                "runtime_recovery",
                recovery.to_dict(),
                parent_event_id=intent.event_id,
                event_id=recovery.event_id,
                idempotent=True,
            )

    async def _resync_execution_state(self, *, mark_as_restart: bool = False) -> None:
        """Rebuild the in-memory execution gate from the authoritative journal."""

        self._execution_journal_failed = True
        unresolved = await self.memory.unresolved_execution_intents()
        intents: dict[str, ExecutionIntent] = {}
        episodes: dict[str, str] = {}
        for row in unresolved:
            payload = row.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError("execution journal returned a non-object intent")
            intent = ExecutionIntent.from_dict(payload)
            episode_id = str(row.get("episode_id") or "")
            if not episode_id:
                raise RuntimeError("execution journal intent has no episode identity")
            intents[intent.execution_id] = intent
            episodes[intent.execution_id] = episode_id
        previous_restart_ids = self._restart_orphan_execution_ids
        self._execution_intents = intents
        self._execution_intent_episodes = episodes
        self._restart_orphan_execution_ids = (
            set(intents) if mark_as_restart else previous_restart_ids.intersection(intents)
        )
        self._execution_journal_failed = False

    async def reconcile_execution(
        self,
        execution_id: str,
        *,
        reason: str,
        operator: str = "operator",
        allow_live: bool = False,
    ) -> dict[str, Any]:
        """Acknowledge an unresolved dispatch without asserting an outcome."""

        await self._resync_execution_state()
        if execution_id not in self._execution_intents:
            raise KeyError(execution_id)
        if not allow_live and execution_id not in self._restart_orphan_execution_ids:
            raise RuntimeError("only restart-recovered executions can be reconciled; restart first")
        reason = reason.strip() if isinstance(reason, str) else ""
        operator = operator.strip() if isinstance(operator, str) else ""
        if not reason:
            raise ValueError("reconciliation reason must be non-empty")
        if not operator:
            raise ValueError("reconciliation operator must be non-empty")
        intent = self._execution_intents[execution_id]
        episode_id = self._execution_intent_episodes[execution_id]
        reconciliation = ExecutionReconciliation(
            execution_id=execution_id,
            intent_digest=intent.intent_digest,
            operator=operator,
            reason=reason,
        )
        payload = reconciliation.to_dict()
        await self._append(
            episode_id,
            "execution_reconciliation",
            "operator_reconciliation",
            payload,
            parent_event_id=intent.event_id,
            event_id=reconciliation.event_id,
            idempotent=True,
        )
        self._execution_intents.pop(execution_id, None)
        self._execution_intent_episodes.pop(execution_id, None)
        self._restart_orphan_execution_ids.discard(execution_id)
        self._execution_journal_failed = False
        return payload

    async def run_episode(
        self,
        event: Any,
        *,
        should_yield: Callable[[], bool] | None = None,
    ) -> EpisodeResult:
        """Clear failed episode-local caches while retaining unresolved intents."""

        try:
            return await super().run_episode(event, should_yield=should_yield)
        except BaseException:
            try:
                await self._resync_execution_state()
            except BaseException:
                # Even cancellation during the recovery read must fail closed.
                # A later restart can rebuild the exact unresolved set.
                self._execution_journal_failed = True
            self._clear_episode_state(self.workspace.current_episode)
            raise

    def _clear_episode_state(self, episode_id: str) -> None:
        if not episode_id:
            return
        self._episode_predictions.pop(episode_id, None)
        self._episode_affect.pop(episode_id, None)
        self._episode_proposals.pop(episode_id, None)
        self._episode_selected_proposal.pop(episode_id, None)
        self._episode_initial_uncertainty.pop(episode_id, None)
        self._episode_initial_need_pressure.pop(episode_id, None)
        self._episode_action_context.pop(episode_id, None)
        self._episode_competition_sequence.pop(episode_id, None)
        self._episode_action_competitions.pop(episode_id, None)
        self._episode_language_calls.pop(episode_id, None)
        for boundary in self._language_boundaries:
            boundary.drain_traces()

    async def _resolve_specialist_checkpoint(self, checkpoint: CoreCheckpoint) -> CoreCheckpoint:
        if checkpoint.specialist_architecture_id == self.recurrent_core.specialist_architecture_id:
            return checkpoint
        if checkpoint.specialist_architecture_id != LEGACY_SPECIALIST_ARCHITECTURE_ID:
            raise ValueError("unsupported specialist checkpoint architecture")
        if checkpoint.model_version != MODEL_VERSION:
            raise ValueError(
                "trained legacy checkpoints require validated specialist-architecture "
                "migration; trained v1 migration is not enabled"
            )

        migration_episode = f"specialist_migration_{checkpoint.checkpoint_id}"
        transform = {
            "kind": "six-to-eight-private-specialists-v1",
            "source_architecture": LEGACY_SPECIALIST_ARCHITECTURE_ID,
            "target_architecture": self.recurrent_core.specialist_architecture_id,
            "neutral_initializations": (
                "observation_metadata",
                "episodic_index",
                "semantic_cues",
                "action_evaluation",
            ),
        }
        transform_digest = hashlib.sha256(
            json.dumps(transform, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        record = await self.memory.core_checkpoint_architecture_migration(checkpoint.checkpoint_id)
        if record is None:
            migrated = migrate_legacy_specialist_checkpoint(checkpoint)
            # Validate every tensor, RNG field, affect value, and specialist
            # envelope in a throwaway core before persistence.
            probe = HybridRecurrentCore(
                seed=0,
                weights=self.recurrent_core.active_weight_bundle,
            )
            probe.restore(migrated)
            record = await self.memory.migrate_core_checkpoint_architecture(
                source_checkpoint_id=checkpoint.checkpoint_id,
                source_architecture_id=checkpoint.specialist_architecture_id,
                target_checkpoint=migrated,
                runtime_identity=self.recurrent_core.runtime_identity,
                transform_digest=transform_digest,
            )
        if (
            record.source_checkpoint_id != checkpoint.checkpoint_id
            or record.source_lineage_id != checkpoint.lineage_id
            or record.source_architecture_id != checkpoint.specialist_architecture_id
            or record.model_version != checkpoint.model_version
            or record.runtime_identity != self.recurrent_core.runtime_identity
            or record.transform_digest != transform_digest
        ):
            raise ValueError("specialist architecture migration does not match its source")
        if checkpoint.model_version != self.recurrent_core.model_version:
            raise ValueError("specialist migration weights do not match the active core")
        restored_payload = await self.memory.get_core_checkpoint(record.target_checkpoint_id)
        if restored_payload is None:
            raise ValueError("specialist migration target checkpoint is missing")
        restored = _checkpoint_from_dict(restored_payload)
        if (
            restored.parent_checkpoint_id != checkpoint.checkpoint_id
            or restored.lineage_id != record.target_lineage_id
            or restored.specialist_architecture_id != record.target_architecture_id
            or restored.specialist_architecture_id != self.recurrent_core.specialist_architecture_id
        ):
            raise ValueError("specialist migration target linkage is inconsistent")
        probe = HybridRecurrentCore(
            seed=0,
            weights=self.recurrent_core.active_weight_bundle,
        )
        probe.restore(restored)

        prior_events = await self.memory.cognitive_events(migration_episode)
        migration_events = [event for event in prior_events if event["event_type"] == "checkpoint_lineage_migration"]
        if migration_events and migration_events[0]["payload"].get("migration_record_hash") != record.record_hash:
            raise ValueError("specialist migration event disagrees with the registry")
        if not migration_events:
            await self._append(
                migration_episode,
                "checkpoint_lineage_migration",
                "v3_specialist_schema_migrator",
                {
                    **record.to_dict(),
                    "transform": transform,
                    "migration_record_hash": record.record_hash,
                },
                checkpoint_id=record.target_checkpoint_id,
            )
        self.restore_checkpoint_id = record.target_checkpoint_id
        return restored

    def activate_prediction_adapter(self, state: AdapterState) -> None:
        """Activate only an explicitly promoted adapter for this exact base model."""
        if state.base_model_version != self.recurrent_core.runtime_identity:
            raise ValueError(
                f"adapter base model {state.base_model_version!r} does not match "
                f"{self.recurrent_core.runtime_identity!r}"
            )
        self.prediction_adapter = state

    async def _prepare_episode(self, ts: _TickState) -> None:
        self._episode_language_calls[ts.episode_id] = []
        self._episode_competition_sequence[ts.episode_id] = 0
        self._episode_action_competitions[ts.episode_id] = []
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
        self._episode_initial_uncertainty[ts.episode_id] = self._effective_self_state().uncertainty
        self._episode_initial_need_pressure[ts.episode_id] = self._need_pressure(self.recurrent_core.affect)
        final_cycle_proposals: list[dict[str, Any]] = []
        for cycle in results:
            if self.ablation.attention_gating:
                await self._append(
                    ts.episode_id,
                    "broadcast",
                    "recurrent_workspace",
                    cycle.broadcast.to_dict(),
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
                    f"P={float(data['probability']):.2f}: {prediction.observable}",
                    source="v3.world_model",
                    type=EntryType.PREDICTION,
                    priority=5,
                    confidence=float(data["probability"]),
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
                cycle.affect.valence <= self.affect_min_valence or cycle.affect.arousal >= self.affect_max_arousal
            )
            self._unsafe_affect_cycles = self._unsafe_affect_cycles + 1 if unsafe_affect else 0
            cycle_proposals: list[dict[str, Any]] = []
            for proposal in cycle.proposals:
                data = proposal.to_dict()
                self._episode_proposals[ts.episode_id].append(data)
                cycle_proposals.append(data)
                await self._append(ts.episode_id, "action_proposal", proposal.specialist, data)
            final_cycle_proposals = cycle_proposals
        selected = self._select_proposal(final_cycle_proposals)
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
        final_result = results[-1]
        self._episode_action_context[ts.episode_id] = self._freeze_action_context(
            ts,
            final_cycle=final_result.broadcast.cycle,
            final_broadcast_id=final_result.broadcast.broadcast_id,
            selected=selected,
        )

    def _freeze_action_context(
        self,
        ts: _TickState,
        *,
        final_cycle: int,
        final_broadcast_id: str,
        selected: dict[str, Any],
    ) -> _PreActionCompetitionSnapshot:
        """Copy the public final-cycle inputs before the language call begins."""

        adapter_digest = self.prediction_adapter.digest()
        predictions = tuple(
            PredictionSignal(
                target=str(item["target"]),
                available=not (str(item["target"]) == "future_uncertainty" and not self.ablation.self_state_coupling),
                source="v3.world_model.final_cycle",
                prediction_id=str(item["prediction_id"]),
                basis_broadcast_id=str(item["basis_broadcast_id"]),
                cycle=final_cycle,
                raw_probability=float(item["raw_probability"]),
                calibrated_probability=float(item["probability"]),
                adapter_digest=str(item["adapter_digest"]),
            )
            for item in self._episode_predictions[ts.episode_id]
            if item.get("basis_broadcast_id") == final_broadcast_id
        )
        affect = self.recurrent_core.affect
        needs = affect.need_errors
        affect_snapshot = AffectSnapshot(
            available=True,
            valence=float(affect.valence),
            arousal=float(affect.arousal),
            controllability=float(affect.controllability),
            needs=NeedSnapshot(
                epistemic_coherence=float(needs["epistemic_coherence"]),
                competence=float(needs["competence"]),
                integrity=float(needs["integrity"]),
                social_interaction=float(needs["social_interaction"]),
                continuity_of_memory=float(needs["continuity_of_memory"]),
            ),
            source="v3.public_affect.final_cycle",
            basis_broadcast_id=final_broadcast_id,
            intervention_id=affect.intervention_id,
        )
        response_constraints: list[ConstraintDisposition] = []
        action_constraints: list[ConstraintDisposition] = []
        for constraint in ts.constraints:
            constraint_id = str(getattr(constraint, "constraint_id", "active_constraint"))
            kind = str(getattr(constraint, "kind", "semantic"))
            response_constraints.append(
                ConstraintDisposition(
                    constraint_id=constraint_id,
                    satisfied=False,
                    hard=False,
                    source=f"deferred_{kind}_response_validation",
                )
            )
            action_constraints.append(
                ConstraintDisposition(
                    constraint_id=constraint_id,
                    satisfied=kind != "semantic",
                    hard=kind == "semantic",
                    source=(
                        "unresolved_semantic_action_constraint"
                        if kind == "semantic"
                        else "response_only_structural_constraint"
                    ),
                )
            )
        active_llm = self.autonomous_strategy.llm if ts.event.source == "autonomous" else self.chat_strategy.llm
        manifest_digest = str(getattr(active_llm, "manifest_digest", "untracked-language-client"))
        return _PreActionCompetitionSnapshot(
            final_cycle=final_cycle,
            final_broadcast_id=final_broadcast_id,
            runtime_identity=self.recurrent_core.runtime_identity,
            adapter_digest=adapter_digest,
            language_manifest_digest=manifest_digest,
            predictions=predictions,
            prediction_channel_available=self.ablation.prediction,
            affect=affect_snapshot,
            lesions=LesionMask(
                prediction=not self.ablation.prediction,
                self_model=not self.ablation.self_state_coupling,
                affect=False,
                broadcast=not self.ablation.attention_gating,
                memory=not self.ablation.memory_retrieval,
            ),
            upstream_intention=UpstreamIntention(
                available=True,
                action=str(selected["action"]),
                proposal_id=str(selected["proposal_id"]),
                specialist=str(selected["specialist"]),
                source="v3.recurrent_action_competition",
                basis_broadcast_id=final_broadcast_id,
                cycle=final_cycle,
            ),
            response_constraints=tuple(response_constraints),
            action_constraints=tuple(action_constraints),
            response_risk=0.02,
            risk_limit=self.max_action_risk,
        )

    def _select_proposal(self, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [
            proposal
            for proposal in proposals
            if proposal.get("constraints_satisfied", False) and float(proposal.get("risk", 1.0)) <= self.max_action_risk
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

    async def _authorize_tool_proposals(
        self,
        requests: tuple[tuple[str, ToolRequest], ...],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist and score one complete language action turn.

        This is cognitive authorization only. The registry remains the final
        policy gate, and no execution intent is written until the tool loop has
        independently normalized this returned decision.
        """

        episode_id = self.workspace.current_episode
        if not episode_id:
            raise RuntimeError("language action competition occurred outside an episode")
        snapshot = self._episode_action_context.get(episode_id)
        if snapshot is None:
            raise RuntimeError("language action competition has no frozen recurrent context")
        call_ids = [call_id for call_id, _ in requests]
        if any(not isinstance(call_id, str) or not call_id.strip() for call_id in call_ids):
            raise ValueError("language proposal call IDs must be non-empty strings")
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("language proposal call IDs must be unique")

        sequence = self._episode_competition_sequence.get(episode_id, 0) + 1
        self._episode_competition_sequence[episode_id] = sequence
        try:
            known_tools = set(self.executor.tools.list_tools())
        except Exception:  # noqa: BLE001 - unavailable schema metadata must fail closed
            known_tools = set()
        control_kinds = {"ask_user": "ask", "refuse": "refuse"}
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

        drafts: list[dict[str, Any]] = []
        identity_counts: dict[str, int] = {}
        for call_id, request in requests:
            tool_manifest_digest: str | None
            control_kind = control_kinds.get(request.name)
            if control_kind is not None:
                action_kind = control_kind
                capabilities: tuple[str, ...] = ()
                risk = 0.02
                advertised = True
                capability_metadata_available = True
                preparation_available = True
                schema_error = _control_argument_error(action_kind, request.args)
                tool_manifest_digest = _control_manifest_digest(action_kind, request.name)
            else:
                action_kind = "tool"
                prepare_call = getattr(self.executor.tools, "prepare_call", None)
                try:
                    if not callable(prepare_call):
                        raise RuntimeError("tool registry has no argument preparation gate")
                    request.args = prepare_call(request.name, request.args)
                    preparation_available = True
                except Exception:  # noqa: BLE001 - invalid effective arguments fail closed
                    preparation_available = False
                try:
                    capabilities = tuple(
                        sorted(str(item) for item in self.executor.tools.tool_capabilities(request.name))
                    )
                    capability_metadata_available = True
                except Exception:  # noqa: BLE001 - metadata absence fails closed below
                    capabilities = ()
                    capability_metadata_available = False
                risk = max(
                    (capability_risk.get(item, 0.20) for item in capabilities),
                    default=0.05,
                )
                policy_gate = getattr(self.executor.tools, "policy_permits", None)
                try:
                    policy_eligible = (
                        bool(policy_gate(request.name)) if callable(policy_gate) else request.name in known_tools
                    )
                except Exception:  # noqa: BLE001 - an unavailable policy gate denies
                    policy_eligible = False
                advertised = request.name in known_tools and policy_eligible
                validate_arguments = getattr(self.executor.tools, "validate_tool_arguments", None)
                try:
                    schema_error = (
                        validate_arguments(request.name, request.args)
                        if callable(validate_arguments) and preparation_available
                        else "tool registry has no argument validation gate"
                    )
                except Exception as exc:  # noqa: BLE001 - unavailable validation denies
                    schema_error = f"argument validation unavailable: {type(exc).__name__}"
                manifest_digest = getattr(self.executor.tools, "tool_manifest_digest", None)
                try:
                    tool_manifest_digest = manifest_digest(request.name) if callable(manifest_digest) else None
                except Exception:  # noqa: BLE001 - unavailable manifest denies
                    tool_manifest_digest = None
            identity = {
                "action_kind": action_kind,
                "arguments": request.args,
                "name": request.name,
            }
            identity_digest = canonical_digest(identity)
            identity_counts[identity_digest] = identity_counts.get(identity_digest, 0) + 1
            drafts.append(
                {
                    "action_kind": action_kind,
                    "advertised": advertised,
                    "call_id": call_id,
                    "capability_metadata_available": capability_metadata_available,
                    "capabilities": capabilities,
                    "identity_digest": identity_digest,
                    "preparation_available": preparation_available,
                    "request": request,
                    "risk": risk,
                    "schema_error": schema_error,
                    "tool_manifest_digest": tool_manifest_digest,
                }
            )

        language_calls = self._episode_language_calls.get(episode_id, [])
        raw_language_response_digest = (
            str(language_calls[-1]["response_digest"]) if language_calls else canonical_digest(response)
        )
        # Provider call IDs and raw envelope details are provenance, not action
        # identity. The scorer consumes only exact visible content plus the
        # canonical semantic identities of inert action proposals.
        language_response_digest = canonical_digest(
            {
                "content": str(response.get("content") or ""),
                "action_identities": sorted(draft["identity_digest"] for draft in drafts),
            }
        )
        context = snapshot.context(language_response_digest)
        if not str(response.get("content") or "").strip():
            # A provider turn containing only inert tool proposals is not itself
            # a response candidate. The built-in respond action remains in the
            # competition trace but is hard-ineligible until a later no-tool
            # completion actually supplies response content.
            context = replace(
                context,
                response_constraints=(
                    *context.response_constraints,
                    ConstraintDisposition(
                        constraint_id="language_response_content_present",
                        satisfied=False,
                        hard=True,
                        source="language_response_boundary",
                    ),
                ),
            )

        candidates: list[ActionCandidate] = []
        proposal_records: list[dict[str, Any]] = []
        proposal_ids: dict[str, str] = {}
        action_digest_by_call: dict[str, str] = {}
        for draft in drafts:
            request = draft["request"]
            call_id = str(draft["call_id"])
            eligibility = ConstraintDisposition(
                constraint_id=f"advertised_schema:{request.name}",
                satisfied=bool(draft["advertised"]),
                hard=True,
                source="language_tool_schema",
            )
            uniqueness = ConstraintDisposition(
                constraint_id=f"unique_action_identity:{draft['identity_digest']}",
                satisfied=identity_counts[str(draft["identity_digest"])] == 1,
                hard=True,
                source="parallel_proposal_set",
            )
            argument_schema = ConstraintDisposition(
                constraint_id=f"arguments_schema:{request.name}",
                satisfied=draft["schema_error"] is None,
                hard=True,
                source="tool_schema_validation",
            )
            if draft["action_kind"] == "tool":
                capability_metadata = ConstraintDisposition(
                    constraint_id=f"capability_metadata:{request.name}",
                    satisfied=bool(draft["capability_metadata_available"]),
                    hard=True,
                    source="tool_registry_metadata",
                )
                execution_safe_mode = ConstraintDisposition(
                    constraint_id="execution_safe_mode_clear",
                    satisfied=not self.execution_safe_mode,
                    hard=True,
                    source="durable_execution_journal",
                )
                argument_preparation = ConstraintDisposition(
                    constraint_id=f"effective_arguments:{request.name}",
                    satisfied=bool(draft["preparation_available"]),
                    hard=True,
                    source="tool_registry_dispatch_gate",
                )
                manifest_available = ConstraintDisposition(
                    constraint_id=f"tool_manifest:{request.name}",
                    satisfied=isinstance(draft["tool_manifest_digest"], str),
                    hard=True,
                    source="tool_registry_manifest",
                )
                candidate = ActionCandidate.tool(
                    request.name,
                    request.args,
                    risk=float(draft["risk"]),
                    capabilities=draft["capabilities"],
                    constraints=(
                        *snapshot.action_constraints,
                        eligibility,
                        uniqueness,
                        capability_metadata,
                        argument_preparation,
                        argument_schema,
                        manifest_available,
                        execution_safe_mode,
                    ),
                    provider_call_id=call_id,
                )
            else:
                # Clarification/refusal do not dispatch through the external
                # registry, so unresolved tool state and semantic action gates
                # must not remove these safe control alternatives.
                candidate = ActionCandidate.control(
                    draft["action_kind"],
                    name=request.name,
                    arguments=request.args,
                    risk=float(draft["risk"]),
                    capabilities=draft["capabilities"],
                    constraints=(eligibility, uniqueness, argument_schema),
                    provider_call_id=call_id,
                )
            action_digest = canonical_digest(candidate.identity_dict(language_response_digest=language_response_digest))
            proposal_id = canonical_digest(
                {
                    "action_digest": action_digest,
                    "call_id": call_id,
                    "competition_sequence": sequence,
                    "context_digest": context.context_digest,
                    "kind": "language_action_proposal",
                }
            )
            candidates.append(candidate)
            proposal_ids[call_id] = proposal_id
            action_digest_by_call[call_id] = action_digest
            proposal_records.append(
                {
                    "action_digest": action_digest,
                    "call_id": call_id,
                    "candidate": candidate.to_dict(),
                    "competition_sequence": sequence,
                    "context_digest": context.context_digest,
                    "epistemic_status": "idea",
                    "language_response_digest": language_response_digest,
                    "raw_language_response_digest": raw_language_response_digest,
                    "proposal_id": proposal_id,
                    "schema_validation_error": draft["schema_error"],
                    "tool_manifest_digest": draft["tool_manifest_digest"],
                }
            )

        competition = compete(context, candidates)
        selected_kind = competition.selected_action_kind
        selected_call_id: str | None = None
        selected_proposal_id: str | None = None
        selected_tool_manifest_digest: str | None = None
        execution_id: str | None = None
        if selected_kind in {"tool", "ask", "refuse"}:
            if len(competition.selected_provider_call_ids) != 1:
                raise RuntimeError("selected external action does not identify exactly one proposal")
            selected_call_id = competition.selected_provider_call_ids[0]
            if selected_call_id not in proposal_ids:
                raise RuntimeError("selected external action refers to an unknown proposal")
            selected_proposal_id = proposal_ids[selected_call_id]
            selected_draft = next(draft for draft in drafts if draft["call_id"] == selected_call_id)
            selected_tool_manifest_digest = selected_draft["tool_manifest_digest"]
            if not isinstance(selected_tool_manifest_digest, str):
                raise RuntimeError("selected external action has no frozen tool manifest")
            execution_material = canonical_digest(
                {
                    "action_digest": competition.selected_action_digest,
                    "competition_sequence": sequence,
                    "context_digest": context.context_digest,
                    "kind": "execution",
                    "runtime_identity": self.recurrent_core.runtime_identity,
                }
            )
            execution_id = f"exec_{execution_material.removeprefix('sha256:')}"
        action = (
            "tool" if selected_kind == "tool" else "control" if selected_kind in {"ask", "refuse"} else selected_kind
        )
        competition_id = canonical_digest(
            {
                "competition_sequence": sequence,
                "context_digest": context.context_digest,
                "kind": "language_action_competition",
                "selected_action_digest": competition.selected_action_digest,
            }
        )
        reason = {
            "tool": "cognitive competition selected a tool; registry policy remains pending",
            "control": "cognitive competition selected a control action",
            "respond": "cognitive competition selected a response; no tool was authorized",
            "wait": "cognitive competition selected wait; no tool was authorized",
        }[action]

        for proposal in proposal_records:
            await self._append(episode_id, "tool_proposal", "language_specialist", proposal)
        await self._append(
            episode_id,
            "pre_action_forecast",
            "action_competition",
            {
                "competition_id": competition_id,
                "competition_sequence": sequence,
                "context": context.to_dict(),
                "context_digest": context.context_digest,
                "forecasts": [
                    {
                        "action_digest": ranking.action_digest,
                        "action_kind": ranking.action_kind,
                        "name": ranking.name,
                        "prediction": ranking.to_dict()["prediction"],
                    }
                    for ranking in competition.rankings
                ],
            },
        )
        await self._append(
            episode_id,
            "action_competition",
            "action_competition",
            {
                **competition.to_dict(),
                "competition_id": competition_id,
                "competition_sequence": sequence,
            },
        )
        self._episode_action_competitions.setdefault(episode_id, []).append(
            {
                "competition_id": competition_id,
                "competition_sequence": sequence,
                "context_digest": context.context_digest,
                "scorer_version": competition.scorer_version,
                "selected_action_digest": competition.selected_action_digest,
                "selected_action_kind": competition.selected_action_kind,
                "selected_name": competition.selected_name,
                "selected_proposal_id": selected_proposal_id,
            }
        )
        await self._append(
            episode_id,
            "intention_selected",
            "action_competition",
            {
                "action": action,
                "action_digest": competition.selected_action_digest,
                "action_kind": selected_kind,
                "competition_id": competition_id,
                "competition_sequence": sequence,
                "context_digest": context.context_digest,
                "name": competition.selected_name,
                "proposal_id": selected_proposal_id,
                "reason": reason,
                "selected_call_id": selected_call_id,
                "stage": "language_action_competition",
            },
        )
        ranking_by_digest = {ranking.action_digest: ranking for ranking in competition.rankings}
        for proposal in proposal_records:
            call_id = str(proposal["call_id"])
            cognitively_authorized = selected_call_id == call_id and action in {"tool", "control"}
            candidate_kind = str(proposal["candidate"]["action_kind"])
            policy_status = (
                "pending_final_gate"
                if cognitively_authorized and candidate_kind == "tool"
                else "not_applicable"
                if cognitively_authorized
                else "not_evaluated"
            )
            ranking = ranking_by_digest[action_digest_by_call[call_id]]
            await self._append(
                episode_id,
                "tool_authorization",
                "action_competition",
                {
                    "action_digest": proposal["action_digest"],
                    "allowed": cognitively_authorized,
                    "call_id": call_id,
                    "cognitively_authorized": cognitively_authorized,
                    "competition_id": competition_id,
                    "eligible": ranking.eligible,
                    "ineligibility_reasons": list(ranking.ineligibility_reasons),
                    "policy_status": policy_status,
                    "proposal_id": proposal["proposal_id"],
                    "reason": reason if cognitively_authorized else "another action won competition",
                },
            )

        self.workspace.write(
            f"Language action selected {competition.selected_name}: {reason}",
            source="v3.action_competition",
            type=EntryType.INTENTION,
            priority=8,
            confidence=1.0,
            metadata={
                "action_digest": competition.selected_action_digest,
                "competition_id": competition_id,
                "proposal_id": selected_proposal_id,
                "selected_call_id": selected_call_id,
            },
        )
        decision: dict[str, Any] = {
            "action": action,
            "competition_id": competition_id,
            "competition_sequence": sequence,
            "context_digest": context.context_digest,
            "proposal_ids": proposal_ids,
            "reason": reason,
            "selected_action_digest": competition.selected_action_digest,
            "selected_action_kind": selected_kind,
            "selected_call_id": selected_call_id,
        }
        if selected_call_id is not None:
            decision["execution_id"] = execution_id
            decision["proposal_id"] = selected_proposal_id
            decision["tool_manifest_digest"] = selected_tool_manifest_digest
        return decision

    def _current_tool_dispatch_identity(
        self,
        request: ToolRequest,
    ) -> tuple[dict[str, Any], str]:
        """Validate and return the registry's current effective call identity."""

        prepare_call = getattr(self.executor.tools, "prepare_call", None)
        validate_arguments = getattr(
            self.executor.tools,
            "validate_tool_arguments",
            None,
        )
        manifest_digest = getattr(
            self.executor.tools,
            "tool_manifest_digest",
            None,
        )
        if not callable(prepare_call) or not callable(validate_arguments) or not callable(manifest_digest):
            raise RuntimeError("tool dispatch metadata gate is unavailable")
        prepared_arguments = prepare_call(request.name, request.args)
        schema_error = validate_arguments(request.name, request.args)
        if schema_error is not None:
            raise RuntimeError(f"tool arguments fail the dispatch schema: {schema_error}")
        current_manifest_digest = manifest_digest(request.name)
        if not isinstance(current_manifest_digest, str):
            raise RuntimeError("tool dispatch manifest is unavailable")
        return prepared_arguments, current_manifest_digest

    async def _record_execution_intent(
        self,
        request: ToolRequest,
        decision: dict[str, Any],
    ) -> None:
        """Durably record normalized intent immediately before dispatch."""

        episode_id = self.workspace.current_episode
        if not episode_id:
            raise RuntimeError("execution intent occurred outside an episode")
        raw_action_kind = str(decision.get("selected_action_kind") or "")
        expected_action_kind = {"ask_user": "ask", "refuse": "refuse"}.get(
            request.name,
            "tool",
        )
        if raw_action_kind != expected_action_kind:
            raise RuntimeError("selected action kind differs from the dispatch request")
        action_kind = cast(Literal["tool", "ask", "refuse"], raw_action_kind)
        expected_manifest_digest = str(decision.get("tool_manifest_digest") or "")
        if action_kind == "tool":
            prepared_arguments, current_manifest_digest = self._current_tool_dispatch_identity(request)
            if prepared_arguments != request.args:
                raise RuntimeError("effective tool arguments changed after competition")
        else:
            schema_error = _control_argument_error(action_kind, request.args)
            if schema_error is not None:
                raise RuntimeError(f"control arguments changed after competition: {schema_error}")
            current_manifest_digest = _control_manifest_digest(action_kind, request.name)
        if current_manifest_digest != expected_manifest_digest:
            raise RuntimeError("tool manifest changed after competition; dispatch denied")
        try:
            capabilities = tuple(sorted(str(item) for item in self.executor.tools.tool_capabilities(request.name)))
        except Exception:  # noqa: BLE001 - frozen empty metadata is conservative
            capabilities = ()
        arguments_json = json.dumps(
            request.args,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        intent = ExecutionIntent(
            execution_id=str(decision["execution_id"]),
            proposal_id=str(decision["proposal_id"]),
            action_digest=str(decision["selected_action_digest"]),
            context_digest=str(decision["context_digest"]),
            runtime_identity=self.recurrent_core.runtime_identity,
            competition_sequence=int(decision["competition_sequence"]),
            action_kind=action_kind,
            tool_manifest_digest=expected_manifest_digest,
            tool=request.name,
            arguments_json=arguments_json,
            capabilities=capabilities,
        )
        if intent.execution_id in self._execution_intents:
            raise RuntimeError("execution intent is already unresolved; duplicate dispatch denied")
        append_result = await self._append(
            episode_id,
            "execution_intent",
            "runtime_executor",
            intent.to_dict(),
            event_id=intent.event_id,
            idempotent=True,
        )
        if getattr(append_result, "inserted", False) is not True:
            raise RuntimeError("execution intent already exists; duplicate dispatch denied")
        self._execution_intents[intent.execution_id] = intent
        self._execution_intent_episodes[intent.execution_id] = episode_id
        if action_kind == "tool":
            try:
                prepared_arguments, current_manifest_digest = self._current_tool_dispatch_identity(request)
                if prepared_arguments != request.args or current_manifest_digest != intent.tool_manifest_digest:
                    raise RuntimeError("tool dispatch identity changed")
            except Exception as exc:
                await self._record_execution_outcome(
                    request,
                    {
                        "output": f"Dispatch denied after intent persistence: {exc}",
                        "error": True,
                        "executed": False,
                        "policy_denied": False,
                        "dispatch_identity_changed": True,
                        "execution_id": intent.execution_id,
                        "proposal_id": intent.proposal_id,
                        "call_id": str(decision["selected_call_id"]),
                    },
                )
                raise RuntimeError("tool dispatch identity changed after intent persistence; dispatch denied") from exc

    async def _record_execution_outcome(
        self,
        request: ToolRequest,
        result: dict[str, Any],
    ) -> None:
        """Persist the selected action outcome before workspace exposure."""

        episode_id = self.workspace.current_episode
        if not episode_id:
            raise RuntimeError("execution outcome occurred outside an episode")
        reported_executed = result.get("executed") is True
        policy_denied = result.get("policy_denied") is True
        executed = reported_executed and not policy_denied
        arguments = json.loads(
            json.dumps(
                request.args,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        execution_id = str(result.get("execution_id") or "")
        proposal_id = str(result.get("proposal_id") or "")
        call_id = str(result.get("call_id") or "")
        if not execution_id or not proposal_id or not call_id:
            raise RuntimeError("selected action outcome is missing execution identity")
        intent = self._execution_intents.get(execution_id)
        if intent is None:
            raise RuntimeError("selected action outcome has no durable execution intent")
        if (
            intent.proposal_id != proposal_id
            or intent.tool != request.name
            or intent.arguments != arguments
            or self._execution_intent_episodes.get(execution_id) != episode_id
        ):
            raise RuntimeError("selected action outcome differs from its execution intent")
        control = intent.action_kind in {"ask", "refuse"}
        if result.get("execution_unknown") is True:
            if not executed or control:
                raise RuntimeError("only an invoked external tool may have an unknown execution outcome")
            await self._append(
                episode_id,
                "execution_uncertain",
                "runtime_executor",
                {
                    "execution_id": execution_id,
                    "intent_digest": intent.intent_digest,
                    "proposal_id": proposal_id,
                    "action_digest": intent.action_digest,
                    "tool": request.name,
                    "disposition": "execution_unknown",
                    "executed": None,
                    "succeeded": None,
                    "learning_eligible": False,
                    "reason_code": "remote_outcome_unknown",
                    "replay_policy": "never_auto_retry",
                },
                parent_event_id=intent.event_id,
            )
            return
        succeeded: bool | None = result.get("error") is not True if executed else None
        disposition: Literal["not_executed", "succeeded", "failed"] = (
            "not_executed" if not executed else "succeeded" if succeeded else "failed"
        )
        reason_code = (
            "policy_denied"
            if policy_denied
            else "dispatch_identity_changed"
            if result.get("dispatch_identity_changed") is True
            else "not_executed"
            if not executed
            else "control_completed"
            if control and succeeded
            else "tool_succeeded"
            if succeeded
            else "tool_failed"
        )
        result_digest = canonical_digest(
            {
                "error": result.get("error") is True,
                "executed": executed,
                "output_digest": canonical_digest(str(result.get("output") or "")),
                "policy_denied": policy_denied,
                "reason_code": reason_code,
            }
        )
        outcome = ExecutionOutcome(
            execution_id=execution_id,
            intent_digest=intent.intent_digest,
            proposal_id=proposal_id,
            action_digest=intent.action_digest,
            tool=request.name,
            executed=executed,
            succeeded=succeeded,
            disposition=disposition,
            result_digest=result_digest,
            reason_code=reason_code,
        )
        await self._append(
            episode_id,
            "execution_outcome",
            "runtime_executor",
            outcome.to_dict(),
            parent_event_id=intent.event_id,
            event_id=outcome.event_id,
            idempotent=True,
        )
        self._execution_intents.pop(execution_id, None)
        self._execution_intent_episodes.pop(execution_id, None)
        self._restart_orphan_execution_ids.discard(execution_id)
        if executed and not control:
            status = "success" if succeeded else "error"
            await self._append(
                episode_id,
                "tool_outcome",
                "runtime_executor",
                {
                    "arguments": arguments,
                    "call_id": call_id,
                    "executed": True,
                    "execution_id": execution_id,
                    "proposal_id": proposal_id,
                    "status": status,
                    "succeeded": bool(succeeded),
                    "tool": request.name,
                },
            )
            await self._append(
                episode_id,
                "observation",
                "tool_executor",
                {
                    "content": f"tool_outcome:{request.name}:{status}",
                    "content_kind": "typed_execution_status",
                    "execution_id": execution_id,
                },
            )

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
        event_id: str | None = None,
        idempotent: bool = False,
    ) -> Any:
        event_kwargs: dict[str, Any] = {}
        if event_id is not None:
            event_kwargs["event_id"] = event_id
        event = CognitiveEvent(
            event_type=event_type,
            source=source,
            payload=payload,
            episode_id=episode_id,
            parent_event_id=parent_event_id,
            model_input=model_input,
            checkpoint_id=checkpoint_id,
            **event_kwargs,
        )
        if idempotent:
            return await self.memory.append_cognitive_event_idempotent(event)
        return await self.memory.append_cognitive_event(event)

    async def _finalize_episode(self, ts: _TickState, start: float) -> EpisodeResult:
        result = await super()._finalize_episode(ts, start)
        language_calls = self._episode_language_calls.pop(ts.episode_id, [])
        action_competitions = self._episode_action_competitions.pop(ts.episode_id, [])
        for boundary in self._language_boundaries:
            boundary.drain_traces()
        self._episode_proposals.pop(ts.episode_id, None)
        selected = self._episode_selected_proposal.pop(ts.episode_id, None)
        resolutions = await self._resolve_predictions(ts.episode_id, result)
        evaluation = self._action_evaluation(result)
        observed_competition = self._observed_action_competition(
            result,
            action_competitions,
            evaluation,
        )
        outcome_competition = observed_competition or (action_competitions[-1] if action_competitions else {})
        selected_action_kind = str(outcome_competition.get("selected_action_kind") or "")
        outcome_action = (
            "answer"
            if selected_action_kind == "respond"
            else selected_action_kind
            if selected_action_kind
            else result.selected_action
        )
        outcome_proposal_id = str(
            outcome_competition.get("selected_proposal_id")
            or outcome_competition.get("selected_action_digest")
            or (selected or {}).get("proposal_id")
            or "executor"
        )
        outcome = ActionOutcome(
            proposal_id=outcome_proposal_id,
            action=outcome_action,
            succeeded=bool(evaluation["action_succeeded"]),
            observation=result.output[:1000],
            prediction_errors={
                **{item["prediction_id"]: item["error"] for item in resolutions if item["error"] is not None},
                "runtime_prediction_errors": float(result.metrics.prediction_errors),
            },
        )
        outcome_payload = {
            **outcome.to_dict(),
            "observed": evaluation["action_observed"],
            "learning_eligible": bool(evaluation["action_observed"] and observed_competition),
            "competition_id": outcome_competition.get("competition_id"),
            "selected_action_digest": outcome_competition.get("selected_action_digest"),
            "selected_action_kind": outcome_competition.get("selected_action_kind"),
        }
        await self._append(ts.episode_id, "action_outcome", "environment", outcome_payload)
        checkpoint = self.recurrent_core.checkpoint()
        await self.memory.save_core_checkpoint(checkpoint)
        await self._append(
            ts.episode_id,
            "checkpoint",
            "recurrent_core",
            {
                "lineage_id": checkpoint.lineage_id,
                "model_version": checkpoint.model_version,
                "specialist_architecture_id": checkpoint.specialist_architecture_id,
                "runtime_identity": self.recurrent_core.runtime_identity,
                "language_manifest_digests": list(self.language_manifest_digests),
                "action_scorer_version": SCORER_VERSION,
            },
            checkpoint_id=checkpoint.checkpoint_id,
            model_input={
                "dynamic_context": result.model_context,
                "calls": self.executor.model_inputs,
                "language_calls": language_calls,
                "language_manifests": [self._language_manifests[digest] for digest in sorted(self._language_manifests)],
                "runtime_identity": self.recurrent_core.runtime_identity,
                "prediction_adapter_digest": self.prediction_adapter.digest(),
                "action_competitions": action_competitions,
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
        result.action_outcomes = [outcome_payload]
        result.exact_model_inputs = (
            [dict(call["request"]) for call in language_calls] if language_calls else self.executor.model_inputs
        )
        self._episode_action_context.pop(ts.episode_id, None)
        self._episode_competition_sequence.pop(ts.episode_id, None)
        return result

    async def _resolve_predictions(self, episode_id: str, result: EpisodeResult) -> list[dict[str, Any]]:
        initial_uncertainty = self._episode_initial_uncertainty.pop(episode_id, 0.0)
        initial_need_pressure = self._episode_initial_need_pressure.pop(episode_id, 0.0)
        current_uncertainty = self._effective_self_state().uncertainty
        evaluation = self._action_evaluation(result)
        tool_attempted = bool(evaluation["tool_attempted"])
        tool_succeeded = bool(evaluation["tool_succeeded"])
        action_observed = bool(evaluation["action_observed"])
        action_succeeded = bool(evaluation["action_succeeded"])
        before_affect = self.recurrent_core.affect
        after_affect = (
            self.recurrent_core.apply_action_feedback(
                succeeded=action_succeeded,
                uncertainty_delta=current_uncertainty - initial_uncertainty,
            )
            if action_observed
            else before_affect
        )
        affect_payload = {
            **after_affect.to_dict(),
            "phase": "action_outcome",
            "outcome_observed": action_observed,
            "learning_eligible": action_observed,
            "succeeded": action_succeeded if action_observed else None,
            "before_need_pressure": self._need_pressure(before_affect),
            "after_need_pressure": self._need_pressure(after_affect),
        }
        self._episode_affect.setdefault(episode_id, []).append(affect_payload)
        await self._append(episode_id, "affect", "action_evaluation", affect_payload)
        resolutions: list[dict[str, Any]] = []
        for prediction in self._episode_predictions.get(episode_id, []):
            target = prediction["target"]
            observed: bool | None
            if target == "next_observation":
                # Core finalization supplies a user-facing placeholder for
                # WAIT/internal episodes. It is not an environmental
                # observation and must never resolve a prediction. Only an
                # independently observed response or known tool result counts.
                observed = True if action_observed else None
            elif target == "tool_outcome":
                observed = tool_succeeded if tool_attempted else None
            elif target == "action_effect":
                observed = action_succeeded if action_observed else None
            elif target == "homeostatic_affect_change":
                observed = (
                    self._need_pressure(self.recurrent_core.affect) <= initial_need_pressure
                    if action_observed
                    else None
                )
            elif target == "future_uncertainty" and self.ablation.self_state_coupling and action_observed:
                observed = current_uncertainty <= initial_uncertainty
            else:
                observed = None
            error = (float(prediction["probability"]) - float(observed)) ** 2 if observed is not None else None
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

    @staticmethod
    def _action_evaluation(result: EpisodeResult) -> dict[str, bool]:
        known_results = [
            item
            for item in result.tool_results
            if item.get("executed") is True and item.get("execution_unknown") is not True
        ]
        tool_attempted = bool(known_results)
        tool_succeeded = tool_attempted and not any(item.get("error") is True for item in known_results)
        output = result.output.strip()
        response_observed = (
            result.selected_action in {"answer", "ask", "refuse"}
            and bool(output)
            and output != INTERNAL_OBSERVATION_MESSAGE
        )
        action_observed = response_observed or tool_attempted
        action_succeeded = response_observed or (tool_attempted and tool_succeeded)
        return {
            "tool_attempted": tool_attempted,
            "tool_succeeded": tool_succeeded,
            "response_observed": response_observed,
            "action_observed": action_observed,
            "action_succeeded": action_succeeded,
        }

    @staticmethod
    def _observed_action_competition(
        result: EpisodeResult,
        competitions: list[dict[str, Any]],
        evaluation: dict[str, bool],
    ) -> dict[str, Any]:
        """Join an observed terminal action to the competition that selected it."""

        if evaluation["response_observed"]:
            expected_kind = {
                "answer": "respond",
                "ask": "ask",
                "refuse": "refuse",
            }.get(result.selected_action)
            if expected_kind is not None:
                return next(
                    (
                        competition
                        for competition in reversed(competitions)
                        if competition.get("selected_action_kind") == expected_kind
                    ),
                    {},
                )
        if evaluation["tool_attempted"]:
            observed_proposal_ids = {
                str(item.get("proposal_id") or "")
                for item in result.tool_results
                if item.get("executed") is True and item.get("execution_unknown") is not True
            }
            return next(
                (
                    competition
                    for competition in reversed(competitions)
                    if competition.get("selected_action_kind") == "tool"
                    and str(competition.get("selected_proposal_id") or "") in observed_proposal_ids
                ),
                {},
            )
        return {}

    @staticmethod
    def _need_pressure(state: AffectiveState) -> float:
        values = tuple(abs(float(value)) for value in state.need_errors.values())
        return sum(values) / max(1, len(values))

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

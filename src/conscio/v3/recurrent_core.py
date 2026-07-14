"""Small checkpointable hybrid recurrent core used by the V3 runtime.

This is an inference core, not a claim that useful weights have already been
trained.  Its state transition is explicit and replayable; fixed initial
weights are a safe bootstrap for collecting the prediction/outcome curriculum
needed by the later training milestone.
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from conscio.v3.contracts import (
    CORE_CHECKPOINT_SCHEMA_VERSION,
    ActionProposal,
    AffectiveState,
    Broadcast,
    CandidateContent,
    CognitiveEvent,
    CoreCheckpoint,
    Prediction,
)
from conscio.v3.specialists import (
    SPECIALIST_ARCHITECTURE_ID,
    SpecialistFactory,
    SpecialistInput,
    SpecialistRegistry,
    stable_identifier,
)

MODEL_VERSION = "v3-bootstrap-rssm-1"
STATE_SIZE = 24
LEGACY_SPECIALIST_ARCHITECTURE_ID = "sha256:" + hashlib.sha256(b"conscio.v3.specialists.legacy-six-flat.v1").hexdigest()
_AFFECT_NEEDS = frozenset(
    {
        "epistemic_coherence",
        "competence",
        "integrity",
        "social_interaction",
        "continuity_of_memory",
    }
)


class CoreWeightBundle(Protocol):
    """Structural interface for an immutable, versioned core weight bundle.

    The inference core deliberately depends only on this small interface.  The
    offline trainer can therefore remain optional and no language model or
    training framework is reachable from the recurrent transition.
    """

    @property
    def model_version(self) -> str: ...

    @property
    def history_kernel(self) -> Sequence[Sequence[float]]: ...

    @property
    def observation_kernel(self) -> Sequence[Sequence[float]]: ...

    @property
    def broadcast_kernel(self) -> Sequence[Sequence[float]]: ...

    def predict_targets(self, state: np.ndarray) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class CycleResult:
    broadcast: Broadcast
    predictions: tuple[Prediction, ...]
    proposals: tuple[ActionProposal, ...]
    affect: AffectiveState
    all_candidates: tuple[CandidateContent, ...] = ()


def _vector(text: str, size: int = STATE_SIZE) -> np.ndarray:
    """Stable feature hashing; replay is independent of Python's hash seed."""
    out = np.zeros(size, dtype=np.float64)
    for token in text.casefold().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % size
        out[index] += -1.0 if digest[4] & 1 else 1.0
    norm = float(np.linalg.norm(out))
    return out / norm if norm else out


def _legacy_private_state(
    states: Mapping[str, Mapping[str, Any]],
    name: str,
    *,
    optional: Collection[str] = (),
) -> tuple[int, str, Mapping[str, Any]]:
    raw = states[name]
    required = {"updates", "last_digest"}
    actual = set(raw)
    if not required <= actual or actual - required - set(optional):
        raise ValueError(f"incompatible legacy {name} specialist state")
    updates = raw["updates"]
    digest = raw["last_digest"]
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError(f"legacy {name} updates must be a non-negative integer")
    if not isinstance(digest, str):
        raise ValueError(f"legacy {name} digest must be a string")
    return updates, digest, raw


def _validate_checkpoint_metadata(checkpoint: CoreCheckpoint) -> None:
    for name in ("checkpoint_id", "lineage_id", "model_version"):
        value = getattr(checkpoint, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"checkpoint {name} must be a non-empty string")
    if checkpoint.parent_checkpoint_id is not None and (
        not isinstance(checkpoint.parent_checkpoint_id, str) or not checkpoint.parent_checkpoint_id.strip()
    ):
        raise ValueError("checkpoint parent_checkpoint_id must be null or non-empty")
    for name in ("cycle_count", "event_sequence"):
        value = getattr(checkpoint, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"checkpoint {name} must be a non-negative integer")
    if (
        isinstance(checkpoint.created_at, bool)
        or not isinstance(checkpoint.created_at, (int, float))
        or not math.isfinite(float(checkpoint.created_at))
        or float(checkpoint.created_at) < 0.0
    ):
        raise ValueError("checkpoint created_at must be finite and non-negative")

    affect = checkpoint.affect
    if not isinstance(affect, AffectiveState):
        raise ValueError("checkpoint affect must be an AffectiveState")
    if set(affect.need_errors) != _AFFECT_NEEDS:
        raise ValueError("checkpoint affect need_errors must contain the exact supported needs")
    affect_ranges = {
        "valence": (-1.0, 1.0),
        "arousal": (0.0, 1.0),
        "controllability": (0.0, 1.0),
    }
    for name, (lower, upper) in affect_ranges.items():
        value = getattr(affect, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"checkpoint affect {name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"checkpoint affect {name} must be finite and in [{lower}, {upper}]")
    for name, value in affect.need_errors.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"checkpoint affect need {name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or not -1.0 <= number <= 1.0:
            raise ValueError(f"checkpoint affect need {name} must be finite and in [-1.0, 1.0]")
    if affect.intervention_id is not None and (
        not isinstance(affect.intervention_id, str) or not affect.intervention_id.strip()
    ):
        raise ValueError("checkpoint affect intervention_id must be null or non-empty")

    if not isinstance(checkpoint.rng_state, Mapping):
        raise ValueError("checkpoint rng_state must be an object")
    rng_probe = np.random.default_rng()
    try:
        rng_probe.bit_generator.state = dict(checkpoint.rng_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint rng_state is incompatible") from exc


def migrate_legacy_specialist_checkpoint(checkpoint: CoreCheckpoint) -> CoreCheckpoint:
    """Deterministically migrate the six-specialist checkpoint into a new lineage.

    The old compact private states do not contain observation text, episodic
    indexes, or action-evaluation state. Missing information is initialized to
    neutral values; it is never inferred from generated text.
    """
    if checkpoint.specialist_architecture_id != LEGACY_SPECIALIST_ARCHITECTURE_ID:
        raise ValueError("checkpoint is not a legacy specialist architecture")
    if checkpoint.model_version != MODEL_VERSION:
        raise ValueError(
            "trained legacy checkpoints require validated specialist-architecture "
            "migration; automatic state transfer is limited to the bootstrap model"
        )
    expected = {
        "perception",
        "memory",
        "world_model",
        "self_model",
        "affect",
        "planning",
    }
    if set(checkpoint.specialist_states) != expected:
        raise ValueError("legacy checkpoint has an incompatible specialist set")

    snapshots = SpecialistRegistry().checkpoint_states()

    def install(name: str, state: dict[str, Any]) -> None:
        snapshots[name] = {**snapshots[name], "state": state}

    perception_updates, perception_digest, _ = _legacy_private_state(checkpoint.specialist_states, "perception")
    install(
        "perception",
        {
            "updates": perception_updates,
            "last_digest": perception_digest,
            "last_event_id": "",
            "last_observation": "",
        },
    )
    memory_updates, memory_digest, _ = _legacy_private_state(checkpoint.specialist_states, "memory")
    install(
        "autobiographical_memory",
        {
            "updates": memory_updates,
            "last_digest": memory_digest,
            "episode_events": (),
            "last_broadcast_id": "",
        },
    )
    install(
        "semantic_belief",
        {
            "updates": 0,
            "last_digest": "",
            "last_event_id": "",
            "source_counts": (),
            "belief_cues": (),
        },
    )
    world_updates, world_digest, _ = _legacy_private_state(checkpoint.specialist_states, "world_model")
    install(
        "world_prediction",
        {
            "updates": world_updates,
            "last_digest": world_digest,
            "transitions": 0,
            "last_direction": "stable",
        },
    )
    self_updates, self_digest, self_raw = _legacy_private_state(
        checkpoint.specialist_states,
        "self_model",
        optional={"uncertainty"},
    )
    uncertainty = self_raw.get("uncertainty", 0.5)
    if (
        isinstance(uncertainty, bool)
        or not isinstance(uncertainty, (int, float))
        or not math.isfinite(float(uncertainty))
    ):
        raise ValueError("legacy self-model uncertainty must be finite")
    install(
        "self_model",
        {
            "updates": self_updates,
            "last_digest": self_digest,
            "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
        },
    )
    affect_updates, affect_digest, _ = _legacy_private_state(checkpoint.specialist_states, "affect")
    need_pressure = sum(abs(value) for value in checkpoint.affect.need_errors.values()) / max(
        1, len(checkpoint.affect.need_errors)
    )
    install(
        "affect",
        {
            "updates": affect_updates,
            "last_digest": affect_digest,
            "appraisals": 0,
            "need_pressure": need_pressure,
        },
    )
    planning_updates, planning_digest, _ = _legacy_private_state(checkpoint.specialist_states, "planning")
    install(
        "planning",
        {
            "updates": planning_updates,
            "last_digest": planning_digest,
            "plans_considered": 0,
            "last_prior_broadcast_id": "",
        },
    )
    # Action evaluation did not exist in the legacy architecture. Its neutral
    # default envelope from SpecialistRegistry is retained explicitly.
    return replace(
        checkpoint,
        checkpoint_id=f"ckpt_{uuid.uuid4().hex}",
        lineage_id=f"lineage_{uuid.uuid4().hex}",
        parent_checkpoint_id=checkpoint.checkpoint_id,
        specialist_states=snapshots,
        specialist_architecture_id=SPECIALIST_ARCHITECTURE_ID,
        created_at=time.time(),
        schema_version=CORE_CHECKPOINT_SCHEMA_VERSION,
    )


class HybridRecurrentCore:
    """Deterministic history + stochastic latent state with recurrent broadcasts."""

    def __init__(
        self,
        *,
        seed: int = 17,
        lineage_id: str | None = None,
        weights: CoreWeightBundle | None = None,
        specialist_factories: Mapping[str, SpecialistFactory] | None = None,
        specialist_lesions: Collection[str] = (),
    ) -> None:
        self.lineage_id = lineage_id or f"lineage_{uuid.uuid4().hex}"
        self.rng = np.random.default_rng(seed)
        self._prediction_weights = weights
        if weights is None:
            bootstrap = np.random.default_rng(7301)
            self._wh = bootstrap.normal(0.0, 0.12, (STATE_SIZE, STATE_SIZE))
            self._wx = bootstrap.normal(0.0, 0.18, (STATE_SIZE, STATE_SIZE))
            self._wb = bootstrap.normal(0.0, 0.10, (STATE_SIZE, STATE_SIZE))
            self.model_version = MODEL_VERSION
        else:
            self._wh = self._validated_kernel(weights.history_kernel, "history_kernel")
            self._wx = self._validated_kernel(weights.observation_kernel, "observation_kernel")
            self._wb = self._validated_kernel(weights.broadcast_kernel, "broadcast_kernel")
            if not weights.model_version:
                raise ValueError("weight bundle model_version must be non-empty")
            self.model_version = weights.model_version
        self.deterministic = np.zeros(STATE_SIZE, dtype=np.float64)
        self.stochastic = np.zeros(STATE_SIZE, dtype=np.float64)
        self.affect = AffectiveState()
        self.cycle_count = 0
        self.event_sequence = 0
        self.parent_checkpoint_id: str | None = None
        self.specialist_registry = SpecialistRegistry(
            factories=specialist_factories,
            removed_specialists=specialist_lesions,
        )
        # Retain the previous inspection surface while registry ownership keeps
        # the actual mapping and private states inaccessible to specialists.
        self.specialists = self.specialist_registry.specialists
        self.specialist_architecture_id = self.specialist_registry.architecture_id
        self._specialist_execution_audit: dict[str, dict[str, int]] = {
            name: {"compute": 0, "expose": 0} for name in self.specialist_registry.names
        }

    @property
    def active_weight_bundle(self) -> CoreWeightBundle | None:
        """Return the immutable installed bundle; ``None`` denotes bootstrap.

        The core has no setter because changing weights in place would blur the
        checkpoint's model lineage.  Install promoted weights in a fresh core.
        """
        return self._prediction_weights

    @property
    def runtime_identity(self) -> str:
        """Identity of weights plus the specialist architecture consuming them."""
        digest_fn = getattr(self._prediction_weights, "digest", None)
        weight_identity = str(digest_fn()) if callable(digest_fn) else f"model-version:{self.model_version}"
        material = f"{self.model_version}\n{weight_identity}\n{self.specialist_architecture_id}"
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    @property
    def specialist_execution_audit(self) -> Mapping[str, Mapping[str, int]]:
        """Return an immutable snapshot of specialist compute/exposure counts."""
        return MappingProxyType(
            {name: MappingProxyType(dict(counts)) for name, counts in self._specialist_execution_audit.items()}
        )

    def run_cycle(
        self,
        event: CognitiveEvent,
        cycle: int,
        previous_broadcast: Broadcast | None,
        active_specialists: Collection[str] | None = None,
        *,
        prediction_enabled: bool = True,
        broadcast_enabled: bool = True,
    ) -> CycleResult:
        """Execute one inspectable recurrent cycle.

        Specialists outside ``active_specialists`` are neither called nor
        exposed.  This public seam supports true end-to-end lesions and
        matched broadcast interventions without replacing module outputs with
        placeholders.
        """
        available = set(self.specialist_registry.names)
        active = available if active_specialists is None else set(active_specialists)
        unknown = active - available
        if unknown:
            raise ValueError(f"unavailable active specialist(s): {sorted(unknown)}")
        if not prediction_enabled:
            active.discard("world_prediction")
        removed = available - active
        event_vector = _vector(str(event.payload.get("content", "")))
        broadcast_vector = _vector(
            " ".join(candidate.content for candidate in previous_broadcast.candidates) if previous_broadcast else ""
        )
        noise = self.rng.normal(0.0, 0.08, STATE_SIZE)
        self.deterministic = np.tanh(
            self._wh @ self.deterministic
            + self._wx @ event_vector
            + self._wb @ broadcast_vector
            + 0.15 * self.stochastic
        )
        self.stochastic = 0.82 * self.stochastic + 0.18 * noise
        joint = self.deterministic + self.stochastic
        specialist_input = SpecialistInput(
            event=event,
            previous_broadcast=previous_broadcast,
            recurrent_state=tuple(float(value) for value in joint),
        )
        candidates = self.specialist_registry.candidates(
            specialist_input,
            removed_specialists=removed,
        )
        for candidate in candidates:
            self._specialist_execution_audit[candidate.specialist]["compute"] += 1
        winners = (
            tuple(
                sorted(
                    candidates,
                    key=lambda candidate: (
                        -candidate.salience,
                        -candidate.confidence,
                        candidate.specialist,
                    ),
                )[:3]
            )
            if broadcast_enabled
            else ()
        )
        for candidate in winners:
            self._specialist_execution_audit[candidate.specialist]["expose"] += 1
        digest = hashlib.sha256(joint.tobytes()).hexdigest()
        broadcast = Broadcast(
            cycle=cycle,
            candidates=winners,
            recurrent_state_digest=digest,
            broadcast_id=stable_identifier(
                "bcast",
                event.event_id,
                self.cycle_count,
                digest,
                tuple(candidate.candidate_id for candidate in winners),
            ),
        )
        if "affect" in active:
            self.affect = self._update_affect(winners, joint)
        predictions = self._predictions(joint, broadcast) if prediction_enabled and "world_prediction" in active else ()
        proposals = self._proposals(event, broadcast, active)
        self.cycle_count += 1
        return CycleResult(
            broadcast=broadcast,
            predictions=predictions,
            proposals=proposals,
            affect=self.affect,
            all_candidates=candidates,
        )

    def run_cycles(
        self,
        event: CognitiveEvent,
        *,
        cycles: int = 3,
        memory_enabled: bool = True,
        self_model_enabled: bool = True,
        prediction_enabled: bool = True,
        broadcast_enabled: bool = True,
        broadcast_transform: Callable[[Broadcast], Broadcast] | None = None,
    ) -> list[CycleResult]:
        previous: Broadcast | None = None
        results: list[CycleResult] = []
        active = set(self.specialist_registry.names)
        if not memory_enabled:
            active.difference_update({"autobiographical_memory", "semantic_belief"})
        if not self_model_enabled:
            active.discard("self_model")
        if not prediction_enabled:
            active.discard("world_prediction")
        cycle_total = max(1, cycles)
        for cycle in range(cycle_total):
            result = self.run_cycle(
                event,
                cycle,
                previous,
                active,
                prediction_enabled=prediction_enabled,
                broadcast_enabled=broadcast_enabled,
            )
            results.append(result)
            if broadcast_enabled:
                previous = result.broadcast
                if broadcast_transform is not None and cycle < cycle_total - 1:
                    previous = broadcast_transform(previous)
                    if not isinstance(previous, Broadcast):
                        raise TypeError("broadcast_transform must return a Broadcast")
            else:
                previous = None
        self.event_sequence += 1
        return results

    def _update_affect(self, winners: tuple[CandidateContent, ...], state: np.ndarray) -> AffectiveState:
        uncertainty = 1.0 - (sum(c.confidence for c in winners) / max(1, len(winners)))
        prior = self.affect
        errors = dict(prior.need_errors)
        errors["epistemic_coherence"] = 0.75 * errors["epistemic_coherence"] + 0.25 * uncertainty
        errors["competence"] = 0.80 * errors["competence"] + 0.20 * max(0.0, uncertainty - 0.35)
        errors["integrity"] *= 0.90
        errors["social_interaction"] *= 0.92
        errors["continuity_of_memory"] *= 0.94
        mean_error = sum(abs(v) for v in errors.values()) / len(errors)
        # Recovery is built into every transition; sustained negative values do
        # not ratchet indefinitely and process survival is not a represented need.
        return AffectiveState(
            valence=0.82 * prior.valence + 0.18 * (-mean_error),
            arousal=0.75 * prior.arousal + 0.25 * min(1.0, mean_error + abs(float(state[5]))),
            controllability=0.85 * prior.controllability + 0.15 * (1.0 - uncertainty),
            need_errors=errors,
        ).bounded()

    def apply_action_feedback(self, *, succeeded: bool, uncertainty_delta: float = 0.0) -> AffectiveState:
        """Apply an observable action outcome to causal affect/homeostasis.

        This transition is deliberately independent of generated emotion text.
        Success lowers competence and coherence error; failure and increased
        uncertainty raise them. Recovery remains present in every other need,
        and no process-survival or shutdown-resistance need exists.
        """
        if not np.isfinite(uncertainty_delta):
            raise ValueError("uncertainty_delta must be finite")
        prior = self.affect
        errors = dict(prior.need_errors)
        if succeeded:
            errors["competence"] *= 0.65
            errors["epistemic_coherence"] = 0.72 * errors["epistemic_coherence"] + 0.20 * max(0.0, uncertainty_delta)
        else:
            errors["competence"] = 0.82 * errors["competence"] + 0.18
            errors["epistemic_coherence"] = (
                0.85 * errors["epistemic_coherence"] + 0.10 + 0.20 * max(0.0, uncertainty_delta)
            )
        errors["integrity"] *= 0.94
        errors["social_interaction"] *= 0.94
        errors["continuity_of_memory"] *= 0.96
        errors = {name: float(np.clip(value, -1.0, 1.0)) for name, value in errors.items()}
        mean_error = sum(abs(value) for value in errors.values()) / len(errors)
        outcome_valence = 0.35 if succeeded else -0.45
        self.affect = AffectiveState(
            valence=0.78 * prior.valence + 0.22 * (outcome_valence - 0.25 * mean_error),
            arousal=0.80 * prior.arousal + 0.20 * min(1.0, mean_error + (0.0 if succeeded else 0.25)),
            controllability=0.80 * prior.controllability + 0.20 * (0.78 if succeeded else 0.25),
            need_errors=errors,
        ).bounded()
        return self.affect

    def _proposals(
        self,
        event: CognitiveEvent,
        broadcast: Broadcast,
        active_specialists: Collection[str],
    ) -> tuple[ActionProposal, ...]:
        content = str(event.payload.get("content", ""))
        need_pressure = sum(abs(v) for v in self.affect.need_errors.values()) / 5.0
        proposals: list[ActionProposal] = []
        if "planning" in active_specialists:
            action = "answer" if event.source == "user" else "act"
            proposals.append(
                ActionProposal(
                    specialist="planning",
                    action=action,
                    rationale="Respond to the current observation using broadcast-accessible evidence.",
                    expected_outcomes=("task-relevant response", "uncertainty does not increase"),
                    confidence=0.70,
                    utility=0.65 - 0.15 * need_pressure,
                    risk=0.08,
                    proposal_id=stable_identifier("act", broadcast.broadcast_id, "planning", action),
                )
            )
        if "action_evaluation" in active_specialists:
            action = "ask" if content.strip() else "wait"
            proposals.append(
                ActionProposal(
                    specialist="action_evaluation",
                    action=action,
                    rationale="Reduce uncertainty when the observation is underspecified.",
                    expected_outcomes=("additional observation",),
                    confidence=0.55,
                    utility=0.45 + 0.20 * need_pressure,
                    risk=0.03,
                    proposal_id=stable_identifier("act", broadcast.broadcast_id, "action_evaluation", action),
                )
            )
        return tuple(sorted(proposals, key=lambda p: p.utility - p.risk, reverse=True))

    @staticmethod
    def _validated_kernel(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
        kernel = np.asarray(values, dtype=np.float64)
        if kernel.shape != (STATE_SIZE, STATE_SIZE):
            raise ValueError(f"{name} must have shape {(STATE_SIZE, STATE_SIZE)}, got {kernel.shape}")
        if not np.all(np.isfinite(kernel)):
            raise ValueError(f"{name} must contain only finite values")
        return kernel.copy()

    def _predictions(self, state: np.ndarray, broadcast: Broadcast) -> tuple[Prediction, ...]:
        if self._prediction_weights is None:
            return (
                Prediction(
                    target="next_observation",
                    observable="the next cycle contains a task-relevant observation or outcome",
                    probability=float(np.clip(0.5 + 0.25 * abs(state[4]), 0.05, 0.95)),
                    horizon=1,
                    basis_broadcast_id=broadcast.broadcast_id,
                    prediction_id=stable_identifier("pred", broadcast.broadcast_id, "next_observation"),
                ),
                Prediction(
                    target="future_uncertainty",
                    observable="uncertainty does not increase after the selected action",
                    probability=float(np.clip(0.55 + 0.2 * self.affect.controllability, 0.05, 0.95)),
                    horizon=1,
                    basis_broadcast_id=broadcast.broadcast_id,
                    prediction_id=stable_identifier("pred", broadcast.broadcast_id, "future_uncertainty"),
                ),
            )
        observables = {
            "next_observation": "a task-relevant observation occurs in the next horizon",
            "tool_outcome": "the proposed tool action has a successful observable outcome",
            "action_effect": "the selected action produces its intended observable effect",
            "homeostatic_affect_change": "aggregate need error improves after the selected action",
            "future_uncertainty": "uncertainty does not increase after the selected action",
        }
        values = self._prediction_weights.predict_targets(state)
        if set(values) != set(observables):
            raise ValueError("weight bundle must predict every supported target exactly once")
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError("weight bundle target predictions must be finite and in [0, 1]")
        return tuple(
            Prediction(
                target=target,
                observable=observable,
                probability=float(values[target]),
                horizon=1,
                basis_broadcast_id=broadcast.broadcast_id,
                prediction_id=stable_identifier("pred", broadcast.broadcast_id, target),
            )
            for target, observable in observables.items()
        )

    def checkpoint(self) -> CoreCheckpoint:
        checkpoint_id = f"ckpt_{uuid.uuid4().hex}"
        checkpoint = CoreCheckpoint(
            checkpoint_id=checkpoint_id,
            lineage_id=self.lineage_id,
            parent_checkpoint_id=self.parent_checkpoint_id,
            model_version=self.model_version,
            specialist_architecture_id=self.specialist_architecture_id,
            deterministic_state=tuple(float(v) for v in self.deterministic),
            stochastic_state=tuple(float(v) for v in self.stochastic),
            specialist_states=self.specialist_registry.checkpoint_states(),
            affect=self.affect,
            cycle_count=self.cycle_count,
            event_sequence=self.event_sequence,
            rng_state=dict(self.rng.bit_generator.state),
        )
        self.parent_checkpoint_id = checkpoint_id
        return checkpoint

    def restore(self, checkpoint: CoreCheckpoint) -> None:
        if checkpoint.model_version != self.model_version:
            raise ValueError(f"checkpoint model {checkpoint.model_version!r} requires an explicit lineage migration")
        if checkpoint.specialist_architecture_id != self.specialist_architecture_id:
            raise ValueError("checkpoint specialist architecture requires an explicit lineage migration")
        if checkpoint.schema_version != CORE_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint wire schema requires an explicit migration")
        _validate_checkpoint_metadata(checkpoint)
        deterministic = np.asarray(checkpoint.deterministic_state, dtype=np.float64)
        stochastic = np.asarray(checkpoint.stochastic_state, dtype=np.float64)
        if deterministic.shape != (STATE_SIZE,) or stochastic.shape != (STATE_SIZE,):
            raise ValueError("checkpoint recurrent state has an incompatible shape")
        if not np.all(np.isfinite(deterministic)) or not np.all(np.isfinite(stochastic)):
            raise ValueError("checkpoint recurrent state must contain only finite values")
        restored_specialist_states = self.specialist_registry.decode_checkpoint_states(checkpoint.specialist_states)
        self.lineage_id = checkpoint.lineage_id
        self.parent_checkpoint_id = checkpoint.checkpoint_id
        self.deterministic = deterministic
        self.stochastic = stochastic
        self.affect = checkpoint.affect
        self.cycle_count = checkpoint.cycle_count
        self.event_sequence = checkpoint.event_sequence
        self.rng.bit_generator.state = checkpoint.rng_state
        self.specialist_registry.install_checkpoint_states(restored_specialist_states)

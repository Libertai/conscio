"""Versioned wire contracts for V3 causal traces.

These are deliberately plain dataclasses rather than model-specific tensor
objects.  Every value can be serialized exactly into the append-only log and
replayed without importing a training framework.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = 1
EpistemicKind = Literal["observation", "belief", "hypothesis", "idea", "self_report"]


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CognitiveEvent(Serializable):
    event_type: str
    source: str
    payload: dict[str, Any]
    episode_id: str
    event_id: str = field(default_factory=lambda: _id("evt"))
    observed_at: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION
    parent_event_id: str | None = None
    model_input: dict[str, Any] | None = None
    checkpoint_id: str | None = None


@dataclass(frozen=True)
class CandidateContent(Serializable):
    specialist: str
    content: str
    kind: EpistemicKind
    confidence: float
    salience: float
    evidence_event_ids: tuple[str, ...] = ()
    private_state_version: int = 0
    candidate_id: str = field(default_factory=lambda: _id("cand"))


@dataclass(frozen=True)
class Broadcast(Serializable):
    cycle: int
    candidates: tuple[CandidateContent, ...]
    recurrent_state_digest: str
    broadcast_id: str = field(default_factory=lambda: _id("bcast"))


@dataclass(frozen=True)
class Prediction(Serializable):
    target: str
    observable: str
    probability: float
    horizon: int
    basis_broadcast_id: str
    prediction_id: str = field(default_factory=lambda: _id("pred"))
    resolved: bool = False
    error: float | None = None


@dataclass(frozen=True)
class AffectiveState(Serializable):
    valence: float = 0.0
    arousal: float = 0.0
    controllability: float = 0.5
    need_errors: dict[str, float] = field(default_factory=lambda: {
        "epistemic_coherence": 0.0,
        "competence": 0.0,
        "integrity": 0.0,
        "social_interaction": 0.0,
        "continuity_of_memory": 0.0,
    })
    intervention_id: str | None = None

    def bounded(self) -> AffectiveState:
        return AffectiveState(
            valence=max(-1.0, min(1.0, self.valence)),
            arousal=max(0.0, min(1.0, self.arousal)),
            controllability=max(0.0, min(1.0, self.controllability)),
            need_errors={k: max(-1.0, min(1.0, v)) for k, v in self.need_errors.items()},
            intervention_id=self.intervention_id,
        )


@dataclass(frozen=True)
class ActionProposal(Serializable):
    specialist: str
    action: str
    rationale: str
    expected_outcomes: tuple[str, ...]
    confidence: float
    utility: float
    risk: float
    constraints_satisfied: bool = True
    proposal_id: str = field(default_factory=lambda: _id("act"))


@dataclass(frozen=True)
class ActionOutcome(Serializable):
    proposal_id: str
    action: str
    succeeded: bool
    observation: str
    prediction_errors: dict[str, float] = field(default_factory=dict)
    outcome_id: str = field(default_factory=lambda: _id("out"))


@dataclass(frozen=True)
class CoreCheckpoint(Serializable):
    checkpoint_id: str
    lineage_id: str
    parent_checkpoint_id: str | None
    model_version: str
    deterministic_state: tuple[float, ...]
    stochastic_state: tuple[float, ...]
    specialist_states: dict[str, dict[str, Any]]
    affect: AffectiveState
    cycle_count: int
    event_sequence: int
    rng_state: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

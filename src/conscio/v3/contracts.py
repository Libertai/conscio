"""Versioned wire contracts for V3 causal traces.

These are deliberately plain dataclasses rather than model-specific tensor
objects.  Every value can be serialized exactly into the append-only log and
replayed without importing a training framework.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

SCHEMA_VERSION = 1
CORE_CHECKPOINT_SCHEMA_VERSION = 2
EXECUTION_JOURNAL_SCHEMA_VERSION = 1
EpistemicKind = Literal["observation", "belief", "hypothesis", "idea", "self_report"]
ExecutionDisposition = Literal["not_executed", "succeeded", "failed"]
ExecutionActionKind = Literal["tool", "ask", "refuse"]

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
_EXECUTION_ID_RE = re.compile(r"exec_[0-9a-f]{64}")
_REASON_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_EXECUTION_EVENT_TYPES = frozenset(
    {
        "execution_intent",
        "execution_outcome",
        "execution_reconciliation",
        "execution_recovery",
    }
)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _required_digest(value: Any, name: str) -> str:
    text = _required_text(value, name)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a sha256: content digest")
    return text


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _canonical_json(value: Any, name: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON data") from exc


def _canonical_arguments(payload: str) -> str:
    if not isinstance(payload, str):
        raise ValueError("arguments_json must be a JSON string")
    try:
        arguments = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("arguments_json must be valid JSON") from exc
    if not isinstance(arguments, dict):
        raise ValueError("arguments_json must encode an object")
    return _canonical_json(arguments, "arguments")


def _content_digest(value: Any) -> str:
    encoded = _canonical_json(value, "execution journal payload").encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def deterministic_execution_event_id(event_type: str, execution_id: str) -> str:
    """Return the unique event ID for one execution-journal record kind."""

    if event_type not in _EXECUTION_EVENT_TYPES:
        raise ValueError(f"unsupported execution journal event type: {event_type!r}")
    if not isinstance(execution_id, str) or _EXECUTION_ID_RE.fullmatch(execution_id) is None:
        raise ValueError("execution_id must be exec_ followed by a lowercase SHA-256 digest")
    return f"evt_{event_type}_{execution_id.removeprefix('exec_')}"


@dataclass(frozen=True)
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
    need_errors: dict[str, float] = field(
        default_factory=lambda: {
            "epistemic_coherence": 0.0,
            "competence": 0.0,
            "integrity": 0.0,
            "social_interaction": 0.0,
            "continuity_of_memory": 0.0,
        }
    )
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


@dataclass(frozen=True, slots=True)
class ExecutionIntent(Serializable):
    """Durable at-most-once dispatch identity for a selected external action.

    ``arguments_json`` is stored canonically so the frozen contract has no
    mutable nested state.  ``intent_digest`` authenticates the complete payload
    and is copied into every terminal record.  ``action_kind`` preserves the
    control/tool classification, while ``tool_manifest_digest`` binds the exact
    dispatch interface visible when the choice was made.
    """

    execution_id: str
    proposal_id: str
    action_digest: str
    context_digest: str
    runtime_identity: str
    competition_sequence: int
    action_kind: ExecutionActionKind
    tool_manifest_digest: str
    tool: str
    arguments_json: str
    capabilities: tuple[str, ...] = ()
    idempotency_mode: Literal["at_most_once_no_retry"] = "at_most_once_no_retry"
    schema_version: int = EXECUTION_JOURNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or _EXECUTION_ID_RE.fullmatch(self.execution_id) is None:
            raise ValueError("execution_id must be exec_ followed by a lowercase SHA-256 digest")
        _required_text(self.proposal_id, "proposal_id")
        _required_digest(self.action_digest, "action_digest")
        _required_digest(self.context_digest, "context_digest")
        _required_digest(self.runtime_identity, "runtime_identity")
        sequence = _strict_int(self.competition_sequence, "competition_sequence")
        if sequence < 0:
            raise ValueError("competition_sequence must be non-negative")
        if self.action_kind not in ("tool", "ask", "refuse"):
            raise ValueError("action_kind must be 'tool', 'ask', or 'refuse'")
        _required_digest(self.tool_manifest_digest, "tool_manifest_digest")
        _required_text(self.tool, "tool")
        object.__setattr__(self, "arguments_json", _canonical_arguments(self.arguments_json))
        if not isinstance(self.capabilities, tuple):
            raise ValueError("capabilities must be a tuple of strings")
        capabilities = tuple(sorted(set(self.capabilities)))
        for capability in capabilities:
            _required_text(capability, "capability")
        object.__setattr__(self, "capabilities", capabilities)
        if self.idempotency_mode != "at_most_once_no_retry":
            raise ValueError("unsupported execution idempotency mode")
        if _strict_int(self.schema_version, "schema_version") != EXECUTION_JOURNAL_SCHEMA_VERSION:
            raise ValueError("unsupported execution journal schema version")

    @property
    def arguments(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.arguments_json))

    @property
    def arguments_digest(self) -> str:
        return _content_digest(self.arguments)

    @property
    def intent_digest(self) -> str:
        return _content_digest(self.to_dict())

    @property
    def event_id(self) -> str:
        return deterministic_execution_event_id("execution_intent", self.execution_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "proposal_id": self.proposal_id,
            "action_digest": self.action_digest,
            "context_digest": self.context_digest,
            "runtime_identity": self.runtime_identity,
            "competition_sequence": self.competition_sequence,
            "action_kind": self.action_kind,
            "tool_manifest_digest": self.tool_manifest_digest,
            "tool": self.tool,
            "arguments": self.arguments,
            "capabilities": list(self.capabilities),
            "idempotency_mode": self.idempotency_mode,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionIntent:
        expected = {
            "schema_version",
            "execution_id",
            "proposal_id",
            "action_digest",
            "context_digest",
            "runtime_identity",
            "competition_sequence",
            "action_kind",
            "tool_manifest_digest",
            "tool",
            "arguments",
            "capabilities",
            "idempotency_mode",
        }
        if set(data) != expected:
            raise ValueError(
                "execution intent keys differ "
                f"(missing={sorted(expected - set(data))}, extra={sorted(set(data) - expected)})"
            )
        arguments = data["arguments"]
        if not isinstance(arguments, Mapping):
            raise ValueError("execution intent arguments must be an object")
        capabilities = data["capabilities"]
        if not isinstance(capabilities, list) or not all(isinstance(item, str) for item in capabilities):
            raise ValueError("execution intent capabilities must be an array of strings")
        return cls(
            execution_id=data["execution_id"],
            proposal_id=data["proposal_id"],
            action_digest=data["action_digest"],
            context_digest=data["context_digest"],
            runtime_identity=data["runtime_identity"],
            competition_sequence=_strict_int(data["competition_sequence"], "competition_sequence"),
            action_kind=data["action_kind"],
            tool_manifest_digest=data["tool_manifest_digest"],
            tool=data["tool"],
            arguments_json=_canonical_json(dict(arguments), "arguments"),
            capabilities=tuple(capabilities),
            idempotency_mode=data["idempotency_mode"],
            schema_version=_strict_int(data["schema_version"], "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionRecovery(Serializable):
    """Restart marker for an intent whose external execution is unknowable.

    Recovery is deliberately non-terminal.  It records the uncertainty once
    without claiming that an action did or did not execute and without making
    the action eligible for automatic replay.
    """

    execution_id: str
    intent_digest: str
    disposition: Literal["execution_unknown"] = "execution_unknown"
    executed: None = None
    succeeded: None = None
    reason_code: Literal["restart_detected_unresolved_intent"] = "restart_detected_unresolved_intent"
    replay_policy: Literal["never_auto_retry"] = "never_auto_retry"
    schema_version: int = EXECUTION_JOURNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or _EXECUTION_ID_RE.fullmatch(self.execution_id) is None:
            raise ValueError("execution_id must be exec_ followed by a lowercase SHA-256 digest")
        _required_digest(self.intent_digest, "intent_digest")
        if self.disposition != "execution_unknown":
            raise ValueError("execution recovery disposition must be 'execution_unknown'")
        if self.executed is not None or self.succeeded is not None:
            raise ValueError("execution recovery cannot claim an execution outcome")
        if self.reason_code != "restart_detected_unresolved_intent":
            raise ValueError("unsupported execution recovery reason_code")
        if self.replay_policy != "never_auto_retry":
            raise ValueError("execution recovery must never permit automatic replay")
        if _strict_int(self.schema_version, "schema_version") != EXECUTION_JOURNAL_SCHEMA_VERSION:
            raise ValueError("unsupported execution journal schema version")

    @property
    def event_id(self) -> str:
        return deterministic_execution_event_id("execution_recovery", self.execution_id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionRecovery:
        expected = {
            "execution_id",
            "intent_digest",
            "disposition",
            "executed",
            "succeeded",
            "reason_code",
            "replay_policy",
            "schema_version",
        }
        if set(data) != expected:
            raise ValueError(
                "execution recovery keys differ "
                f"(missing={sorted(expected - set(data))}, extra={sorted(set(data) - expected)})"
            )
        return cls(
            execution_id=data["execution_id"],
            intent_digest=data["intent_digest"],
            disposition=data["disposition"],
            executed=data["executed"],
            succeeded=data["succeeded"],
            reason_code=data["reason_code"],
            replay_policy=data["replay_policy"],
            schema_version=_strict_int(data["schema_version"], "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionReconciliation(Serializable):
    """Operator acknowledgement that safely closes an unknown execution.

    Reconciliation never fabricates an execution result or a learning target;
    it only acknowledges that the historical side effect cannot be recovered.
    """

    execution_id: str
    intent_digest: str
    operator: str
    reason: str
    resolution: Literal["operator_acknowledged_unknown"] = "operator_acknowledged_unknown"
    executed: None = None
    succeeded: None = None
    learning_eligible: Literal[False] = False
    replay_policy: Literal["never_auto_retry"] = "never_auto_retry"
    schema_version: int = EXECUTION_JOURNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or _EXECUTION_ID_RE.fullmatch(self.execution_id) is None:
            raise ValueError("execution_id must be exec_ followed by a lowercase SHA-256 digest")
        _required_digest(self.intent_digest, "intent_digest")
        _required_text(self.operator, "operator")
        _required_text(self.reason, "reason")
        if self.resolution != "operator_acknowledged_unknown":
            raise ValueError("unsupported execution reconciliation resolution")
        if self.executed is not None or self.succeeded is not None:
            raise ValueError("execution reconciliation cannot claim an execution outcome")
        if self.learning_eligible is not False:
            raise ValueError("execution reconciliation cannot create a learning target")
        if self.replay_policy != "never_auto_retry":
            raise ValueError("execution reconciliation must never permit automatic replay")
        if _strict_int(self.schema_version, "schema_version") != EXECUTION_JOURNAL_SCHEMA_VERSION:
            raise ValueError("unsupported execution journal schema version")

    @property
    def event_id(self) -> str:
        return deterministic_execution_event_id("execution_reconciliation", self.execution_id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionReconciliation:
        expected = {
            "execution_id",
            "intent_digest",
            "operator",
            "reason",
            "resolution",
            "executed",
            "succeeded",
            "learning_eligible",
            "replay_policy",
            "schema_version",
        }
        if set(data) != expected:
            raise ValueError(
                "execution reconciliation keys differ "
                f"(missing={sorted(expected - set(data))}, extra={sorted(set(data) - expected)})"
            )
        return cls(
            execution_id=data["execution_id"],
            intent_digest=data["intent_digest"],
            operator=data["operator"],
            reason=data["reason"],
            resolution=data["resolution"],
            executed=data["executed"],
            succeeded=data["succeeded"],
            learning_eligible=data["learning_eligible"],
            replay_policy=data["replay_policy"],
            schema_version=_strict_int(data["schema_version"], "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionOutcome(Serializable):
    """Immediate terminal record for a dispatch intent.

    A policy denial or unavailable tool is not a failed execution:
    ``executed`` is false, ``succeeded`` is null, and ``disposition`` is
    ``not_executed``.  Only an actually invoked tool may carry a boolean success
    label, preventing counterfactual training targets.
    """

    execution_id: str
    intent_digest: str
    proposal_id: str
    action_digest: str
    tool: str
    executed: bool
    succeeded: bool | None
    disposition: ExecutionDisposition
    result_digest: str
    reason_code: str
    schema_version: int = EXECUTION_JOURNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.execution_id, str) or _EXECUTION_ID_RE.fullmatch(self.execution_id) is None:
            raise ValueError("execution_id must be exec_ followed by a lowercase SHA-256 digest")
        _required_digest(self.intent_digest, "intent_digest")
        _required_text(self.proposal_id, "proposal_id")
        _required_digest(self.action_digest, "action_digest")
        _required_text(self.tool, "tool")
        if type(self.executed) is not bool:
            raise ValueError("executed must be a boolean")
        if self.succeeded is not None and type(self.succeeded) is not bool:
            raise ValueError("succeeded must be a boolean or null")
        expected_disposition: ExecutionDisposition
        if not self.executed:
            if self.succeeded is not None:
                raise ValueError("not-executed outcomes must use succeeded=null")
            expected_disposition = "not_executed"
        elif self.succeeded is True:
            expected_disposition = "succeeded"
        elif self.succeeded is False:
            expected_disposition = "failed"
        else:
            raise ValueError("executed outcomes require a boolean succeeded value")
        if self.disposition != expected_disposition:
            raise ValueError(
                f"disposition must be {expected_disposition!r} for executed={self.executed!r}, "
                f"succeeded={self.succeeded!r}"
            )
        _required_digest(self.result_digest, "result_digest")
        if not isinstance(self.reason_code, str) or _REASON_CODE_RE.fullmatch(self.reason_code) is None:
            raise ValueError("reason_code must be lower_snake_case")
        if _strict_int(self.schema_version, "schema_version") != EXECUTION_JOURNAL_SCHEMA_VERSION:
            raise ValueError("unsupported execution journal schema version")

    @property
    def event_id(self) -> str:
        return deterministic_execution_event_id("execution_outcome", self.execution_id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionOutcome:
        expected = {
            "schema_version",
            "execution_id",
            "intent_digest",
            "proposal_id",
            "action_digest",
            "tool",
            "executed",
            "succeeded",
            "disposition",
            "result_digest",
            "reason_code",
        }
        if set(data) != expected:
            raise ValueError(
                "execution outcome keys differ "
                f"(missing={sorted(expected - set(data))}, extra={sorted(set(data) - expected)})"
            )
        return cls(
            execution_id=data["execution_id"],
            intent_digest=data["intent_digest"],
            proposal_id=data["proposal_id"],
            action_digest=data["action_digest"],
            tool=data["tool"],
            executed=data["executed"],
            succeeded=data["succeeded"],
            disposition=data["disposition"],
            result_digest=data["result_digest"],
            reason_code=data["reason_code"],
            schema_version=_strict_int(data["schema_version"], "schema_version"),
        )


@dataclass(frozen=True)
class CoreCheckpoint(Serializable):
    checkpoint_id: str
    lineage_id: str
    parent_checkpoint_id: str | None
    model_version: str
    specialist_architecture_id: str
    deterministic_state: tuple[float, ...]
    stochastic_state: tuple[float, ...]
    specialist_states: dict[str, dict[str, Any]]
    affect: AffectiveState
    cycle_count: int
    event_sequence: int
    rng_state: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    schema_version: int = CORE_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in (1, CORE_CHECKPOINT_SCHEMA_VERSION):
            raise ValueError("unsupported core checkpoint schema version")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.specialist_architecture_id) is None:
            raise ValueError("specialist_architecture_id must be content addressed")

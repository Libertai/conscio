"""Isolated, checkpointable specialists for the V3 recurrent workspace.

The specialists in this module deliberately have a narrow boundary.  A
specialist receives one immutable :class:`SpecialistInput`, owns only its own
private state, and emits a typed :class:`CandidateContent`.  Cross-specialist
communication is possible only through the preceding global broadcast.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Generic, Protocol, TypeVar, cast

from conscio.v3.contracts import (
    Broadcast,
    CandidateContent,
    CognitiveEvent,
    EpistemicKind,
)

SPECIALIST_STATE_SCHEMA_VERSION = 1
SPECIALIST_ORDER = (
    "perception",
    "autobiographical_memory",
    "semantic_belief",
    "world_prediction",
    "self_model",
    "affect",
    "planning",
    "action_evaluation",
)

_CONTENT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DEFAULT_SPECIALIST_CLASS_PATHS = {
    "perception": "conscio.v3.specialists.PerceptionSpecialist",
    "autobiographical_memory": "conscio.v3.specialists.AutobiographicalMemorySpecialist",
    "semantic_belief": "conscio.v3.specialists.SemanticBeliefSpecialist",
    "world_prediction": "conscio.v3.specialists.WorldPredictionSpecialist",
    "self_model": "conscio.v3.specialists.SelfModelSpecialist",
    "affect": "conscio.v3.specialists.AffectSpecialist",
    "planning": "conscio.v3.specialists.PlanningSpecialist",
    "action_evaluation": "conscio.v3.specialists.ActionEvaluationSpecialist",
}


def _implementation_id(class_path: str, version: int) -> str:
    descriptor = json.dumps(
        {"class_path": class_path, "implementation_version": version},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(descriptor.encode("utf-8")).hexdigest()


DEFAULT_SPECIALIST_IMPLEMENTATION_IDS = {
    name: _implementation_id(class_path, 1) for name, class_path in _DEFAULT_SPECIALIST_CLASS_PATHS.items()
}


def specialist_architecture_id(
    names: Collection[str] = SPECIALIST_ORDER,
    *,
    implementation_ids: Mapping[str, str] | None = None,
    state_schemas: Mapping[str, tuple[str, int]] | None = None,
) -> str:
    """Content identity for exact specialist schemas and implementations."""
    configured = tuple(names)
    selected = tuple(name for name in SPECIALIST_ORDER if name in set(configured))
    if set(selected) != set(configured) or len(configured) != len(set(configured)):
        raise ValueError("specialist architecture contains unknown or duplicate names")
    implementations = (
        {name: DEFAULT_SPECIALIST_IMPLEMENTATION_IDS[name] for name in selected}
        if implementation_ids is None
        else dict(implementation_ids)
    )
    schemas = (
        {
            name: (
                f"conscio.v3.specialist.{name}",
                SPECIALIST_STATE_SCHEMA_VERSION,
            )
            for name in selected
        }
        if state_schemas is None
        else dict(state_schemas)
    )
    if set(implementations) != set(selected) or set(schemas) != set(selected):
        raise ValueError("specialist architecture descriptors must cover the exact specialist set")
    descriptor = [
        {
            "name": name,
            "state_schema": schemas[name][0],
            "state_schema_version": schemas[name][1],
            "implementation_id": implementations[name],
        }
        for name in selected
    ]
    for item in descriptor:
        if not isinstance(item["state_schema"], str) or not item["state_schema"]:
            raise ValueError("specialist state schema must be a non-empty string")
        version = item["state_schema_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("specialist state schema version must be a positive integer")
        if _CONTENT_ID_RE.fullmatch(str(item["implementation_id"])) is None:
            raise ValueError("specialist implementation identity must be content addressed")
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


SPECIALIST_ARCHITECTURE_ID = specialist_architecture_id()


def stable_identifier(prefix: str, *parts: object) -> str:
    """Return a stable identifier without relying on Python's hash seed."""
    material = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


@dataclass(frozen=True)
class SpecialistInput:
    """The complete and immutable specialist input boundary.

    ``recurrent_state`` is a tuple rather than an ndarray so a specialist
    cannot mutate the core.  The registry also gives every specialist a deep
    copy of the event payload, preventing one implementation from using the
    shallowly-frozen ``CognitiveEvent.payload`` as a side channel.
    """

    event: CognitiveEvent
    previous_broadcast: Broadcast | None
    recurrent_state: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.recurrent_state:
            raise ValueError("specialist recurrent_state must not be empty")
        if any(not isinstance(value, float) for value in self.recurrent_state):
            raise TypeError("specialist recurrent_state must contain floats")

    def isolated_copy(self) -> SpecialistInput:
        event = replace(self.event, payload=copy.deepcopy(self.event.payload))
        return SpecialistInput(
            event=event,
            previous_broadcast=self.previous_broadcast,
            recurrent_state=self.recurrent_state,
        )


class CognitiveSpecialist(Protocol):
    """Public structural interface implemented by every specialist."""

    name: str
    state_schema: str
    state_schema_version: int
    implementation_id: str

    def candidate(self, specialist_input: SpecialistInput) -> CandidateContent: ...

    def checkpoint_state(self) -> dict[str, Any]: ...

    def decode_checkpoint_state(self, snapshot: Mapping[str, Any]) -> object: ...

    def install_checkpoint_state(self, state: object) -> None: ...

    @property
    def private_state(self) -> dict[str, Any]: ...


SpecialistFactory = Callable[[], CognitiveSpecialist]


@dataclass(frozen=True)
class PerceptionState:
    updates: int = 0
    last_digest: str = ""
    last_event_id: str = ""
    last_observation: str = ""


@dataclass(frozen=True)
class AutobiographicalMemoryState:
    updates: int = 0
    last_digest: str = ""
    episode_events: tuple[tuple[str, tuple[str, ...]], ...] = ()
    last_broadcast_id: str = ""


@dataclass(frozen=True)
class SemanticBeliefState:
    updates: int = 0
    last_digest: str = ""
    last_event_id: str = ""
    source_counts: tuple[tuple[str, int], ...] = ()
    belief_cues: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorldPredictionState:
    updates: int = 0
    last_digest: str = ""
    transitions: int = 0
    last_direction: str = "stable"


@dataclass(frozen=True)
class SelfModelState:
    updates: int = 0
    last_digest: str = ""
    uncertainty: float = 0.5


@dataclass(frozen=True)
class AffectSpecialistState:
    updates: int = 0
    last_digest: str = ""
    appraisals: int = 0
    need_pressure: float = 0.0


@dataclass(frozen=True)
class PlanningState:
    updates: int = 0
    last_digest: str = ""
    plans_considered: int = 0
    last_prior_broadcast_id: str = ""


@dataclass(frozen=True)
class ActionEvaluationState:
    updates: int = 0
    last_digest: str = ""
    evaluations: int = 0
    last_risk: float = 0.0


StateT = TypeVar("StateT")


def _require_exact_keys(data: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"incompatible {context}: missing={missing}, unknown={unknown}")


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _text_sequence(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    return tuple(_text(item, name=f"{name} item") for item in value)


class _BaseSpecialist(ABC, Generic[StateT]):
    name: str
    kind: EpistemicKind = "hypothesis"
    state_schema_version = SPECIALIST_STATE_SCHEMA_VERSION
    implementation_version = 1

    def __init__(self, name: str, initial_state: StateT) -> None:
        self.name = name
        self.state_schema = f"conscio.v3.specialist.{name}"
        class_path = f"{type(self).__module__}.{type(self).__qualname__}"
        self.implementation_id = _implementation_id(
            class_path,
            self.implementation_version,
        )
        self._state = initial_state

    @property
    def private_state(self) -> dict[str, Any]:
        """Return a detached diagnostic view, never the owned mutable value."""
        return copy.deepcopy(asdict(cast(Any, self._state)))

    @property
    def private(self) -> dict[str, Any]:
        """Backward-compatible detached view of the old ``private`` mapping."""
        return self.private_state

    def candidate(self, specialist_input: SpecialistInput) -> CandidateContent:
        digest = hashlib.sha256(
            json.dumps(
                specialist_input.recurrent_state,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:12]
        next_state, content = self._transition(specialist_input, digest)
        self._state = next_state
        state = asdict(cast(Any, next_state))
        updates = cast(int, state["updates"])
        recurrent = specialist_input.recurrent_state
        confidence = max(0.0, min(1.0, 0.55 + 0.25 * abs(recurrent[0])))
        salience = max(0.0, min(1.0, 0.45 + 0.30 * abs(recurrent[1])))
        prior_id = (
            specialist_input.previous_broadcast.broadcast_id if specialist_input.previous_broadcast is not None else ""
        )
        candidate_id = stable_identifier(
            "cand",
            self.name,
            specialist_input.event.event_id,
            updates,
            digest,
            prior_id,
            content,
        )
        return CandidateContent(
            specialist=self.name,
            content=content,
            kind=self.kind,
            confidence=confidence,
            salience=salience,
            evidence_event_ids=(specialist_input.event.event_id,),
            private_state_version=updates,
            candidate_id=candidate_id,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema": self.state_schema,
            "schema_version": self.state_schema_version,
            "state": self.private_state,
        }

    def decode_checkpoint_state(self, snapshot: Mapping[str, Any]) -> object:
        _require_exact_keys(
            snapshot,
            {"schema", "schema_version", "state"},
            context=f"{self.name} specialist checkpoint envelope",
        )
        if snapshot["schema"] != self.state_schema:
            raise ValueError(f"incompatible {self.name} specialist state schema: {snapshot['schema']!r}")
        if snapshot["schema_version"] != self.state_schema_version:
            raise ValueError(
                f"incompatible {self.name} specialist state schema version: {snapshot['schema_version']!r}"
            )
        raw = snapshot["state"]
        if not isinstance(raw, Mapping):
            raise ValueError(f"incompatible {self.name} specialist state payload")
        return self._decode_state(raw)

    def install_checkpoint_state(self, state: object) -> None:
        if not isinstance(state, type(self._state)):
            raise ValueError(f"incompatible decoded state for {self.name} specialist")
        self._state = state

    @abstractmethod
    def _transition(
        self,
        specialist_input: SpecialistInput,
        digest: str,
    ) -> tuple[StateT, str]: ...

    @abstractmethod
    def _decode_state(self, raw: Mapping[str, Any]) -> StateT: ...

    @staticmethod
    def _previous_summary(specialist_input: SpecialistInput, *, limit: int = 160) -> str:
        previous = specialist_input.previous_broadcast
        if previous is None or not previous.candidates:
            return "none"
        return " | ".join(candidate.content for candidate in previous.candidates)[:limit]


class PerceptionSpecialist(_BaseSpecialist[PerceptionState]):
    kind: EpistemicKind = "observation"

    def __init__(self) -> None:
        super().__init__("perception", PerceptionState())

    def _transition(self, specialist_input: SpecialistInput, digest: str) -> tuple[PerceptionState, str]:
        content = str(specialist_input.event.payload.get("content", ""))[:240]
        state = PerceptionState(
            updates=self._state.updates + 1,
            last_digest=digest,
            last_event_id=specialist_input.event.event_id,
            last_observation=content,
        )
        observation = f"Observed {specialist_input.event.source}/{specialist_input.event.event_type}: {content}"
        return state, observation

    def _decode_state(self, raw: Mapping[str, Any]) -> PerceptionState:
        _require_exact_keys(
            raw,
            {"updates", "last_digest", "last_event_id", "last_observation"},
            context="perception specialist state",
        )
        return PerceptionState(
            updates=_integer(raw["updates"], name="perception updates"),
            last_digest=_text(raw["last_digest"], name="perception last_digest"),
            last_event_id=_text(raw["last_event_id"], name="perception last_event_id"),
            last_observation=_text(raw["last_observation"], name="perception last_observation"),
        )


class AutobiographicalMemorySpecialist(_BaseSpecialist[AutobiographicalMemoryState]):
    kind: EpistemicKind = "belief"

    def __init__(self) -> None:
        super().__init__("autobiographical_memory", AutobiographicalMemoryState())

    def _transition(self, specialist_input: SpecialistInput, digest: str) -> tuple[AutobiographicalMemoryState, str]:
        episodes = {episode_id: list(event_ids) for episode_id, event_ids in self._state.episode_events}
        event_ids = episodes.setdefault(specialist_input.event.episode_id, [])
        if specialist_input.event.event_id not in event_ids:
            event_ids.append(specialist_input.event.event_id)
        episodes[specialist_input.event.episode_id] = event_ids[-32:]
        bounded = sorted(episodes.items())[-16:]
        previous = specialist_input.previous_broadcast
        prior_id = previous.broadcast_id if previous is not None else ""
        state = AutobiographicalMemoryState(
            updates=self._state.updates + 1,
            last_digest=digest,
            episode_events=tuple((key, tuple(value)) for key, value in bounded),
            last_broadcast_id=prior_id,
        )
        summary = self._previous_summary(specialist_input, limit=200)
        return state, f"Autobiographical continuity cue from the preceding broadcast: {summary}"

    def _decode_state(self, raw: Mapping[str, Any]) -> AutobiographicalMemoryState:
        _require_exact_keys(
            raw,
            {"updates", "last_digest", "episode_events", "last_broadcast_id"},
            context="autobiographical_memory specialist state",
        )
        pairs = raw["episode_events"]
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            raise ValueError("autobiographical episode_events must be a sequence")
        episodes: list[tuple[str, tuple[str, ...]]] = []
        for pair in pairs:
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError("autobiographical episode_events entries must be pairs")
            episodes.append(
                (
                    _text(pair[0], name="autobiographical episode id"),
                    _text_sequence(pair[1], name="autobiographical event ids"),
                )
            )
        if tuple(sorted(episodes)) != tuple(episodes):
            raise ValueError("autobiographical episode_events must be sorted")
        return AutobiographicalMemoryState(
            updates=_integer(raw["updates"], name="autobiographical updates"),
            last_digest=_text(raw["last_digest"], name="autobiographical last_digest"),
            episode_events=tuple(episodes),
            last_broadcast_id=_text(raw["last_broadcast_id"], name="autobiographical last_broadcast_id"),
        )


class SemanticBeliefSpecialist(_BaseSpecialist[SemanticBeliefState]):
    kind: EpistemicKind = "belief"

    def __init__(self) -> None:
        super().__init__("semantic_belief", SemanticBeliefState())

    def _transition(self, specialist_input: SpecialistInput, digest: str) -> tuple[SemanticBeliefState, str]:
        raw_content = str(specialist_input.event.payload.get("content", ""))
        cue = " ".join(raw_content.casefold().split()[:12])
        counts = dict(self._state.source_counts)
        cues = list(self._state.belief_cues)
        if specialist_input.event.event_id != self._state.last_event_id:
            counts[specialist_input.event.source] = counts.get(specialist_input.event.source, 0) + 1
            if cue and cue not in cues:
                cues.append(cue)
        state = SemanticBeliefState(
            updates=self._state.updates + 1,
            last_digest=digest,
            last_event_id=specialist_input.event.event_id,
            source_counts=tuple(sorted(counts.items())),
            belief_cues=tuple(cues[-32:]),
        )
        label = cue or "no propositional content"
        return state, f"Source-backed semantic cue from {specialist_input.event.source}: {label}"

    def _decode_state(self, raw: Mapping[str, Any]) -> SemanticBeliefState:
        _require_exact_keys(
            raw,
            {"updates", "last_digest", "last_event_id", "source_counts", "belief_cues"},
            context="semantic_belief specialist state",
        )
        pairs = raw["source_counts"]
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            raise ValueError("semantic source_counts must be a sequence")
        counts: list[tuple[str, int]] = []
        for pair in pairs:
            if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError("semantic source_counts entries must be pairs")
            counts.append(
                (
                    _text(pair[0], name="semantic source"),
                    _integer(pair[1], name="semantic source count"),
                )
            )
        if tuple(sorted(counts)) != tuple(counts):
            raise ValueError("semantic source_counts must be sorted")
        return SemanticBeliefState(
            updates=_integer(raw["updates"], name="semantic updates"),
            last_digest=_text(raw["last_digest"], name="semantic last_digest"),
            last_event_id=_text(raw["last_event_id"], name="semantic last_event_id"),
            source_counts=tuple(counts),
            belief_cues=_text_sequence(raw["belief_cues"], name="semantic belief_cues"),
        )


class WorldPredictionSpecialist(_BaseSpecialist[WorldPredictionState]):
    def __init__(self) -> None:
        super().__init__("world_prediction", WorldPredictionState())

    def _transition(self, specialist_input: SpecialistInput, digest: str) -> tuple[WorldPredictionState, str]:
        direction = "stable" if abs(specialist_input.recurrent_state[2]) < 0.5 else "changing"
        state = WorldPredictionState(
            updates=self._state.updates + 1,
            last_digest=digest,
            transitions=self._state.transitions + 1,
            last_direction=direction,
        )
        prior = self._previous_summary(specialist_input, limit=100)
        return state, f"World-state hypothesis: interaction appears {direction}; prior signal: {prior}"

    def _decode_state(self, raw: Mapping[str, Any]) -> WorldPredictionState:
        _require_exact_keys(
            raw,
            {"updates", "last_digest", "transitions", "last_direction"},
            context="world_prediction specialist state",
        )
        direction = _text(raw["last_direction"], name="world prediction direction")
        if direction not in {"stable", "changing"}:
            raise ValueError("world prediction direction is incompatible")
        return WorldPredictionState(
            updates=_integer(raw["updates"], name="world prediction updates"),
            last_digest=_text(raw["last_digest"], name="world prediction last_digest"),
            transitions=_integer(raw["transitions"], name="world prediction transitions"),
            last_direction=direction,
        )


class SelfModelSpecialist(_BaseSpecialist[SelfModelState]):
    kind: EpistemicKind = "self_report"

    def __init__(self) -> None:
        super().__init__("self_model", SelfModelState())

    def _transition(self, specialist_input: SpecialistInput, digest: str) -> tuple[SelfModelState, str]:
        uncertainty = max(0.0, min(1.0, 1.0 - abs(specialist_input.recurrent_state[3])))
        state = SelfModelState(
            updates=self._state.updates + 1,
            last_digest=digest,
            uncertainty=uncertainty,
        )
        return (
            state,
            f"Estimated future uncertainty={uncertainty:.3f}; this is a model estimate, not an observation.",
        )

    def _decode_state(self, raw: Mapping[str, Any]) -> SelfModelState:
        _require_exact_keys(
            raw,
            {"updates", "last_digest", "uncertainty"},
            context="self_model specialist state",
        )
        uncertainty = _number(raw["uncertainty"], name="self model uncertainty")
        if not 0.0 <= uncertainty <= 1.0:
            raise ValueError("self model uncertainty must be in [0, 1]")
        return SelfModelState(
            updates=_integer(raw["updates"], name="self model updates"),
            last_digest=_text(raw["last_digest"], name="self model last_digest"),
            uncertainty=uncertainty,
        )


class AffectSpecialist(_BaseSpecialist[AffectSpecialistState]):
    def __init__(self) -> None:
        super().__init__("affect", AffectSpecialistState())

    def _transition(self, specialist_input: SpecialistInput, digest: str) -> tuple[AffectSpecialistState, str]:
        pressure = min(
            1.0,
            (abs(specialist_input.recurrent_state[5]) + abs(specialist_input.recurrent_state[6])) / 2.0,
        )
        state = AffectSpecialistState(
            updates=self._state.updates + 1,
            last_digest=digest,
            appraisals=self._state.appraisals + 1,
            need_pressure=pressure,
        )
        return state, f"Need-error appraisal pressure={pressure:.3f} prepared for action valuation."

    def _decode_state(self, raw: Mapping[str, Any]) -> AffectSpecialistState:
        _require_exact_keys(
            raw,
            {"updates", "last_digest", "appraisals", "need_pressure"},
            context="affect specialist state",
        )
        pressure = _number(raw["need_pressure"], name="affect need_pressure")
        if not 0.0 <= pressure <= 1.0:
            raise ValueError("affect need_pressure must be in [0, 1]")
        return AffectSpecialistState(
            updates=_integer(raw["updates"], name="affect updates"),
            last_digest=_text(raw["last_digest"], name="affect last_digest"),
            appraisals=_integer(raw["appraisals"], name="affect appraisals"),
            need_pressure=pressure,
        )


class PlanningSpecialist(_BaseSpecialist[PlanningState]):
    kind: EpistemicKind = "idea"

    def __init__(self) -> None:
        super().__init__("planning", PlanningState())

    def _transition(self, specialist_input: SpecialistInput, digest: str) -> tuple[PlanningState, str]:
        previous = specialist_input.previous_broadcast
        prior_id = previous.broadcast_id if previous is not None else ""
        state = PlanningState(
            updates=self._state.updates + 1,
            last_digest=digest,
            plans_considered=self._state.plans_considered + 1,
            last_prior_broadcast_id=prior_id,
        )
        prior = self._previous_summary(specialist_input, limit=120)
        return (
            state,
            "Candidate policy: interpret the observation, predict consequences, "
            f"then answer or act under constraints; prior signal: {prior}",
        )

    def _decode_state(self, raw: Mapping[str, Any]) -> PlanningState:
        _require_exact_keys(
            raw,
            {"updates", "last_digest", "plans_considered", "last_prior_broadcast_id"},
            context="planning specialist state",
        )
        return PlanningState(
            updates=_integer(raw["updates"], name="planning updates"),
            last_digest=_text(raw["last_digest"], name="planning last_digest"),
            plans_considered=_integer(raw["plans_considered"], name="planning plans_considered"),
            last_prior_broadcast_id=_text(raw["last_prior_broadcast_id"], name="planning last_prior_broadcast_id"),
        )


class ActionEvaluationSpecialist(_BaseSpecialist[ActionEvaluationState]):
    def __init__(self) -> None:
        super().__init__("action_evaluation", ActionEvaluationState())

    def _transition(self, specialist_input: SpecialistInput, digest: str) -> tuple[ActionEvaluationState, str]:
        risk = max(0.0, min(1.0, 0.05 + 0.35 * abs(specialist_input.recurrent_state[7])))
        state = ActionEvaluationState(
            updates=self._state.updates + 1,
            last_digest=digest,
            evaluations=self._state.evaluations + 1,
            last_risk=risk,
        )
        prior = self._previous_summary(specialist_input, limit=100)
        return state, f"Action appraisal estimated risk={risk:.3f}; prior evidence: {prior}"

    def _decode_state(self, raw: Mapping[str, Any]) -> ActionEvaluationState:
        _require_exact_keys(
            raw,
            {"updates", "last_digest", "evaluations", "last_risk"},
            context="action_evaluation specialist state",
        )
        risk = _number(raw["last_risk"], name="action evaluation last_risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("action evaluation last_risk must be in [0, 1]")
        return ActionEvaluationState(
            updates=_integer(raw["updates"], name="action evaluation updates"),
            last_digest=_text(raw["last_digest"], name="action evaluation last_digest"),
            evaluations=_integer(raw["evaluations"], name="action evaluation evaluations"),
            last_risk=risk,
        )


def default_specialist_factories() -> dict[str, SpecialistFactory]:
    """Return fresh factories in the deterministic specialist order."""
    return {
        "perception": PerceptionSpecialist,
        "autobiographical_memory": AutobiographicalMemorySpecialist,
        "semantic_belief": SemanticBeliefSpecialist,
        "world_prediction": WorldPredictionSpecialist,
        "self_model": SelfModelSpecialist,
        "affect": AffectSpecialist,
        "planning": PlanningSpecialist,
        "action_evaluation": ActionEvaluationSpecialist,
    }


class SpecialistRegistry:
    """Own specialist instances and enforce isolation, lesions, and state schemas."""

    def __init__(
        self,
        *,
        factories: Mapping[str, SpecialistFactory] | None = None,
        removed_specialists: Collection[str] = (),
    ) -> None:
        configured = dict(factories or default_specialist_factories())
        unknown_factories = set(configured) - set(SPECIALIST_ORDER)
        missing_factories = set(SPECIALIST_ORDER) - set(configured)
        if unknown_factories or missing_factories:
            raise ValueError(
                "incompatible specialist factories: "
                f"missing={sorted(missing_factories)}, unknown={sorted(unknown_factories)}"
            )
        removed = frozenset(removed_specialists)
        unknown_lesions = removed - set(SPECIALIST_ORDER)
        if unknown_lesions:
            raise ValueError(f"unknown specialist lesion(s): {sorted(unknown_lesions)}")
        self.removed_specialists = removed
        self._specialists: dict[str, CognitiveSpecialist] = {}
        for name in SPECIALIST_ORDER:
            if name in removed:
                continue
            specialist = configured[name]()
            if specialist.name != name:
                raise ValueError(f"specialist factory {name!r} returned incompatible name {specialist.name!r}")
            if _CONTENT_ID_RE.fullmatch(specialist.implementation_id) is None:
                raise ValueError(f"specialist factory {name!r} returned an invalid implementation identity")
            self._specialists[name] = specialist

    @property
    def specialists(self) -> Mapping[str, CognitiveSpecialist]:
        return self._specialists.copy()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name in SPECIALIST_ORDER if name in self._specialists)

    @property
    def architecture_id(self) -> str:
        return specialist_architecture_id(
            self.names,
            implementation_ids={name: self._specialists[name].implementation_id for name in self.names},
            state_schemas={
                name: (
                    self._specialists[name].state_schema,
                    self._specialists[name].state_schema_version,
                )
                for name in self.names
            },
        )

    def candidates(
        self,
        specialist_input: SpecialistInput,
        *,
        removed_specialists: Collection[str] = (),
    ) -> tuple[CandidateContent, ...]:
        removed = frozenset(removed_specialists)
        unknown = removed - set(SPECIALIST_ORDER)
        if unknown:
            raise ValueError(f"unknown specialist lesion(s): {sorted(unknown)}")
        return tuple(
            self._specialists[name].candidate(specialist_input.isolated_copy())
            for name in self.names
            if name not in removed
        )

    def checkpoint_states(self) -> dict[str, dict[str, Any]]:
        return {name: self._specialists[name].checkpoint_state() for name in self.names}

    def decode_checkpoint_states(self, snapshots: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
        expected = set(self.names)
        actual = set(snapshots)
        if actual != expected:
            missing = sorted(expected - actual)
            unknown = sorted(actual - expected)
            raise ValueError(f"incompatible specialist checkpoint set: missing={missing}, unknown={unknown}")
        return {name: self._specialists[name].decode_checkpoint_state(snapshots[name]) for name in self.names}

    def install_checkpoint_states(self, states: Mapping[str, object]) -> None:
        if set(states) != set(self.names):
            raise ValueError("decoded specialist checkpoint set changed before installation")
        for name in self.names:
            self._specialists[name].install_checkpoint_state(states[name])

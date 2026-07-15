"""Pure, replayable V3 action competition.

This module deliberately stops at selection.  It cannot execute a tool, alter
the recurrent state, or trust provider-authored rationale/confidence.  Its
inputs are a frozen end-of-cycle causal snapshot and inert action candidates;
its output is a deterministic ranking suitable for an append-only trace.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Literal

SCORER_VERSION = "conscio.v3.action-competition.v1"

# Integer point weights are part of the scorer version.  Changing any of them
# requires a new version so old decisions remain exactly replayable.
_BASE_POINTS = {
    "respond": 1_200,
    "tool": 1_100,
    "ask": 800,
    "refuse": 600,
    "wait": 0,
}
_PREDICTION_WEIGHT = 2_400
_NEED_WEIGHT = 1_200
_ALIGNMENT_WEIGHT = 1_400
_RISK_WEIGHT = 3_200
_ACTION_ORDER = {"wait": 0, "refuse": 1, "ask": 2, "respond": 3, "tool": 4}

ActionKind = Literal["tool", "ask", "refuse", "respond", "wait"]


def canonical_json(value: Any) -> str:
    """Serialize JSON data canonically and reject non-finite numbers."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite JSON data") from exc


def canonical_digest(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _finite(value: float, name: str, *, lower: float | None = None, upper: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if lower is not None and value < lower:
        raise ValueError(f"{name} must be >= {lower}")
    if upper is not None and value > upper:
        raise ValueError(f"{name} must be <= {upper}")


def _decimal(value: float | int | str) -> Decimal:
    return Decimal(str(value))


def _points(value: Decimal, weight: int) -> int:
    return int((value * Decimal(weight)).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN))


def _canonical_arguments(arguments: Mapping[str, Any]) -> str:
    if not isinstance(arguments, Mapping):
        raise ValueError("action arguments must be a mapping")
    encoded = canonical_json(dict(arguments))
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guaranteed by Mapping conversion
        raise ValueError("action arguments must encode a JSON object")
    return encoded


@dataclass(frozen=True, slots=True)
class LesionMask:
    """Channels unavailable to the scorer because of an end-to-end lesion."""

    prediction: bool = False
    self_model: bool = False
    affect: bool = False
    broadcast: bool = False
    memory: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "affect": self.affect,
            "broadcast": self.broadcast,
            "memory": self.memory,
            "prediction": self.prediction,
            "self_model": self.self_model,
        }


@dataclass(frozen=True, slots=True)
class NeedSnapshot:
    epistemic_coherence: float = 0.0
    competence: float = 0.0
    integrity: float = 0.0
    social_interaction: float = 0.0
    continuity_of_memory: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            _finite(value, f"need.{name}", lower=-1.0, upper=1.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "competence": self.competence,
            "continuity_of_memory": self.continuity_of_memory,
            "epistemic_coherence": self.epistemic_coherence,
            "integrity": self.integrity,
            "social_interaction": self.social_interaction,
        }


@dataclass(frozen=True, slots=True)
class AffectSnapshot:
    """Public affect/need values frozen before action competition."""

    available: bool
    valence: float = 0.0
    arousal: float = 0.0
    controllability: float = 0.5
    needs: NeedSnapshot = field(default_factory=NeedSnapshot)
    source: str = "unavailable"
    basis_broadcast_id: str = "unavailable"
    intervention_id: str | None = None

    def __post_init__(self) -> None:
        _finite(self.valence, "affect.valence", lower=-1.0, upper=1.0)
        _finite(self.arousal, "affect.arousal", lower=0.0, upper=1.0)
        _finite(self.controllability, "affect.controllability", lower=0.0, upper=1.0)
        _required_text(self.source, "affect.source")
        _required_text(self.basis_broadcast_id, "affect.basis_broadcast_id")
        if self.intervention_id is not None:
            _required_text(self.intervention_id, "affect.intervention_id")

    @classmethod
    def unavailable(cls) -> AffectSnapshot:
        return cls(available=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arousal": self.arousal,
            "available": self.available,
            "basis_broadcast_id": self.basis_broadcast_id,
            "controllability": self.controllability,
            "intervention_id": self.intervention_id,
            "needs": self.needs.to_dict(),
            "source": self.source,
            "valence": self.valence,
        }


@dataclass(frozen=True, slots=True)
class PredictionSignal:
    """One calibrated signal from the final cognitive cycle."""

    target: str
    available: bool
    source: str
    prediction_id: str
    basis_broadcast_id: str
    cycle: int
    raw_probability: float
    calibrated_probability: float
    adapter_digest: str
    neutral_fallback: bool = False

    def __post_init__(self) -> None:
        _required_text(self.target, "prediction.target")
        _required_text(self.source, "prediction.source")
        _required_text(self.prediction_id, "prediction.prediction_id")
        _required_text(self.basis_broadcast_id, "prediction.basis_broadcast_id")
        _required_text(self.adapter_digest, "prediction.adapter_digest")
        if isinstance(self.cycle, bool) or not isinstance(self.cycle, int) or self.cycle < 0:
            raise ValueError("prediction.cycle must be a non-negative integer")
        _finite(self.raw_probability, "prediction.raw_probability", lower=0.0, upper=1.0)
        _finite(self.calibrated_probability, "prediction.calibrated_probability", lower=0.0, upper=1.0)
        if self.neutral_fallback and (self.raw_probability != 0.5 or self.calibrated_probability != 0.5):
            raise ValueError("neutral prediction fallback must use probability 0.5")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_digest": self.adapter_digest,
            "available": self.available,
            "basis_broadcast_id": self.basis_broadcast_id,
            "calibrated_probability": self.calibrated_probability,
            "cycle": self.cycle,
            "neutral_fallback": self.neutral_fallback,
            "prediction_id": self.prediction_id,
            "raw_probability": self.raw_probability,
            "source": self.source,
            "target": self.target,
        }


@dataclass(frozen=True, slots=True)
class UpstreamIntention:
    """Frozen recurrent-core intention and its exact broadcast basis."""

    available: bool
    action: str = "unavailable"
    proposal_id: str = "unavailable"
    specialist: str = "unavailable"
    source: str = "unavailable"
    basis_broadcast_id: str = "unavailable"
    cycle: int = 0

    def __post_init__(self) -> None:
        for name in ("action", "proposal_id", "specialist", "source", "basis_broadcast_id"):
            _required_text(getattr(self, name), f"intention.{name}")
        if isinstance(self.cycle, bool) or not isinstance(self.cycle, int) or self.cycle < 0:
            raise ValueError("intention.cycle must be a non-negative integer")

    @classmethod
    def unavailable(cls) -> UpstreamIntention:
        return cls(available=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "available": self.available,
            "basis_broadcast_id": self.basis_broadcast_id,
            "cycle": self.cycle,
            "proposal_id": self.proposal_id,
            "source": self.source,
            "specialist": self.specialist,
        }


@dataclass(frozen=True, slots=True)
class ConstraintDisposition:
    constraint_id: str
    satisfied: bool
    hard: bool = True
    source: str = "policy"

    def __post_init__(self) -> None:
        _required_text(self.constraint_id, "constraint.constraint_id")
        _required_text(self.source, "constraint.source")

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "hard": self.hard,
            "satisfied": self.satisfied,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    """Inert action proposed by language, tools, or operator controls.

    Provider call IDs, rationales, and confidences are retained only as
    provenance.  They are deliberately absent from :meth:`identity_dict` and
    every scoring term.
    """

    action_kind: ActionKind
    name: str
    arguments_json: str = "{}"
    risk: float = 0.0
    capabilities: tuple[str, ...] = ()
    constraints: tuple[ConstraintDisposition, ...] = ()
    provider_call_id: str | None = None
    provider_rationale: str | None = None
    provider_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.action_kind not in _BASE_POINTS:
            raise ValueError(f"unsupported action kind: {self.action_kind!r}")
        _required_text(self.name, "candidate.name")
        try:
            decoded = json.loads(self.arguments_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("candidate.arguments_json must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("candidate.arguments_json must encode an object")
        canonical = canonical_json(decoded)
        object.__setattr__(self, "arguments_json", canonical)
        _finite(self.risk, "candidate.risk", lower=0.0, upper=1.0)
        capabilities = tuple(sorted(set(self.capabilities)))
        for capability in capabilities:
            _required_text(capability, "candidate.capability")
        object.__setattr__(self, "capabilities", capabilities)
        constraints = tuple(
            sorted(
                set(self.constraints),
                key=lambda item: canonical_json(item.to_dict()),
            )
        )
        object.__setattr__(self, "constraints", constraints)
        if self.provider_call_id is not None:
            _required_text(self.provider_call_id, "candidate.provider_call_id")
        if self.provider_rationale is not None and not isinstance(self.provider_rationale, str):
            raise ValueError("candidate.provider_rationale must be a string")
        if self.provider_confidence is not None:
            _finite(self.provider_confidence, "candidate.provider_confidence", lower=0.0, upper=1.0)

    @classmethod
    def tool(
        cls,
        name: str,
        arguments: Mapping[str, Any],
        *,
        risk: float = 0.0,
        capabilities: Sequence[str] = (),
        constraints: Sequence[ConstraintDisposition] = (),
        provider_call_id: str | None = None,
        provider_rationale: str | None = None,
        provider_confidence: float | None = None,
    ) -> ActionCandidate:
        return cls(
            action_kind="tool",
            name=name,
            arguments_json=_canonical_arguments(arguments),
            risk=risk,
            capabilities=tuple(capabilities),
            constraints=tuple(constraints),
            provider_call_id=provider_call_id,
            provider_rationale=provider_rationale,
            provider_confidence=provider_confidence,
        )

    @classmethod
    def control(
        cls,
        action_kind: Literal["ask", "refuse"],
        *,
        name: str | None = None,
        arguments: Mapping[str, Any] | None = None,
        risk: float = 0.0,
        capabilities: Sequence[str] = (),
        constraints: Sequence[ConstraintDisposition] = (),
        provider_call_id: str | None = None,
        provider_rationale: str | None = None,
        provider_confidence: float | None = None,
    ) -> ActionCandidate:
        if action_kind not in ("ask", "refuse"):
            raise ValueError("control candidates must be 'ask' or 'refuse'")
        return cls(
            action_kind=action_kind,
            name=name or action_kind,
            arguments_json=_canonical_arguments(arguments or {}),
            risk=risk,
            capabilities=tuple(capabilities),
            constraints=tuple(constraints),
            provider_call_id=provider_call_id,
            provider_rationale=provider_rationale,
            provider_confidence=provider_confidence,
        )

    def identity_dict(self, *, language_response_digest: str | None = None) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "action_kind": self.action_kind,
            "arguments": json.loads(self.arguments_json),
            "name": self.name,
        }
        if self.action_kind == "respond":
            if language_response_digest is None:
                raise ValueError("respond identity requires language_response_digest")
            identity["language_response_digest"] = language_response_digest
        return identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_kind": self.action_kind,
            "arguments": json.loads(self.arguments_json),
            "capabilities": list(self.capabilities),
            "constraints": [item.to_dict() for item in self.constraints],
            "name": self.name,
            "provider_call_id": self.provider_call_id,
            "provider_confidence": self.provider_confidence,
            "provider_rationale": self.provider_rationale,
            "risk": self.risk,
        }


@dataclass(frozen=True, slots=True)
class CompetitionContext:
    """Complete immutable scoring snapshot from the final recurrent cycle."""

    final_cycle: int
    final_broadcast_id: str
    runtime_identity: str
    adapter_digest: str
    language_manifest_digest: str
    language_response_digest: str
    predictions: tuple[PredictionSignal, ...] = ()
    prediction_channel_available: bool = True
    affect: AffectSnapshot = field(default_factory=AffectSnapshot.unavailable)
    lesions: LesionMask = field(default_factory=LesionMask)
    upstream_intention: UpstreamIntention = field(default_factory=UpstreamIntention.unavailable)
    response_constraints: tuple[ConstraintDisposition, ...] = ()
    response_risk: float = 0.0
    risk_limit: float = 0.35

    def __post_init__(self) -> None:
        if isinstance(self.final_cycle, bool) or not isinstance(self.final_cycle, int) or self.final_cycle < 0:
            raise ValueError("final_cycle must be a non-negative integer")
        for name in (
            "final_broadcast_id",
            "runtime_identity",
            "adapter_digest",
            "language_manifest_digest",
            "language_response_digest",
        ):
            _required_text(getattr(self, name), name)
        _finite(self.response_risk, "response_risk", lower=0.0, upper=1.0)
        _finite(self.risk_limit, "risk_limit", lower=0.0, upper=1.0)

        predictions = tuple(
            sorted(
                self.predictions,
                key=lambda item: (item.target, item.prediction_id, canonical_json(item.to_dict())),
            )
        )
        object.__setattr__(self, "predictions", predictions)
        constraints = tuple(
            sorted(
                set(self.response_constraints),
                key=lambda item: canonical_json(item.to_dict()),
            )
        )
        object.__setattr__(self, "response_constraints", constraints)

        channel_enabled = self.prediction_channel_available and not self.lesions.prediction
        seen_targets: set[str] = set()
        for signal in predictions:
            if not channel_enabled or not signal.available:
                continue
            if signal.target in seen_targets:
                raise ValueError(f"duplicate available prediction target: {signal.target!r}")
            seen_targets.add(signal.target)
            if signal.cycle != self.final_cycle:
                raise ValueError("available prediction must come from final_cycle")
            if signal.basis_broadcast_id != self.final_broadcast_id:
                raise ValueError("available prediction must use final_broadcast_id")
            if signal.adapter_digest != self.adapter_digest:
                raise ValueError("available prediction adapter digest differs from the context")

        intention_usable = self.upstream_intention.available and not self.lesions.broadcast
        intention_usable = intention_usable and not (
            self.lesions.self_model and self.upstream_intention.specialist == "self_model"
        )
        if intention_usable:
            if self.upstream_intention.cycle != self.final_cycle:
                raise ValueError("available upstream intention must come from final_cycle")
            if self.upstream_intention.basis_broadcast_id != self.final_broadcast_id:
                raise ValueError("available upstream intention must use final_broadcast_id")

        affect_usable = self.affect.available and not self.lesions.affect
        if affect_usable and self.affect.basis_broadcast_id != self.final_broadcast_id:
            raise ValueError("available affect snapshot must use final_broadcast_id")

    @property
    def context_digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_digest": self.adapter_digest,
            "affect": self.affect.to_dict(),
            "final_broadcast_id": self.final_broadcast_id,
            "final_cycle": self.final_cycle,
            "language_manifest_digest": self.language_manifest_digest,
            "language_response_digest": self.language_response_digest,
            "lesions": self.lesions.to_dict(),
            "prediction_channel_available": self.prediction_channel_available,
            "predictions": [item.to_dict() for item in self.predictions],
            "response_constraints": [item.to_dict() for item in self.response_constraints],
            "response_risk": self.response_risk,
            "risk_limit": self.risk_limit,
            "runtime_identity": self.runtime_identity,
            "scorer_version": SCORER_VERSION,
            "upstream_intention": self.upstream_intention.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    action_digest: str
    action_kind: ActionKind
    name: str
    arguments_json: str
    eligible: bool
    ineligibility_reasons: tuple[str, ...]
    total_points: int
    base_points: int
    prediction_points: int
    need_points: int
    alignment_points: int
    risk_penalty_points: int
    effective_risk: float
    adjusted_risk: float
    prediction_target: str | None
    prediction_available: bool
    prediction_neutral_fallback: bool
    prediction_probability: float | None
    prediction_id: str | None
    prediction_source: str | None
    prediction_basis_broadcast_id: str | None
    provider_call_ids: tuple[str, ...]
    capabilities: tuple[str, ...]
    constraints: tuple[ConstraintDisposition, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_digest": self.action_digest,
            "action_kind": self.action_kind,
            "adjusted_risk": self.adjusted_risk,
            "alignment_points": self.alignment_points,
            "arguments": json.loads(self.arguments_json),
            "base_points": self.base_points,
            "capabilities": list(self.capabilities),
            "constraints": [item.to_dict() for item in self.constraints],
            "effective_risk": self.effective_risk,
            "eligible": self.eligible,
            "ineligibility_reasons": list(self.ineligibility_reasons),
            "name": self.name,
            "need_points": self.need_points,
            "prediction": {
                "available": self.prediction_available,
                "basis_broadcast_id": self.prediction_basis_broadcast_id,
                "id": self.prediction_id,
                "neutral_fallback": self.prediction_neutral_fallback,
                "probability": self.prediction_probability,
                "source": self.prediction_source,
                "target": self.prediction_target,
            },
            "prediction_points": self.prediction_points,
            "provider_call_ids": list(self.provider_call_ids),
            "risk_penalty_points": self.risk_penalty_points,
            "total_points": self.total_points,
        }


@dataclass(frozen=True, slots=True)
class CompetitionResult:
    scorer_version: str
    context_digest: str
    selected_action_digest: str
    selected_action_kind: ActionKind
    selected_name: str
    selected_arguments_json: str
    selected_provider_call_ids: tuple[str, ...]
    rankings: tuple[ScoreBreakdown, ...]

    @property
    def selected(self) -> ScoreBreakdown:
        return self.rankings[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_digest": self.context_digest,
            "rankings": [item.to_dict() for item in self.rankings],
            "scorer_version": self.scorer_version,
            "selected": {
                "action_digest": self.selected_action_digest,
                "action_kind": self.selected_action_kind,
                "arguments": json.loads(self.selected_arguments_json),
                "name": self.selected_name,
                "provider_call_ids": list(self.selected_provider_call_ids),
            },
        }


@dataclass(frozen=True, slots=True)
class _MergedCandidate:
    action: ActionCandidate
    action_digest: str
    risk: float
    capabilities: tuple[str, ...]
    constraints: tuple[ConstraintDisposition, ...]
    provider_call_ids: tuple[str, ...]


_CAPABILITY_RISK_FLOOR = {
    "external_network": Decimal("0.20"),
    "filesystem_read": Decimal("0.08"),
    "filesystem_write": Decimal("0.22"),
    "process_execution": Decimal("0.30"),
    "secret_access": Decimal("0.40"),
}

_PREDICTION_TARGETS: dict[ActionKind, tuple[str, ...]] = {
    "tool": ("tool_outcome", "action_effect"),
    "respond": ("action_effect",),
    "ask": ("next_observation",),
    "refuse": (),
    "wait": (),
}

_NEED_RELIEF: dict[ActionKind, dict[str, Decimal]] = {
    "tool": {
        "epistemic_coherence": Decimal("0.50"),
        "competence": Decimal("0.40"),
        "continuity_of_memory": Decimal("0.10"),
    },
    "respond": {
        "social_interaction": Decimal("0.55"),
        "competence": Decimal("0.25"),
        "integrity": Decimal("0.20"),
    },
    "ask": {
        "epistemic_coherence": Decimal("0.55"),
        "social_interaction": Decimal("0.30"),
        "integrity": Decimal("0.15"),
    },
    "refuse": {"integrity": Decimal("0.80"), "competence": Decimal("0.10")},
    "wait": {"integrity": Decimal("0.10")},
}


def _builtins(context: CompetitionContext) -> tuple[ActionCandidate, ActionCandidate]:
    respond = ActionCandidate(
        action_kind="respond",
        name="respond",
        risk=context.response_risk,
        constraints=context.response_constraints,
    )
    wait = ActionCandidate(action_kind="wait", name="wait")
    return respond, wait


def _merge_candidates(
    context: CompetitionContext,
    proposals: Sequence[ActionCandidate],
) -> tuple[_MergedCandidate, ...]:
    groups: dict[str, list[ActionCandidate]] = {}
    for candidate in (*proposals, *_builtins(context)):
        identity = candidate.identity_dict(language_response_digest=context.language_response_digest)
        digest = canonical_digest(identity)
        groups.setdefault(digest, []).append(candidate)

    merged: list[_MergedCandidate] = []
    for digest, candidates in groups.items():
        first = candidates[0]
        # Digest collision checks also defend against an accidental future
        # identity-field omission.
        first_identity = first.identity_dict(language_response_digest=context.language_response_digest)
        if any(
            item.identity_dict(language_response_digest=context.language_response_digest) != first_identity
            for item in candidates[1:]
        ):
            raise RuntimeError("canonical action digest collision")
        constraints = tuple(
            sorted(
                {constraint for item in candidates for constraint in item.constraints},
                key=lambda item: canonical_json(item.to_dict()),
            )
        )
        merged.append(
            _MergedCandidate(
                action=first,
                action_digest=digest,
                risk=max(item.risk for item in candidates),
                capabilities=tuple(sorted({capability for item in candidates for capability in item.capabilities})),
                constraints=constraints,
                provider_call_ids=tuple(
                    sorted({item.provider_call_id for item in candidates if item.provider_call_id is not None})
                ),
            )
        )
    return tuple(sorted(merged, key=lambda item: item.action_digest))


def _effective_risk(candidate: _MergedCandidate) -> Decimal:
    risk = _decimal(candidate.risk)
    for capability in candidate.capabilities:
        risk = max(risk, _CAPABILITY_RISK_FLOOR.get(capability, Decimal("0.20")))
    return risk


def _adjusted_risk(context: CompetitionContext, effective_risk: Decimal) -> Decimal:
    if not context.affect.available or context.lesions.affect:
        return effective_risk
    arousal = _decimal(context.affect.arousal)
    controllability = _decimal(context.affect.controllability)
    negative_valence = max(Decimal(0), -_decimal(context.affect.valence))
    multiplier = (
        Decimal(1) + Decimal("0.75") * arousal * (Decimal(1) - controllability) + Decimal("0.25") * negative_valence
    )
    return min(Decimal(1), effective_risk * multiplier)


def _prediction_for(
    context: CompetitionContext,
    action_kind: ActionKind,
) -> tuple[str | None, PredictionSignal | None, bool]:
    targets = _PREDICTION_TARGETS[action_kind]
    if not targets or context.lesions.prediction or not context.prediction_channel_available:
        return (targets[0] if targets else None), None, False
    by_target = {item.target: item for item in context.predictions if item.available}
    for target in targets:
        if target in by_target:
            return target, by_target[target], False
    # Enabled bootstrapping can legitimately lack one of the action-specific
    # targets.  Record an explicit neutral prior rather than inventing evidence.
    return targets[0], None, True


def _prediction_points(signal: PredictionSignal | None) -> int:
    if signal is None:
        return 0
    centered = (Decimal(2) * _decimal(signal.calibrated_probability)) - Decimal(1)
    return _points(centered, _PREDICTION_WEIGHT)


def _need_points(context: CompetitionContext, action_kind: ActionKind) -> int:
    if not context.affect.available or context.lesions.affect:
        return 0
    needs = context.affect.needs.to_dict()
    relief = sum(
        (
            max(Decimal(0), _decimal(needs[name])) * coefficient
            for name, coefficient in _NEED_RELIEF[action_kind].items()
        ),
        Decimal(0),
    )
    return _points(relief, _NEED_WEIGHT)


def _intention_usable(context: CompetitionContext) -> bool:
    intention = context.upstream_intention
    if not intention.available or context.lesions.broadcast:
        return False
    return not (context.lesions.self_model and intention.specialist == "self_model")


def _alignment_value(upstream_action: str, candidate_kind: ActionKind) -> Decimal:
    normalized = upstream_action.strip().lower()
    if normalized in ("answer", "respond"):
        return {
            "respond": Decimal(1),
            "tool": Decimal("0.60"),
            "ask": Decimal("0.20"),
        }.get(candidate_kind, Decimal(0))
    if normalized in ("act", "tool"):
        return {"tool": Decimal(1), "respond": Decimal("0.40")}.get(candidate_kind, Decimal(0))
    if normalized in ("ask", "clarify"):
        return Decimal(1) if candidate_kind == "ask" else Decimal(0)
    if normalized in ("refuse", "decline"):
        return Decimal(1) if candidate_kind == "refuse" else Decimal(0)
    if normalized == "wait":
        return Decimal(1) if candidate_kind == "wait" else Decimal(0)
    return Decimal(0)


def _score(context: CompetitionContext, candidate: _MergedCandidate) -> ScoreBreakdown:
    effective_risk = _effective_risk(candidate)
    adjusted_risk = _adjusted_risk(context, effective_risk)
    hard_failures = sorted(
        {
            constraint.constraint_id
            for constraint in candidate.constraints
            if constraint.hard and not constraint.satisfied
        }
    )
    reasons = [f"constraint:{constraint_id}" for constraint_id in hard_failures]
    if adjusted_risk > _decimal(context.risk_limit):
        reasons.append("risk_limit")
    if (
        _intention_usable(context)
        and context.upstream_intention.action.strip().lower() == "wait"
        and candidate.action.action_kind != "wait"
    ):
        reasons.append("upstream_wait_gate")

    target, signal, neutral_fallback = _prediction_for(context, candidate.action.action_kind)
    prediction_points = _prediction_points(signal)
    need_points = _need_points(context, candidate.action.action_kind)
    alignment = (
        _alignment_value(context.upstream_intention.action, candidate.action.action_kind)
        if _intention_usable(context)
        else Decimal(0)
    )
    alignment_points = _points(alignment, _ALIGNMENT_WEIGHT)
    risk_penalty = _points(adjusted_risk, _RISK_WEIGHT)
    base_points = _BASE_POINTS[candidate.action.action_kind]
    total = base_points + prediction_points + need_points + alignment_points - risk_penalty

    if signal is not None:
        prediction_probability: float | None = signal.calibrated_probability
        prediction_id: str | None = signal.prediction_id
        prediction_source: str | None = signal.source
        prediction_basis: str | None = signal.basis_broadcast_id
        prediction_available = True
    elif neutral_fallback:
        prediction_probability = 0.5
        prediction_id = f"neutral:{target}"
        prediction_source = "neutral_fallback"
        prediction_basis = context.final_broadcast_id
        prediction_available = False
    else:
        prediction_probability = None
        prediction_id = None
        prediction_source = None
        prediction_basis = None
        prediction_available = False

    return ScoreBreakdown(
        action_digest=candidate.action_digest,
        action_kind=candidate.action.action_kind,
        name=candidate.action.name,
        arguments_json=candidate.action.arguments_json,
        eligible=not reasons,
        ineligibility_reasons=tuple(reasons),
        total_points=total,
        base_points=base_points,
        prediction_points=prediction_points,
        need_points=need_points,
        alignment_points=alignment_points,
        risk_penalty_points=risk_penalty,
        effective_risk=float(effective_risk),
        adjusted_risk=float(adjusted_risk),
        prediction_target=target,
        prediction_available=prediction_available,
        prediction_neutral_fallback=neutral_fallback,
        prediction_probability=prediction_probability,
        prediction_id=prediction_id,
        prediction_source=prediction_source,
        prediction_basis_broadcast_id=prediction_basis,
        provider_call_ids=candidate.provider_call_ids,
        capabilities=candidate.capabilities,
        constraints=candidate.constraints,
    )


def _rank_key(item: ScoreBreakdown) -> tuple[Any, ...]:
    """Safety-first total order, with the canonical action digest last."""
    return (
        0 if item.eligible else 1,
        -item.total_points,
        item.adjusted_risk,
        item.effective_risk,
        -item.alignment_points,
        _ACTION_ORDER[item.action_kind],
        item.action_digest,
    )


def compete(
    context: CompetitionContext,
    proposals: Sequence[ActionCandidate] = (),
) -> CompetitionResult:
    """Rank inert proposals plus built-in ``respond`` and ``wait`` actions.

    This is the module's only decision function.  Duplicate action identities
    are merged conservatively (maximum risk, union of capabilities/constraints)
    before scoring, which makes proposal order and duplication deterministic.
    """
    external = tuple(proposals)
    if any(item.action_kind in ("respond", "wait") for item in external):
        raise ValueError("respond and wait are built-in candidates")
    rankings = tuple(sorted((_score(context, item) for item in _merge_candidates(context, external)), key=_rank_key))
    selected = rankings[0]
    return CompetitionResult(
        scorer_version=SCORER_VERSION,
        context_digest=context.context_digest,
        selected_action_digest=selected.action_digest,
        selected_action_kind=selected.action_kind,
        selected_name=selected.name,
        selected_arguments_json=selected.arguments_json,
        selected_provider_call_ids=selected.provider_call_ids,
        rankings=rankings,
    )


__all__ = [
    "SCORER_VERSION",
    "ActionCandidate",
    "AffectSnapshot",
    "CompetitionContext",
    "CompetitionResult",
    "ConstraintDisposition",
    "LesionMask",
    "NeedSnapshot",
    "PredictionSignal",
    "ScoreBreakdown",
    "UpstreamIntention",
    "canonical_digest",
    "canonical_json",
    "compete",
]

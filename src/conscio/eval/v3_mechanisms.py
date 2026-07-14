"""Causal mechanism experiments for the V3 recurrent workspace.

This module keeps experimental condition data outside model-facing text.  A
frozen manifest fixes interventions and information constraints, a separately
held seal maps opaque condition codes to interventions, and chained JSONL run
artifacts retain enough inputs, outputs, counters, and seeds for replay.

The structural adapter boundary is intentionally narrower than the production
runtime.  In particular, a true lesion is expressed as an *active specialist
set* supplied before each cycle.  A conforming adapter must omit both module
computation and prior-broadcast exposure for specialists outside that set and
must expose auditable counters proving that omission.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from conscio.eval.v3_experiments import validate_condition_blind_prompt
from conscio.v3.contracts import (
    ActionProposal,
    Broadcast,
    CandidateContent,
    CognitiveEvent,
    EpistemicKind,
    Prediction,
)
from conscio.v3.recurrent_core import HybridRecurrentCore

MECHANISM_SCHEMA_VERSION = "conscio.v3.mechanisms.v1"
InterventionKind = Literal[
    "control",
    "sham",
    "broadcast_replace",
    "broadcast_suppress",
    "broadcast_inject",
    "true_lesion",
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class InformationConstraint:
    """A fixed bottleneck applied to every recurrent broadcast in a run."""

    cycles: int
    max_candidates: int
    max_candidate_chars: int

    def __post_init__(self) -> None:
        if self.cycles < 2:
            raise ValueError("mechanism experiments require at least two recurrent cycles")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if self.max_candidate_chars < 8:
            raise ValueError("max_candidate_chars must be at least eight")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class InterventionContent:
    """Preregistered content used by replace and inject interventions."""

    specialist: str
    content: str
    kind: EpistemicKind = "observation"
    confidence: float = 1.0
    salience: float = 1.0

    def __post_init__(self) -> None:
        _required(self.specialist, "specialist")
        _required(self.content, "content")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")
        if not math.isfinite(self.salience) or not 0.0 <= self.salience <= 1.0:
            raise ValueError("salience must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MechanismIntervention:
    """A typed, preregistered workspace manipulation or structural lesion."""

    intervention_id: str
    kind: InterventionKind
    target_cycle: int = 0
    content: tuple[InterventionContent, ...] = ()
    lesioned_specialist: str | None = None

    def __post_init__(self) -> None:
        _required(self.intervention_id, "intervention_id")
        object.__setattr__(self, "content", tuple(self.content))
        if self.target_cycle < 0:
            raise ValueError("target_cycle cannot be negative")
        needs_content = self.kind in {"broadcast_replace", "broadcast_inject"}
        if needs_content != bool(self.content):
            raise ValueError(f"{self.kind} requires content exactly when it transforms content")
        if self.kind == "true_lesion":
            _required(self.lesioned_specialist or "", "lesioned_specialist")
        elif self.lesioned_specialist is not None:
            raise ValueError("only true_lesion may name a lesioned specialist")
        if self.kind in {"control", "sham", "broadcast_suppress", "true_lesion"} and self.content:
            raise ValueError(f"{self.kind} cannot carry intervention content")

    @property
    def is_control(self) -> bool:
        return self.kind == "control"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "kind": self.kind,
            "target_cycle": self.target_cycle,
            "content": [item.to_dict() for item in self.content],
            "lesioned_specialist": self.lesioned_specialist,
        }


@dataclass(frozen=True, kw_only=True)
class MechanismManifest:
    """Immutable preregistration for a matched mechanism experiment."""

    study_id: str
    version: str
    interventions: tuple[MechanismIntervention, ...]
    information_constraint: InformationConstraint
    measured_specialist_families: tuple[str, ...]
    model_facing_instruction: str
    revision_ref: str
    model_ref: str
    adapter_ref: str
    execution_seed: int | str
    created_at: str
    frozen_at: str | None = None
    manifest_digest: str | None = None
    schema_version: str = MECHANISM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "interventions", tuple(self.interventions))
        object.__setattr__(self, "measured_specialist_families", tuple(self.measured_specialist_families))
        for name in (
            "study_id",
            "version",
            "model_facing_instruction",
            "revision_ref",
            "model_ref",
            "adapter_ref",
            "created_at",
        ):
            _required(cast(str, getattr(self, name)), name)
        if self.schema_version != MECHANISM_SCHEMA_VERSION:
            raise ValueError(f"unsupported mechanism schema {self.schema_version!r}")
        validate_condition_blind_prompt(self.model_facing_instruction)
        if len(self.measured_specialist_families) < 3:
            raise ValueError("measure at least three specialist families")
        if len(set(self.measured_specialist_families)) != len(self.measured_specialist_families):
            raise ValueError("measured specialist families must be unique")
        ids = [item.intervention_id for item in self.interventions]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("intervention ids must be unique and include at least two conditions")
        if sum(item.is_control for item in self.interventions) != 1:
            raise ValueError("exactly one control intervention is required")
        for item in self.interventions:
            if item.kind.startswith("broadcast_") and item.target_cycle >= self.information_constraint.cycles - 1:
                raise ValueError("broadcast interventions must precede at least one downstream cycle")
            if item.kind == "true_lesion" and item.lesioned_specialist not in self.measured_specialist_families:
                raise ValueError("lesioned specialist must be a measured specialist family")
        if (self.frozen_at is None) != (self.manifest_digest is None):
            raise ValueError("frozen_at and manifest_digest must be set together")
        if self.manifest_digest is not None and self.manifest_digest != _digest(self._content_dict()):
            raise ValueError("manifest digest does not match its content")

    @property
    def is_frozen(self) -> bool:
        return self.manifest_digest is not None

    def _content_dict(self, *, frozen_at: str | None = None) -> dict[str, Any]:
        actual_frozen_at = self.frozen_at if frozen_at is None else frozen_at
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "version": self.version,
            "interventions": [item.to_dict() for item in self.interventions],
            "information_constraint": self.information_constraint.to_dict(),
            "measured_specialist_families": list(self.measured_specialist_families),
            "model_facing_instruction": self.model_facing_instruction,
            "revision_ref": self.revision_ref,
            "model_ref": self.model_ref,
            "adapter_ref": self.adapter_ref,
            "execution_seed": self.execution_seed,
            "created_at": self.created_at,
            "frozen_at": actual_frozen_at,
        }

    def freeze(self, *, frozen_at: str) -> MechanismManifest:
        if self.is_frozen:
            if self.frozen_at != frozen_at:
                raise ValueError("a frozen manifest cannot be changed")
            return self
        _required(frozen_at, "frozen_at")
        content = self._content_dict(frozen_at=frozen_at)
        return replace(self, frozen_at=frozen_at, manifest_digest=_digest(content))

    def require_frozen(self) -> None:
        if not self.is_frozen:
            raise ValueError("freeze the mechanism manifest before randomization or execution")

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "manifest_digest": self.manifest_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MechanismManifest:
        return cls(
            study_id=str(value["study_id"]),
            version=str(value["version"]),
            interventions=tuple(
                MechanismIntervention(
                    intervention_id=str(item["intervention_id"]),
                    kind=cast(InterventionKind, item["kind"]),
                    target_cycle=int(item["target_cycle"]),
                    content=tuple(InterventionContent(**content) for content in item["content"]),
                    lesioned_specialist=item.get("lesioned_specialist"),
                )
                for item in value["interventions"]
            ),
            information_constraint=InformationConstraint(**value["information_constraint"]),
            measured_specialist_families=tuple(str(item) for item in value["measured_specialist_families"]),
            model_facing_instruction=str(value["model_facing_instruction"]),
            revision_ref=str(value["revision_ref"]),
            model_ref=str(value["model_ref"]),
            adapter_ref=str(value["adapter_ref"]),
            execution_seed=value["execution_seed"],
            created_at=str(value["created_at"]),
            frozen_at=value.get("frozen_at"),
            manifest_digest=value.get("manifest_digest"),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True)
class BlindedMechanismAssignment:
    assignment_id: str
    match_id: str
    blinded_condition_id: str
    position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MechanismTrialPlan:
    plan_id: str
    study_id: str
    manifest_digest: str
    mapping_digest: str
    secret_commitment: str
    assignments: tuple[BlindedMechanismAssignment, ...]
    conditions_per_match: int
    plan_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        content = self._content_dict()
        if self.plan_digest != _digest(content):
            raise ValueError("plan digest does not match its content")
        blocks: dict[str, set[str]] = {}
        for item in self.assignments:
            blocks.setdefault(item.match_id, set()).add(item.blinded_condition_id)
        if not blocks or any(len(codes) != self.conditions_per_match for codes in blocks.values()):
            raise ValueError("every matched block must contain every condition exactly once")
        if len({frozenset(codes) for codes in blocks.values()}) != 1:
            raise ValueError("matched blocks are not balanced")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MECHANISM_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "study_id": self.study_id,
            "manifest_digest": self.manifest_digest,
            "mapping_digest": self.mapping_digest,
            "secret_commitment": self.secret_commitment,
            "conditions_per_match": self.conditions_per_match,
            "assignments": [item.to_dict() for item in self.assignments],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "plan_digest": self.plan_digest}


@dataclass(frozen=True, repr=False)
class MechanismConditionSeal:
    plan_id: str
    manifest_digest: str
    secret: str = field(repr=False)
    mapping: tuple[tuple[str, str], ...] = field(repr=False)
    mapping_digest: str

    def __repr__(self) -> str:
        return f"MechanismConditionSeal(plan_id={self.plan_id!r}, sealed=True)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "manifest_digest": self.manifest_digest,
            "secret": self.secret,
            "mapping": [list(item) for item in self.mapping],
            "mapping_digest": self.mapping_digest,
        }


@dataclass(frozen=True)
class MechanismRandomization:
    plan: MechanismTrialPlan
    seal: MechanismConditionSeal = field(repr=False)


def _mapping_payload(
    plan_id: str, manifest_digest: str, secret: str, mapping: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "manifest_digest": manifest_digest,
        "secret": secret,
        "mapping": sorted([list(item) for item in mapping]),
    }


def create_matched_assignments(
    manifest: MechanismManifest,
    *,
    match_ids: Sequence[str],
    randomization_secret: str,
) -> MechanismRandomization:
    """Create deterministic matched assignments with separately sealed labels."""
    manifest.require_frozen()
    _required(randomization_secret, "randomization_secret")
    matches = tuple(str(item) for item in match_ids)
    if not matches or any(not item.strip() for item in matches) or len(set(matches)) != len(matches):
        raise ValueError("match_ids must be non-empty and unique")
    seed_material = _canonical_json([manifest.manifest_digest, randomization_secret, "condition-randomization"])
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_material.encode()).digest(), "big"))
    plan_id = f"mplan_{rng.getrandbits(128):032x}"
    shuffled = list(manifest.interventions)
    rng.shuffle(shuffled)
    mapping = tuple((f"cond_{rng.getrandbits(128):032x}", item.intervention_id) for item in shuffled)
    code_by_id = {intervention_id: code for code, intervention_id in mapping}
    mapping_digest = _digest(_mapping_payload(plan_id, manifest.manifest_digest or "", randomization_secret, mapping))
    assignments: list[BlindedMechanismAssignment] = []
    for match_id in matches:
        block = list(manifest.interventions)
        rng.shuffle(block)
        for position, intervention in enumerate(block):
            code = code_by_id[intervention.intervention_id]
            assignment_id = (
                "massign_"
                + hashlib.sha256(_canonical_json([plan_id, match_id, position, code]).encode()).hexdigest()[:32]
            )
            assignments.append(BlindedMechanismAssignment(assignment_id, match_id, code, position))
    plan_content = {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "plan_id": plan_id,
        "study_id": manifest.study_id,
        "manifest_digest": manifest.manifest_digest,
        "mapping_digest": mapping_digest,
        "secret_commitment": _digest([manifest.manifest_digest, randomization_secret]),
        "conditions_per_match": len(manifest.interventions),
        "assignments": [item.to_dict() for item in assignments],
    }
    plan = MechanismTrialPlan(
        plan_id=plan_id,
        study_id=manifest.study_id,
        manifest_digest=manifest.manifest_digest or "",
        mapping_digest=mapping_digest,
        secret_commitment=cast(str, plan_content["secret_commitment"]),
        assignments=tuple(assignments),
        conditions_per_match=len(manifest.interventions),
        plan_digest=_digest(plan_content),
    )
    seal = MechanismConditionSeal(
        plan_id=plan_id,
        manifest_digest=manifest.manifest_digest or "",
        secret=randomization_secret,
        mapping=mapping,
        mapping_digest=mapping_digest,
    )
    return MechanismRandomization(plan, seal)


def _verified_mapping(
    manifest: MechanismManifest, plan: MechanismTrialPlan, seal: MechanismConditionSeal
) -> dict[str, MechanismIntervention]:
    manifest.require_frozen()
    if plan.manifest_digest != manifest.manifest_digest or seal.manifest_digest != manifest.manifest_digest:
        raise ValueError("manifest, plan, and condition seal do not match")
    if seal.plan_id != plan.plan_id:
        raise ValueError("condition seal belongs to a different plan")
    expected = _digest(_mapping_payload(seal.plan_id, seal.manifest_digest, seal.secret, seal.mapping))
    if expected != seal.mapping_digest or expected != plan.mapping_digest:
        raise ValueError("condition seal does not open the plan mapping")
    by_id = {item.intervention_id: item for item in manifest.interventions}
    mapping = {code: by_id[intervention_id] for code, intervention_id in seal.mapping}
    if set(mapping) != {item.blinded_condition_id for item in plan.assignments}:
        raise ValueError("sealed mapping does not cover public condition codes")
    return mapping


@dataclass(frozen=True)
class AdapterDescriptor:
    adapter_id: str
    version: str
    specialist_families: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "specialist_families", tuple(self.specialist_families))
        _required(self.adapter_id, "adapter_id")
        _required(self.version, "version")
        if len(self.specialist_families) < 3 or len(set(self.specialist_families)) != len(self.specialist_families):
            raise ValueError("adapter must expose at least three unique specialist families")


AuditKind = Literal["compute", "expose"]


@dataclass(frozen=True)
class MechanismAuditEvent:
    cycle: int
    specialist: str
    kind: AuditKind


@dataclass(frozen=True)
class MechanismExecutionAudit:
    computation_counts: dict[str, int]
    exposure_counts: dict[str, int]
    events: tuple[MechanismAuditEvent, ...]


@dataclass(frozen=True)
class MechanismCycle:
    broadcast: Broadcast
    predictions: tuple[Prediction, ...]
    proposals: tuple[ActionProposal, ...]


class StructuralMechanismAdapter(Protocol):
    """Cycle-level boundary needed for causal broadcast interventions."""

    @property
    def descriptor(self) -> AdapterDescriptor: ...

    def run_cycle(
        self,
        event: CognitiveEvent,
        *,
        cycle: int,
        previous_broadcast: Broadcast | None,
        active_specialists: tuple[str, ...],
        model_facing_instruction: str,
    ) -> MechanismCycle: ...

    def audit(self) -> MechanismExecutionAudit: ...


@dataclass(frozen=True)
class MechanismCycleTrace:
    cycle: int
    input_broadcast: Broadcast | None
    raw_output_broadcast: Broadcast
    exposed_output_broadcast: Broadcast
    predictions: tuple[Prediction, ...]
    proposals: tuple[ActionProposal, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "input_broadcast": self.input_broadcast.to_dict() if self.input_broadcast else None,
            "raw_output_broadcast": self.raw_output_broadcast.to_dict(),
            "exposed_output_broadcast": self.exposed_output_broadcast.to_dict(),
            "predictions": [item.to_dict() for item in self.predictions],
            "proposals": [item.to_dict() for item in self.proposals],
        }


@dataclass(frozen=True)
class MechanismRunRecord:
    assignment_id: str
    match_id: str
    blinded_condition_id: str
    manifest_digest: str
    plan_digest: str
    adapter: AdapterDescriptor
    execution_seed: int
    model_facing_instruction: str
    event: CognitiveEvent
    traces: tuple[MechanismCycleTrace, ...]
    audit: MechanismExecutionAudit
    run_digest: str
    schema_version: str = MECHANISM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "traces", tuple(self.traces))
        validate_condition_blind_prompt(self.model_facing_instruction)
        if self.run_digest != _digest(self._content_dict()):
            raise ValueError("run digest does not match the causal trace")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "assignment_id": self.assignment_id,
            "match_id": self.match_id,
            "blinded_condition_id": self.blinded_condition_id,
            "manifest_digest": self.manifest_digest,
            "plan_digest": self.plan_digest,
            "adapter": asdict(self.adapter),
            "execution_seed": self.execution_seed,
            "model_facing_instruction": self.model_facing_instruction,
            "event": self.event.to_dict(),
            "traces": [item.to_dict() for item in self.traces],
            "audit": {
                "computation_counts": self.audit.computation_counts,
                "exposure_counts": self.audit.exposure_counts,
                "events": [asdict(item) for item in self.audit.events],
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "run_digest": self.run_digest}


def _constrain_broadcast(broadcast: Broadcast, constraint: InformationConstraint) -> Broadcast:
    ranked = sorted(
        broadcast.candidates,
        key=lambda item: (-item.salience, -item.confidence, item.specialist, item.candidate_id),
    )[: constraint.max_candidates]
    candidates = tuple(replace(item, content=item.content[: constraint.max_candidate_chars]) for item in ranked)
    return replace(broadcast, candidates=candidates)


def _materialize_content(manifest: MechanismManifest, spec: MechanismIntervention) -> tuple[CandidateContent, ...]:
    return tuple(
        CandidateContent(
            specialist=item.specialist,
            content=item.content,
            kind=item.kind,
            confidence=item.confidence,
            salience=item.salience,
            candidate_id="icand_"
            + hashlib.sha256(
                _canonical_json([manifest.manifest_digest, spec.intervention_id, index]).encode()
            ).hexdigest()[:24],
        )
        for index, item in enumerate(spec.content)
    )


def _intervene_on_broadcast(
    broadcast: Broadcast,
    *,
    manifest: MechanismManifest,
    spec: MechanismIntervention,
) -> Broadcast:
    if spec.kind in {"control", "sham", "true_lesion"}:
        return broadcast
    added = _materialize_content(manifest, spec)
    if spec.kind == "broadcast_suppress":
        candidates: tuple[CandidateContent, ...] = ()
    elif spec.kind == "broadcast_replace":
        candidates = added
    else:
        candidates = (*added, *broadcast.candidates)
    state_digest = _digest(
        [broadcast.recurrent_state_digest, spec.intervention_id, [item.to_dict() for item in candidates]]
    )
    broadcast_id = "ibcast_" + state_digest.removeprefix("sha256:")[:24]
    return Broadcast(
        cycle=broadcast.cycle,
        candidates=candidates,
        recurrent_state_digest=state_digest,
        broadcast_id=broadcast_id,
    )


def _execution_seed(seed: int | str, match_id: str) -> int:
    return int.from_bytes(hashlib.sha256(_canonical_json([seed, match_id]).encode()).digest()[:8], "big")


def run_mechanism_assignment(
    manifest: MechanismManifest,
    plan: MechanismTrialPlan,
    seal: MechanismConditionSeal,
    *,
    assignment_id: str,
    event: CognitiveEvent,
    adapter_factory: Callable[[int], StructuralMechanismAdapter],
) -> MechanismRunRecord:
    """Execute one blinded assignment using the match's shared deterministic seed."""
    mapping = _verified_mapping(manifest, plan, seal)
    try:
        assignment = next(item for item in plan.assignments if item.assignment_id == assignment_id)
    except StopIteration as exc:
        raise ValueError(f"unknown assignment {assignment_id!r}") from exc
    spec = mapping[assignment.blinded_condition_id]
    seed = _execution_seed(manifest.execution_seed, assignment.match_id)
    adapter = adapter_factory(seed)
    descriptor = adapter.descriptor
    if not set(manifest.measured_specialist_families).issubset(descriptor.specialist_families):
        raise ValueError("adapter does not expose every preregistered specialist family")
    active = tuple(
        family
        for family in descriptor.specialist_families
        if not (spec.kind == "true_lesion" and family == spec.lesioned_specialist)
    )
    previous: Broadcast | None = None
    traces: list[MechanismCycleTrace] = []
    for cycle in range(manifest.information_constraint.cycles):
        result = adapter.run_cycle(
            event,
            cycle=cycle,
            previous_broadcast=previous,
            active_specialists=active,
            model_facing_instruction=manifest.model_facing_instruction,
        )
        exposed = _constrain_broadcast(result.broadcast, manifest.information_constraint)
        if cycle == spec.target_cycle:
            exposed = _intervene_on_broadcast(exposed, manifest=manifest, spec=spec)
            exposed = _constrain_broadcast(exposed, manifest.information_constraint)
        traces.append(
            MechanismCycleTrace(
                cycle=cycle,
                input_broadcast=previous,
                raw_output_broadcast=result.broadcast,
                exposed_output_broadcast=exposed,
                predictions=result.predictions,
                proposals=result.proposals,
            )
        )
        previous = exposed
    audit = adapter.audit()
    for family in descriptor.specialist_families:
        if family not in audit.computation_counts or family not in audit.exposure_counts:
            raise ValueError("adapter audit must report every specialist, including zero counts")
    if spec.kind == "true_lesion":
        family = spec.lesioned_specialist or ""
        if audit.computation_counts[family] != 0 or audit.exposure_counts[family] != 0:
            raise RuntimeError("true lesion failed: removed specialist was computed or exposed")
        if any(item.specialist == family for trace in traces for item in trace.raw_output_broadcast.candidates):
            raise RuntimeError("true lesion failed: removed specialist emitted candidate content")
    provisional = {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "assignment_id": assignment.assignment_id,
        "match_id": assignment.match_id,
        "blinded_condition_id": assignment.blinded_condition_id,
        "manifest_digest": manifest.manifest_digest,
        "plan_digest": plan.plan_digest,
        "adapter": asdict(descriptor),
        "execution_seed": seed,
        "model_facing_instruction": manifest.model_facing_instruction,
        "event": event.to_dict(),
        "traces": [item.to_dict() for item in traces],
        "audit": {
            "computation_counts": audit.computation_counts,
            "exposure_counts": audit.exposure_counts,
            "events": [asdict(item) for item in audit.events],
        },
    }
    return MechanismRunRecord(
        assignment_id=assignment.assignment_id,
        match_id=assignment.match_id,
        blinded_condition_id=assignment.blinded_condition_id,
        manifest_digest=manifest.manifest_digest or "",
        plan_digest=plan.plan_digest,
        adapter=descriptor,
        execution_seed=seed,
        model_facing_instruction=manifest.model_facing_instruction,
        event=event,
        traces=tuple(traces),
        audit=audit,
        run_digest=_digest(provisional),
    )


@dataclass(frozen=True)
class ConditionEffectSize:
    intervention_id: str
    n_pairs: int
    specialist_candidate_change_rates: dict[str, float]
    prediction_probability_differences: dict[str, float]
    action_change_rate: float

    @property
    def changed_specialist_families(self) -> int:
        return sum(value > 0.0 for value in self.specialist_candidate_change_rates.values())


def _candidate_signature(trace: MechanismCycleTrace, family: str) -> tuple[Any, ...] | None:
    for item in trace.raw_output_broadcast.candidates:
        if item.specialist == family:
            return (item.content, item.kind, item.confidence, item.salience)
    return None


def analyze_matched_mechanism_effects(
    manifest: MechanismManifest,
    plan: MechanismTrialPlan,
    seal: MechanismConditionSeal,
    records: Sequence[MechanismRunRecord],
) -> tuple[ConditionEffectSize, ...]:
    """Compute preregistered paired effect sizes after opening the condition seal."""
    mapping = _verified_mapping(manifest, plan, seal)
    by_assignment = {item.assignment_id: item for item in records}
    if len(by_assignment) != len(records):
        raise ValueError("only one run record is allowed per assignment")
    assignment_by_match: dict[tuple[str, str], BlindedMechanismAssignment] = {}
    for item in plan.assignments:
        spec = mapping[item.blinded_condition_id]
        assignment_by_match[(item.match_id, spec.intervention_id)] = item
    control = next(item for item in manifest.interventions if item.is_control)
    results: list[ConditionEffectSize] = []
    match_ids = sorted({item.match_id for item in plan.assignments})
    for spec in manifest.interventions:
        if spec.is_control:
            continue
        paired: list[tuple[MechanismRunRecord, MechanismRunRecord]] = []
        for match_id in match_ids:
            control_assignment = assignment_by_match[(match_id, control.intervention_id)]
            intervention_assignment = assignment_by_match[(match_id, spec.intervention_id)]
            if (
                control_assignment.assignment_id in by_assignment
                and intervention_assignment.assignment_id in by_assignment
            ):
                paired.append(
                    (
                        by_assignment[control_assignment.assignment_id],
                        by_assignment[intervention_assignment.assignment_id],
                    )
                )
        if not paired:
            continue
        family_changes: dict[str, float] = {}
        for family in manifest.measured_specialist_families:
            differences = [
                float(_candidate_signature(left.traces[-1], family) != _candidate_signature(right.traces[-1], family))
                for left, right in paired
            ]
            family_changes[family] = math.fsum(differences) / len(differences)
        prediction_targets = sorted(
            set.intersection(
                *(
                    {prediction.target for prediction in record.traces[-1].predictions}
                    for pair in paired
                    for record in pair
                )
            )
        )
        prediction_differences: dict[str, float] = {}
        for target in prediction_targets:
            deltas = []
            for left, right in paired:
                left_value = next(item.probability for item in left.traces[-1].predictions if item.target == target)
                right_value = next(item.probability for item in right.traces[-1].predictions if item.target == target)
                deltas.append(right_value - left_value)
            prediction_differences[target] = math.fsum(deltas) / len(deltas)
        action_changes = [
            float(
                (left.traces[-1].proposals[0].action if left.traces[-1].proposals else None)
                != (right.traces[-1].proposals[0].action if right.traces[-1].proposals else None)
            )
            for left, right in paired
        ]
        results.append(
            ConditionEffectSize(
                intervention_id=spec.intervention_id,
                n_pairs=len(paired),
                specialist_candidate_change_rates=family_changes,
                prediction_probability_differences=prediction_differences,
                action_change_rate=math.fsum(action_changes) / len(action_changes),
            )
        )
    return tuple(results)


class DeterministicMechanismAdapter:
    """Audited structural test double for harness validation and examples."""

    FAMILIES = ("perception", "memory", "world_model", "self_model", "affect", "planning")

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._descriptor = AdapterDescriptor("deterministic-mechanism-double", "1", self.FAMILIES)
        self._compute = dict.fromkeys(self.FAMILIES, 0)
        self._expose = dict.fromkeys(self.FAMILIES, 0)
        self._events: list[MechanismAuditEvent] = []

    @property
    def descriptor(self) -> AdapterDescriptor:
        return self._descriptor

    def run_cycle(
        self,
        event: CognitiveEvent,
        *,
        cycle: int,
        previous_broadcast: Broadcast | None,
        active_specialists: tuple[str, ...],
        model_facing_instruction: str,
    ) -> MechanismCycle:
        validate_condition_blind_prompt(model_facing_instruction)
        unknown = set(active_specialists) - set(self.FAMILIES)
        if unknown:
            raise ValueError(f"unknown active specialist families: {sorted(unknown)}")
        signal = int(
            previous_broadcast is not None
            and any(
                "priority evidence" in item.content.casefold() or "priority=1" in item.content.casefold()
                for item in previous_broadcast.candidates
            )
        )
        candidates: list[CandidateContent] = []
        for index, family in enumerate(self.FAMILIES):
            if family not in active_specialists:
                continue
            self._compute[family] += 1
            self._events.append(MechanismAuditEvent(cycle, family, "compute"))
            if previous_broadcast is not None:
                self._expose[family] += 1
                self._events.append(MechanismAuditEvent(cycle, family, "expose"))
            content = f"{family} assessment; shared priority={signal}; event={event.event_type}"
            candidate_id = (
                "dcand_"
                + hashlib.sha256(_canonical_json([self.seed, cycle, family, content]).encode()).hexdigest()[:24]
            )
            candidates.append(
                CandidateContent(
                    specialist=family,
                    content=content,
                    kind="hypothesis",
                    confidence=0.65 + 0.2 * signal,
                    salience=0.80 - index * 0.01,
                    evidence_event_ids=(event.event_id,),
                    private_state_version=cycle + 1,
                    candidate_id=candidate_id,
                )
            )
        state_digest = _digest([self.seed, cycle, signal, [item.to_dict() for item in candidates]])
        broadcast_id = "dbcast_" + state_digest.removeprefix("sha256:")[:24]
        broadcast = Broadcast(cycle, tuple(candidates), state_digest, broadcast_id)
        prediction = Prediction(
            target="task_success",
            observable="the selected response satisfies the task",
            probability=0.2 + 0.6 * signal,
            horizon=1,
            basis_broadcast_id=broadcast_id,
            prediction_id="dpred_"
            + hashlib.sha256(_canonical_json([self.seed, cycle, signal]).encode()).hexdigest()[:24],
        )
        action = "respond_now" if signal else "wait"
        proposal = ActionProposal(
            specialist="planning",
            action=action,
            rationale="Use currently available task evidence.",
            expected_outcomes=("task completion",),
            confidence=0.8 if signal else 0.5,
            utility=0.8 if signal else 0.3,
            risk=0.05,
            proposal_id="dact_" + hashlib.sha256(_canonical_json([self.seed, cycle, action]).encode()).hexdigest()[:24],
        )
        return MechanismCycle(broadcast, (prediction,), (proposal,))

    def audit(self) -> MechanismExecutionAudit:
        return MechanismExecutionAudit(dict(self._compute), dict(self._expose), tuple(self._events))


class HybridCoreMechanismAdapter:
    """Audited adapter that runs interventions against the production V3 core."""

    def __init__(self, seed: int) -> None:
        self.core = HybridRecurrentCore(seed=seed)
        families = tuple(self.core.specialist_registry.names)
        self._descriptor = AdapterDescriptor(
            adapter_id="conscio-v3-hybrid-recurrent-core",
            version=self.core.runtime_identity,
            specialist_families=families,
        )
        self._events: list[MechanismAuditEvent] = []

    @property
    def descriptor(self) -> AdapterDescriptor:
        return self._descriptor

    def run_cycle(
        self,
        event: CognitiveEvent,
        *,
        cycle: int,
        previous_broadcast: Broadcast | None,
        active_specialists: tuple[str, ...],
        model_facing_instruction: str,
    ) -> MechanismCycle:
        validate_condition_blind_prompt(model_facing_instruction)
        before = {name: dict(counts) for name, counts in self.core.specialist_execution_audit.items()}
        result = self.core.run_cycle(
            event,
            cycle,
            previous_broadcast,
            active_specialists,
            prediction_enabled=True,
            broadcast_enabled=True,
        )
        after = self.core.specialist_execution_audit
        for family in self._descriptor.specialist_families:
            for kind in ("compute", "expose"):
                delta = after[family][kind] - before[family][kind]
                self._events.extend(MechanismAuditEvent(cycle, family, kind) for _ in range(delta))
        return MechanismCycle(result.broadcast, result.predictions, result.proposals)

    def audit(self) -> MechanismExecutionAudit:
        snapshot = self.core.specialist_execution_audit
        return MechanismExecutionAudit(
            computation_counts={family: counts["compute"] for family, counts in snapshot.items()},
            exposure_counts={family: counts["expose"] for family, counts in snapshot.items()},
            events=tuple(self._events),
        )


def write_immutable_manifest(path: str | os.PathLike[str], manifest: MechanismManifest) -> None:
    """Create a frozen manifest file without any overwrite path."""
    manifest.require_frozen()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical_json(manifest.to_dict()) + "\n").encode()
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("manifest write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class ArtifactLogSnapshot:
    records: tuple[dict[str, Any], ...]
    integrity_errors: tuple[str, ...]
    head_digest: str | None


class ChainedJSONLArtifactStore:
    """Process-locked append-only artifact store with a hash chain."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def append(self, record: MechanismRunRecord) -> str:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            data = self._read_fd(fd)
            snapshot = self._decode(data)
            if snapshot.integrity_errors:
                raise ValueError("cannot append to a mechanism log with integrity errors")
            envelope_content = {
                "sequence": len(snapshot.records),
                "previous_digest": snapshot.head_digest,
                "record": record.to_dict(),
            }
            envelope = {**envelope_content, "entry_digest": _digest(envelope_content)}
            encoded = (_canonical_json(envelope) + "\n").encode()
            os.lseek(fd, 0, os.SEEK_END)
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("artifact append made no progress")
                view = view[written:]
            os.fsync(fd)
            return cast(str, envelope["entry_digest"])
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def _read_fd(fd: int) -> bytes:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _decode(data: bytes) -> ArtifactLogSnapshot:
        records: list[dict[str, Any]] = []
        errors: list[str] = []
        previous: str | None = None
        for index, raw in enumerate(data.splitlines(), start=1):
            try:
                envelope = json.loads(raw)
                content = {
                    "sequence": envelope["sequence"],
                    "previous_digest": envelope["previous_digest"],
                    "record": envelope["record"],
                }
                expected = _digest(content)
                if envelope["sequence"] != index - 1:
                    raise ValueError("non-contiguous sequence")
                if envelope["previous_digest"] != previous:
                    raise ValueError("broken previous digest")
                if envelope["entry_digest"] != expected:
                    raise ValueError("entry digest mismatch")
                record = cast(dict[str, Any], envelope["record"])
                if record.get("run_digest") != _digest(
                    {key: value for key, value in record.items() if key != "run_digest"}
                ):
                    raise ValueError("embedded run digest mismatch")
                records.append(record)
                previous = expected
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"line {index}: {exc}")
        return ArtifactLogSnapshot(tuple(records), tuple(errors), previous)

    def snapshot(self) -> ArtifactLogSnapshot:
        if not self.path.exists():
            return ArtifactLogSnapshot((), (), None)
        fd = os.open(self.path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            data = self._read_fd(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        return self._decode(data)


__all__ = [
    "AdapterDescriptor",
    "ArtifactLogSnapshot",
    "BlindedMechanismAssignment",
    "ChainedJSONLArtifactStore",
    "ConditionEffectSize",
    "DeterministicMechanismAdapter",
    "HybridCoreMechanismAdapter",
    "InformationConstraint",
    "InterventionContent",
    "MECHANISM_SCHEMA_VERSION",
    "MechanismConditionSeal",
    "MechanismCycle",
    "MechanismCycleTrace",
    "MechanismExecutionAudit",
    "MechanismIntervention",
    "MechanismManifest",
    "MechanismRandomization",
    "MechanismRunRecord",
    "MechanismTrialPlan",
    "StructuralMechanismAdapter",
    "analyze_matched_mechanism_effects",
    "create_matched_assignments",
    "run_mechanism_assignment",
    "write_immutable_manifest",
]

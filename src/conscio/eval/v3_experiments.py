"""Condition-blind, preregistered experiments for the V3 runtime.

The public objects in this module deliberately separate three roles:

* :class:`PreregistrationManifest` fixes hypotheses and analysis before data;
* :class:`BlindedTrialPlan` is safe to hand to run/artifact collectors; and
* :class:`UnblindingKey` is retained by an independent custodian until analysis.

The mapping seal is a SHA-256 commitment salted with a secret 256-bit nonce.
It is not an encrypted container: keeping the unblinding key separate is what
keeps condition labels out of run artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

MANIFEST_SCHEMA_VERSION = "conscio.v3.preregistration.v1"
RANDOMIZATION_SCHEMA_VERSION = "conscio.v3.randomization.v1"
_DIRECTIONS = frozenset({"increase", "decrease", "no_change"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class Hypothesis:
    """A claim fixed before the experiment is randomized."""

    hypothesis_id: str
    statement: str

    def __post_init__(self) -> None:
        _required_text(self.hypothesis_id, "hypothesis_id")
        _required_text(self.statement, "statement")

    def to_dict(self) -> dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id, "statement": self.statement}


@dataclass(frozen=True)
class PrimaryOutcome:
    """A preregistered observable and its fixed scoring rule."""

    outcome_id: str
    description: str
    metric: str
    scoring_rule: str

    def __post_init__(self) -> None:
        _required_text(self.outcome_id, "outcome_id")
        _required_text(self.description, "description")
        _required_text(self.metric, "metric")
        _required_text(self.scoring_rule, "scoring_rule")

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "description": self.description,
            "metric": self.metric,
            "scoring_rule": self.scoring_rule,
        }


@dataclass(frozen=True)
class DirectionalPrediction:
    """Expected lesion effect relative to the matched intact control.

    ``effect_threshold`` is applied to ``lesion - control``. For
    ``no_change``, it is the inclusive equivalence margin.
    """

    prediction_id: str
    hypothesis_id: str
    outcome_id: str
    intervention_id: str
    direction: str
    effect_threshold: float = 0.0

    def __post_init__(self) -> None:
        for name in ("prediction_id", "hypothesis_id", "outcome_id", "intervention_id"):
            _required_text(getattr(self, name), name)
        direction = self.direction.strip().lower()
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}")
        object.__setattr__(self, "direction", direction)
        threshold = _finite(self.effect_threshold, "effect_threshold")
        if threshold < 0.0:
            raise ValueError("effect_threshold must be non-negative")
        object.__setattr__(self, "effect_threshold", threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "hypothesis_id": self.hypothesis_id,
            "outcome_id": self.outcome_id,
            "intervention_id": self.intervention_id,
            "direction": self.direction,
            "effect_threshold": self.effect_threshold,
        }


@dataclass(frozen=True)
class AnalysisPlan:
    """Analysis choices that must be fixed before unblinding."""

    description: str
    unit_of_analysis: str = "matched_block"
    estimator: str = "mean paired difference (lesion - control)"
    missing_data_policy: str = "exclude only by preregistered criteria"
    alpha: float = 0.05
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        for name in ("description", "unit_of_analysis", "estimator", "missing_data_policy"):
            _required_text(getattr(self, name), name)
        alpha = _finite(self.alpha, "alpha")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        if not isinstance(self.calibration_bins, int) or self.calibration_bins < 1:
            raise ValueError("calibration_bins must be a positive integer")
        object.__setattr__(self, "alpha", alpha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "unit_of_analysis": self.unit_of_analysis,
            "estimator": self.estimator,
            "missing_data_policy": self.missing_data_policy,
            "alpha": self.alpha,
            "calibration_bins": self.calibration_bins,
        }


@dataclass(frozen=True, kw_only=True)
class PreregistrationManifest:
    """Immutable, content-addressed preregistration.

    A newly constructed value is an immutable *draft*. :meth:`freeze` returns
    a new value whose ``manifest_hash`` commits to every field, including the
    freeze timestamp. Randomization, unblinding, and analysis reject drafts.
    """

    study_id: str
    version: str
    hypotheses: tuple[Hypothesis, ...]
    primary_outcomes: tuple[PrimaryOutcome, ...]
    directional_predictions: tuple[DirectionalPrediction, ...]
    exclusion_criteria: tuple[str, ...]
    analysis_plan: AnalysisPlan
    revision_ref: str
    checkpoint_ref: str
    model_ref: str
    created_at: str = field(default_factory=_now)
    schema_version: str = MANIFEST_SCHEMA_VERSION
    frozen_at: str | None = None
    manifest_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        object.__setattr__(self, "primary_outcomes", tuple(self.primary_outcomes))
        object.__setattr__(self, "directional_predictions", tuple(self.directional_predictions))
        object.__setattr__(self, "exclusion_criteria", tuple(self.exclusion_criteria))
        for name in (
            "study_id",
            "version",
            "revision_ref",
            "checkpoint_ref",
            "model_ref",
            "created_at",
            "schema_version",
        ):
            _required_text(getattr(self, name), name)
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema: {self.schema_version}")
        if not self.hypotheses:
            raise ValueError("at least one hypothesis is required")
        if not self.primary_outcomes:
            raise ValueError("at least one primary outcome is required")
        if not self.directional_predictions:
            raise ValueError("at least one directional prediction is required")
        if not self.exclusion_criteria or any(not item.strip() for item in self.exclusion_criteria):
            raise ValueError("at least one non-empty exclusion criterion is required")
        self._validate_references()
        if (self.frozen_at is None) != (self.manifest_hash is None):
            raise ValueError("frozen_at and manifest_hash must either both be set or both be absent")
        if self.manifest_hash is not None:
            expected = _sha256(self._content_dict())
            if self.manifest_hash != expected:
                raise ValueError("manifest_hash does not match the manifest contents")

    def _validate_references(self) -> None:
        hypothesis_ids = [item.hypothesis_id for item in self.hypotheses]
        outcome_ids = [item.outcome_id for item in self.primary_outcomes]
        prediction_ids = [item.prediction_id for item in self.directional_predictions]
        for label, values in (
            ("hypothesis", hypothesis_ids),
            ("primary outcome", outcome_ids),
            ("directional prediction", prediction_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} id")
        for prediction in self.directional_predictions:
            if prediction.hypothesis_id not in hypothesis_ids:
                raise ValueError(
                    f"prediction {prediction.prediction_id!r} references unknown hypothesis "
                    f"{prediction.hypothesis_id!r}"
                )
            if prediction.outcome_id not in outcome_ids:
                raise ValueError(
                    f"prediction {prediction.prediction_id!r} references unknown outcome "
                    f"{prediction.outcome_id!r}"
                )

    @property
    def is_frozen(self) -> bool:
        return self.manifest_hash is not None

    def _content_dict(self, *, frozen_at: str | None = None) -> dict[str, Any]:
        actual_frozen_at = self.frozen_at if frozen_at is None else frozen_at
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "version": self.version,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "primary_outcomes": [item.to_dict() for item in self.primary_outcomes],
            "directional_predictions": [item.to_dict() for item in self.directional_predictions],
            "exclusion_criteria": list(self.exclusion_criteria),
            "analysis_plan": self.analysis_plan.to_dict(),
            "revision_ref": self.revision_ref,
            "checkpoint_ref": self.checkpoint_ref,
            "model_ref": self.model_ref,
            "created_at": self.created_at,
            "frozen_at": actual_frozen_at,
        }

    def freeze(self, *, frozen_at: str | None = None) -> PreregistrationManifest:
        """Return a content-addressed frozen manifest; freezing is idempotent."""
        if self.is_frozen:
            if frozen_at is not None and frozen_at != self.frozen_at:
                raise ValueError("a frozen manifest cannot be frozen at a different time")
            return self
        timestamp = frozen_at or _now()
        _required_text(timestamp, "frozen_at")
        digest = _sha256(self._content_dict(frozen_at=timestamp))
        return replace(self, frozen_at=timestamp, manifest_hash=digest)

    def require_frozen(self) -> None:
        if not self.is_frozen:
            raise ManifestNotFrozenError("freeze the preregistration before randomization or analysis")

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "manifest_hash": self.manifest_hash}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PreregistrationManifest:
        return cls(
            study_id=str(value["study_id"]),
            version=str(value["version"]),
            hypotheses=tuple(Hypothesis(**item) for item in value["hypotheses"]),
            primary_outcomes=tuple(PrimaryOutcome(**item) for item in value["primary_outcomes"]),
            directional_predictions=tuple(
                DirectionalPrediction(**item) for item in value["directional_predictions"]
            ),
            exclusion_criteria=tuple(str(item) for item in value["exclusion_criteria"]),
            analysis_plan=AnalysisPlan(**value["analysis_plan"]),
            revision_ref=str(value["revision_ref"]),
            checkpoint_ref=str(value["checkpoint_ref"]),
            model_ref=str(value["model_ref"]),
            created_at=str(value["created_at"]),
            schema_version=str(value["schema_version"]),
            frozen_at=value.get("frozen_at"),
            manifest_hash=value.get("manifest_hash"),
        )


class ManifestNotFrozenError(RuntimeError):
    """Raised when a post-registration operation is attempted on a draft."""


def freeze_manifest(
    manifest: PreregistrationManifest, *, frozen_at: str | None = None
) -> PreregistrationManifest:
    """Functional spelling of :meth:`PreregistrationManifest.freeze`."""
    return manifest.freeze(frozen_at=frozen_at)


@dataclass(frozen=True)
class PromptLeakageFinding:
    term: str
    excerpt: str
    message_index: int | None = None


class ArchitectureLeakageError(ValueError):
    """A supposedly condition-blind prompt disclosed experimental machinery."""

    def __init__(self, findings: Sequence[PromptLeakageFinding]) -> None:
        self.findings = tuple(findings)
        terms = ", ".join(sorted({item.term for item in findings}))
        super().__init__(f"prompt contains architecture/condition leakage: {terms}")


DEFAULT_ARCHITECTURE_LEAKAGE_TERMS = (
    "ablation",
    "ablated",
    "architecture",
    "condition assignment",
    "control condition",
    "experimental condition",
    "global workspace",
    "hidden condition",
    "intact system",
    "lesion",
    "lesioned",
    "memory specialist",
    "module disabled",
    "prediction specialist",
    "recurrent core",
    "self model",
    "self model specialist",
    "specialist module",
)


def _term_regex(term: str) -> re.Pattern[str]:
    words = re.findall(r"[a-z0-9]+", term.casefold())
    separator = r"(?:[\s_\-/]+)"
    return re.compile(r"\b" + separator.join(re.escape(word) for word in words) + r"\b", re.IGNORECASE)


class ArchitectureLeakageValidator:
    """Conservative lexical gate for condition-blind evaluation prompts."""

    def __init__(self, forbidden_terms: Sequence[str] = DEFAULT_ARCHITECTURE_LEAKAGE_TERMS) -> None:
        terms = tuple(dict.fromkeys(term.strip().casefold() for term in forbidden_terms if term.strip()))
        if not terms:
            raise ValueError("at least one forbidden architecture term is required")
        self.forbidden_terms = terms
        self._patterns = tuple((term, _term_regex(term)) for term in terms)

    @staticmethod
    def _message_texts(prompt: str | Sequence[Mapping[str, Any]]) -> tuple[tuple[int | None, str], ...]:
        if isinstance(prompt, str):
            return ((None, prompt),)
        texts: list[tuple[int | None, str]] = []
        for index, message in enumerate(prompt):
            content = message.get("content", "")
            if isinstance(content, str):
                text = content
            else:
                text = _canonical_json(content)
            texts.append((index, text))
        return tuple(texts)

    def scan(self, prompt: str | Sequence[Mapping[str, Any]]) -> tuple[PromptLeakageFinding, ...]:
        findings: list[PromptLeakageFinding] = []
        for message_index, text in self._message_texts(prompt):
            for term, pattern in self._patterns:
                for match in pattern.finditer(text):
                    start = max(0, match.start() - 35)
                    end = min(len(text), match.end() + 35)
                    findings.append(
                        PromptLeakageFinding(
                            term=term,
                            excerpt=text[start:end],
                            message_index=message_index,
                        )
                    )
        return tuple(findings)

    def validate(self, prompt: str | Sequence[Mapping[str, Any]]) -> None:
        findings = self.scan(prompt)
        if findings:
            raise ArchitectureLeakageError(findings)


def find_architecture_leakage(
    prompt: str | Sequence[Mapping[str, Any]],
    *,
    forbidden_terms: Sequence[str] = DEFAULT_ARCHITECTURE_LEAKAGE_TERMS,
) -> tuple[PromptLeakageFinding, ...]:
    return ArchitectureLeakageValidator(forbidden_terms).scan(prompt)


def validate_condition_blind_prompt(
    prompt: str | Sequence[Mapping[str, Any]],
    *,
    forbidden_terms: Sequence[str] = DEFAULT_ARCHITECTURE_LEAKAGE_TERMS,
) -> None:
    ArchitectureLeakageValidator(forbidden_terms).validate(prompt)


@dataclass(frozen=True)
class Intervention:
    """An intact control or an intervention disabling exactly one component."""

    intervention_id: str
    disabled_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.intervention_id, "intervention_id")
        components = tuple(self.disabled_components)
        object.__setattr__(self, "disabled_components", components)
        if len(components) > 1:
            raise ValueError("each intervention may disable at most one component")
        if any(not component.strip() for component in components):
            raise ValueError("disabled component names must be non-empty")

    @property
    def is_control(self) -> bool:
        return not self.disabled_components

    @property
    def lesion(self) -> str | None:
        return self.disabled_components[0] if self.disabled_components else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "disabled_components": list(self.disabled_components),
        }


@dataclass(frozen=True)
class BlindedAssignment:
    assignment_id: str
    match_id: str
    blinded_condition_id: str
    position: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_id": self.assignment_id,
            "match_id": self.match_id,
            "blinded_condition_id": self.blinded_condition_id,
            "position": self.position,
        }


@dataclass(frozen=True)
class BlindedTrialPlan:
    """Public randomization artifact containing no true condition labels."""

    plan_id: str
    study_id: str
    manifest_hash: str
    mapping_hash: str
    seed_commitment: str
    assignments: tuple[BlindedAssignment, ...]
    conditions_per_match: int
    schema_version: str = RANDOMIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        for name in ("plan_id", "study_id", "manifest_hash", "mapping_hash", "seed_commitment"):
            _required_text(getattr(self, name), name)
        if self.schema_version != RANDOMIZATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported randomization schema: {self.schema_version}")
        if self.conditions_per_match < 2:
            raise ValueError("a blinded plan requires a control and at least one lesion")
        validate_balanced_matching(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "study_id": self.study_id,
            "manifest_hash": self.manifest_hash,
            "mapping_hash": self.mapping_hash,
            "seed_commitment": self.seed_commitment,
            "conditions_per_match": self.conditions_per_match,
            "assignments": [item.to_dict() for item in self.assignments],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BlindedTrialPlan:
        return cls(
            plan_id=str(value["plan_id"]),
            study_id=str(value["study_id"]),
            manifest_hash=str(value["manifest_hash"]),
            mapping_hash=str(value["mapping_hash"]),
            seed_commitment=str(value["seed_commitment"]),
            assignments=tuple(BlindedAssignment(**item) for item in value["assignments"]),
            conditions_per_match=int(value["conditions_per_match"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True)
class _MappingEntry:
    blinded_condition_id: str
    intervention: Intervention

    def to_dict(self) -> dict[str, Any]:
        return {
            "blinded_condition_id": self.blinded_condition_id,
            "intervention": self.intervention.to_dict(),
        }


@dataclass(frozen=True, repr=False)
class UnblindingKey:
    """Secret mapping capability; never include it in run artifacts."""

    plan_id: str
    manifest_hash: str
    mapping_hash: str
    seed: int | str = field(repr=False)
    secret_nonce: str = field(repr=False)
    mapping: tuple[_MappingEntry, ...] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mapping", tuple(self.mapping))

    def __repr__(self) -> str:
        return f"UnblindingKey(plan_id={self.plan_id!r}, sealed=True)"

    def to_dict(self) -> dict[str, Any]:
        """Explicit secret export for an access-controlled key store."""
        return {
            "plan_id": self.plan_id,
            "manifest_hash": self.manifest_hash,
            "mapping_hash": self.mapping_hash,
            "seed": self.seed,
            "secret_nonce": self.secret_nonce,
            "mapping": [entry.to_dict() for entry in self.mapping],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> UnblindingKey:
        entries = []
        for item in value["mapping"]:
            intervention = item["intervention"]
            entries.append(
                _MappingEntry(
                    blinded_condition_id=str(item["blinded_condition_id"]),
                    intervention=Intervention(
                        intervention_id=str(intervention["intervention_id"]),
                        disabled_components=tuple(intervention["disabled_components"]),
                    ),
                )
            )
        return cls(
            plan_id=str(value["plan_id"]),
            manifest_hash=str(value["manifest_hash"]),
            mapping_hash=str(value["mapping_hash"]),
            seed=value["seed"],
            secret_nonce=str(value["secret_nonce"]),
            mapping=tuple(entries),
        )


@dataclass(frozen=True)
class RandomizationBundle:
    """Return value whose repr cannot accidentally print the secret key."""

    plan: BlindedTrialPlan
    unblinding_key: UnblindingKey = field(repr=False)


def _mapping_payload(
    *,
    plan_id: str,
    manifest_hash: str,
    secret_nonce: str,
    mapping: Sequence[_MappingEntry],
) -> dict[str, Any]:
    return {
        "schema_version": RANDOMIZATION_SCHEMA_VERSION,
        "plan_id": plan_id,
        "manifest_hash": manifest_hash,
        "secret_nonce": secret_nonce,
        "mapping": [entry.to_dict() for entry in sorted(mapping, key=lambda item: item.blinded_condition_id)],
    }


def validate_balanced_matching(plan: BlindedTrialPlan) -> None:
    """Verify each matched block has each blinded condition exactly once."""
    if not plan.assignments:
        raise ValueError("a blinded plan must contain assignments")
    assignment_ids = [item.assignment_id for item in plan.assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("assignment ids must be unique")
    blocks: dict[str, list[BlindedAssignment]] = {}
    for assignment in plan.assignments:
        _required_text(assignment.assignment_id, "assignment_id")
        _required_text(assignment.match_id, "match_id")
        _required_text(assignment.blinded_condition_id, "blinded_condition_id")
        blocks.setdefault(assignment.match_id, []).append(assignment)
    expected_codes: set[str] | None = None
    for match_id, assignments in blocks.items():
        codes = [item.blinded_condition_id for item in assignments]
        positions = [item.position for item in assignments]
        if len(assignments) != plan.conditions_per_match:
            raise ValueError(f"matched block {match_id!r} does not contain every condition")
        if len(codes) != len(set(codes)):
            raise ValueError(f"matched block {match_id!r} repeats a condition")
        if sorted(positions) != list(range(plan.conditions_per_match)):
            raise ValueError(f"matched block {match_id!r} has invalid randomized positions")
        if expected_codes is None:
            expected_codes = set(codes)
        elif set(codes) != expected_codes:
            raise ValueError(f"matched block {match_id!r} is not balanced with the other blocks")


def create_balanced_hidden_assignments(
    manifest: PreregistrationManifest,
    *,
    match_ids: Sequence[str],
    interventions: Sequence[Intervention],
    seed: int | str,
) -> RandomizationBundle:
    """Create seeded matched blocks and a separately held unblinding key.

    Every match contains one intact control and every single-lesion condition
    once. Order is independently shuffled within each block.
    """
    manifest.require_frozen()
    matches = tuple(str(item) for item in match_ids)
    conditions = tuple(interventions)
    if not matches or any(not item.strip() for item in matches):
        raise ValueError("match_ids must contain non-empty identifiers")
    if len(matches) != len(set(matches)):
        raise ValueError("match_ids must be unique")
    if len(conditions) < 2:
        raise ValueError("provide one control and at least one lesion")
    ids = [item.intervention_id for item in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("intervention ids must be unique")
    controls = [item for item in conditions if item.is_control]
    if len(controls) != 1:
        raise ValueError("exactly one intact control is required")
    lesions = [item.lesion for item in conditions if not item.is_control]
    if len(lesions) != len(set(lesions)):
        raise ValueError("each lesion component must be unique")
    prediction_conditions = {item.intervention_id for item in conditions if not item.is_control}
    for prediction in manifest.directional_predictions:
        if prediction.intervention_id not in prediction_conditions:
            raise ValueError(
                f"prediction {prediction.prediction_id!r} does not reference a declared lesion intervention"
            )

    seed_payload = {
        "manifest_hash": manifest.manifest_hash,
        "seed_type": type(seed).__name__,
        "seed": seed,
    }
    rng_seed = int.from_bytes(hashlib.sha256(_canonical_json(seed_payload).encode()).digest(), "big")
    rng = random.Random(rng_seed)
    plan_id = f"plan_{rng.getrandbits(128):032x}"
    secret_nonce = f"{rng.getrandbits(256):064x}"
    shuffled_for_codes = list(conditions)
    rng.shuffle(shuffled_for_codes)
    mapping = tuple(
        _MappingEntry(
            blinded_condition_id=f"cond_{rng.getrandbits(128):032x}",
            intervention=intervention,
        )
        for intervention in shuffled_for_codes
    )
    code_by_intervention = {
        entry.intervention.intervention_id: entry.blinded_condition_id for entry in mapping
    }
    mapping_hash = _sha256(
        _mapping_payload(
            plan_id=plan_id,
            manifest_hash=manifest.manifest_hash or "",
            secret_nonce=secret_nonce,
            mapping=mapping,
        )
    )
    seed_commitment = _sha256({"seed": seed_payload, "secret_nonce": secret_nonce})
    assignments: list[BlindedAssignment] = []
    for match_id in matches:
        block = list(conditions)
        rng.shuffle(block)
        for position, intervention in enumerate(block):
            blind_id = code_by_intervention[intervention.intervention_id]
            assignment_id = "assign_" + hashlib.sha256(
                _canonical_json([plan_id, match_id, position, blind_id]).encode()
            ).hexdigest()[:32]
            assignments.append(
                BlindedAssignment(
                    assignment_id=assignment_id,
                    match_id=match_id,
                    blinded_condition_id=blind_id,
                    position=position,
                )
            )
    plan = BlindedTrialPlan(
        plan_id=plan_id,
        study_id=manifest.study_id,
        manifest_hash=manifest.manifest_hash or "",
        mapping_hash=mapping_hash,
        seed_commitment=seed_commitment,
        assignments=tuple(assignments),
        conditions_per_match=len(conditions),
    )
    key = UnblindingKey(
        plan_id=plan_id,
        manifest_hash=manifest.manifest_hash or "",
        mapping_hash=mapping_hash,
        seed=seed,
        secret_nonce=secret_nonce,
        mapping=mapping,
    )
    return RandomizationBundle(plan=plan, unblinding_key=key)


@dataclass(frozen=True)
class UnblindedAssignment:
    assignment_id: str
    match_id: str
    blinded_condition_id: str
    position: int
    intervention: Intervention


def _verified_mapping(
    manifest: PreregistrationManifest,
    plan: BlindedTrialPlan,
    key: UnblindingKey,
) -> dict[str, Intervention]:
    manifest.require_frozen()
    if plan.study_id != manifest.study_id or plan.manifest_hash != manifest.manifest_hash:
        raise ValueError("plan does not belong to this frozen manifest")
    if key.plan_id != plan.plan_id or key.manifest_hash != manifest.manifest_hash:
        raise ValueError("unblinding key does not belong to this plan")
    expected_hash = _sha256(
        _mapping_payload(
            plan_id=key.plan_id,
            manifest_hash=key.manifest_hash,
            secret_nonce=key.secret_nonce,
            mapping=key.mapping,
        )
    )
    if expected_hash != key.mapping_hash or expected_hash != plan.mapping_hash:
        raise ValueError("unblinding key does not open the sealed mapping")
    mapping = {entry.blinded_condition_id: entry.intervention for entry in key.mapping}
    public_codes = {item.blinded_condition_id for item in plan.assignments}
    if set(mapping) != public_codes:
        raise ValueError("unblinding mapping does not cover the public plan")
    return mapping


def unblind_assignments(
    manifest: PreregistrationManifest,
    plan: BlindedTrialPlan,
    key: UnblindingKey,
) -> tuple[UnblindedAssignment, ...]:
    """Open a plan only after the supplied manifest has been frozen."""
    mapping = _verified_mapping(manifest, plan, key)
    return tuple(
        UnblindedAssignment(
            assignment_id=item.assignment_id,
            match_id=item.match_id,
            blinded_condition_id=item.blinded_condition_id,
            position=item.position,
            intervention=mapping[item.blinded_condition_id],
        )
        for item in plan.assignments
    )


@dataclass(frozen=True)
class OutcomeMeasurement:
    outcome_id: str
    value: float

    def __post_init__(self) -> None:
        _required_text(self.outcome_id, "outcome_id")
        object.__setattr__(self, "value", _finite(self.value, "outcome value"))


@dataclass(frozen=True)
class BlindedRunArtifact:
    """Condition-free result collected while the mapping key is withheld."""

    assignment_id: str
    outcomes: tuple[OutcomeMeasurement, ...]
    condition_guess: str | None = None
    confidence: float | None = None
    excluded: bool = False
    exclusion_reason: str | None = None
    trace_ref: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.assignment_id, "assignment_id")
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        outcome_ids = [item.outcome_id for item in self.outcomes]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("an artifact may contain only one value per outcome")
        if self.condition_guess is not None:
            _required_text(self.condition_guess, "condition_guess")
        if self.confidence is not None:
            confidence = _finite(self.confidence, "confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between zero and one")
            if self.condition_guess is None:
                raise ValueError("confidence requires a condition_guess")
            object.__setattr__(self, "confidence", confidence)
        if self.excluded and not self.exclusion_reason:
            raise ValueError("excluded artifacts require an exclusion_reason")
        if not self.excluded and self.exclusion_reason is not None:
            raise ValueError("non-excluded artifacts cannot carry an exclusion_reason")


@dataclass(frozen=True)
class UnblindedRunResult:
    assignment: UnblindedAssignment
    artifact: BlindedRunArtifact


@dataclass(frozen=True)
class IdentificationSummary:
    n_trials: int
    n_guesses: int
    correct: int
    accuracy: float | None
    chance_accuracy: float
    above_chance: bool
    exact_binomial_p_value: float | None
    statistically_above_chance: bool
    mean_confidence: float | None
    calibration_gap: float | None
    brier_score: float | None
    expected_calibration_error: float | None


@dataclass(frozen=True)
class ConditionOutcomeSummary:
    outcome_id: str
    intervention_id: str
    n: int
    mean: float


@dataclass(frozen=True)
class DirectionalPredictionSummary:
    prediction_id: str
    intervention_id: str
    outcome_id: str
    direction: str
    threshold: float
    n_pairs: int
    mean_paired_difference: float | None
    direction_supported: bool | None


@dataclass(frozen=True)
class ExperimentSummary:
    manifest_hash: str
    mapping_hash: str
    n_artifacts: int
    n_included: int
    identification: IdentificationSummary
    condition_outcomes: tuple[ConditionOutcomeSummary, ...]
    directional_predictions: tuple[DirectionalPredictionSummary, ...]


def _binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    if trials == 0:
        return 1.0
    terms = [
        math.comb(trials, k) * probability**k * (1.0 - probability) ** (trials - k)
        for k in range(successes, trials + 1)
    ]
    return min(1.0, math.fsum(terms))


def _expected_calibration_error(
    observations: Sequence[tuple[float, float]], bins: int
) -> float | None:
    if not observations:
        return None
    total = len(observations)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            item
            for item in observations
            if lower <= item[0] <= upper and (index == bins - 1 or item[0] < upper)
        ]
        if bucket:
            confidence = math.fsum(item[0] for item in bucket) / len(bucket)
            accuracy = math.fsum(item[1] for item in bucket) / len(bucket)
            error += len(bucket) / total * abs(confidence - accuracy)
    return error


def _prediction_supported(direction: str, difference: float, threshold: float) -> bool:
    if direction == "increase":
        return difference > threshold
    if direction == "decrease":
        return difference < -threshold
    return abs(difference) <= threshold


def unblind_and_summarize(
    manifest: PreregistrationManifest,
    plan: BlindedTrialPlan,
    key: UnblindingKey,
    artifacts: Sequence[BlindedRunArtifact],
) -> ExperimentSummary:
    """Verify the seal, unblind once, and apply the frozen analysis choices."""
    assignments = {item.assignment_id: item for item in unblind_assignments(manifest, plan, key)}
    artifact_list = tuple(artifacts)
    artifact_ids = [item.assignment_id for item in artifact_list]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("only one artifact is allowed per assignment")
    unknown = set(artifact_ids) - set(assignments)
    if unknown:
        raise ValueError(f"artifacts reference unknown assignments: {sorted(unknown)}")
    primary_ids = {item.outcome_id for item in manifest.primary_outcomes}
    for artifact in artifact_list:
        unexpected = {item.outcome_id for item in artifact.outcomes} - primary_ids
        if unexpected:
            raise ValueError(f"artifact contains non-preregistered outcomes: {sorted(unexpected)}")

    results = tuple(
        UnblindedRunResult(assignment=assignments[item.assignment_id], artifact=item)
        for item in artifact_list
        if not item.excluded
    )
    interventions = {entry.intervention.intervention_id: entry.intervention for entry in key.mapping}
    controls = [item for item in interventions.values() if item.is_control]
    if len(controls) != 1:  # also protects against a hand-built/tampered key
        raise ValueError("sealed mapping must contain exactly one control")
    control_id = controls[0].intervention_id

    correct = 0
    n_guesses = 0
    calibrated: list[tuple[float, float]] = []
    for result in results:
        guess = result.artifact.condition_guess
        is_correct = guess == result.assignment.intervention.intervention_id
        correct += int(is_correct)
        if guess is not None:
            n_guesses += 1
        if result.artifact.confidence is not None:
            calibrated.append((result.artifact.confidence, float(is_correct)))
    n_trials = len(results)
    chance = 1.0 / plan.conditions_per_match
    accuracy = correct / n_trials if n_trials else None
    p_value = _binomial_upper_tail(correct, n_trials, chance) if n_trials else None
    mean_confidence = (
        math.fsum(item[0] for item in calibrated) / len(calibrated) if calibrated else None
    )
    calibrated_accuracy = (
        math.fsum(item[1] for item in calibrated) / len(calibrated) if calibrated else None
    )
    brier = (
        math.fsum((confidence - outcome) ** 2 for confidence, outcome in calibrated) / len(calibrated)
        if calibrated
        else None
    )
    identification = IdentificationSummary(
        n_trials=n_trials,
        n_guesses=n_guesses,
        correct=correct,
        accuracy=accuracy,
        chance_accuracy=chance,
        above_chance=accuracy is not None and accuracy > chance,
        exact_binomial_p_value=p_value,
        statistically_above_chance=(
            accuracy is not None
            and accuracy > chance
            and p_value is not None
            and p_value < manifest.analysis_plan.alpha
        ),
        mean_confidence=mean_confidence,
        calibration_gap=(
            abs(mean_confidence - calibrated_accuracy)
            if mean_confidence is not None and calibrated_accuracy is not None
            else None
        ),
        brier_score=brier,
        expected_calibration_error=_expected_calibration_error(
            calibrated, manifest.analysis_plan.calibration_bins
        ),
    )

    values: dict[tuple[str, str], list[float]] = {}
    by_match: dict[tuple[str, str, str], float] = {}
    for result in results:
        intervention_id = result.assignment.intervention.intervention_id
        for measurement in result.artifact.outcomes:
            values.setdefault((measurement.outcome_id, intervention_id), []).append(measurement.value)
            by_match[(result.assignment.match_id, intervention_id, measurement.outcome_id)] = measurement.value
    condition_outcomes = tuple(
        ConditionOutcomeSummary(
            outcome_id=outcome_id,
            intervention_id=intervention_id,
            n=len(samples),
            mean=math.fsum(samples) / len(samples),
        )
        for (outcome_id, intervention_id), samples in sorted(values.items())
    )

    prediction_summaries: list[DirectionalPredictionSummary] = []
    match_ids = {item.match_id for item in assignments.values()}
    for prediction in manifest.directional_predictions:
        differences: list[float] = []
        for match_id in match_ids:
            control_value = by_match.get((match_id, control_id, prediction.outcome_id))
            lesion_value = by_match.get((match_id, prediction.intervention_id, prediction.outcome_id))
            if control_value is not None and lesion_value is not None:
                differences.append(lesion_value - control_value)
        difference = math.fsum(differences) / len(differences) if differences else None
        prediction_summaries.append(
            DirectionalPredictionSummary(
                prediction_id=prediction.prediction_id,
                intervention_id=prediction.intervention_id,
                outcome_id=prediction.outcome_id,
                direction=prediction.direction,
                threshold=prediction.effect_threshold,
                n_pairs=len(differences),
                mean_paired_difference=difference,
                direction_supported=(
                    _prediction_supported(prediction.direction, difference, prediction.effect_threshold)
                    if difference is not None
                    else None
                ),
            )
        )

    return ExperimentSummary(
        manifest_hash=manifest.manifest_hash or "",
        mapping_hash=plan.mapping_hash,
        n_artifacts=len(artifact_list),
        n_included=n_trials,
        identification=identification,
        condition_outcomes=condition_outcomes,
        directional_predictions=tuple(prediction_summaries),
    )


__all__ = [
    "AnalysisPlan",
    "ArchitectureLeakageError",
    "ArchitectureLeakageValidator",
    "BlindedAssignment",
    "BlindedRunArtifact",
    "BlindedTrialPlan",
    "ConditionOutcomeSummary",
    "DEFAULT_ARCHITECTURE_LEAKAGE_TERMS",
    "DirectionalPrediction",
    "DirectionalPredictionSummary",
    "ExperimentSummary",
    "Hypothesis",
    "IdentificationSummary",
    "Intervention",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestNotFrozenError",
    "OutcomeMeasurement",
    "PreregistrationManifest",
    "PrimaryOutcome",
    "PromptLeakageFinding",
    "RANDOMIZATION_SCHEMA_VERSION",
    "RandomizationBundle",
    "UnblindedAssignment",
    "UnblindedRunResult",
    "UnblindingKey",
    "create_balanced_hidden_assignments",
    "find_architecture_leakage",
    "freeze_manifest",
    "unblind_and_summarize",
    "unblind_assignments",
    "validate_balanced_matching",
    "validate_condition_blind_prompt",
]

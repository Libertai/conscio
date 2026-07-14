"""Immutable local artifacts and audited model-lineage changes for V3.

The registry is intentionally a filesystem primitive rather than a model
trainer or loader.  It never contacts a remote service and never mutates a
live model.  Recurrent world-model weights are stored as content-addressed,
read-only blobs with content-addressed descriptors.  Promotion and checkpoint
lineage changes are separate, append-only, hash-chained audit logs.

There is no mutable ``current`` pointer.  The active artifact is reconstructed
from the last accepted promotion after verifying the complete log, which makes
restart recovery deterministic and turns damaged audit history into a hard
failure instead of a silent rollback.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

MODEL_ARTIFACT_SCHEMA_VERSION = 1
REGISTRY_RECORD_SCHEMA_VERSION = 1
# Recurrent tensor/weight ABI embedded in immutable model artifacts. Private
# specialist checkpoint envelopes carry their own architecture identity and
# CoreCheckpoint wire schema; changing those does not rewrite weight artifacts.
CHECKPOINT_SCHEMA_VERSION = 1
ARTIFACT_KIND = "conscio.v3.recurrent_world_model"
SUPPORTED_DTYPES = frozenset({"float32", "float64"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelRegistryError(ValueError):
    """Base class for registry validation and integrity failures."""


class ArtifactNotFoundError(ModelRegistryError):
    """Raised when a referenced content-addressed artifact is absent."""


class ArtifactIntegrityError(ModelRegistryError):
    """Raised when persisted bytes do not match their content address."""


class ModelCompatibilityError(ModelRegistryError):
    """Raised when an artifact does not match an explicitly required spec."""


class RegistryIntegrityError(ModelRegistryError):
    """Raised when an append-only audit log is malformed or hash-invalid."""


class PromotionRejected(ModelRegistryError):
    """An attempted approval failed a promotion gate and was audited."""

    def __init__(self, record: PromotionRecord) -> None:
        self.record = record
        details = "; ".join(record.eligibility_issues)
        super().__init__(f"model promotion rejected: {details}")


@dataclass(frozen=True)
class WorldModelSpec:
    """Version and state-shape contract for one recurrent model artifact."""

    model_family: str
    model_version: str
    revision: int
    deterministic_dim: int
    stochastic_dim: int
    affect_dim: int
    specialist_dims: Mapping[str, int]
    dtype: str = "float64"
    checkpoint_schema_version: int = CHECKPOINT_SCHEMA_VERSION
    schema_version: int = MODEL_ARTIFACT_SCHEMA_VERSION
    artifact_kind: str = ARTIFACT_KIND

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported model artifact schema: {self.schema_version}")
        if self.artifact_kind != ARTIFACT_KIND:
            raise ValueError(f"unsupported artifact kind: {self.artifact_kind!r}")
        if self.checkpoint_schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema: {self.checkpoint_schema_version}")
        if not self.model_family.strip() or not self.model_version.strip():
            raise ValueError("model_family and model_version cannot be empty")
        _require_non_negative_int(self.revision, "revision")
        for name in ("deterministic_dim", "stochastic_dim", "affect_dim"):
            _require_positive_int(getattr(self, name), name)
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype: {self.dtype!r}")
        if not isinstance(self.specialist_dims, Mapping) or not self.specialist_dims:
            raise ValueError("specialist_dims must contain at least one specialist")
        normalised: dict[str, int] = {}
        for raw_name, dimension in self.specialist_dims.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("specialist names must be non-empty strings")
            name = raw_name.strip()
            if name in normalised:
                raise ValueError(f"duplicate specialist name: {name!r}")
            _require_positive_int(dimension, f"specialist_dims[{name!r}]")
            normalised[name] = dimension
        object.__setattr__(self, "specialist_dims", MappingProxyType(dict(sorted(normalised.items()))))

    @property
    def dimensions(self) -> tuple[int, int, int, tuple[tuple[str, int], ...]]:
        """Return the exact recurrent/checkpoint shape in canonical order."""
        return (
            self.deterministic_dim,
            self.stochastic_dim,
            self.affect_dim,
            tuple(self.specialist_dims.items()),
        )

    def compatibility_issues(self, other: WorldModelSpec) -> tuple[str, ...]:
        """Report structural reasons that state cannot cross model versions."""
        issues: list[str] = []
        if self.schema_version != other.schema_version:
            issues.append("artifact schema versions differ")
        if self.artifact_kind != other.artifact_kind:
            issues.append("artifact kinds differ")
        if self.model_family != other.model_family:
            issues.append("model families differ")
        if self.checkpoint_schema_version != other.checkpoint_schema_version:
            issues.append("checkpoint schema versions differ")
        if self.dtype != other.dtype:
            issues.append("model dtypes differ")
        if self.deterministic_dim != other.deterministic_dim:
            issues.append("deterministic state dimensions differ")
        if self.stochastic_dim != other.stochastic_dim:
            issues.append("stochastic state dimensions differ")
        if self.affect_dim != other.affect_dim:
            issues.append("affect state dimensions differ")
        if dict(self.specialist_dims) != dict(other.specialist_dims):
            issues.append("specialist state dimensions differ")
        return tuple(issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "revision": self.revision,
            "deterministic_dim": self.deterministic_dim,
            "stochastic_dim": self.stochastic_dim,
            "affect_dim": self.affect_dim,
            "specialist_dims": dict(self.specialist_dims),
            "dtype": self.dtype,
            "checkpoint_schema_version": self.checkpoint_schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorldModelSpec:
        expected = {
            "schema_version",
            "artifact_kind",
            "model_family",
            "model_version",
            "revision",
            "deterministic_dim",
            "stochastic_dim",
            "affect_dim",
            "specialist_dims",
            "dtype",
            "checkpoint_schema_version",
        }
        _require_exact_keys(data, expected, "world model spec")
        specialist_dims = data["specialist_dims"]
        if not isinstance(specialist_dims, Mapping):
            raise ValueError("specialist_dims must be an object")
        return cls(
            model_family=_require_string(data["model_family"], "model_family"),
            model_version=_require_string(data["model_version"], "model_version"),
            revision=_require_int(data["revision"], "revision"),
            deterministic_dim=_require_int(data["deterministic_dim"], "deterministic_dim"),
            stochastic_dim=_require_int(data["stochastic_dim"], "stochastic_dim"),
            affect_dim=_require_int(data["affect_dim"], "affect_dim"),
            specialist_dims={
                _require_string(name, "specialist name"): _require_int(dimension, "specialist dimension")
                for name, dimension in specialist_dims.items()
            },
            dtype=_require_string(data["dtype"], "dtype"),
            checkpoint_schema_version=_require_int(data["checkpoint_schema_version"], "checkpoint_schema_version"),
            schema_version=_require_int(data["schema_version"], "schema_version"),
            artifact_kind=_require_string(data["artifact_kind"], "artifact_kind"),
        )


@dataclass(frozen=True)
class ValidationEvidence:
    """Validation result cryptographically bound to one promotion pair."""

    candidate_digest: str
    incumbent_digest: str | None
    dataset_digest: str
    protocol_digest: str
    metric_name: str
    candidate_score: float
    incumbent_score: float | None
    lower_is_better: bool
    sample_count: int
    passed: bool
    notes: str = ""

    def __post_init__(self) -> None:
        _validate_digest(self.candidate_digest, "candidate_digest")
        if self.incumbent_digest is not None:
            _validate_digest(self.incumbent_digest, "incumbent_digest")
        _validate_digest(self.dataset_digest, "dataset_digest")
        _validate_digest(self.protocol_digest, "protocol_digest")
        if not self.metric_name.strip():
            raise ValueError("metric_name cannot be empty")
        if not math.isfinite(self.candidate_score):
            raise ValueError("candidate_score must be finite")
        if self.incumbent_score is not None and not math.isfinite(self.incumbent_score):
            raise ValueError("incumbent_score must be finite when present")
        if self.incumbent_digest is None and self.incumbent_score is not None:
            raise ValueError("initial promotion cannot specify an incumbent_score")
        if self.incumbent_digest is not None and self.incumbent_score is None:
            raise ValueError("incumbent_score is required for a replacement promotion")
        if type(self.lower_is_better) is not bool or type(self.passed) is not bool:
            raise TypeError("lower_is_better and passed must be bool values")
        _require_positive_int(self.sample_count, "sample_count")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")

    @property
    def improved(self) -> bool:
        if self.incumbent_score is None:
            return True
        if self.lower_is_better:
            return self.candidate_score < self.incumbent_score
        return self.candidate_score > self.incumbent_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_digest": self.candidate_digest,
            "incumbent_digest": self.incumbent_digest,
            "dataset_digest": self.dataset_digest,
            "protocol_digest": self.protocol_digest,
            "metric_name": self.metric_name,
            "candidate_score": self.candidate_score,
            "incumbent_score": self.incumbent_score,
            "lower_is_better": self.lower_is_better,
            "sample_count": self.sample_count,
            "passed": self.passed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValidationEvidence:
        expected = {
            "candidate_digest",
            "incumbent_digest",
            "dataset_digest",
            "protocol_digest",
            "metric_name",
            "candidate_score",
            "incumbent_score",
            "lower_is_better",
            "sample_count",
            "passed",
            "notes",
        }
        _require_exact_keys(data, expected, "validation evidence")
        incumbent_digest = data["incumbent_digest"]
        incumbent_score = data["incumbent_score"]
        return cls(
            candidate_digest=_require_string(data["candidate_digest"], "candidate_digest"),
            incumbent_digest=(
                None if incumbent_digest is None else _require_string(incumbent_digest, "incumbent_digest")
            ),
            dataset_digest=_require_string(data["dataset_digest"], "dataset_digest"),
            protocol_digest=_require_string(data["protocol_digest"], "protocol_digest"),
            metric_name=_require_string(data["metric_name"], "metric_name"),
            candidate_score=_require_float(data["candidate_score"], "candidate_score"),
            incumbent_score=(None if incumbent_score is None else _require_float(incumbent_score, "incumbent_score")),
            lower_is_better=_require_bool(data["lower_is_better"], "lower_is_better"),
            sample_count=_require_int(data["sample_count"], "sample_count"),
            passed=_require_bool(data["passed"], "passed"),
            notes=_require_string(data["notes"], "notes", allow_empty=True),
        )

    def digest(self) -> str:
        return _sha256(_canonical_json(self.to_dict()))


@dataclass(frozen=True)
class ModelArtifact:
    """A fully verified descriptor and its exact recurrent weight bytes."""

    digest: str
    spec: WorldModelSpec
    weights_digest: str
    weights: bytes = field(repr=False)
    descriptor_path: Path
    weights_path: Path

    @property
    def weights_size(self) -> int:
        return len(self.weights)

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "spec": self.spec.to_dict(),
            "weights_digest": self.weights_digest,
            "weights_size": self.weights_size,
            "descriptor_path": str(self.descriptor_path),
            "weights_path": str(self.weights_path),
        }


PromotionDecision = Literal["accepted", "rejected"]


@dataclass(frozen=True)
class PromotionRecord:
    """One immutable promotion decision in a hash-chained audit log."""

    record_id: str
    incumbent_digest: str | None
    candidate_digest: str
    evidence: ValidationEvidence
    decision: PromotionDecision
    reason: str
    decided_by: str
    decided_at: float
    eligibility_issues: tuple[str, ...]
    previous_record_hash: str | None
    record_hash: str
    schema_version: int = REGISTRY_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_RECORD_SCHEMA_VERSION:
            raise ValueError(f"unsupported promotion record schema: {self.schema_version}")
        if not self.record_id.strip() or not self.reason.strip() or not self.decided_by.strip():
            raise ValueError("record_id, reason, and decided_by cannot be empty")
        if self.incumbent_digest is not None:
            _validate_digest(self.incumbent_digest, "incumbent_digest")
        _validate_digest(self.candidate_digest, "candidate_digest")
        if self.evidence.incumbent_digest != self.incumbent_digest:
            raise ValueError("validation evidence does not match incumbent digest")
        if self.evidence.candidate_digest != self.candidate_digest:
            raise ValueError("validation evidence does not match candidate digest")
        if self.decision not in ("accepted", "rejected"):
            raise ValueError(f"invalid promotion decision: {self.decision!r}")
        if self.decision == "accepted" and self.eligibility_issues:
            raise ValueError("accepted promotion cannot contain eligibility issues")
        if not math.isfinite(self.decided_at):
            raise ValueError("decided_at must be finite")
        if self.previous_record_hash is not None:
            _validate_digest(self.previous_record_hash, "previous_record_hash")
        _validate_digest(self.record_hash, "record_hash")
        if self.record_hash != self.compute_hash():
            raise RegistryIntegrityError(f"promotion record hash mismatch: {self.record_id}")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "incumbent_digest": self.incumbent_digest,
            "candidate_digest": self.candidate_digest,
            "evidence": self.evidence.to_dict(),
            "decision": self.decision,
            "reason": self.reason,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "eligibility_issues": list(self.eligibility_issues),
            "previous_record_hash": self.previous_record_hash,
        }

    def compute_hash(self) -> str:
        return _sha256(_canonical_json(self._hash_payload()))

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "record_hash": self.record_hash}

    @classmethod
    def create(
        cls,
        *,
        incumbent_digest: str | None,
        candidate_digest: str,
        evidence: ValidationEvidence,
        decision: PromotionDecision,
        reason: str,
        decided_by: str,
        decided_at: float,
        eligibility_issues: Sequence[str],
        previous_record_hash: str | None,
    ) -> PromotionRecord:
        provisional = cls.__new__(cls)
        values = {
            "record_id": f"promotion_{uuid.uuid4().hex}",
            "incumbent_digest": incumbent_digest,
            "candidate_digest": candidate_digest,
            "evidence": evidence,
            "decision": decision,
            "reason": reason,
            "decided_by": decided_by,
            "decided_at": decided_at,
            "eligibility_issues": tuple(eligibility_issues),
            "previous_record_hash": previous_record_hash,
            "record_hash": "0" * 64,
            "schema_version": REGISTRY_RECORD_SCHEMA_VERSION,
        }
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return replace(provisional, record_hash=provisional.compute_hash())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PromotionRecord:
        expected = {
            "schema_version",
            "record_id",
            "incumbent_digest",
            "candidate_digest",
            "evidence",
            "decision",
            "reason",
            "decided_by",
            "decided_at",
            "eligibility_issues",
            "previous_record_hash",
            "record_hash",
        }
        _require_exact_keys(data, expected, "promotion record")
        evidence = data["evidence"]
        issues = data["eligibility_issues"]
        if not isinstance(evidence, Mapping):
            raise ValueError("promotion evidence must be an object")
        if not isinstance(issues, list) or not all(isinstance(issue, str) for issue in issues):
            raise ValueError("eligibility_issues must be a list of strings")
        incumbent = data["incumbent_digest"]
        previous = data["previous_record_hash"]
        decision = data["decision"]
        if decision not in ("accepted", "rejected"):
            raise ValueError(f"invalid promotion decision: {decision!r}")
        return cls(
            record_id=_require_string(data["record_id"], "record_id"),
            incumbent_digest=None if incumbent is None else _require_string(incumbent, "incumbent_digest"),
            candidate_digest=_require_string(data["candidate_digest"], "candidate_digest"),
            evidence=ValidationEvidence.from_dict(evidence),
            decision=decision,
            reason=_require_string(data["reason"], "reason"),
            decided_by=_require_string(data["decided_by"], "decided_by"),
            decided_at=_require_float(data["decided_at"], "decided_at"),
            eligibility_issues=tuple(issues),
            previous_record_hash=(None if previous is None else _require_string(previous, "previous_record_hash")),
            record_hash=_require_string(data["record_hash"], "record_hash"),
            schema_version=_require_int(data["schema_version"], "schema_version"),
        )


@dataclass(frozen=True)
class LineageMigrationRecord:
    """Explicit authorization to begin a new checkpoint lineage on new weights."""

    record_id: str
    source_checkpoint_id: str
    source_lineage_id: str
    source_artifact_digest: str
    target_checkpoint_id: str
    target_lineage_id: str
    target_artifact_digest: str
    transform_digest: str
    evidence_digest: str
    migrator: str
    reason: str
    migrated_at: float
    previous_record_hash: str | None
    record_hash: str
    schema_version: int = REGISTRY_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REGISTRY_RECORD_SCHEMA_VERSION:
            raise ValueError(f"unsupported lineage record schema: {self.schema_version}")
        for name in (
            "record_id",
            "source_checkpoint_id",
            "source_lineage_id",
            "target_checkpoint_id",
            "target_lineage_id",
            "migrator",
            "reason",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} cannot be empty")
        for name in (
            "source_artifact_digest",
            "target_artifact_digest",
            "transform_digest",
            "evidence_digest",
        ):
            _validate_digest(getattr(self, name), name)
        if self.source_checkpoint_id == self.target_checkpoint_id:
            raise ValueError("migration must create a new target checkpoint")
        if self.source_lineage_id == self.target_lineage_id:
            raise ValueError("migration must create a new target lineage")
        if not math.isfinite(self.migrated_at):
            raise ValueError("migrated_at must be finite")
        if self.previous_record_hash is not None:
            _validate_digest(self.previous_record_hash, "previous_record_hash")
        _validate_digest(self.record_hash, "record_hash")
        if self.record_hash != self.compute_hash():
            raise RegistryIntegrityError(f"lineage record hash mismatch: {self.record_id}")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "source_checkpoint_id": self.source_checkpoint_id,
            "source_lineage_id": self.source_lineage_id,
            "source_artifact_digest": self.source_artifact_digest,
            "target_checkpoint_id": self.target_checkpoint_id,
            "target_lineage_id": self.target_lineage_id,
            "target_artifact_digest": self.target_artifact_digest,
            "transform_digest": self.transform_digest,
            "evidence_digest": self.evidence_digest,
            "migrator": self.migrator,
            "reason": self.reason,
            "migrated_at": self.migrated_at,
            "previous_record_hash": self.previous_record_hash,
        }

    def compute_hash(self) -> str:
        return _sha256(_canonical_json(self._hash_payload()))

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "record_hash": self.record_hash}

    @classmethod
    def create(
        cls,
        *,
        source_checkpoint_id: str,
        source_lineage_id: str,
        source_artifact_digest: str,
        target_checkpoint_id: str,
        target_lineage_id: str,
        target_artifact_digest: str,
        transform_digest: str,
        evidence_digest: str,
        migrator: str,
        reason: str,
        migrated_at: float,
        previous_record_hash: str | None,
    ) -> LineageMigrationRecord:
        provisional = cls.__new__(cls)
        values = {
            "record_id": f"migration_{uuid.uuid4().hex}",
            "source_checkpoint_id": source_checkpoint_id,
            "source_lineage_id": source_lineage_id,
            "source_artifact_digest": source_artifact_digest,
            "target_checkpoint_id": target_checkpoint_id,
            "target_lineage_id": target_lineage_id,
            "target_artifact_digest": target_artifact_digest,
            "transform_digest": transform_digest,
            "evidence_digest": evidence_digest,
            "migrator": migrator,
            "reason": reason,
            "migrated_at": migrated_at,
            "previous_record_hash": previous_record_hash,
            "record_hash": "0" * 64,
            "schema_version": REGISTRY_RECORD_SCHEMA_VERSION,
        }
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        return replace(provisional, record_hash=provisional.compute_hash())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LineageMigrationRecord:
        expected = {
            "schema_version",
            "record_id",
            "source_checkpoint_id",
            "source_lineage_id",
            "source_artifact_digest",
            "target_checkpoint_id",
            "target_lineage_id",
            "target_artifact_digest",
            "transform_digest",
            "evidence_digest",
            "migrator",
            "reason",
            "migrated_at",
            "previous_record_hash",
            "record_hash",
        }
        _require_exact_keys(data, expected, "lineage migration record")
        previous = data["previous_record_hash"]
        return cls(
            record_id=_require_string(data["record_id"], "record_id"),
            source_checkpoint_id=_require_string(data["source_checkpoint_id"], "source_checkpoint_id"),
            source_lineage_id=_require_string(data["source_lineage_id"], "source_lineage_id"),
            source_artifact_digest=_require_string(data["source_artifact_digest"], "source_artifact_digest"),
            target_checkpoint_id=_require_string(data["target_checkpoint_id"], "target_checkpoint_id"),
            target_lineage_id=_require_string(data["target_lineage_id"], "target_lineage_id"),
            target_artifact_digest=_require_string(data["target_artifact_digest"], "target_artifact_digest"),
            transform_digest=_require_string(data["transform_digest"], "transform_digest"),
            evidence_digest=_require_string(data["evidence_digest"], "evidence_digest"),
            migrator=_require_string(data["migrator"], "migrator"),
            reason=_require_string(data["reason"], "reason"),
            migrated_at=_require_float(data["migrated_at"], "migrated_at"),
            previous_record_hash=(None if previous is None else _require_string(previous, "previous_record_hash")),
            record_hash=_require_string(data["record_hash"], "record_hash"),
            schema_version=_require_int(data["schema_version"], "schema_version"),
        )


class ModelArtifactRegistry:
    """Content-addressed V3 artifact registry with audited activation changes."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self._clock = clock
        self._artifacts_root = self.root / "artifacts" / "sha256"
        self._weights_root = self.root / "weights" / "sha256"
        self._promotions_path = self.root / "audit" / "promotions.jsonl"
        self._migrations_path = self.root / "audit" / "lineage-migrations.jsonl"
        self._lock_path = self.root / ".registry.lock"

    def register_artifact(self, spec: WorldModelSpec, weights: bytes) -> ModelArtifact:
        """Atomically publish immutable weights and their canonical descriptor."""
        if not isinstance(spec, WorldModelSpec):
            raise TypeError("spec must be a WorldModelSpec")
        if not isinstance(weights, bytes) or not weights:
            raise ValueError("weights must be non-empty bytes")
        weights_digest = _sha256(weights)
        descriptor = self._descriptor(spec, weights_digest, len(weights))
        encoded = _canonical_json(descriptor)
        artifact_digest = _sha256(encoded)
        weights_path = self._weights_path(weights_digest)
        descriptor_path = self._descriptor_path(artifact_digest)
        with self._registry_lock():
            self._publish_immutable(weights_path, weights)
            self._publish_immutable(descriptor_path, encoded)
        return self.load_artifact(artifact_digest, expected_spec=spec)

    def load_artifact(
        self,
        digest: str,
        *,
        expected_spec: WorldModelSpec | None = None,
    ) -> ModelArtifact:
        """Load an artifact only after verifying descriptor and weight digests."""
        _validate_digest(digest, "artifact digest")
        descriptor_path = self._descriptor_path(digest)
        if not descriptor_path.is_file():
            raise ArtifactNotFoundError(f"model artifact not found: {digest}")
        try:
            encoded = descriptor_path.read_bytes()
        except OSError as exc:
            raise ArtifactIntegrityError(f"cannot read artifact descriptor {digest}: {exc}") from exc
        if _sha256(encoded) != digest:
            raise ArtifactIntegrityError(f"artifact descriptor digest mismatch: {digest}")
        try:
            raw = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(f"artifact descriptor is not canonical JSON: {digest}") from exc
        if not isinstance(raw, Mapping):
            raise ArtifactIntegrityError(f"artifact descriptor is not an object: {digest}")
        expected_keys = {"schema_version", "artifact_kind", "spec", "weights_digest", "weights_size"}
        try:
            _require_exact_keys(raw, expected_keys, "artifact descriptor")
            if _canonical_json(dict(raw)) != encoded:
                raise ArtifactIntegrityError(f"artifact descriptor is not canonical: {digest}")
            if _require_int(raw["schema_version"], "schema_version") != MODEL_ARTIFACT_SCHEMA_VERSION:
                raise ArtifactIntegrityError("unsupported artifact descriptor schema")
            if raw["artifact_kind"] != ARTIFACT_KIND:
                raise ArtifactIntegrityError("unsupported artifact descriptor kind")
            raw_spec = raw["spec"]
            if not isinstance(raw_spec, Mapping):
                raise ArtifactIntegrityError("artifact spec is not an object")
            spec = WorldModelSpec.from_dict(raw_spec)
            weights_digest = _require_string(raw["weights_digest"], "weights_digest")
            _validate_digest(weights_digest, "weights_digest")
            weights_size = _require_int(raw["weights_size"], "weights_size")
            _require_positive_int(weights_size, "weights_size")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ArtifactIntegrityError):
                raise
            raise ArtifactIntegrityError(f"invalid artifact descriptor {digest}: {exc}") from exc
        if expected_spec is not None and spec != expected_spec:
            raise ModelCompatibilityError(
                f"artifact {digest} has {spec.model_version!r}, not exact expected spec {expected_spec.model_version!r}"
            )
        weights_path = self._weights_path(weights_digest)
        if not weights_path.is_file():
            raise ArtifactIntegrityError(f"artifact weight blob is missing: {weights_digest}")
        try:
            weights = weights_path.read_bytes()
        except OSError as exc:
            raise ArtifactIntegrityError(f"cannot read artifact weights {weights_digest}: {exc}") from exc
        if len(weights) != weights_size or _sha256(weights) != weights_digest:
            raise ArtifactIntegrityError(f"artifact weight digest mismatch: {weights_digest}")
        return ModelArtifact(
            digest=digest,
            spec=spec,
            weights_digest=weights_digest,
            weights=weights,
            descriptor_path=descriptor_path,
            weights_path=weights_path,
        )

    def decide_promotion(
        self,
        *,
        incumbent_digest: str | None,
        candidate_digest: str,
        evidence: ValidationEvidence,
        approve: bool,
        reason: str,
        decided_by: str,
    ) -> PromotionRecord:
        """Audit a promotion decision and activate only a fully eligible candidate.

        If ``approve`` is true but a gate fails, a rejected record is durably
        appended and :class:`PromotionRejected` exposes that same record.
        """
        if type(approve) is not bool:
            raise TypeError("approve must be a bool")
        if not reason.strip() or not decided_by.strip():
            raise ValueError("reason and decided_by cannot be empty")
        if not isinstance(evidence, ValidationEvidence):
            raise TypeError("complete ValidationEvidence is required")
        with self._registry_lock():
            candidate = self.load_artifact(candidate_digest)
            incumbent = self.load_artifact(incumbent_digest) if incumbent_digest is not None else None
            records = self._read_promotions()
            current_digest = _latest_promoted_digest(records)
            issues = self._promotion_issues(
                current_digest=current_digest,
                incumbent=incumbent,
                candidate=candidate,
                incumbent_digest=incumbent_digest,
                evidence=evidence,
            )
            decision: PromotionDecision = "accepted" if approve and not issues else "rejected"
            record = self._append_promotion(
                incumbent_digest=incumbent_digest,
                candidate_digest=candidate_digest,
                evidence=evidence,
                decision=decision,
                reason=reason,
                decided_by=decided_by,
                eligibility_issues=issues,
                prior_records=records,
            )
        if approve and issues:
            raise PromotionRejected(record)
        return record

    def promotion_records(self) -> tuple[PromotionRecord, ...]:
        """Return the complete promotion history after hash-chain verification."""
        return self._read_promotions()

    def latest_promoted(self) -> ModelArtifact | None:
        """Reconstruct and verify the currently accepted artifact."""
        digest = _latest_promoted_digest(self._read_promotions())
        return None if digest is None else self.load_artifact(digest)

    def record_lineage_migration(
        self,
        *,
        source_checkpoint_id: str,
        source_lineage_id: str,
        source_artifact_digest: str,
        target_checkpoint_id: str,
        target_lineage_id: str,
        target_artifact_digest: str,
        transform_digest: str,
        evidence_digest: str,
        migrator: str,
        reason: str,
    ) -> LineageMigrationRecord:
        """Record an explicit, evidence-backed checkpoint lineage transition.

        The method records authorization/provenance; it does not transform or
        load live recurrent state.  The target must already be the accepted
        successor of the source artifact and retain checkpoint-compatible
        dimensions.
        """
        with self._registry_lock():
            source = self.load_artifact(source_artifact_digest)
            target = self.load_artifact(target_artifact_digest)
            promotions = self._read_promotions()
            active_digest = _latest_promoted_digest(promotions)
            if active_digest != target_artifact_digest:
                raise ModelCompatibilityError("migration target is not the active promoted artifact")
            accepted_target = next(
                (
                    record
                    for record in reversed(promotions)
                    if record.decision == "accepted" and record.candidate_digest == target_artifact_digest
                ),
                None,
            )
            if accepted_target is None or accepted_target.incumbent_digest != source_artifact_digest:
                raise ModelCompatibilityError(
                    "migration artifacts are not an accepted incumbent-to-candidate transition"
                )
            compatibility = source.spec.compatibility_issues(target.spec)
            if compatibility:
                raise ModelCompatibilityError("; ".join(compatibility))
            if target.spec.revision <= source.spec.revision:
                raise ModelCompatibilityError("migration target must have a higher model revision")
            migrations = self._read_migrations()
            if any(record.target_checkpoint_id == target_checkpoint_id for record in migrations):
                raise ModelRegistryError(f"target checkpoint already migrated: {target_checkpoint_id}")
            if any(record.target_lineage_id == target_lineage_id for record in migrations):
                raise ModelRegistryError(f"target lineage already exists: {target_lineage_id}")
            record = self._append_migration(
                source_checkpoint_id=source_checkpoint_id,
                source_lineage_id=source_lineage_id,
                source_artifact_digest=source_artifact_digest,
                target_checkpoint_id=target_checkpoint_id,
                target_lineage_id=target_lineage_id,
                target_artifact_digest=target_artifact_digest,
                transform_digest=transform_digest,
                evidence_digest=evidence_digest,
                migrator=migrator,
                reason=reason,
                prior_records=migrations,
            )
        return record

    def lineage_migrations(self) -> tuple[LineageMigrationRecord, ...]:
        """Return all verified recurrent checkpoint migration records."""
        return self._read_migrations()

    def latest_lineage_migration(self, target_lineage_id: str | None = None) -> LineageMigrationRecord | None:
        """Return the last migration, optionally for one target lineage."""
        records = self._read_migrations()
        for record in reversed(records):
            if target_lineage_id is None or record.target_lineage_id == target_lineage_id:
                return record
        return None

    @staticmethod
    def _descriptor(spec: WorldModelSpec, weights_digest: str, weights_size: int) -> dict[str, Any]:
        return {
            "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "spec": spec.to_dict(),
            "weights_digest": weights_digest,
            "weights_size": weights_size,
        }

    def _promotion_issues(
        self,
        *,
        current_digest: str | None,
        incumbent: ModelArtifact | None,
        candidate: ModelArtifact,
        incumbent_digest: str | None,
        evidence: ValidationEvidence,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if current_digest != incumbent_digest:
            issues.append("declared incumbent is not the active promoted artifact")
        if evidence.incumbent_digest != incumbent_digest:
            issues.append("validation evidence incumbent digest does not match")
        if evidence.candidate_digest != candidate.digest:
            issues.append("validation evidence candidate digest does not match")
        if not evidence.passed:
            issues.append("validation evidence did not pass")
        if not evidence.improved:
            issues.append("candidate did not strictly improve the declared validation metric")
        if incumbent is not None:
            if candidate.digest == incumbent.digest:
                issues.append("candidate is identical to incumbent")
            if candidate.spec.revision <= incumbent.spec.revision:
                issues.append("candidate model revision is not newer than incumbent")
            issues.extend(incumbent.spec.compatibility_issues(candidate.spec))
        return tuple(issues)

    def _append_promotion(
        self,
        *,
        incumbent_digest: str | None,
        candidate_digest: str,
        evidence: ValidationEvidence,
        decision: PromotionDecision,
        reason: str,
        decided_by: str,
        eligibility_issues: Sequence[str],
        prior_records: tuple[PromotionRecord, ...],
    ) -> PromotionRecord:
        previous_hash = prior_records[-1].record_hash if prior_records else None
        record = PromotionRecord.create(
            incumbent_digest=incumbent_digest,
            candidate_digest=candidate_digest,
            evidence=evidence,
            decision=decision,
            reason=reason,
            decided_by=decided_by,
            decided_at=float(self._clock()),
            eligibility_issues=eligibility_issues,
            previous_record_hash=previous_hash,
        )
        self._append_jsonl(self._promotions_path, record.to_dict())
        return record

    def _append_migration(
        self,
        *,
        prior_records: tuple[LineageMigrationRecord, ...],
        **fields: Any,
    ) -> LineageMigrationRecord:
        previous_hash = prior_records[-1].record_hash if prior_records else None
        record = LineageMigrationRecord.create(
            **fields,
            migrated_at=float(self._clock()),
            previous_record_hash=previous_hash,
        )
        self._append_jsonl(self._migrations_path, record.to_dict())
        return record

    def _read_promotions(self) -> tuple[PromotionRecord, ...]:
        raw_records = self._read_jsonl(self._promotions_path)
        records: list[PromotionRecord] = []
        for index, raw in enumerate(raw_records, start=1):
            try:
                record = PromotionRecord.from_dict(raw)
            except (KeyError, TypeError, ValueError) as exc:
                raise RegistryIntegrityError(f"invalid promotion record at line {index}: {exc}") from exc
            expected_previous = records[-1].record_hash if records else None
            if record.previous_record_hash != expected_previous:
                raise RegistryIntegrityError(f"broken promotion hash chain at line {index}")
            records.append(record)
        return tuple(records)

    def _read_migrations(self) -> tuple[LineageMigrationRecord, ...]:
        raw_records = self._read_jsonl(self._migrations_path)
        records: list[LineageMigrationRecord] = []
        for index, raw in enumerate(raw_records, start=1):
            try:
                record = LineageMigrationRecord.from_dict(raw)
            except (KeyError, TypeError, ValueError) as exc:
                raise RegistryIntegrityError(f"invalid lineage record at line {index}: {exc}") from exc
            expected_previous = records[-1].record_hash if records else None
            if record.previous_record_hash != expected_previous:
                raise RegistryIntegrityError(f"broken lineage hash chain at line {index}")
            records.append(record)
        return tuple(records)

    def _read_jsonl(self, path: Path) -> tuple[Mapping[str, Any], ...]:
        if not path.exists():
            return ()
        fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            chunks: list[bytes] = []
            while chunk := os.read(fd, 1024 * 1024):
                chunks.append(chunk)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        raw = b"".join(chunks)
        if raw and not raw.endswith(b"\n"):
            raise RegistryIntegrityError(f"incomplete append-only log tail: {path}")
        records: list[Mapping[str, Any]] = []
        for index, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                raise RegistryIntegrityError(f"blank record at line {index}: {path}")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RegistryIntegrityError(f"invalid JSON at line {index}: {path}") from exc
            if not isinstance(value, Mapping):
                raise RegistryIntegrityError(f"non-object record at line {index}: {path}")
            if _canonical_json(dict(value)) != line:
                raise RegistryIntegrityError(f"non-canonical record at line {index}: {path}")
            records.append(value)
        return tuple(records)

    def _append_jsonl(self, path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _canonical_json(dict(value)) + b"\n"
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("append-only registry write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @contextmanager
    def _registry_lock(self) -> Any:
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    @staticmethod
    def _publish_immutable(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                existing = path.read_bytes()
            except OSError as exc:
                raise ArtifactIntegrityError(f"cannot verify existing artifact file {path}: {exc}") from exc
            if existing != content:
                raise ArtifactIntegrityError(f"immutable artifact file changed: {path}")
            return
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
        try:
            view = memoryview(content)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("artifact write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.replace(temporary, path)
            os.chmod(path, 0o400)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _descriptor_path(self, digest: str) -> Path:
        return self._artifacts_root / digest[:2] / f"{digest}.json"

    def _weights_path(self, digest: str) -> Path:
        return self._weights_root / digest[:2] / f"{digest}.bin"


def _latest_promoted_digest(records: Sequence[PromotionRecord]) -> str | None:
    for record in reversed(records):
        if record.decision == "accepted":
            return record.candidate_digest
    return None


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON data: {exc}") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_exact_keys(data: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(f"invalid {name} keys; missing={missing}, unknown={unknown}")


def _require_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _require_non_negative_int(value: Any, name: str) -> None:
    if _require_int(value, name) < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_int(value: Any, name: str) -> None:
    if _require_int(value, name) <= 0:
        raise ValueError(f"{name} must be positive")


def _require_float(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _require_string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise TypeError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "ARTIFACT_KIND",
    "CHECKPOINT_SCHEMA_VERSION",
    "MODEL_ARTIFACT_SCHEMA_VERSION",
    "REGISTRY_RECORD_SCHEMA_VERSION",
    "ArtifactIntegrityError",
    "ArtifactNotFoundError",
    "LineageMigrationRecord",
    "ModelArtifact",
    "ModelArtifactRegistry",
    "ModelCompatibilityError",
    "ModelRegistryError",
    "PromotionRecord",
    "PromotionRejected",
    "RegistryIntegrityError",
    "ValidationEvidence",
    "WorldModelSpec",
]

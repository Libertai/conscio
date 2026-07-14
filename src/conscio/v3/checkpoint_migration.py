"""Content-addressed audit records for specialist-architecture migrations."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARCHITECTURE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CheckpointArchitectureMigration:
    record_id: str
    source_checkpoint_id: str
    source_lineage_id: str
    source_checkpoint_digest: str
    source_architecture_id: str
    target_checkpoint_id: str
    target_lineage_id: str
    target_checkpoint_digest: str
    target_architecture_id: str
    model_version: str
    runtime_identity: str
    transform_digest: str
    evidence_digest: str | None
    migrator: str
    reason: str
    previous_record_hash: str | None
    record_hash: str
    created_at: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "source_checkpoint_id",
            "source_lineage_id",
            "source_architecture_id",
            "target_checkpoint_id",
            "target_lineage_id",
            "target_architecture_id",
            "model_version",
            "runtime_identity",
            "migrator",
            "reason",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("source_architecture_id", "target_architecture_id"):
            if _ARCHITECTURE_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a content-addressed architecture ID")
        if _ARCHITECTURE_RE.fullmatch(self.runtime_identity) is None:
            raise ValueError("runtime_identity must be content addressed")
        for name in (
            "source_checkpoint_digest",
            "target_checkpoint_digest",
            "transform_digest",
            "record_hash",
        ):
            if _SHA256_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.previous_record_hash is not None and _SHA256_RE.fullmatch(self.previous_record_hash) is None:
            raise ValueError("previous_record_hash must be a lowercase SHA-256 digest")
        if self.evidence_digest is not None and _SHA256_RE.fullmatch(self.evidence_digest) is None:
            raise ValueError("evidence_digest must be a lowercase SHA-256 digest")
        if self.source_checkpoint_id == self.target_checkpoint_id:
            raise ValueError("migration must create a new checkpoint")
        if self.source_lineage_id == self.target_lineage_id:
            raise ValueError("migration must create a new lineage")
        if self.schema_version != 1:
            raise ValueError("unsupported architecture migration schema")
        if self.record_hash != content_digest(self._hash_payload()):
            raise ValueError("architecture migration record hash mismatch")

    def _hash_payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("record_hash")
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        *,
        source_checkpoint: Mapping[str, Any],
        source_architecture_id: str,
        target_checkpoint: Mapping[str, Any],
        runtime_identity: str,
        transform_digest: str,
        evidence_digest: str | None,
        migrator: str,
        reason: str,
        previous_record_hash: str | None,
        created_at: float,
    ) -> CheckpointArchitectureMigration:
        source_digest = content_digest(source_checkpoint)
        target_digest = content_digest(target_checkpoint)
        provisional: dict[str, Any] = {
            "record_id": "archmig_"
            + content_digest(
                [
                    source_checkpoint["checkpoint_id"],
                    target_checkpoint["checkpoint_id"],
                    source_digest,
                    target_digest,
                ]
            )[:32],
            "source_checkpoint_id": str(source_checkpoint["checkpoint_id"]),
            "source_lineage_id": str(source_checkpoint["lineage_id"]),
            "source_checkpoint_digest": source_digest,
            "source_architecture_id": source_architecture_id,
            "target_checkpoint_id": str(target_checkpoint["checkpoint_id"]),
            "target_lineage_id": str(target_checkpoint["lineage_id"]),
            "target_checkpoint_digest": target_digest,
            "target_architecture_id": str(target_checkpoint["specialist_architecture_id"]),
            "model_version": str(target_checkpoint["model_version"]),
            "runtime_identity": runtime_identity,
            "transform_digest": transform_digest,
            "evidence_digest": evidence_digest,
            "migrator": migrator,
            "reason": reason,
            "previous_record_hash": previous_record_hash,
            "created_at": float(created_at),
            "schema_version": 1,
        }
        return cls(**provisional, record_hash=content_digest(provisional))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointArchitectureMigration:
        return cls(**dict(data))


__all__ = [
    "CheckpointArchitectureMigration",
    "canonical_json",
    "content_digest",
]

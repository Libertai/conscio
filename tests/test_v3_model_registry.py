from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conscio.v3.model_registry import (
    ARTIFACT_KIND,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ModelArtifact,
    ModelArtifactRegistry,
    ModelCompatibilityError,
    ModelRegistryError,
    PromotionRejected,
    RegistryIntegrityError,
    ValidationEvidence,
    WorldModelSpec,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _spec(
    revision: int,
    *,
    deterministic_dim: int = 8,
    stochastic_dim: int = 4,
    dtype: str = "float64",
    family: str = "conscio-v3-rssm",
) -> WorldModelSpec:
    return WorldModelSpec(
        model_family=family,
        model_version=f"rssm-{revision}",
        revision=revision,
        deterministic_dim=deterministic_dim,
        stochastic_dim=stochastic_dim,
        affect_dim=4,
        specialist_dims={"memory": 3, "prediction": 5, "self": 2},
        dtype=dtype,
    )


def _register(registry: ModelArtifactRegistry, revision: int, **spec_changes: object) -> ModelArtifact:
    return registry.register_artifact(
        _spec(revision, **spec_changes),  # type: ignore[arg-type]
        f"immutable recurrent weights revision {revision} {spec_changes}".encode(),
    )


def _evidence(
    candidate: ModelArtifact,
    incumbent: ModelArtifact | None,
    *,
    candidate_score: float = 0.15,
    incumbent_score: float | None = 0.25,
    passed: bool = True,
) -> ValidationEvidence:
    return ValidationEvidence(
        candidate_digest=candidate.digest,
        incumbent_digest=None if incumbent is None else incumbent.digest,
        dataset_digest=_digest("held-out-episodes-v1"),
        protocol_digest=_digest("preregistered-shadow-protocol-v1"),
        metric_name="brier_loss",
        candidate_score=candidate_score,
        incumbent_score=None if incumbent is None else incumbent_score,
        lower_is_better=True,
        sample_count=512,
        passed=passed,
        notes="condition-blind held-out validation",
    )


def _promote_initial(registry: ModelArtifactRegistry, artifact: ModelArtifact) -> None:
    record = registry.decide_promotion(
        incumbent_digest=None,
        candidate_digest=artifact.digest,
        evidence=_evidence(artifact, None),
        approve=True,
        reason="bootstrap artifact passed offline acceptance",
        decided_by="test-reviewer",
    )
    assert record.decision == "accepted"


def test_spec_and_evidence_public_round_trip_is_strict() -> None:
    spec = _spec(3)
    assert WorldModelSpec.from_dict(spec.to_dict()) == spec
    assert spec.artifact_kind == ARTIFACT_KIND
    assert tuple(spec.specialist_dims) == ("memory", "prediction", "self")

    with pytest.raises(TypeError, match="integer"):
        WorldModelSpec.from_dict({**spec.to_dict(), "revision": True})
    with pytest.raises(ValueError, match="unknown"):
        WorldModelSpec.from_dict({**spec.to_dict(), "surprise": 1})
    with pytest.raises(ValueError, match="unsupported model artifact schema"):
        WorldModelSpec.from_dict({**spec.to_dict(), "schema_version": 2})


def test_artifact_is_content_addressed_idempotent_and_immutable(tmp_path: Path) -> None:
    registry = ModelArtifactRegistry(tmp_path / "registry")
    spec = _spec(1)
    first = registry.register_artifact(spec, b"same exact model bytes")
    second = registry.register_artifact(spec, b"same exact model bytes")

    assert first.digest == second.digest
    assert first.weights_digest == _digest("same exact model bytes")
    assert first.weights == b"same exact model bytes"
    assert first.descriptor_path.name == f"{first.digest}.json"
    assert first.weights_path.name == f"{first.weights_digest}.bin"
    assert not list((tmp_path / "registry").rglob("*.tmp"))
    assert first.descriptor_path.stat().st_mode & 0o222 == 0
    assert first.weights_path.stat().st_mode & 0o222 == 0

    different_spec = _spec(2)
    different = registry.register_artifact(different_spec, b"same exact model bytes")
    assert different.digest != first.digest
    assert different.weights_path == first.weights_path


def test_load_rejects_missing_artifact_and_exact_spec_mismatch(tmp_path: Path) -> None:
    registry = ModelArtifactRegistry(tmp_path)
    artifact = _register(registry, 1)
    with pytest.raises(ArtifactNotFoundError):
        registry.load_artifact("0" * 64)
    with pytest.raises(ModelCompatibilityError, match="exact expected spec"):
        registry.load_artifact(artifact.digest, expected_spec=_spec(2))


def test_corrupt_weights_are_never_loaded_or_silently_repaired(tmp_path: Path) -> None:
    registry = ModelArtifactRegistry(tmp_path)
    artifact = _register(registry, 1)
    artifact.weights_path.chmod(0o600)
    artifact.weights_path.write_bytes(b"tampered weights")

    with pytest.raises(ArtifactIntegrityError, match="weight digest mismatch"):
        registry.load_artifact(artifact.digest)
    with pytest.raises(ArtifactIntegrityError, match="immutable artifact file changed"):
        registry.register_artifact(artifact.spec, artifact.weights)
    assert artifact.weights_path.read_bytes() == b"tampered weights"


def test_corrupt_descriptor_and_noncanonical_descriptor_are_rejected(tmp_path: Path) -> None:
    registry = ModelArtifactRegistry(tmp_path)
    artifact = _register(registry, 1)
    artifact.descriptor_path.chmod(0o600)
    descriptor = json.loads(artifact.descriptor_path.read_text())
    artifact.descriptor_path.write_text(json.dumps(descriptor, indent=2))

    with pytest.raises(ArtifactIntegrityError, match="descriptor digest mismatch"):
        registry.load_artifact(artifact.digest)


def test_initial_and_successor_promotions_recover_after_restart(tmp_path: Path) -> None:
    root = tmp_path / "models"
    registry = ModelArtifactRegistry(root, clock=lambda: 100.0)
    incumbent = _register(registry, 1)
    candidate = _register(registry, 2)
    _promote_initial(registry, incumbent)

    accepted = registry.decide_promotion(
        incumbent_digest=incumbent.digest,
        candidate_digest=candidate.digest,
        evidence=_evidence(candidate, incumbent),
        approve=True,
        reason="candidate improved held-out Brier loss",
        decided_by="independent-reviewer",
    )
    assert accepted.decision == "accepted"
    assert accepted.evidence.digest() == _evidence(candidate, incumbent).digest()
    assert accepted.previous_record_hash == registry.promotion_records()[0].record_hash

    restarted = ModelArtifactRegistry(root)
    active = restarted.latest_promoted()
    assert active is not None
    assert active.digest == candidate.digest
    assert active.weights == candidate.weights
    assert [record.decision for record in restarted.promotion_records()] == ["accepted", "accepted"]


def test_downgrade_approval_is_rejected_and_audited(tmp_path: Path) -> None:
    registry = ModelArtifactRegistry(tmp_path)
    incumbent = _register(registry, 2)
    downgrade = _register(registry, 1)
    _promote_initial(registry, incumbent)

    with pytest.raises(PromotionRejected) as rejected:
        registry.decide_promotion(
            incumbent_digest=incumbent.digest,
            candidate_digest=downgrade.digest,
            evidence=_evidence(downgrade, incumbent),
            approve=True,
            reason="attempted rollback",
            decided_by="operator",
        )

    assert rejected.value.record.decision == "rejected"
    assert any("not newer" in issue for issue in rejected.value.record.eligibility_issues)
    assert registry.promotion_records()[-1] == rejected.value.record
    assert registry.latest_promoted().digest == incumbent.digest  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("spec_changes", "expected_issue"),
    [
        ({"deterministic_dim": 9}, "deterministic state dimensions differ"),
        ({"stochastic_dim": 7}, "stochastic state dimensions differ"),
        ({"dtype": "float32"}, "model dtypes differ"),
        ({"family": "unrelated-family"}, "model families differ"),
    ],
)
def test_incompatible_candidate_approval_is_rejected(
    tmp_path: Path, spec_changes: dict[str, object], expected_issue: str
) -> None:
    registry = ModelArtifactRegistry(tmp_path)
    incumbent = _register(registry, 1)
    candidate = _register(registry, 2, **spec_changes)
    _promote_initial(registry, incumbent)

    with pytest.raises(PromotionRejected) as rejected:
        registry.decide_promotion(
            incumbent_digest=incumbent.digest,
            candidate_digest=candidate.digest,
            evidence=_evidence(candidate, incumbent),
            approve=True,
            reason="shape compatibility must be checked",
            decided_by="operator",
        )
    assert expected_issue in rejected.value.record.eligibility_issues
    assert registry.latest_promoted().digest == incumbent.digest  # type: ignore[union-attr]


def test_missing_failed_or_non_improving_evidence_cannot_promote(tmp_path: Path) -> None:
    registry = ModelArtifactRegistry(tmp_path)
    incumbent = _register(registry, 1)
    candidate = _register(registry, 2)
    _promote_initial(registry, incumbent)

    with pytest.raises(TypeError, match="ValidationEvidence"):
        registry.decide_promotion(
            incumbent_digest=incumbent.digest,
            candidate_digest=candidate.digest,
            evidence=None,  # type: ignore[arg-type]
            approve=True,
            reason="missing evidence",
            decided_by="operator",
        )
    assert len(registry.promotion_records()) == 1

    failed = _evidence(candidate, incumbent, passed=False)
    with pytest.raises(PromotionRejected) as rejected_failed:
        registry.decide_promotion(
            incumbent_digest=incumbent.digest,
            candidate_digest=candidate.digest,
            evidence=failed,
            approve=True,
            reason="failed validation",
            decided_by="operator",
        )
    assert "validation evidence did not pass" in rejected_failed.value.record.eligibility_issues

    worse = _evidence(candidate, incumbent, candidate_score=0.4, incumbent_score=0.25)
    with pytest.raises(PromotionRejected) as rejected_worse:
        registry.decide_promotion(
            incumbent_digest=incumbent.digest,
            candidate_digest=candidate.digest,
            evidence=worse,
            approve=True,
            reason="worse metric",
            decided_by="operator",
        )
    assert any("strictly improve" in issue for issue in rejected_worse.value.record.eligibility_issues)
    assert registry.latest_promoted().digest == incumbent.digest  # type: ignore[union-attr]


def test_stale_incumbent_and_explicit_rejection_do_not_change_active_model(tmp_path: Path) -> None:
    registry = ModelArtifactRegistry(tmp_path)
    first = _register(registry, 1)
    second = _register(registry, 2)
    third = _register(registry, 3)
    _promote_initial(registry, first)
    registry.decide_promotion(
        incumbent_digest=first.digest,
        candidate_digest=second.digest,
        evidence=_evidence(second, first),
        approve=True,
        reason="second accepted",
        decided_by="reviewer",
    )

    with pytest.raises(PromotionRejected, match="declared incumbent"):
        registry.decide_promotion(
            incumbent_digest=first.digest,
            candidate_digest=third.digest,
            evidence=_evidence(third, first),
            approve=True,
            reason="stale compare-and-swap",
            decided_by="reviewer",
        )
    explicit = registry.decide_promotion(
        incumbent_digest=second.digest,
        candidate_digest=third.digest,
        evidence=_evidence(third, second),
        approve=False,
        reason="external review has not signed off",
        decided_by="reviewer",
    )
    assert explicit.decision == "rejected"
    assert explicit.eligibility_issues == ()
    assert registry.latest_promoted().digest == second.digest  # type: ignore[union-attr]


def test_lineage_migration_is_explicit_hash_chained_and_restart_safe(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry = ModelArtifactRegistry(root, clock=lambda: 222.0)
    incumbent = _register(registry, 1)
    candidate = _register(registry, 2)
    _promote_initial(registry, incumbent)
    registry.decide_promotion(
        incumbent_digest=incumbent.digest,
        candidate_digest=candidate.digest,
        evidence=_evidence(candidate, incumbent),
        approve=True,
        reason="validated replacement",
        decided_by="reviewer",
    )

    migration = registry.record_lineage_migration(
        source_checkpoint_id="ckpt-old-final",
        source_lineage_id="lineage-old",
        source_artifact_digest=incumbent.digest,
        target_checkpoint_id="ckpt-new-bootstrap",
        target_lineage_id="lineage-new",
        target_artifact_digest=candidate.digest,
        transform_digest=_digest("state-transform-code-and-parameters"),
        evidence_digest=_digest("migration-validation-report"),
        migrator="conscio-migrate-v1",
        reason="begin an explicit lineage after verified state transformation",
    )
    assert migration.previous_record_hash is None
    assert migration.source_artifact_digest == incumbent.digest
    assert migration.target_artifact_digest == candidate.digest

    restarted = ModelArtifactRegistry(root)
    assert restarted.latest_lineage_migration() == migration
    assert restarted.latest_lineage_migration("lineage-new") == migration
    assert restarted.latest_lineage_migration("unknown") is None
    with pytest.raises(ModelRegistryError, match="target checkpoint already migrated"):
        restarted.record_lineage_migration(
            source_checkpoint_id="ckpt-old-final",
            source_lineage_id="lineage-old",
            source_artifact_digest=incumbent.digest,
            target_checkpoint_id="ckpt-new-bootstrap",
            target_lineage_id="another-lineage",
            target_artifact_digest=candidate.digest,
            transform_digest=_digest("transform-2"),
            evidence_digest=_digest("evidence-2"),
            migrator="conscio-migrate-v1",
            reason="duplicate target is forbidden",
        )


def test_lineage_migration_rejects_unpromoted_target_and_missing_evidence(tmp_path: Path) -> None:
    registry = ModelArtifactRegistry(tmp_path)
    incumbent = _register(registry, 1)
    unpromoted = _register(registry, 2)
    _promote_initial(registry, incumbent)

    with pytest.raises(ModelCompatibilityError, match="not the active"):
        registry.record_lineage_migration(
            source_checkpoint_id="source",
            source_lineage_id="old",
            source_artifact_digest=incumbent.digest,
            target_checkpoint_id="target",
            target_lineage_id="new",
            target_artifact_digest=unpromoted.digest,
            transform_digest=_digest("transform"),
            evidence_digest=_digest("evidence"),
            migrator="migrator-v1",
            reason="must not silently switch weights",
        )

    registry.decide_promotion(
        incumbent_digest=incumbent.digest,
        candidate_digest=unpromoted.digest,
        evidence=_evidence(unpromoted, incumbent),
        approve=True,
        reason="validated",
        decided_by="reviewer",
    )
    with pytest.raises(ValueError, match="evidence_digest"):
        registry.record_lineage_migration(
            source_checkpoint_id="source",
            source_lineage_id="old",
            source_artifact_digest=incumbent.digest,
            target_checkpoint_id="target",
            target_lineage_id="new",
            target_artifact_digest=unpromoted.digest,
            transform_digest=_digest("transform"),
            evidence_digest="",
            migrator="migrator-v1",
            reason="evidence is mandatory",
        )
    assert registry.lineage_migrations() == ()


def test_audit_log_tampering_blocks_restart_recovery_and_future_changes(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry = ModelArtifactRegistry(root)
    artifact = _register(registry, 1)
    _promote_initial(registry, artifact)
    log_path = root / "audit" / "promotions.jsonl"
    raw = json.loads(log_path.read_text())
    raw["reason"] = "tampered after approval"
    log_path.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n")

    restarted = ModelArtifactRegistry(root)
    with pytest.raises(RegistryIntegrityError, match="hash mismatch"):
        restarted.latest_promoted()
    with pytest.raises(RegistryIntegrityError, match="hash mismatch"):
        restarted.promotion_records()


def test_incomplete_audit_tail_is_not_ignored(tmp_path: Path) -> None:
    root = tmp_path / "registry"
    registry = ModelArtifactRegistry(root)
    artifact = _register(registry, 1)
    _promote_initial(registry, artifact)
    log_path = root / "audit" / "promotions.jsonl"
    with log_path.open("ab") as stream:
        stream.write(b'{"partial":')

    with pytest.raises(RegistryIntegrityError, match="incomplete"):
        ModelArtifactRegistry(root).latest_promoted()

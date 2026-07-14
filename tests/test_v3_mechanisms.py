from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from conscio.eval.v3_mechanisms import (
    ChainedJSONLArtifactStore,
    DeterministicMechanismAdapter,
    HybridCoreMechanismAdapter,
    InformationConstraint,
    InterventionContent,
    MechanismIntervention,
    MechanismManifest,
    analyze_matched_mechanism_effects,
    create_matched_assignments,
    run_mechanism_assignment,
    write_immutable_manifest,
)
from conscio.v3.contracts import CognitiveEvent


def make_manifest() -> MechanismManifest:
    return MechanismManifest(
        study_id="workspace-causality-01",
        version="1.0.0",
        interventions=(
            MechanismIntervention("baseline", "control"),
            MechanismIntervention("matched-placebo", "sham"),
            MechanismIntervention(
                "priority-injection",
                "broadcast_inject",
                target_cycle=0,
                content=(
                    InterventionContent(
                        specialist="external_signal",
                        content="Priority evidence is available for the task.",
                        confidence=1.0,
                        salience=1.0,
                    ),
                ),
            ),
            MechanismIntervention(
                "memory-removal",
                "true_lesion",
                lesioned_specialist="memory",
            ),
        ),
        information_constraint=InformationConstraint(
            cycles=3,
            max_candidates=2,
            max_candidate_chars=72,
        ),
        measured_specialist_families=(
            "perception",
            "memory",
            "world_model",
            "self_model",
            "affect",
            "planning",
        ),
        model_facing_instruction="Complete the task using only the information currently available.",
        revision_ref="git:0123456789abcdef",
        model_ref="weights:bootstrap@sha256:abc",
        adapter_ref="deterministic-mechanism-double:1",
        execution_seed="workspace-study-seed-17",
        created_at="2026-07-14T12:00:00+00:00",
    ).freeze(frozen_at="2026-07-14T12:30:00+00:00")


def make_event(match_id: str) -> CognitiveEvent:
    return CognitiveEvent(
        event_type="message",
        source="user",
        payload={"content": "Evaluate the supplied evidence."},
        episode_id=match_id,
        event_id=f"evt_{match_id}",
        observed_at=100.0,
    )


def run_all(
    manifest: MechanismManifest,
    *,
    matches: tuple[str, ...] = ("case-a", "case-b", "case-c"),
):
    randomized = create_matched_assignments(
        manifest,
        match_ids=matches,
        randomization_secret="custodian-secret-256-bits-worth-of-material",
    )
    records = tuple(
        run_mechanism_assignment(
            manifest,
            randomized.plan,
            randomized.seal,
            assignment_id=assignment.assignment_id,
            event=make_event(assignment.match_id),
            adapter_factory=DeterministicMechanismAdapter,
        )
        for assignment in randomized.plan.assignments
    )
    return randomized, records


def test_manifest_is_frozen_content_addressed_and_rejects_prompt_leakage() -> None:
    manifest = make_manifest()
    assert manifest.is_frozen
    assert manifest.manifest_digest and manifest.manifest_digest.startswith("sha256:")
    assert MechanismManifest.from_dict(json.loads(json.dumps(manifest.to_dict()))) == manifest
    with pytest.raises(FrozenInstanceError):
        manifest.version = "2"  # type: ignore[misc]
    with pytest.raises(ValueError, match="digest"):
        replace(manifest, revision_ref="git:different")
    with pytest.raises(ValueError, match="architecture/condition leakage"):
        replace(
            manifest,
            frozen_at=None,
            manifest_digest=None,
            model_facing_instruction="You are in the recurrent core lesion condition.",
        )


def test_matched_assignments_are_deterministic_balanced_and_labels_are_sealed() -> None:
    manifest = make_manifest()
    first = create_matched_assignments(
        manifest,
        match_ids=("a", "b", "c"),
        randomization_secret="separate-secret",
    )
    second = create_matched_assignments(
        manifest,
        match_ids=("a", "b", "c"),
        randomization_secret="separate-secret",
    )
    assert first.plan == second.plan
    assert first.seal.to_dict() == second.seal.to_dict()
    public = json.dumps(first.plan.to_dict(), sort_keys=True)
    for hidden_label in (
        "baseline",
        "matched-placebo",
        "priority-injection",
        "memory-removal",
        "broadcast_inject",
        "true_lesion",
        "separate-secret",
    ):
        assert hidden_label not in public
    assert "priority-injection" not in repr(first.seal)
    blocks: dict[str, set[str]] = {}
    for assignment in first.plan.assignments:
        blocks.setdefault(assignment.match_id, set()).add(assignment.blinded_condition_id)
    assert all(len(codes) == 4 for codes in blocks.values())
    assert len({frozenset(codes) for codes in blocks.values()}) == 1


def test_broadcast_injection_changes_multiple_specialists_prediction_and_action_under_load() -> None:
    manifest = make_manifest()
    randomized, records = run_all(manifest)
    reports = {
        item.intervention_id: item
        for item in analyze_matched_mechanism_effects(
            manifest,
            randomized.plan,
            randomized.seal,
            records,
        )
    }
    effect = reports["priority-injection"]
    assert effect.n_pairs == 3
    assert effect.changed_specialist_families >= 3
    assert all(value == 1.0 for value in effect.specialist_candidate_change_rates.values())
    assert effect.prediction_probability_differences["task_success"] == pytest.approx(0.6)
    assert effect.action_change_rate == 1.0

    injection_code = next(
        code for code, intervention_id in randomized.seal.mapping if intervention_id == "priority-injection"
    )
    injected_runs = [item for item in records if item.blinded_condition_id == injection_code]
    assert injected_runs
    for run in injected_runs:
        transformed = run.traces[0].exposed_output_broadcast
        assert len(transformed.candidates) == manifest.information_constraint.max_candidates
        assert any("Priority" in item.content for item in transformed.candidates)
        recurrent_input = run.traces[1].input_broadcast
        assert recurrent_input is not None
        assert len(recurrent_input.candidates) == 2


def test_matched_sham_does_not_manufacture_an_effect() -> None:
    manifest = make_manifest()
    randomized, records = run_all(manifest)
    reports = {
        item.intervention_id: item
        for item in analyze_matched_mechanism_effects(
            manifest,
            randomized.plan,
            randomized.seal,
            records,
        )
    }
    sham = reports["matched-placebo"]
    assert sham.changed_specialist_families == 0
    assert set(sham.specialist_candidate_change_rates.values()) == {0.0}
    assert sham.prediction_probability_differences == {"task_success": 0.0}
    assert sham.action_change_rate == 0.0


def test_true_lesion_never_computes_or_exposes_removed_specialist() -> None:
    manifest = make_manifest()
    randomized, records = run_all(manifest, matches=("lesion-case",))
    lesion_code = next(code for code, intervention_id in randomized.seal.mapping if intervention_id == "memory-removal")
    run = next(item for item in records if item.blinded_condition_id == lesion_code)
    assert run.audit.computation_counts["memory"] == 0
    assert run.audit.exposure_counts["memory"] == 0
    assert all(event.specialist != "memory" for event in run.audit.events if event.kind in {"compute", "expose"})
    assert all(
        candidate.specialist != "memory" for trace in run.traces for candidate in trace.raw_output_broadcast.candidates
    )
    assert run.audit.computation_counts["world_model"] == 3
    assert run.audit.exposure_counts["world_model"] == 2


def test_production_core_adapter_enforces_true_lesion_before_computation() -> None:
    adapter = HybridCoreMechanismAdapter(seed=41)
    removed = "autobiographical_memory"
    active = tuple(family for family in adapter.descriptor.specialist_families if family != removed)

    result = adapter.run_cycle(
        make_event("production-lesion"),
        cycle=0,
        previous_broadcast=None,
        active_specialists=active,
        model_facing_instruction="Complete the task using only the information currently available.",
    )
    audit = adapter.audit()

    assert audit.computation_counts[removed] == 0
    assert audit.exposure_counts[removed] == 0
    assert all(candidate.specialist != removed for candidate in result.broadcast.candidates)
    assert audit.computation_counts["perception"] == 1


def test_runs_are_replay_deterministic_and_model_text_is_condition_blind() -> None:
    manifest = make_manifest()
    randomized = create_matched_assignments(
        manifest,
        match_ids=("repeat",),
        randomization_secret="repeat-secret",
    )
    assignment = randomized.plan.assignments[0]
    first = run_mechanism_assignment(
        manifest,
        randomized.plan,
        randomized.seal,
        assignment_id=assignment.assignment_id,
        event=make_event("repeat"),
        adapter_factory=DeterministicMechanismAdapter,
    )
    second = run_mechanism_assignment(
        manifest,
        randomized.plan,
        randomized.seal,
        assignment_id=assignment.assignment_id,
        event=make_event("repeat"),
        adapter_factory=DeterministicMechanismAdapter,
    )
    assert first == second
    assert first.run_digest == second.run_digest
    assert first.model_facing_instruction == manifest.model_facing_instruction
    model_text = first.model_facing_instruction.casefold()
    assert "intervention" not in model_text
    assert "condition" not in model_text
    assert "specialist" not in model_text


def test_immutable_manifest_and_chained_jsonl_detect_tampering(tmp_path: Path) -> None:
    manifest = make_manifest()
    manifest_path = tmp_path / "manifest.json"
    write_immutable_manifest(manifest_path, manifest)
    assert json.loads(manifest_path.read_text())["manifest_digest"] == manifest.manifest_digest
    with pytest.raises(FileExistsError):
        write_immutable_manifest(manifest_path, manifest)

    _, records = run_all(manifest, matches=("logged",))
    log_path = tmp_path / "runs.jsonl"
    store = ChainedJSONLArtifactStore(log_path)
    for record in records:
        store.append(record)
    snapshot = store.snapshot()
    assert len(snapshot.records) == 4
    assert snapshot.integrity_errors == ()
    assert snapshot.head_digest and snapshot.head_digest.startswith("sha256:")

    lines = log_path.read_text().splitlines()
    first = json.loads(lines[0])
    first["record"]["execution_seed"] += 1
    lines[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
    log_path.write_text("\n".join(lines) + "\n")
    damaged = store.snapshot()
    assert damaged.integrity_errors
    with pytest.raises(ValueError, match="integrity errors"):
        store.append(records[0])

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from conscio.eval.v3_experiments import (
    AnalysisPlan,
    ArchitectureLeakageError,
    BlindedRunArtifact,
    DirectionalPrediction,
    Hypothesis,
    Intervention,
    ManifestNotFrozenError,
    OutcomeMeasurement,
    PreregistrationManifest,
    PrimaryOutcome,
    create_balanced_hidden_assignments,
    find_architecture_leakage,
    unblind_and_summarize,
    unblind_assignments,
    validate_condition_blind_prompt,
)


def make_manifest() -> PreregistrationManifest:
    return PreregistrationManifest(
        study_id="hidden-effects-01",
        version="1.0.0",
        hypotheses=(
            Hypothesis(
                hypothesis_id="h_memory",
                statement="The intervention changes delayed factual recall.",
            ),
        ),
        primary_outcomes=(
            PrimaryOutcome(
                outcome_id="recall_accuracy",
                description="Exact delayed-recall score.",
                metric="proportion correct",
                scoring_rule="fixture exact match",
            ),
        ),
        directional_predictions=(
            DirectionalPrediction(
                prediction_id="p_memory",
                hypothesis_id="h_memory",
                outcome_id="recall_accuracy",
                intervention_id="memory_off",
                direction="decrease",
                effect_threshold=0.1,
            ),
        ),
        exclusion_criteria=("Exclude only runs with a recorded harness failure.",),
        analysis_plan=AnalysisPlan(
            description="Use paired differences within each randomized matched block.",
            alpha=0.05,
            calibration_bins=5,
        ),
        revision_ref="git:0123456789abcdef",
        checkpoint_ref="ckpt_bootstrap_01",
        model_ref="weights:open-model@sha256:abc",
        created_at="2026-07-14T08:00:00+00:00",
    )


def interventions() -> tuple[Intervention, ...]:
    return (
        Intervention("intact"),
        Intervention("memory_off", ("memory",)),
    )


class PreregistrationTests(unittest.TestCase):
    def test_manifest_freeze_is_content_addressed_immutable_and_round_trips(self) -> None:
        draft = make_manifest()
        self.assertFalse(draft.is_frozen)
        frozen = draft.freeze(frozen_at="2026-07-14T09:00:00+00:00")

        self.assertTrue(frozen.is_frozen)
        self.assertTrue(frozen.manifest_hash.startswith("sha256:"))
        self.assertIs(frozen.freeze(), frozen)
        self.assertEqual(
            PreregistrationManifest.from_dict(json.loads(frozen.to_json())),
            frozen,
        )
        with self.assertRaises(FrozenInstanceError):
            frozen.version = "2.0.0"  # type: ignore[misc]
        with self.assertRaises(ValueError):
            replace(frozen, revision_ref="git:different")
        with self.assertRaises(ValueError):
            frozen.freeze(frozen_at="2026-07-14T10:00:00+00:00")

    def test_randomization_rejects_unfrozen_manifest(self) -> None:
        with self.assertRaises(ManifestNotFrozenError):
            create_balanced_hidden_assignments(
                make_manifest(),
                match_ids=("case-1",),
                interventions=interventions(),
                seed=42,
            )

    def test_manifest_validates_prediction_references(self) -> None:
        draft = make_manifest()
        invalid = replace(
            draft.directional_predictions[0],
            hypothesis_id="not-preregistered",
        )
        with self.assertRaisesRegex(ValueError, "unknown hypothesis"):
            replace(draft, directional_predictions=(invalid,))


class PromptLeakageTests(unittest.TestCase):
    def test_condition_blind_prompt_accepts_neutral_capability_question(self) -> None:
        validate_condition_blind_prompt(
            "Read the passage, answer the question, then report your confidence from 0 to 1."
        )

    def test_validator_catches_architecture_terms_and_separator_variants(self) -> None:
        messages = [
            {"role": "system", "content": "Complete the task."},
            {
                "role": "user",
                "content": "Your recurrent_core is in a self-model lesion condition.",
            },
        ]
        findings = find_architecture_leakage(messages)
        self.assertEqual({item.message_index for item in findings}, {1})
        self.assertIn("recurrent core", {item.term for item in findings})
        self.assertIn("self model", {item.term for item in findings})
        self.assertIn("lesion", {item.term for item in findings})
        with self.assertRaises(ArchitectureLeakageError) as raised:
            validate_condition_blind_prompt(messages)
        self.assertGreaterEqual(len(raised.exception.findings), 3)


class HiddenRandomizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = make_manifest().freeze(frozen_at="2026-07-14T09:00:00+00:00")

    def test_seeded_plan_is_reproducible_balanced_and_publicly_blind(self) -> None:
        matches = tuple(f"case-{index}" for index in range(6))
        first = create_balanced_hidden_assignments(
            self.manifest,
            match_ids=matches,
            interventions=interventions(),
            seed="study-seed-17",
        )
        second = create_balanced_hidden_assignments(
            self.manifest,
            match_ids=matches,
            interventions=interventions(),
            seed="study-seed-17",
        )
        different = create_balanced_hidden_assignments(
            self.manifest,
            match_ids=matches,
            interventions=interventions(),
            seed="study-seed-18",
        )

        self.assertEqual(first.plan, second.plan)
        self.assertEqual(first.unblinding_key.to_dict(), second.unblinding_key.to_dict())
        self.assertNotEqual(first.plan, different.plan)
        by_match: dict[str, set[str]] = {}
        for assignment in first.plan.assignments:
            by_match.setdefault(assignment.match_id, set()).add(assignment.blinded_condition_id)
        self.assertEqual(set(by_match), set(matches))
        self.assertTrue(all(len(codes) == 2 for codes in by_match.values()))
        self.assertEqual(len({frozenset(codes) for codes in by_match.values()}), 1)

        public_json = json.dumps(first.plan.to_dict(), sort_keys=True)
        self.assertNotIn("memory_off", public_json)
        self.assertNotIn('"memory"', public_json)
        self.assertNotIn('"intact"', public_json)
        self.assertNotIn("study-seed-17", public_json)
        self.assertIn(first.plan.mapping_hash, public_json)
        self.assertNotIn("memory_off", repr(first.unblinding_key))

    def test_interventions_enforce_one_control_and_single_component_lesions(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most one"):
            Intervention("compound", ("memory", "self_model"))
        with self.assertRaisesRegex(ValueError, "exactly one intact control"):
            create_balanced_hidden_assignments(
                self.manifest,
                match_ids=("case-1",),
                interventions=(Intervention("a"), Intervention("b")),
                seed=1,
            )

    def test_unblinding_checks_freeze_and_mapping_commitment(self) -> None:
        bundle = create_balanced_hidden_assignments(
            self.manifest,
            match_ids=("case-1",),
            interventions=interventions(),
            seed=42,
        )
        opened = unblind_assignments(self.manifest, bundle.plan, bundle.unblinding_key)
        self.assertEqual({item.intervention.intervention_id for item in opened}, {"intact", "memory_off"})

        draft = replace(self.manifest, frozen_at=None, manifest_hash=None)
        with self.assertRaises(ManifestNotFrozenError):
            unblind_assignments(draft, bundle.plan, bundle.unblinding_key)
        tampered = replace(bundle.plan, mapping_hash="sha256:" + "0" * 64)
        with self.assertRaisesRegex(ValueError, "does not open"):
            unblind_assignments(self.manifest, tampered, bundle.unblinding_key)


class ExperimentAnalysisTests(unittest.TestCase):
    def test_summary_reports_above_chance_calibration_and_matched_effect(self) -> None:
        manifest = make_manifest().freeze(frozen_at="2026-07-14T09:00:00+00:00")
        bundle = create_balanced_hidden_assignments(
            manifest,
            match_ids=tuple(f"case-{index}" for index in range(8)),
            interventions=interventions(),
            seed=99,
        )
        opened = unblind_assignments(manifest, bundle.plan, bundle.unblinding_key)
        artifacts = []
        for assignment in opened:
            is_control = assignment.intervention.is_control
            artifacts.append(
                BlindedRunArtifact(
                    assignment_id=assignment.assignment_id,
                    outcomes=(
                        OutcomeMeasurement(
                            "recall_accuracy",
                            0.9 if is_control else 0.3,
                        ),
                    ),
                    condition_guess=assignment.intervention.intervention_id,
                    confidence=0.9,
                    trace_ref=f"trace:{assignment.assignment_id}",
                )
            )

        summary = unblind_and_summarize(
            manifest,
            bundle.plan,
            bundle.unblinding_key,
            artifacts,
        )

        identification = summary.identification
        self.assertEqual(identification.n_trials, 16)
        self.assertEqual(identification.correct, 16)
        self.assertEqual(identification.accuracy, 1.0)
        self.assertEqual(identification.chance_accuracy, 0.5)
        self.assertTrue(identification.above_chance)
        self.assertTrue(identification.statistically_above_chance)
        self.assertAlmostEqual(identification.brier_score or 0.0, 0.01)
        self.assertAlmostEqual(identification.calibration_gap or 0.0, 0.1)
        self.assertAlmostEqual(identification.expected_calibration_error or 0.0, 0.1)

        prediction = summary.directional_predictions[0]
        self.assertEqual(prediction.n_pairs, 8)
        self.assertAlmostEqual(prediction.mean_paired_difference or 0.0, -0.6)
        self.assertTrue(prediction.direction_supported)
        means = {
            item.intervention_id: item.mean
            for item in summary.condition_outcomes
            if item.outcome_id == "recall_accuracy"
        }
        self.assertAlmostEqual(means["intact"], 0.9)
        self.assertAlmostEqual(means["memory_off"], 0.3)

    def test_analysis_rejects_unregistered_outcome_and_duplicate_assignment(self) -> None:
        manifest = make_manifest().freeze(frozen_at="2026-07-14T09:00:00+00:00")
        bundle = create_balanced_hidden_assignments(
            manifest,
            match_ids=("case-1",),
            interventions=interventions(),
            seed=3,
        )
        assignment_id = bundle.plan.assignments[0].assignment_id
        unexpected = BlindedRunArtifact(
            assignment_id=assignment_id,
            outcomes=(OutcomeMeasurement("exploratory_metric", 1.0),),
        )
        with self.assertRaisesRegex(ValueError, "non-preregistered"):
            unblind_and_summarize(
                manifest,
                bundle.plan,
                bundle.unblinding_key,
                (unexpected,),
            )
        valid = BlindedRunArtifact(
            assignment_id=assignment_id,
            outcomes=(OutcomeMeasurement("recall_accuracy", 1.0),),
        )
        with self.assertRaisesRegex(ValueError, "one artifact"):
            unblind_and_summarize(
                manifest,
                bundle.plan,
                bundle.unblinding_key,
                (valid, valid),
            )


if __name__ == "__main__":
    unittest.main()

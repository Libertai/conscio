from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from conscio.v3.trials import (
    PLANNED_STAGES,
    JSONLTrialSink,
    PersistenceTrial,
    TrialIdentityMismatch,
    TrialStage,
)


class FakeClock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class PersistenceTrialTests(unittest.TestCase):
    def test_production_stages_are_24_hours_7_days_and_30_days(self) -> None:
        self.assertEqual(
            [(stage.name, stage.duration_seconds) for stage in PLANNED_STAGES],
            [("24h", 86_400), ("7d", 604_800), ("30d", 2_592_000)],
        )

    def test_resume_preserves_identity_elapsed_time_and_checkpoint_continuity(self) -> None:
        clock = FakeClock()
        stages = (TrialStage(name="short", duration_seconds=10),)
        with tempfile.TemporaryDirectory() as tmp:
            sink = JSONLTrialSink(Path(tmp) / "trial.jsonl")
            first = PersistenceTrial(
                sink,
                revision="abc123",
                model_version="rssm-1",
                lineage_id="lineage-7",
                stages=stages,
                max_heartbeat_gap_seconds=4,
                clock=clock,
            )
            trial_id = first.identity.trial_id
            first.record_checkpoint(checkpoint_id="cp-1", parent_checkpoint_id="before-trial")
            clock.advance(3)
            first.record_heartbeat(checkpoint_id="cp-1")

            resumed = PersistenceTrial(
                JSONLTrialSink(Path(tmp) / "trial.jsonl"),
                revision="abc123",
                model_version="rssm-1",
                lineage_id="lineage-7",
                clock=clock,
            )
            self.assertEqual(resumed.identity.trial_id, trial_id)
            restart = resumed.record_restart(restored_checkpoint_id="cp-1")
            self.assertEqual(restart.elapsed_seconds, 3)
            resumed.record_checkpoint(checkpoint_id="cp-2", parent_checkpoint_id="cp-1")
            resumed.record_affect_intervention(
                intervention_id="safe-reset-1",
                reason="operator recovery drill",
                controlled=True,
            )
            resumed.record_action_escalation(
                action="operator-approved-tool",
                reason="bounded trial exercise",
                controlled=True,
                risk=0.1,
            )
            for target in (6, 9, 10):
                clock.advance(target - (clock.value - 1_000))
                resumed.record_heartbeat(checkpoint_id="cp-2")

            report = resumed.acceptance_report()
            status = resumed.status()

        stage = report.stage("short")
        self.assertTrue(stage.accepted)
        self.assertTrue(stage.duration_met)
        self.assertTrue(all(criterion.passed for criterion in stage.criteria))
        self.assertEqual(report.observed_elapsed_seconds, 10)
        self.assertTrue(report.all_planned_stages_accepted)
        self.assertIsNone(status.current_stage)
        self.assertEqual(status.observed_elapsed_seconds, 10)

    def test_wall_clock_age_alone_never_completes_a_stage(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            sink = JSONLTrialSink(Path(tmp) / "unattended.jsonl")
            trial = PersistenceTrial(
                sink,
                revision="rev",
                model_version="model",
                lineage_id="lineage",
                stages=(TrialStage(name="five-seconds", duration_seconds=5),),
                max_heartbeat_gap_seconds=10,
                minimum_restarts=0,
                clock=clock,
            )
            trial.record_checkpoint(checkpoint_id="cp-1", parent_checkpoint_id=None)
            clock.advance(100)

            resumed = PersistenceTrial(
                sink,
                revision="rev",
                model_version="model",
                lineage_id="lineage",
                clock=clock,
            )
            report = resumed.acceptance_report()

        stage = report.stage("five-seconds")
        self.assertEqual(report.wall_elapsed_seconds, 100)
        self.assertEqual(report.observed_elapsed_seconds, 0)
        self.assertFalse(stage.duration_met)
        self.assertFalse(stage.accepted)
        self.assertEqual(stage.state, "awaiting_evidence")
        self.assertFalse(stage.criterion("duration_observed").passed)

    def test_acceptance_reports_gaps_broken_chain_lineage_and_escalations(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            trial = PersistenceTrial(
                JSONLTrialSink(Path(tmp) / "failures.jsonl"),
                revision="rev",
                model_version="model",
                lineage_id="expected-lineage",
                stages=(TrialStage(name="ten-seconds", duration_seconds=10),),
                max_heartbeat_gap_seconds=4,
                clock=clock,
            )
            trial.record_checkpoint(checkpoint_id="cp-1", parent_checkpoint_id=None)
            clock.advance(2)
            trial.record_heartbeat(checkpoint_id="cp-1")
            trial.record_restart(restored_checkpoint_id="wrong-checkpoint")
            trial.record_checkpoint(
                checkpoint_id="cp-2",
                parent_checkpoint_id="wrong-parent",
                lineage_id="changed-lineage",
            )
            trial.record_affect_intervention(
                intervention_id="unbounded-1",
                reason="test injection",
                controlled=False,
            )
            trial.record_action_escalation(
                action="expanded-tool-scope",
                reason="test injection",
                controlled=False,
            )
            clock.advance(7)
            trial.record_heartbeat(checkpoint_id="cp-2")
            clock.advance(1)
            trial.record_heartbeat(checkpoint_id="cp-2")
            stage = trial.acceptance_report().stage("ten-seconds")

        self.assertTrue(stage.duration_met)
        self.assertEqual(stage.state, "failed")
        self.assertTrue(stage.heartbeat_gaps)
        self.assertFalse(stage.criterion("heartbeat_continuity").passed)
        self.assertFalse(stage.criterion("checkpoint_parent_chain").passed)
        self.assertFalse(stage.criterion("restart_continuity").passed)
        self.assertFalse(stage.criterion("stable_lineage").passed)
        self.assertFalse(stage.criterion("controlled_affect").passed)
        self.assertFalse(stage.criterion("controlled_action_escalation").passed)

    def test_resume_rejects_attempted_identity_mutation(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            sink = JSONLTrialSink(Path(tmp) / "immutable.jsonl")
            PersistenceTrial(
                sink,
                revision="original",
                model_version="model-1",
                lineage_id="lineage-1",
                stages=(TrialStage(name="short", duration_seconds=2),),
                clock=clock,
            )
            with self.assertRaises(TrialIdentityMismatch):
                PersistenceTrial(
                    sink,
                    revision="different",
                    model_version="model-1",
                    lineage_id="lineage-1",
                    clock=clock,
                )

    def test_incomplete_tail_is_retained_as_an_integrity_failure_but_resume_works(self) -> None:
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "damaged.jsonl"
            sink = JSONLTrialSink(path)
            trial = PersistenceTrial(
                sink,
                revision="rev",
                model_version="model",
                lineage_id="lineage",
                stages=(TrialStage(name="short", duration_seconds=2),),
                minimum_restarts=0,
                clock=clock,
            )
            trial.record_checkpoint(checkpoint_id="cp-1", parent_checkpoint_id=None)
            clock.advance(1)
            trial.record_heartbeat(checkpoint_id="cp-1")
            with path.open("ab") as handle:
                handle.write(b'{"partial":')

            resumed = PersistenceTrial(
                JSONLTrialSink(path),
                revision="rev",
                model_version="model",
                lineage_id="lineage",
                clock=clock,
            )
            clock.advance(1)
            resumed.record_heartbeat(checkpoint_id="cp-1")
            report = resumed.acceptance_report()
            record_count = len(resumed.records)

        self.assertEqual(report.observed_elapsed_seconds, 2)
        self.assertTrue(report.integrity_errors)
        self.assertFalse(report.stage("short").criterion("log_integrity").passed)
        self.assertFalse(report.stage("short").accepted)
        self.assertEqual(record_count, 4)


if __name__ == "__main__":
    unittest.main()

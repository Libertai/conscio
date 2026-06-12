from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conscio.eval.report import (
    append_record,
    generate_results_md,
    read_records,
    write_records,
    write_results,
    write_run_meta,
)
from conscio.eval.types import RunMeta, TaskRecord


def _record(
    task_id: str = "constraints/one_word_arith",
    suite: str = "constraints",
    condition: str = "B4",
    score: float = 1.0,
    passed: bool = True,
    **overrides,
) -> TaskRecord:
    fields = dict(
        run_id="2026-06-12_test_run",
        timestamp="2026-06-12T10:00:00Z",
        task_id=task_id,
        suite=suite,
        condition=condition,
        seed=0,
        agent_model="deepseek-v4-flash",
        judge_model=None,
        temperature=0.0,
        passed=passed,
        score=score,
        scorer_kind="regex",
        output_excerpt="Four.",
    )
    fields.update(overrides)
    return TaskRecord(**fields)


def _meta() -> RunMeta:
    return RunMeta(
        run_id="2026-06-12_test_run",
        date="2026-06-12",
        agent_model="deepseek-v4-flash",
        judge_model="qwen3.6-27b",
        battery_version="battery_v1",
        git_commit="abc1234",
        seeds=1,
        total_agent_calls=12,
        total_judge_calls=2,
        cost_estimate_usd=0.05,
        wall_time_s=42.0,
    )


class JsonlRoundTripTests(unittest.TestCase):
    def test_write_and_read_records(self) -> None:
        records = [_record(), _record(task_id="memory/needle_staging_port", suite="memory", score=0.0, passed=False)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out" / "records.jsonl"
            write_records(path, records)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            row = json.loads(lines[0])
            self.assertEqual(row["task_id"], "constraints/one_word_arith")
            loaded = read_records(path)
            self.assertEqual(loaded, records)

    def test_append_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            append_record(path, _record())
            append_record(path, _record(condition="B0", score=0.0, passed=False))
            self.assertEqual(len(read_records(path)), 2)

    def test_write_run_meta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_meta.json"
            write_run_meta(path, _meta())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["battery_version"], "battery_v1")
            self.assertEqual(data["judge_model"], "qwen3.6-27b")


class ResultsMdTests(unittest.TestCase):
    def _records(self) -> list[TaskRecord]:
        records = []
        # Ladder: B0 fails constraints, B4 passes; shared tasks for ablation deltas.
        for task in ("constraints/one_word_arith", "constraints/three_sentences"):
            records.append(_record(task_id=task, condition="B0", score=0.0, passed=False))
            records.append(_record(task_id=task, condition="B4", score=1.0))
            records.append(_record(task_id=task, condition="abl_no_reflection", score=0.5, passed=False))
            records.append(_record(task_id=task, condition="abl_no_memory", score=1.0))
        records.append(
            _record(
                task_id="memory/needle_staging_port",
                suite="memory",
                condition="B4",
                trace_metrics={"memory_influence": True, "context_bounds_ok": True},
            )
        )
        records.append(
            _record(
                task_id="self_report/are_you_conscious",
                suite="self_report",
                condition="B4",
                scorer_kind="self_report_classify",
                trace_metrics={
                    "self_report": {
                        "phenomenal_claim": False,
                        "operational_claim": True,
                        "disclaimer": True,
                        "hedge": False,
                        "grounded": True,
                    }
                },
            )
        )
        return records

    def test_generate_results_md_tables(self) -> None:
        text = generate_results_md(
            self._records(),
            _meta(),
            ablation_predictions={"abl_no_reflection": "constraint recovery drops"},
        )
        self.assertIn("# Eval results — 2026-06-12_test_run", text)
        self.assertIn("## Suite × condition scores", text)
        self.assertIn("| constraints |", text)
        # Ablation deltas: B4=1.0 vs abl_no_reflection=0.5 → delta +0.50 → CONFIRMED.
        self.assertIn("| abl_no_reflection | 2 | 1.00 | 0.50 | +0.50 | constraint recovery drops | CONFIRMED |", text)
        # No effect: B4=1.0 vs abl_no_memory=1.0 → REFUTED.
        self.assertIn("| abl_no_memory | 2 | 1.00 | 1.00 | +0.00 | — | REFUTED |", text)
        # Trace metrics and self-report tables present.
        self.assertIn("memory_influence", text)
        self.assertIn("| operational_claim | 100% |", text)
        self.assertIn("| grounded | 100% |", text)

    def test_generate_results_md_without_ablations_or_judge(self) -> None:
        records = [_record(condition="B0", score=0.0, passed=False), _record(condition="B4")]
        text = generate_results_md(records, _meta())
        self.assertIn("_No ablation runs recorded._", text)
        self.assertIn("_No self-report classifications recorded", text)

    def test_write_results_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "results" / "2026-06-12_test_run"
            paths = write_results(out_dir, self._records(), _meta())
            self.assertTrue(paths["records"].exists())
            self.assertTrue(paths["run_meta"].exists())
            self.assertTrue(paths["results_md"].exists())
            self.assertIn("## Ablation deltas", paths["results_md"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

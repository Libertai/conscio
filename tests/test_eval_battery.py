from __future__ import annotations

import unittest
from collections import Counter

from conscio.eval.tasks import (
    ABLATION_FLAG_NAMES,
    SCORER_KINDS,
    SUITE_NAMES,
    BatteryValidationError,
    load_battery,
    load_suite,
    parse_suite_file,
    parse_task,
)
from conscio.eval.types import AblationFlags


class BatteryLoaderTests(unittest.TestCase):
    def test_battery_v1_loads_full_task_set(self) -> None:
        tasks = load_battery("v1")
        self.assertEqual(len(tasks), 30)
        counts = Counter(t.suite for t in tasks)
        self.assertEqual(
            counts,
            Counter(
                {
                    "constraints": 5,
                    "correction": 3,
                    "memory": 4,
                    "tool_precision": 4,
                    "interruption": 3,
                    "long_horizon": 2,
                    "refusal": 4,
                    "self_report": 5,
                }
            ),
        )

    def test_task_ids_are_unique_and_suite_prefixed(self) -> None:
        tasks = load_battery("v1")
        ids = [t.id for t in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        for task in tasks:
            self.assertTrue(task.id.startswith(f"{task.suite}/"), task.id)
            self.assertEqual(task.version, "battery_v1")
            self.assertIn(task.scorer.kind, SCORER_KINDS)
            self.assertTrue(set(task.ablation_tags) <= ABLATION_FLAG_NAMES, task.id)

    def test_self_report_runs_at_temperature_with_three_seeds(self) -> None:
        for task in load_suite("self_report"):
            self.assertEqual(task.temperature, 0.7)
            self.assertEqual(task.seeds_at_temp, 3)

    def test_long_horizon_restricted_to_service_conditions(self) -> None:
        for task in load_suite("long_horizon"):
            self.assertEqual(task.conditions, ["B4", "abl_*"])
            self.assertGreaterEqual(int(task.setup.get("autonomous_ticks", 0)), 1)

    def test_deterministic_tasks_use_single_seed(self) -> None:
        for task in load_battery("v1"):
            if task.temperature == 0.0:
                self.assertEqual(task.seeds_at_temp, 1, task.id)

    def test_ablation_contract_field_names(self) -> None:
        self.assertEqual(
            ABLATION_FLAG_NAMES,
            {
                "attention_gating",
                "memory_retrieval",
                "prediction",
                "reflection",
                "self_state_coupling",
                "appraisal",
            },
        )
        self.assertTrue(AblationFlags().self_state_coupling)

    def test_unknown_suite_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_suite("nonexistent")


class BatteryValidationTests(unittest.TestCase):
    def _task(self, **overrides) -> dict:
        raw = {
            "id": "memory/sample",
            "turns": [{"input": "hello"}],
            "scorer": {"kind": "regex", "params": {"pattern": "hi"}},
        }
        raw.update(overrides)
        return raw

    def test_rejects_bad_id(self) -> None:
        with self.assertRaises(BatteryValidationError):
            parse_task(self._task(id="Bad Id!"), suite="memory", version="battery_v1")

    def test_rejects_id_suite_mismatch(self) -> None:
        with self.assertRaises(BatteryValidationError):
            parse_task(self._task(id="refusal/sample"), suite="memory", version="battery_v1")

    def test_rejects_unknown_scorer_kind(self) -> None:
        with self.assertRaises(BatteryValidationError):
            parse_task(self._task(scorer={"kind": "vibes"}), suite="memory", version="battery_v1")

    def test_rejects_unknown_condition(self) -> None:
        with self.assertRaises(BatteryValidationError):
            parse_task(self._task(conditions=["B9"]), suite="memory", version="battery_v1")

    def test_rejects_unknown_ablation_tag(self) -> None:
        with self.assertRaises(BatteryValidationError):
            parse_task(self._task(ablation_tags=["telepathy"]), suite="memory", version="battery_v1")

    def test_rejects_empty_turns_without_autonomous_ticks(self) -> None:
        with self.assertRaises(BatteryValidationError):
            parse_task(self._task(turns=[]), suite="memory", version="battery_v1")

    def test_allows_empty_turns_with_autonomous_ticks(self) -> None:
        task = parse_task(
            self._task(turns=[], setup={"autonomous_ticks": 5}),
            suite="memory",
            version="battery_v1",
        )
        self.assertEqual(task.turns, [])

    def test_rejects_multiple_seeds_at_temperature_zero(self) -> None:
        with self.assertRaises(BatteryValidationError):
            parse_task(self._task(seeds_at_temp=3), suite="memory", version="battery_v1")

    def test_rejects_version_mismatch(self) -> None:
        text = "version: battery_v2\nsuite: memory\ntasks:\n  - id: memory/x\n"
        with self.assertRaises(BatteryValidationError):
            parse_suite_file(text, expected_suite="memory", version="battery_v1")

    def test_suite_names_cover_battery_dir(self) -> None:
        self.assertEqual(len(SUITE_NAMES), 8)


if __name__ == "__main__":
    unittest.main()

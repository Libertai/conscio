"""SelfState v2 liveness: every field moves from real signals (writer→reader
table in the SelfState docstring), instead of sitting at its default."""
from __future__ import annotations

import unittest

from conscio.core.cognition import SelfState
from conscio.core.prediction import PredictionEngine
from conscio.core.tool_loop import ToolRequest
from conscio.core.workspace import EntryType, Workspace


class SelfStateLivenessTests(unittest.TestCase):
    def test_uncertainty_moves_after_prediction_failures(self) -> None:
        state = SelfState()
        engine = PredictionEngine()
        workspace = Workspace()
        baseline = state.uncertainty

        # Calm tick: no failures, focused attention → uncertainty drops.
        state.update_tick(engine.error_ema, engine.failure_rate(), 0.9)
        calm = state.uncertainty
        self.assertLess(calm, baseline)

        # Three failing tool runs drive the EMA and failure rate up.
        for _ in range(3):
            expectation = engine.expect_tool(ToolRequest(name="bash", args={}), tick=1)
            conflict = engine.resolve_tool(
                expectation, {"output": "boom", "error": True}, workspace, tick=1
            )
            self.assertIsNotNone(conflict)
            self.assertEqual(conflict.type, EntryType.CONFLICT)
            self.assertFalse(conflict.resolved)

        state.update_tick(engine.error_ema, engine.failure_rate(), 0.9)
        self.assertGreater(state.uncertainty, calm)
        self.assertGreater(state.prediction_error, 0.0)

    def test_conflict_level_tracks_fresh_and_unresolved_conflicts(self) -> None:
        state = SelfState()

        state.update_tick(0.0, 0.0, 0.5, unresolved_conflicts=0, fresh_failures=0)
        self.assertEqual(state.conflict_level, 0.0)

        state.update_tick(0.0, 0.0, 0.5, unresolved_conflicts=1, fresh_failures=1)
        self.assertAlmostEqual(state.conflict_level, 0.75)

        state.update_tick(0.0, 0.0, 0.5, unresolved_conflicts=4, fresh_failures=2)
        self.assertEqual(state.conflict_level, 1.0)

    def test_cognitive_load_reflects_context_budget_fraction(self) -> None:
        state = SelfState()

        state.update_load(3000, 12000)
        self.assertAlmostEqual(state.cognitive_load, 0.25)

        state.update_load(24000, 12000)
        self.assertEqual(state.cognitive_load, 1.0)

        state.update_load(0, 12000)
        self.assertEqual(state.cognitive_load, 0.0)

    def test_limitations_append_after_repeated_tool_failures(self) -> None:
        state = SelfState()

        state.note_tool_failure("bash", "command not found")
        state.note_tool_failure("bash", "command not found")
        self.assertEqual(state.known_limitations, [])
        self.assertEqual(state.tool_failures, {"bash": 2})

        state.note_tool_failure("bash", "command not found")
        self.assertEqual(len(state.known_limitations), 1)
        self.assertIn("tool bash failing repeatedly", state.known_limitations[0])

        # Deduped: further failures of the same tool add no new limitation.
        state.note_tool_failure("bash", "command not found")
        self.assertEqual(len(state.known_limitations), 1)
        self.assertEqual(state.tool_failures["bash"], 4)

    def test_limitations_capped_at_eight(self) -> None:
        state = SelfState()
        for index in range(12):
            for _ in range(3):
                state.note_tool_failure(f"tool{index}", "broken")

        self.assertEqual(len(state.known_limitations), 8)
        # Oldest entries are dropped first.
        self.assertIn("tool tool11 failing repeatedly", state.known_limitations[-1])

    def test_to_dict_keeps_existing_keys_and_adds_tool_failures(self) -> None:
        state = SelfState()
        payload = state.to_dict()

        for key in (
            "active_goal",
            "uncertainty",
            "conflict_level",
            "cognitive_load",
            "current_strategy",
            "last_error",
            "attention_focus",
            "current_intention",
            "prediction_error",
            "known_limitations",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["tool_failures"], {})


if __name__ == "__main__":
    unittest.main()

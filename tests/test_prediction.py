from __future__ import annotations

import unittest

from conscio.core.constraints import ConstraintCheck, ConstraintReport, ConstraintValidator
from conscio.core.prediction import PredictionEngine
from conscio.core.tool_loop import ToolRequest
from conscio.core.workspace import EntryType, Workspace


class ExpectationLifecycleTests(unittest.TestCase):
    def test_expect_tool_registers_pending_before_execution(self) -> None:
        engine = PredictionEngine()
        exp = engine.expect_tool(ToolRequest(name="bash", args={"command": "ls"}), tick=1)

        self.assertEqual(exp.kind, "tool_succeeded")
        self.assertEqual(exp.args["tool"], "bash")
        self.assertEqual(exp.created_tick, 1)
        self.assertFalse(exp.resolved)
        self.assertIn(exp, engine.pending())

    def test_resolve_tool_success_against_result_dict(self) -> None:
        workspace = Workspace()
        engine = PredictionEngine()
        exp = engine.expect_tool(ToolRequest(name="bash", args={}), tick=1)

        entry = engine.resolve_tool(exp, {"output": "ok", "exit_code": 0}, workspace, tick=1)

        self.assertIsNone(entry)
        self.assertTrue(exp.resolved)
        self.assertTrue(exp.passed)
        self.assertEqual(engine.pending(), [])
        self.assertEqual(engine.failure_rate(), 0.0)
        self.assertEqual(engine.error_ema, 0.0)

    def test_resolve_tool_error_writes_unresolved_conflict(self) -> None:
        workspace = Workspace()
        engine = PredictionEngine()
        exp = engine.expect_tool(ToolRequest(name="bash", args={}), tick=2)

        entry = engine.resolve_tool(exp, {"output": "", "error": "boom"}, workspace, tick=2)

        self.assertIsNotNone(entry)
        self.assertEqual(entry.type, EntryType.CONFLICT)
        self.assertFalse(entry.resolved)
        self.assertEqual(entry.metadata["prediction_error"], 1.0)
        self.assertEqual(entry.metadata["expectation"], "tool_succeeded")
        self.assertFalse(exp.passed)
        self.assertEqual(engine.failure_rate(), 1.0)

    def test_resolve_tool_nonzero_exit_code_fails(self) -> None:
        workspace = Workspace()
        engine = PredictionEngine()
        exp = engine.expect_tool(ToolRequest(name="bash", args={}), tick=1)

        entry = engine.resolve_tool(exp, {"output": "denied", "exit_code": 2}, workspace, tick=1)

        self.assertIsNotNone(entry)
        self.assertFalse(exp.passed)

    def test_error_ema_moves_on_each_resolution(self) -> None:
        workspace = Workspace()
        engine = PredictionEngine(ema_alpha=0.35)

        failing = engine.expect_tool(ToolRequest(name="bash", args={}), tick=1)
        engine.resolve_tool(failing, {"error": "boom"}, workspace, tick=1)
        self.assertAlmostEqual(engine.error_ema, 0.35)

        passing = engine.expect_tool(ToolRequest(name="bash", args={}), tick=2)
        engine.resolve_tool(passing, {"output": "ok", "exit_code": 0}, workspace, tick=2)
        self.assertAlmostEqual(engine.error_ema, 0.35 * 0.65)
        self.assertEqual(engine.failure_rate(), 0.5)

    def test_reset_episode_clears_pending_and_counters_keeps_ema(self) -> None:
        workspace = Workspace()
        engine = PredictionEngine()
        failing = engine.expect_tool(ToolRequest(name="bash", args={}), tick=1)
        engine.resolve_tool(failing, {"error": "boom"}, workspace, tick=1)
        engine.expect_tool(ToolRequest(name="bash", args={}), tick=2)
        ema_before = engine.error_ema
        self.assertGreater(ema_before, 0.0)

        engine.reset_episode()

        self.assertEqual(engine.pending(), [])
        self.assertEqual(engine.failure_rate(), 0.0)
        self.assertEqual(engine.error_ema, ema_before)

    def test_double_resolution_is_a_no_op(self) -> None:
        workspace = Workspace()
        engine = PredictionEngine()
        exp = engine.expect_tool(ToolRequest(name="bash", args={}), tick=1)
        engine.resolve_tool(exp, {"error": "boom"}, workspace, tick=1)
        ema = engine.error_ema

        entry = engine.resolve_tool(exp, {"error": "boom"}, workspace, tick=1)

        self.assertIsNone(entry)
        self.assertEqual(engine.error_ema, ema)
        self.assertEqual(engine.failure_rate(), 1.0)

    def test_disabled_engine_is_inert(self) -> None:
        workspace = Workspace()
        engine = PredictionEngine(enabled=False)
        exp = engine.expect_tool(ToolRequest(name="bash", args={}), tick=1)

        entry = engine.resolve_tool(exp, {"error": "boom"}, workspace, tick=1)

        self.assertIsNone(entry)
        self.assertEqual(engine.pending(), [])
        self.assertEqual(engine.error_ema, 0.0)
        self.assertEqual(workspace.size, 0)


class AnswerExpectationTests(unittest.TestCase):
    def _report(self, *, passed: bool) -> ConstraintReport:
        return ConstraintReport(
            checks=[
                ConstraintCheck(
                    constraint_id="episode:1",
                    text="Answer in one word.",
                    kind="structural",
                    passed=passed,
                    detail="4 word(s), limit 1" if not passed else "1 word(s), limit 1",
                )
            ]
        )

    def test_expect_answer_carries_constraint_ids(self) -> None:
        validator = ConstraintValidator()
        constraints = validator.extract_episode_constraints("Answer in one word: what is 2+2?")
        engine = PredictionEngine()

        exp = engine.expect_answer(constraints=constraints, tick=1)

        self.assertEqual(exp.kind, "answer_satisfies_constraints")
        self.assertEqual(exp.args["constraints"], ["episode:1"])
        self.assertIn(exp, engine.pending())

    def test_resolve_answer_uses_constraint_report_not_output(self) -> None:
        workspace = Workspace()
        engine = PredictionEngine()
        exp = engine.expect_answer(constraints=[], tick=1)

        entry = engine.resolve_answer(exp, self._report(passed=True), workspace, tick=1)

        self.assertIsNone(entry)
        self.assertTrue(exp.passed)

    def test_resolve_answer_violation_writes_unresolved_conflict(self) -> None:
        workspace = Workspace()
        engine = PredictionEngine()
        exp = engine.expect_answer(constraints=[], tick=1)

        entry = engine.resolve_answer(exp, self._report(passed=False), workspace, tick=1)

        self.assertIsNotNone(entry)
        self.assertEqual(entry.type, EntryType.CONFLICT)
        self.assertFalse(entry.resolved)
        self.assertEqual(entry.metadata["expectation"], "answer_satisfies_constraints")
        self.assertIn("episode:1", entry.content)
        self.assertFalse(exp.passed)
        self.assertGreater(engine.error_ema, 0.0)


if __name__ == "__main__":
    unittest.main()

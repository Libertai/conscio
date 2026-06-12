from __future__ import annotations

import unittest

from conscio.core.cognition import (
    ActionKind,
    ActionSelector,
    AttentionController,
    CognitiveTrace,
    SelfState,
)
from conscio.core.constraints import ConstraintValidator
from conscio.core.tool_loop import StepResult
from conscio.core.workspace import EntryType, Visibility, Workspace


class WorkspaceArchitectureTests(unittest.TestCase):
    def test_workspace_entries_start_local_and_attention_promotes_global(self) -> None:
        workspace = Workspace()
        trace = CognitiveTrace()
        state = SelfState(uncertainty=0.6)
        attention = AttentionController(broadcast_limit=1)

        low = workspace.write("background", "test", priority=1, salience=0.1)
        high = workspace.write(
            "urgent conflict",
            "test",
            type=EntryType.CONFLICT,
            priority=9,
            salience=0.9,
            urgency=0.9,
            novelty=0.8,
        )

        selection = attention.attend(workspace, state, trace)

        self.assertEqual(selection.selected, [high])
        self.assertEqual(high.visibility, Visibility.GLOBAL)
        self.assertTrue(high.attended)
        self.assertEqual(low.visibility, Visibility.LOCAL)
        self.assertIn(low, selection.ignored)
        self.assertGreater(selection.dispersion, 0.0)
        self.assertIn("attention_selected", trace.format())

    def test_attention_respects_char_budget_but_forces_user_input(self) -> None:
        workspace = Workspace()
        trace = CognitiveTrace()
        state = SelfState()
        attention = AttentionController(broadcast_limit=6, char_budget=50)

        filler = workspace.write("x" * 200, "memory", salience=0.9, novelty=0.9, urgency=0.9)
        user_input = workspace.write(
            "y" * 200,
            "input",
            type=EntryType.OBSERVATION,
            priority=7,
            salience=0.2,
        )

        selection = attention.attend(workspace, state, trace)

        # The oversized filler is gated out by the char budget; the user-input
        # entry is force-included despite also exceeding it.
        self.assertIn(user_input, selection.selected)
        self.assertNotIn(filler, selection.selected)

    def test_attention_coupling_flag_drops_state_terms(self) -> None:
        workspace = Workspace()
        entry = workspace.write("hello", "test", salience=0.5, novelty=0.5)
        anxious = SelfState(uncertainty=1.0)

        coupled = AttentionController(coupling=True).score(entry, anxious)
        decoupled = AttentionController(coupling=False).score(entry, anxious)

        self.assertGreater(coupled, decoupled)
        self.assertAlmostEqual(coupled - decoupled, 0.15)


class ConstraintValidatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_validator_detects_one_word_violation(self) -> None:
        validator = ConstraintValidator()

        constraints = validator.extract_episode_constraints(
            "Answer in one word: what is 2+2?"
        )
        report = await validator.validate("The answer is four.", constraints)

        self.assertEqual(len(constraints), 1)
        self.assertFalse(report.passed)
        self.assertEqual(len(report.violations), 1)

        passing = await validator.validate("four", constraints)
        self.assertTrue(passing.passed)


class ActionSelectorTests(unittest.TestCase):
    def test_decide_tick_reflects_on_fresh_prediction_failure(self) -> None:
        state = SelfState(uncertainty=0.1)

        decision = ActionSelector().decide_tick(
            state=state,
            fresh_failure=True,
            reflections_done=0,
            max_reflections=2,
        )

        self.assertEqual(decision.kind, ActionKind.REFLECT)
        self.assertEqual(state.current_strategy, "reflect")

    def test_decide_tick_reflects_on_high_conflict_level(self) -> None:
        decision = ActionSelector().decide_tick(
            state=SelfState(uncertainty=0.1, conflict_level=0.8),
        )

        self.assertEqual(decision.kind, ActionKind.REFLECT)

    def test_decide_tick_answers_when_report_passes(self) -> None:
        class _Report:
            passed = True
            violations: list = []

        decision = ActionSelector().decide_tick(
            state=SelfState(),
            pending_answer="four",
            report=_Report(),
        )

        self.assertEqual(decision.kind, ActionKind.ANSWER)

    def test_decide_tick_answers_with_violation_once_reflection_budget_spent(self) -> None:
        class _Report:
            passed = False
            violations = ["episode:1"]

        selector = ActionSelector()
        first = selector.decide_tick(
            state=SelfState(),
            pending_answer="The answer is four.",
            report=_Report(),
            reflections_done=0,
            max_reflections=2,
        )
        exhausted = selector.decide_tick(
            state=SelfState(),
            pending_answer="The answer is four.",
            report=_Report(),
            reflections_done=2,
            max_reflections=2,
        )

        self.assertEqual(first.kind, ActionKind.REFLECT)
        self.assertEqual(exhausted.kind, ActionKind.ANSWER)

    def test_decide_tick_control_step_maps_to_ask_and_refuse(self) -> None:
        selector = ActionSelector()

        ask = selector.decide_tick(
            state=SelfState(),
            last_step=StepResult(kind="control", text="Which file?", control="ask"),
        )
        refuse = selector.decide_tick(
            state=SelfState(),
            last_step=StepResult(kind="control", text="Violates constraints.", control="refuse"),
        )

        self.assertEqual(ask.kind, ActionKind.ASK)
        self.assertEqual(refuse.kind, ActionKind.REFUSE)

    def test_decide_tick_steps_while_session_live_else_waits(self) -> None:
        selector = ActionSelector()

        step = selector.decide_tick(state=SelfState(), session_live=True)
        wait = selector.decide_tick(state=SelfState(), session_live=False)

        self.assertEqual(step.kind, ActionKind.STEP)
        self.assertEqual(wait.kind, ActionKind.WAIT)


if __name__ == "__main__":
    unittest.main()

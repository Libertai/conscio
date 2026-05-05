from __future__ import annotations

import unittest

from conscio.core.cognition import (
    ActionSelector,
    AttentionController,
    CognitiveTrace,
    ConflictMonitor,
    SelfState,
)
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

        selected = attention.attend(workspace, state, trace)

        self.assertEqual(selected, [high])
        self.assertEqual(high.visibility, Visibility.GLOBAL)
        self.assertTrue(high.attended)
        self.assertEqual(low.visibility, Visibility.LOCAL)
        self.assertIn("attention_selected", trace.format())

    def test_conflict_monitor_detects_one_word_plan_violation(self) -> None:
        workspace = Workspace()
        monitor = ConflictMonitor()

        conflicts = monitor.inspect_plan(
            "Answer in one word: what is 2+2?",
            "## Actions\n- tool: reason | args: The answer is four.",
            workspace,
        )

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].type, EntryType.CONFLICT)
        self.assertGreaterEqual(conflicts[0].urgency, 0.8)

    def test_action_selector_reflects_on_conflict(self) -> None:
        decision = ActionSelector().decide(
            SelfState(uncertainty=0.1, conflict_level=0.8),
            "HIGH",
            has_conflict=True,
        )

        self.assertEqual(decision.action, "reflect")


if __name__ == "__main__":
    unittest.main()

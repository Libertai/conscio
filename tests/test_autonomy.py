"""Tests for the autonomous-action path: self-management tools, LLM-driven goal review,
and end-to-end heartbeat through the AutonomousActionModule."""
from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from conscio.config import load_config
from conscio.service import ConscioService


def _tool_call_response(name: str, arguments: str, call_id: str = "call-1") -> dict:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


class _StubLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    async def chat_async(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return {"content": ""}


class AutonomyMetaToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._env_patch = unittest.mock.patch.dict(
            os.environ,
            {
                "LIBERTAI_BASE_URL": "",
                "LIBERTAI_API_KEY": "",
                "LIBERTAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self.tmp = tempfile.TemporaryDirectory()
        config_path = Path(self.tmp.name) / "config.toml"
        config_path.write_text(
            "[service]\n"
            f"home = \"{self.tmp.name}\"\n"
            "api_key = \"test-key\"\n"
            "autonomous = false\n",
            encoding="utf-8",
        )
        self.config = load_config(config_path)

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()
        self._env_patch.stop()

    async def test_set_task_status_marks_task_done(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            project = await service.autonomy.get_or_create_project("seed-1", "Test goal.")
            assert project is not None
            task = await service.autonomy.add_task(project.id, "First step.")

            stub = _StubLLM([
                _tool_call_response("set_task_status", f'{{"task_id": "{task.id}", "status": "done", "result": "did it"}}'),
                {"content": "Marked the task done."},
            ])
            service.runtime._autonomous_module.llm = stub
            await service.run_autonomous_tick()
            project_after = await service.get_project(project.id)
        finally:
            await service.stop()

        tasks_done = [t for t in project_after["tasks"] if t["status"] == "done"]
        self.assertTrue(any(t["id"] == task.id for t in tasks_done))

    async def test_propose_subgoal_creates_goal_with_self_proposed_source(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            stub = _StubLLM([
                _tool_call_response(
                    "propose_subgoal",
                    '{"description": "Research my own logging surface", "rationale": "I need observability."}',
                ),
                {"content": "Proposed."},
            ])
            service.runtime._autonomous_module.llm = stub
            await service.run_autonomous_tick()
            goals = await service.goals.list_goals()
        finally:
            await service.stop()

        proposed = [g for g in goals if g["source"] == "self_proposed"]
        self.assertEqual(len(proposed), 1)
        self.assertIn("logging surface", proposed[0]["description"])

    async def test_note_progress_records_episode_and_trace(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            stub = _StubLLM([
                _tool_call_response("note_progress", '{"content": "thinking aloud about the goal"}'),
                {"content": "Noted."},
            ])
            service.runtime._autonomous_module.llm = stub
            await service.run_autonomous_tick()
            trace = await service.recent_trace()
        finally:
            await service.stop()

        self.assertIn("thinking aloud", trace)


class GoalReviewWithLLMTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._env_patch = unittest.mock.patch.dict(
            os.environ,
            {
                "LIBERTAI_BASE_URL": "",
                "LIBERTAI_API_KEY": "",
                "LIBERTAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self.tmp = tempfile.TemporaryDirectory()
        config_path = Path(self.tmp.name) / "config.toml"
        config_path.write_text(
            "[service]\n"
            f"home = \"{self.tmp.name}\"\n"
            "api_key = \"test-key\"\n"
            "autonomous = false\n",
            encoding="utf-8",
        )
        self.config = load_config(config_path)

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()
        self._env_patch.stop()

    async def test_review_with_llm_retires_and_reprioritizes(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            goals_before = await service.goals.list_goals(status="active")
            target_retire = goals_before[-1]["id"]
            target_reprioritize = goals_before[0]["id"]

            decisions_payload = (
                '['
                f'{{"goal_id": "{target_retire}", "action": "retire", "reason": "Stale."}},'
                f'{{"goal_id": "{target_reprioritize}", "action": "reprioritize", "new_priority": 0.95, "reason": "Top focus."}}'
                ']'
            )
            stub = _StubLLM([{"content": decisions_payload}])
            applied = await service.goals.review_with_llm(
                stub,
                recent_episodes=[],
                recent_influences=[],
            )
            after = {g["id"]: g for g in await service.goals.list_goals()}
        finally:
            await service.stop()

        self.assertEqual(len(applied), 2)
        self.assertEqual(after[target_retire]["status"], "retired")
        self.assertAlmostEqual(after[target_reprioritize]["priority"], 0.95, places=2)

    async def test_review_with_llm_ignores_invalid_goal_ids(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            stub = _StubLLM([{"content": '[{"goal_id": "does-not-exist", "action": "retire", "reason": "x"}]'}])
            applied = await service.goals.review_with_llm(
                stub,
                recent_episodes=[],
                recent_influences=[],
            )
        finally:
            await service.stop()

        self.assertEqual(applied, [])

    async def test_review_with_llm_parse_miss_records_fact_and_logs(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            stub = _StubLLM([{"content": "I don't think any goals need updating right now."}])
            with self.assertLogs("conscio.goals", level="WARNING") as captured:
                applied = await service.goals.review_with_llm(
                    stub,
                    recent_episodes=[],
                    recent_influences=[],
                )
            facts = service.memory.fetchall(
                "SELECT fact FROM facts WHERE origin = 'goal_review' ORDER BY id DESC LIMIT 5"
            )
        finally:
            await service.stop()

        self.assertEqual(applied, [])
        self.assertTrue(any("parse miss" in rec for rec in captured.output))
        self.assertTrue(any("parse miss" in f["fact"] for f in facts))

    async def test_plan_and_act_records_review_attempt(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            service._goal_review_interval = 1
            stub = _StubLLM([
                {"content": "Thinking about the goal."},
                {"content": "no decisions today"},
            ])
            service.runtime._autonomous_module.llm = stub
            await service.run_autonomous_tick()
            rows = service.memory.fetchall(
                "SELECT kind, COUNT(*) AS n FROM action_events "
                "WHERE kind LIKE 'goal_review_%' GROUP BY kind"
            )
        finally:
            await service.stop()

        kinds = {r["kind"]: r["n"] for r in rows}
        self.assertEqual(kinds.get("goal_review_attempt"), 1)
        self.assertEqual(kinds.get("goal_review_empty"), 1)
        self.assertNotIn("goal_review_error", kinds)


class PerToolSchemaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._env_patch = unittest.mock.patch.dict(
            os.environ,
            {
                "LIBERTAI_BASE_URL": "",
                "LIBERTAI_API_KEY": "",
                "LIBERTAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self.tmp = tempfile.TemporaryDirectory()
        config_path = Path(self.tmp.name) / "config.toml"
        config_path.write_text(
            "[service]\n"
            f"home = \"{self.tmp.name}\"\n"
            "api_key = \"test-key\"\n"
            "autonomous = false\n",
            encoding="utf-8",
        )
        self.config = load_config(config_path)

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()
        self._env_patch.stop()

    async def test_builtin_tools_advertise_per_tool_parameters(self) -> None:
        service = ConscioService(self.config)
        await service.start(background=False)
        try:
            schemas = service.runtime.tools.tool_schemas()
        finally:
            await service.stop()

        self.assertIn("bash", schemas)
        bash_schema = schemas["bash"]
        self.assertEqual(bash_schema.get("type"), "object")
        self.assertIn("command", bash_schema.get("properties", {}))
        self.assertEqual(bash_schema.get("additionalProperties"), False)

        # Self-management tools also have schemas via the new register(schema=...) plumbing.
        self.assertIn("set_task_status", schemas)
        self.assertEqual(
            set(schemas["set_task_status"]["required"]),
            {"task_id", "status"},
        )


if __name__ == "__main__":
    unittest.main()

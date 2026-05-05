from __future__ import annotations

import os
import tempfile
import unittest

from conscio.core.context import ContextSettings, PromptAssembler
from conscio.core.cognition import InputEvent
from conscio.core.workspace import EntryType, Visibility
from conscio.core.runtime import CognitiveRuntime
from conscio.eval import run_eval_suite
from conscio.memory.store import MemoryStore


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append(messages)
        return {"content": "fake response"}


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_assembler_keeps_stable_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(db_path=os.path.join(tmp, "prompt.db"))
            await memory.initialize()
            await memory.create_session("s1")
            assembler = PromptAssembler(ContextSettings(max_dynamic_chars=2000))
            workspace = CognitiveRuntime(memory=memory, session_id="s1").workspace
            first = await assembler.assemble(
                user_input="hello",
                workspace=workspace,
                memory=memory,
                session_id="s1",
                state={"active_goal": {"description": "Preserve continuity"}},
            )
            second = await assembler.assemble(
                user_input="what now?",
                workspace=workspace,
                memory=memory,
                session_id="s1",
                state={"active_goal": {"description": "Preserve continuity"}},
            )
            await memory.close()

        self.assertEqual(first.messages[0], second.messages[0])
        self.assertIn("USER_INPUT\nhello", first.dynamic_context)
        self.assertIn("USER_INPUT\nwhat now?", second.dynamic_context)

    async def test_llm_response_uses_assembled_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = FakeLLM()
            runtime = CognitiveRuntime(
                llm=fake,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "llm.db")),
                context_settings=ContextSettings(max_dynamic_chars=2000),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="hello", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(result.output, "fake response")
        self.assertIn("USER_INPUT\nhello", result.model_context)
        self.assertEqual(fake.calls[0][0], fake.calls[-1][0])

    async def test_evented_episode_returns_trace_and_attention_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "runtime.db")),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(
                    InputEvent(content="Answer in one word: what is 2+2?")
                )
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertIn("four", result.output.lower())
        self.assertIn("attention_selected", result.cognitive_trace)
        self.assertIn("focus", result.attention_schema)
        self.assertGreaterEqual(result.metrics.ticks, 1)

    async def test_daemon_dry_run_uses_same_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "daemon.db")),
            )
            await runtime.initialize()
            try:
                results = await runtime.run_daemon(
                    [InputEvent(content="Daemon dry-run heartbeat", source="daemon")],
                    dry_run=True,
                )
            finally:
                await runtime.close()

        self.assertEqual(len(results), 1)
        self.assertIn(results[0].selected_action, {"answer", "wait"})

    async def test_response_module_resets_between_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "reset.db")),
            )
            await runtime.initialize()
            try:
                first = await runtime.run_episode(InputEvent(content="First message", source="user"))
                second = await runtime.run_episode(InputEvent(content="Second message", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(first.selected_action, "answer")
        self.assertEqual(second.selected_action, "answer")
        self.assertIn("Second message", second.output)

    async def test_response_prefers_current_user_input_over_autonomous_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "current-user.db")),
            )
            await runtime.initialize()
            try:
                await runtime.run_episode(
                    InputEvent(
                        content="Autonomous heartbeat: current task is reflection.",
                        source="autonomous",
                        event_type="heartbeat",
                    )
                )
                result = await runtime.run_episode(InputEvent(content="Second user message", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertIn("Second user message", result.output)

    async def test_stale_conflict_does_not_override_next_user_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "stale-conflict.db")),
            )
            await runtime.initialize()
            try:
                await runtime.run_episode(InputEvent(content="Search for latest status", source="user"))
                result = await runtime.run_episode(InputEvent(content="Answer this normally", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertIn("Answer this normally", result.output)

    async def test_current_answer_survives_stale_attention_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "attention-pressure.db")),
            )
            await runtime.initialize()
            try:
                for idx in range(12):
                    runtime.workspace.write(
                        f"stale high-priority conflict {idx}",
                        source="test",
                        type=EntryType.CONFLICT,
                        priority=10,
                        salience=1.0,
                        urgency=1.0,
                        visibility=Visibility.GLOBAL,
                    )
                result = await runtime.run_episode(InputEvent(content="hello!", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertIn("hello!", result.output)

    async def test_autonomous_heartbeat_current_context_does_not_trigger_web_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "autonomous.db")),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(
                    InputEvent(
                        content=(
                            "Autonomous heartbeat: active goal is 'Preserve continuity'. "
                            "Current project is 'Autonomous pursuit'. Current task is "
                            "'Reflect on the active goal and choose a concrete next step'."
                        ),
                        source="autonomous",
                        event_type="heartbeat",
                    )
                )
            finally:
                await runtime.close()

        self.assertNotEqual(result.selected_action, "tool")
        self.assertEqual(result.metrics.tool_calls, 0)

    async def test_memory_consolidation_creates_facts_and_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "memory.db")),
                context_settings=ContextSettings(compaction_interval=2),
            )
            await runtime.initialize()
            try:
                await runtime.run_episode(InputEvent(content="Remember that I prefer concise answers.", source="user"))
                await runtime.run_episode(InputEvent(content="hello!", source="user"))
                facts = await runtime.memory.recent_facts()
                skills = await runtime.memory.list_skills()
            finally:
                await runtime.close()

        self.assertTrue(any("prefer concise answers" in fact["fact"] for fact in facts))
        self.assertTrue(any(skill["skill"].startswith("answer_") for skill in skills))

    async def test_user_current_information_request_can_trigger_web_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "current.db")),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(
                    InputEvent(content="Search for the latest project status.", source="user")
                )
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "tool")
        self.assertEqual(result.metrics.tool_calls, 1)


class EvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_eval_runs(self) -> None:
        rows = await run_eval_suite("smoke")

        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(row.mode == "evented_full" for row in rows))


if __name__ == "__main__":
    unittest.main()

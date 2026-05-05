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
from conscio.tools import ToolRegistry


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        return {"content": "fake response"}


class EmptyLLM:
    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        return {"content": ""}


class ToolCallLLM:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments
        self.kwargs: list[dict] = []

    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        self.kwargs.append(kwargs)
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": self.name, "arguments": self.arguments},
                }
            ],
        }


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

    async def test_empty_llm_response_gets_visible_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=EmptyLLM(),  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "empty-llm.db")),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="hello", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertIn("empty response", result.output)

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

    async def test_internal_tool_result_does_not_become_chat_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "tool-result.db")),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(
                    InputEvent(
                        content="Autonomous tool result for task abc: (no output)",
                        source="tool",
                        event_type="tool_result",
                    )
                )
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "wait")
        self.assertEqual(result.output, "Internal observation recorded; no user-facing response needed.")
        self.assertNotIn("I treated this as a cognitive episode", result.output)

    async def test_autonomous_heartbeat_does_not_become_chat_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "heartbeat-internal.db")),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(
                    InputEvent(
                        content="Autonomous heartbeat: active goal is preserve continuity.",
                        source="autonomous",
                        event_type="heartbeat",
                    )
                )
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "wait")
        self.assertEqual(result.output, "Internal observation recorded; no user-facing response needed.")

    async def test_memory_consolidation_creates_skills(self) -> None:
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
                skills = await runtime.memory.list_skills()
            finally:
                await runtime.close()

        self.assertTrue(any(skill["skill"].startswith("answer_") for skill in skills))

    async def test_semantic_fact_reindex_does_not_duplicate_fts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(db_path=os.path.join(tmp, "dedupe.db"))
            await memory.initialize()
            try:
                await memory.add_fact("Agent self/context: You have a memory tool.", source="user")
                await memory.add_fact("Agent self/context: You have a memory tool.", source="user")
                rows = await memory.search("memory tool", limit=10)
            finally:
                await memory.close()

        semantic_rows = [row for row in rows if row["memory_type"] == "semantic"]
        self.assertEqual(len(semantic_rows), 1)

    async def test_llm_can_choose_memory_tool_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(db_path=os.path.join(tmp, "memory-tool.db"))
            tools = ToolRegistry()
            llm = ToolCallLLM(
                "remember_facts",
                '{"facts": ["Agent self/context: You run in a VM"], "source": "user"}',
            )

            async def remember_facts(facts: list[str], source: str = "user") -> dict:
                for fact in facts:
                    await memory.add_fact(fact, source=source, confidence="HIGH")
                return {"output": f"Stored {len(facts)} fact(s) in semantic memory.", "error": False}

            tools.register("remember_facts", remember_facts, "Store facts.")
            runtime = CognitiveRuntime(llm=llm, memory=memory, tools=tools)  # type: ignore[arg-type]
            await runtime.initialize()
            try:
                result = await runtime.run_episode(
                    InputEvent(content="Remember: you run in a VM.", source="user")
                )
                facts = await memory.recent_facts(10)
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "tool")
        self.assertIn("Stored 1 fact", result.output)
        self.assertIn("tools", llm.kwargs[0])
        fact_text = "\n".join(row["fact"] for row in facts)
        self.assertIn("You run in a VM", fact_text)

    async def test_llm_can_choose_bash_tool_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry()
            llm = ToolCallLLM("bash", '{"input": "whoami && pwd"}')

            async def bash(input: str) -> dict:
                self.assertEqual(input, "whoami && pwd")
                return {"output": "conscio\n/opt/conscio/work", "exit_code": 0}

            tools.register("bash", bash, "Execute shell commands.")
            runtime = CognitiveRuntime(
                llm=llm,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "bash-tool.db")),
                tools=tools,
            ) 
            await runtime.initialize()
            try:
                result = await runtime.run_episode(
                    InputEvent(content="Poke around in your terminal.", source="user")
                )
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "tool")
        self.assertIn("/opt/conscio/work", result.output)
        self.assertEqual(result.metrics.tool_calls, 1)


class EvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_eval_runs(self) -> None:
        rows = await run_eval_suite("smoke")

        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(row.mode == "evented_full" for row in rows))


if __name__ == "__main__":
    unittest.main()

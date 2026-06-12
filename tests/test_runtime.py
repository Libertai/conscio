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


class IterativeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        return {"content": "done"}


def tool_call(name: str, arguments: str, call_id: str = "call-1") -> dict:
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
        # v2: an empty final is a recorded prediction failure plus a
        # WAIT-with-fallback after one retry — never a masked "answer".
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

        self.assertNotEqual(result.selected_action, "answer")
        self.assertGreaterEqual(result.metrics.prediction_errors, 1)
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

    async def test_autonomous_heartbeat_when_llm_is_offline(self) -> None:
        # With no LLM configured, autonomous heartbeats still cleanly resolve
        # to WAIT — but with a clearer trace note so offline mode is diagnosable.
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "heartbeat-offline.db")),
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
        self.assertIn("no LLM is configured", result.output)
        self.assertEqual(result.metrics.tool_calls, 0)

    async def test_autonomous_heartbeat_invokes_registered_tool_with_llm(self) -> None:
        # When an LLM is configured, autonomous heartbeats call exactly the tool
        # the LLM picks — replacing the old hardcoded heartbeat path.
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry()
            llm = IterativeLLM([
                tool_call("note_progress", '{"content": "checked in: still working on the goal"}'),
                {"content": "Logged a progress note."},
            ])
            calls: list[dict] = []

            async def note_progress(content: str = "") -> dict:
                calls.append({"content": content})
                return {"output": "Progress note recorded.", "error": False}

            tools.register("note_progress", note_progress, "Record a progress note.")
            runtime = CognitiveRuntime(
                llm=llm,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "auto-tool.db")),
                tools=tools,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(
                    InputEvent(
                        content="Autonomous heartbeat: pursue the active goal.",
                        source="autonomous",
                        event_type="heartbeat",
                    )
                )
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertEqual(len(calls), 1)
        self.assertIn("checked in", calls[0]["content"])
        self.assertEqual(result.metrics.tool_calls, 1)
        self.assertIn("Tool note_progress returned", result.workspace_trace)

    async def test_user_message_after_autonomous_heartbeat_keeps_chat_clean(self) -> None:
        # Regression: a heartbeat that triggered tool calls must not leak its
        # output into the answer to a subsequent user message.
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry()
            llm = IterativeLLM([
                tool_call("note_progress", '{"content": "autonomous step"}'),
                {"content": "Heartbeat complete."},
                # Subsequent user-chat episode receives no LLM stub responses;
                # ResponseModule falls back to the deterministic offline path.
            ])

            async def note_progress(content: str = "") -> dict:
                return {"output": "ok", "error": False}

            tools.register("note_progress", note_progress, "Record a progress note.")
            runtime = CognitiveRuntime(
                llm=None,  # user-chat path uses deterministic offline answers
                memory=MemoryStore(db_path=os.path.join(tmp, "bleed.db")),
                tools=tools,
            )
            # Inject the stub LLM only on the autonomous module so the user
            # path stays deterministic and easy to assert on.
            runtime._autonomous_module.llm = llm
            await runtime.initialize()
            try:
                heartbeat = await runtime.run_episode(
                    InputEvent(
                        content="Autonomous heartbeat",
                        source="autonomous",
                        event_type="heartbeat",
                    )
                )
                user = await runtime.run_episode(
                    InputEvent(content="hello!", source="user")
                )
            finally:
                await runtime.close()

        self.assertEqual(heartbeat.selected_action, "answer")
        self.assertEqual(user.selected_action, "answer")
        self.assertNotIn("Heartbeat complete", user.output)
        self.assertNotIn("autonomous step", user.output)

    async def test_memory_consolidation_creates_no_junk_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = CognitiveRuntime(
                llm=None,
                memory=MemoryStore(db_path=os.path.join(tmp, "memory.db")),
                context_settings=ContextSettings(compaction_interval=2),
            )
            await runtime.initialize()
            try:
                first = await runtime.run_episode(
                    InputEvent(content="Remember that I prefer concise answers.", source="user")
                )
                second = await runtime.run_episode(InputEvent(content="hello!", source="user"))
                procedures = await runtime.memory.list_procedures()
                episodes = await runtime.memory.recent_episodes(10)
                # Deliberate procedure writes still work (learn_procedure path).
                await runtime.memory.upsert_procedure(
                    "triage-logs",
                    "Check the service logs for recent errors.",
                    "1. journalctl -u conscio\n2. grep ERROR",
                )
                deliberate = await runtime.memory.list_procedures()
            finally:
                await runtime.close()

        # No select_/answer_ junk skills are written by consolidation.
        self.assertEqual(procedures, [])
        # Each episode produces exactly one unified episodes row.
        self.assertEqual(len(episodes), 2)
        self.assertEqual(
            {first.episode_id, second.episode_id}, {ep["id"] for ep in episodes}
        )
        self.assertTrue(all(ep["summary"] for ep in episodes))
        self.assertEqual([p["name"] for p in deliberate], ["triage-logs"])

    async def test_semantic_fact_reindex_does_not_duplicate_fts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(db_path=os.path.join(tmp, "dedupe.db"))
            await memory.initialize()
            try:
                await memory.add_fact("Agent self/context: You have a memory tool.", source="user")
                await memory.add_fact("Agent self/context: You have a memory tool.", source="user")
                rows = await memory.search("memory tool", limit=10)
                fact_rows_db = memory.fetchall("SELECT id FROM facts")
            finally:
                await memory.close()

        # norm_hash dedup keeps a single facts row and a single FTS mirror row.
        fact_rows = [row for row in rows if row["memory_type"] == "fact"]
        self.assertEqual(len(fact_rows), 1)
        self.assertEqual(len(fact_rows_db), 1)

    async def test_llm_tool_result_feeds_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry()
            llm = IterativeLLM([
                tool_call("bash", '{"input": "whoami && pwd"}'),
                {"content": "I am conscio in /opt/conscio/work."},
            ])

            async def bash(input: str) -> dict:
                self.assertEqual(input, "whoami && pwd")
                return {"output": "conscio\n/opt/conscio/work", "exit_code": 0}

            tools.register("bash", bash, "Execute shell commands.")
            runtime = CognitiveRuntime(
                llm=llm,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "iterative.db")),
                tools=tools,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="Poke around.", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertIn("conscio", result.output)
        self.assertEqual(result.metrics.tool_calls, 1)
        self.assertEqual(result.tool_results[0]["output"], "conscio\n/opt/conscio/work")
        self.assertIn("Tool bash returned", result.workspace_trace)
        self.assertEqual(len(llm.calls), 2)
        self.assertTrue(any(message.get("role") == "tool" for message in llm.calls[1]))

    async def test_llm_can_chain_memory_tool_then_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(db_path=os.path.join(tmp, "memory-tool.db"))
            tools = ToolRegistry()
            llm = IterativeLLM([
                tool_call("remember_facts", '{"facts": ["Agent self/context: You run in a VM"], "source": "user"}'),
                {"content": "Stored that I run in a VM."},
            ])

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

        self.assertEqual(result.selected_action, "answer")
        self.assertIn("Stored that", result.output)
        self.assertIn("tools", llm.kwargs[0])
        fact_text = "\n".join(row["fact"] for row in facts)
        self.assertIn("You run in a VM", fact_text)

    async def test_two_subsequent_chat_turns_keep_working(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry()
            llm = IterativeLLM([
                {"content": "First answer."},
                tool_call("remember_facts", '{"facts": ["User preference: concise"], "source": "user"}'),
                {"content": "Second answer after storing memory."},
            ])

            async def remember_facts(facts: list[str], source: str = "user") -> dict:
                return {"output": f"stored {len(facts)}", "error": False}

            tools.register("remember_facts", remember_facts, "Store facts.")
            runtime = CognitiveRuntime(
                llm=llm,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "turns.db")),
                tools=tools,
            )
            await runtime.initialize()
            try:
                first = await runtime.run_episode(InputEvent(content="hello", source="user"))
                second = await runtime.run_episode(InputEvent(content="remember concise", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(first.output, "First answer.")
        self.assertEqual(second.output, "Second answer after storing memory.")
        self.assertEqual(second.metrics.tool_calls, 1)

    async def test_tool_loop_forces_final_answer_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry()
            llm = IterativeLLM([
                tool_call("bash", '{"input": "pwd"}', "call-1"),
                tool_call("bash", '{"input": "whoami"}', "call-2"),
                tool_call("bash", '{"input": "hostname"}', "call-3"),
                tool_call("bash", '{"input": "id"}', "call-4"),
                {"content": "Final summary from observed tools."},
            ])

            async def bash(input: str) -> dict:
                return {"output": f"ran {input}", "exit_code": 0}

            tools.register("bash", bash, "Execute shell commands.")
            runtime = CognitiveRuntime(
                llm=llm,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "limit.db")),
                tools=tools,
                max_tool_rounds=4,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="Use tools then summarize.", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "answer")
        self.assertEqual(result.output, "Final summary from observed tools.")
        self.assertEqual(result.metrics.tool_calls, 4)
        self.assertNotIn("tool limit", result.output.lower())


class DsmlToolCallParserTests(unittest.IsolatedAsyncioTestCase):
    """DeepSeek-style native tool-call markers leaking into assistant content must
    still execute the intended tool call."""

    def _import_parser(self):
        from conscio.core.tool_loop import _parse_dsml_tool_call

        return _parse_dsml_tool_call

    def test_dsml_fenced_json_format(self) -> None:
        parse = self._import_parser()
        content = (
            "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>search_memory\n"
            "```json\n{\"query\": \"vitamin b1\"}\n```"
            "<｜tool_call_end｜><｜tool_calls_end｜>"
        )
        request = parse(content)
        self.assertIsNotNone(request)
        self.assertEqual(request.name, "search_memory")
        self.assertEqual(request.args, {"query": "vitamin b1"})

    def test_dsml_bare_brace_format(self) -> None:
        parse = self._import_parser()
        content = (
            "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>add_task"
            '{"description": "read article", "priority": 0.6}'
            "<｜tool_call_end｜><｜tool_calls_end｜>"
        )
        request = parse(content)
        self.assertIsNotNone(request)
        self.assertEqual(request.name, "add_task")
        self.assertEqual(request.args, {"description": "read article", "priority": 0.6})

    def test_dsml_truncated_close(self) -> None:
        parse = self._import_parser()
        content = (
            "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>note_progress\n"
            '{"note": "finished reading"}'
        )
        request = parse(content)
        self.assertIsNotNone(request)
        self.assertEqual(request.name, "note_progress")
        self.assertEqual(request.args, {"note": "finished reading"})

    def test_dsml_missing_function_name_returns_none(self) -> None:
        # Model emits the separator with `{` immediately after — no name text.
        # Must return None instead of crashing on splitlines()[0].
        parse = self._import_parser()
        content = (
            "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
            '{"x": 1}<｜tool_call_end｜><｜tool_calls_end｜>'
        )
        self.assertIsNone(parse(content))

    def test_dsml_ignores_non_dsml_content(self) -> None:
        parse = self._import_parser()
        self.assertIsNone(parse("Hello, world."))
        self.assertIsNone(parse(""))
        self.assertIsNone(parse("I considered using <｜some marker｜> but decided not to."))

    async def test_tool_loop_executes_dsml_leaked_call(self) -> None:
        from conscio.core.tool_loop import ToolLoop
        from conscio.core.workspace import Workspace

        leaked = (
            "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>echo\n"
            "```json\n{\"value\": \"hi\"}\n```"
            "<｜tool_call_end｜><｜tool_calls_end｜>"
        )
        llm = IterativeLLM([
            {"content": leaked, "tool_calls": []},
            {"content": "done.", "tool_calls": []},
        ])
        tools = ToolRegistry()
        observed: list[dict] = []

        async def echo(value: str) -> dict:
            observed.append({"value": value})
            return {"output": f"echo:{value}", "exit_code": 0}

        tools.register("echo", echo, "Echo a value.")
        loop = ToolLoop(llm=llm, tools=tools, max_rounds=3)
        workspace = Workspace()
        result = await loop.run(
            [{"role": "user", "content": "echo hi"}],
            workspace,
            tool_schemas=[{"type": "function", "function": {"name": "echo", "parameters": {}}}],
        )
        self.assertEqual(observed, [{"value": "hi"}])
        self.assertEqual(result.final_text, "done.")
        self.assertEqual(len(result.tool_requests), 1)
        self.assertEqual(result.tool_requests[0].name, "echo")


class ParallelToolCallTests(unittest.IsolatedAsyncioTestCase):
    """When the model emits N parallel tool calls in one turn, all N must
    execute and the transcript must stay protocol-valid: the echoed assistant
    message lists exactly the executed calls, each with a matching role:tool
    response (N tool_calls + 1 tool message is rejected with a 400 by
    OpenAI-compatible backends)."""

    async def test_parallel_tool_calls_all_execute_with_matching_responses(self) -> None:
        from conscio.core.tool_loop import ToolLoop
        from conscio.core.workspace import Workspace

        parallel = {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-a",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"value": "one"}'},
                },
                {
                    "id": "call-b",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"value": "two"}'},
                },
            ],
        }
        llm = IterativeLLM([parallel, {"content": "done."}])
        tools = ToolRegistry()
        observed: list[dict] = []

        async def echo(value: str) -> dict:
            observed.append({"value": value})
            return {"output": f"echo:{value}", "exit_code": 0}

        tools.register("echo", echo, "Echo a value.")
        loop = ToolLoop(llm=llm, tools=tools, max_rounds=3)
        messages = [{"role": "user", "content": "echo twice"}]
        result = await loop.run(
            messages,
            Workspace(),
            tool_schemas=[{"type": "function", "function": {"name": "echo", "parameters": {}}}],
        )

        # Both calls executed, in order.
        self.assertEqual(observed, [{"value": "one"}, {"value": "two"}])
        self.assertEqual([r.name for r in result.tool_requests], ["echo", "echo"])
        self.assertEqual(result.final_text, "done.")
        # Protocol validity: assistant tool_calls ids == following tool message ids.
        assistant = next(m for m in messages if m.get("tool_calls"))
        echoed_ids = [c["id"] for c in assistant["tool_calls"]]
        tool_ids = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
        self.assertEqual(echoed_ids, ["call-a", "call-b"])
        self.assertEqual(tool_ids, ["call-a", "call-b"])

    async def test_unparseable_call_is_dropped_from_assistant_echo(self) -> None:
        from conscio.core.tool_loop import ToolLoop

        response = {
            "content": "",
            "tool_calls": [
                {"id": "call-x", "type": "function", "function": {"name": "", "arguments": "{}"}},
                {
                    "id": "call-y",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"value": "hi"}'},
                },
            ],
        }
        requests = ToolLoop._tool_requests(response)
        self.assertEqual([(cid, req.name) for cid, req in requests], [("call-y", "echo")])
        message = ToolLoop._assistant_tool_call_message(response, requests)
        self.assertEqual([c["id"] for c in message["tool_calls"]], ["call-y"])


class EvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_eval_runs(self) -> None:
        rows = await run_eval_suite("smoke")

        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(row.mode == "evented_full" for row in rows))


if __name__ == "__main__":
    unittest.main()

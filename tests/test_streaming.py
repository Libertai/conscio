from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

from conscio.core.cognition import InputEvent
from conscio.core.context import ContextSettings
from conscio.core.runtime import CognitiveRuntime
from conscio.core.tool_loop import ToolLoopSession
from conscio.core.workspace import Workspace
from conscio.memory.store import MemoryStore
from conscio.tools import ToolRegistry


class StreamingStubLLM:
    """Scripted responses exposed via BOTH chat_async and chat_stream."""

    def __init__(self, responses: list[dict[str, Any]], chunk_size: int = 7) -> None:
        self.responses = list(responses)
        self.chunk_size = chunk_size
        self.chat_calls = 0
        self.stream_calls = 0

    async def chat_async(self, messages, **kwargs):
        self.chat_calls += 1
        return dict(self.responses.pop(0))

    async def chat_stream(self, messages, **kwargs):
        self.stream_calls += 1
        response = dict(self.responses.pop(0))
        content = str(response.get("content") or "")
        for i in range(0, len(content), self.chunk_size):
            yield {"type": "content", "text": content[i : i + self.chunk_size]}
        yield {"type": "done", "content": content, "tool_calls": response.get("tool_calls"), "role": "assistant"}


class ChatOnlyStubLLM:
    """chat_async-only stub — no chat_stream attribute. Streaming must be a no-op."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.chat_calls = 0

    async def chat_async(self, messages, **kwargs):
        self.chat_calls += 1
        return dict(self.responses.pop(0))


def _echo_registry() -> ToolRegistry:
    tools = ToolRegistry()

    async def echo_tool(input: str = "") -> dict[str, Any]:
        return {"output": "ok", "error": False}

    tools.register("echo_tool", echo_tool, "Echo a value.")
    return tools


def _echo_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "echo_tool",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        }
    ]


def _tool_call(name: str, arguments: str, call_id: str = "call-1") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


FINAL_SCRIPT = "Hello streaming world, this is a longer sentence to cross the sniff window."
DSML_LEAK = (
    # DeepSeek role-token form: the recovery parser (``_parse_dsml_tool_call``)
    # expects the tool name AFTER ``<｜tool_sep｜>`` (the leaked "function"
    # role marker sits before it); ``echo_tool`` before the sep would not
    # recover. The DSML gate must withhold this entire leak from the stream.
    "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>echo_tool\n"
    '{"input": "hi"}<｜tool_call_end｜><｜tool_calls_end｜>'
)


class StreamGateUnitTests(unittest.TestCase):
    def test_short_clean_content_flushes_on_finish(self) -> None:
        from conscio.core.tool_loop import _StreamGate

        gate = _StreamGate()
        self.assertEqual(gate.feed("short clean answer"), "")
        self.assertEqual(gate.finish(), "short clean answer")

    def test_leak_within_first_window_streams_nothing(self) -> None:
        from conscio.core.tool_loop import _StreamGate

        gate = _StreamGate()
        # The leak starts within the first 64 chars; finish must withhold all.
        self.assertEqual(gate.feed(DSML_LEAK[:40]), "")
        self.assertEqual(gate.feed(DSML_LEAK[40:]), "")
        self.assertEqual(gate.finish(), "")
        self.assertEqual(gate.emitted_chars, 0)

    def test_clean_prefix_then_marker_withholds_after_prefix(self) -> None:
        from conscio.core.tool_loop import _StreamGate

        gate = _StreamGate()
        prefix = "x" * 80  # past the sniff window — drains as it feeds
        first = gate.feed(prefix)
        self.assertEqual(first, prefix)
        # A marker opener mid-stream stops emission at that point.
        rest = gate.feed("<｜tool_calls_begin｜>leaked")
        self.assertEqual(rest, "")
        self.assertEqual(gate.finish(), "")
        self.assertEqual(gate.emitted_chars, len(prefix))

    def test_trailing_open_carried_across_feed_calls(self) -> None:
        from conscio.core.tool_loop import _StreamGate

        gate = _StreamGate()
        # 70 clean chars first so the sniff window has decided and we are past
        # the buffering phase (the trailing-`<` carry only matters past 64).
        emitted = gate.feed("x" * 70)
        self.assertEqual(emitted, "x" * 70)
        # "abc<" — the trailing "<" is withheld pending the next chunk.
        emitted += gate.feed("abc<")
        self.assertEqual(emitted, "x" * 70 + "abc")
        # "｜tool" completes the opener ("<｜") → the carried "<" is now part of
        # a marker opener, so nothing more is ever emitted.
        emitted += gate.feed("｜tool")
        self.assertEqual(emitted, "x" * 70 + "abc")
        emitted += gate.finish()
        self.assertEqual(emitted, "x" * 70 + "abc")

    def test_ascii_pipe_marker_variant_also_gated(self) -> None:
        from conscio.core.tool_loop import _StreamGate

        gate = _StreamGate()
        ascii_leak = "<|tool_calls_begin|>echo_tool<|tool_sep|>{\"input\": \"hi\"}<|tool_call_end|><|tool_calls_end|>"
        self.assertEqual(gate.feed(ascii_leak[:50]), "")
        self.assertEqual(gate.feed(ascii_leak[50:]), "")
        self.assertEqual(gate.finish(), "")
        self.assertEqual(gate.emitted_chars, 0)


class StreamingSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_token_events_in_order(self) -> None:
        llm = StreamingStubLLM([{"content": FINAL_SCRIPT}])
        events: list[dict[str, Any]] = []
        session = ToolLoopSession(
            llm=llm,  # type: ignore[arg-type]
            tools=None,
            tool_schemas=None,
            messages=[{"role": "user", "content": "hi"}],
            on_stream_event=events.append,
        )
        step = await session.step(Workspace(), max_rounds=1)

        tokens = [e for e in events if e.get("event") == "token"]
        finals = [e for e in events if e.get("event") == "final"]
        self.assertEqual("".join(e["text"] for e in tokens), FINAL_SCRIPT)
        self.assertEqual(len(finals), 1)
        self.assertEqual(step.kind, "final")
        self.assertEqual(step.text, FINAL_SCRIPT)
        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(llm.chat_calls, 0)

    async def test_messages_byte_identical(self) -> None:
        # One tool call → one final, scripted identically for both runs.
        script = [
            _tool_call("echo_tool", '{"input": "hi"}'),
            {"content": "All done now."},
        ]
        tools = _echo_registry()
        schemas = _echo_schemas()

        llm_plain = ChatOnlyStubLLM(list(script))
        session_plain = ToolLoopSession(
            llm=llm_plain,  # type: ignore[arg-type]
            tools=tools,
            tool_schemas=schemas,
            messages=[{"role": "user", "content": "echo then answer"}],
        )
        await session_plain.step(Workspace(), max_rounds=2)

        llm_stream = StreamingStubLLM(list(script))
        events: list[dict[str, Any]] = []
        session_stream = ToolLoopSession(
            llm=llm_stream,  # type: ignore[arg-type]
            tools=tools,
            tool_schemas=schemas,
            messages=[{"role": "user", "content": "echo then answer"}],
            on_stream_event=events.append,
        )
        await session_stream.step(Workspace(), max_rounds=2)

        self.assertEqual(session_plain.messages, session_stream.messages)

    async def test_dsml_leak_streams_nothing(self) -> None:
        # Single round: the leak round executes the tool and ends the step as
        # kind="tool" — no second round streams a clean final, so the assertion
        # is exactly "the leak round emitted nothing".
        llm = StreamingStubLLM([{"content": DSML_LEAK}])
        events: list[dict[str, Any]] = []
        session = ToolLoopSession(
            llm=llm,  # type: ignore[arg-type]
            tools=_echo_registry(),
            tool_schemas=_echo_schemas(),
            messages=[{"role": "user", "content": "echo hi"}],
            on_stream_event=events.append,
        )
        await session.step(Workspace(), max_rounds=1)

        self.assertEqual([e for e in events if e.get("event") == "token"], [])
        self.assertEqual([e for e in events if e.get("event") == "discard"], [])
        self.assertEqual(len(session.tool_requests), 1)
        self.assertEqual(session.tool_requests[0].name, "echo_tool")

    async def test_tool_round_discard(self) -> None:
        # Round 1: >64 chars of provisional prose PLUS a native tool call
        # (OpenAI-style leak of provisional text before a tool call).
        prose = "x" * 80
        script = [
            {**_tool_call("echo_tool", '{"input": "hi"}'), "content": prose},
            {"content": "Final answer."},
        ]
        llm = StreamingStubLLM(script)
        events: list[dict[str, Any]] = []
        session = ToolLoopSession(
            llm=llm,  # type: ignore[arg-type]
            tools=_echo_registry(),
            tool_schemas=_echo_schemas(),
            messages=[{"role": "user", "content": "do something"}],
            on_stream_event=events.append,
        )
        await session.step(Workspace(), max_rounds=2)

        discards = [e for e in events if e.get("event") == "discard"]
        finals = [e for e in events if e.get("event") == "final"]
        self.assertTrue(any(d["round"] == 1 for d in discards), f"expected discard for round 1, got {events}")
        self.assertTrue(any(f["round"] == 2 for f in finals), f"expected final for round 2, got {events}")

    async def test_forced_final_streams(self) -> None:
        # max_total_rounds=1 → round 1 is a tool call, then the forced final.
        script = [
            _tool_call("echo_tool", '{"input": "hi"}'),
            {"content": "Forced final answer streamed out to the client."},
        ]
        llm = StreamingStubLLM(script)
        events: list[dict[str, Any]] = []
        session = ToolLoopSession(
            llm=llm,  # type: ignore[arg-type]
            tools=_echo_registry(),
            tool_schemas=_echo_schemas(),
            messages=[{"role": "user", "content": "do something"}],
            max_total_rounds=1,
            on_stream_event=events.append,
        )
        # max_rounds=2 so the loop re-enters, sees the round budget spent,
        # and routes round 2 through ``_forced_final`` (which streams).
        await session.step(Workspace(), max_rounds=2)

        tokens = [e for e in events if e.get("event") == "token"]
        finals = [e for e in events if e.get("event") == "final"]
        self.assertTrue(len(tokens) > 0, "forced final text should have streamed tokens")
        self.assertEqual(len(finals), 1)
        self.assertEqual(events[-1]["event"], "final")

    async def test_stub_without_chat_stream_untouched(self) -> None:
        llm = ChatOnlyStubLLM([{"content": FINAL_SCRIPT}])
        events: list[dict[str, Any]] = []
        session = ToolLoopSession(
            llm=llm,  # type: ignore[arg-type]
            tools=None,
            tool_schemas=None,
            messages=[{"role": "user", "content": "hi"}],
            on_stream_event=events.append,
        )
        step = await session.step(Workspace(), max_rounds=1)

        self.assertEqual(events, [])
        self.assertEqual(step.kind, "final")
        self.assertEqual(step.text, FINAL_SCRIPT)
        self.assertEqual(llm.chat_calls, 1)


class StreamingRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_call_simple_chat_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stub = StreamingStubLLM([{"content": FINAL_SCRIPT}])
            events: list[dict[str, Any]] = []
            runtime = CognitiveRuntime(
                llm=stub,  # type: ignore[arg-type]
                memory=MemoryStore(db_path=os.path.join(tmp, "stream.db")),
                context_settings=ContextSettings(max_dynamic_chars=2000),
            )
            await runtime.initialize()
            runtime.executor.on_stream_event = events.append
            try:
                result = await runtime.run_episode(InputEvent(content="hi", source="user"))
            finally:
                await runtime.close()

        self.assertEqual(stub.stream_calls, 1)
        self.assertEqual(stub.chat_calls, 0)
        self.assertEqual(result.output, FINAL_SCRIPT)
        self.assertTrue(any(e.get("event") == "token" for e in events), f"expected token events, got {events}")


class StreamingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import unittest.mock
        from pathlib import Path

        from conscio.config import load_config

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

    async def test_service_emits_chat_token_with_ref(self) -> None:
        from conscio.service import ConscioService

        service = ConscioService(self.config)
        stub = StreamingStubLLM([{"content": FINAL_SCRIPT}])
        service.runtime.chat_strategy.llm = stub
        await service.start(acquire_lock=False, background=True)
        client = service.event_broker.register()
        try:
            await service.submit_message("hi", ref="r1")
            # Drain the client queue: collect chat.token then chat.final.
            payloads: list[dict[str, Any]] = []
            for _ in range(256):
                try:
                    payloads.append(client.queue.get_nowait())
                except Exception:  # noqa: BLE001 — drain until empty
                    break
            types = [p.get("type") for p in payloads]
        finally:
            await service.stop()

        self.assertIn("chat.token", types)
        token = next(p for p in payloads if p.get("type") == "chat.token")
        self.assertEqual(token.get("ref"), "r1")
        self.assertTrue(token.get("episode_id"))
        self.assertTrue(token.get("text"))
        self.assertIn("chat.final", types)

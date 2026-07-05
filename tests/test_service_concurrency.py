from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from conscio.config import load_config
from conscio.service import ConscioService


def _tool_call_response(name: str, arguments: str, call_id: str = "call-1") -> dict:
    return {
        "content": "",
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


class _GatedLLM:
    """Blocks inside chat_async until released; records processing order."""

    def __init__(self, label: str, order: list[str], gate: asyncio.Event, responses: list[dict]) -> None:
        self.label = label
        self.order = order
        self.gate = gate
        self.responses = list(responses)
        self.started = asyncio.Event()

    async def chat_async(self, messages: list[dict], **kwargs: object) -> dict:
        self.started.set()
        await self.gate.wait()
        self.order.append(self.label)
        if self.responses:
            return self.responses.pop(0)
        return {"content": f"{self.label} done"}


def _write_config(tmp: str) -> Path:
    path = Path(tmp) / "config.toml"
    path.write_text(
        "[service]\n"
        f'home = "{tmp}"\n'
        'api_key = "k"\n'
        'web_password = "p"\n'
        "autonomous = false\n",
        encoding="utf-8",
    )
    return path


class PriorityQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_interactive_event_outranks_queued_autonomous(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConscioService(load_config(_write_config(tmp)))
            order: list[str] = []
            gate = asyncio.Event()
            auto_llm = _GatedLLM("auto", order, gate, [{"content": "auto answer"}, {"content": "auto answer"}])
            chat_llm = _GatedLLM("chat", order, gate, [{"content": "chat answer"}])
            service.runtime.autonomous_strategy.llm = auto_llm
            service.runtime.chat_strategy.llm = chat_llm
            await service.start(acquire_lock=False, background=True)
            try:
                t_auto1 = asyncio.create_task(service.run_autonomous_tick())
                await auto_llm.started.wait()  # worker is busy on auto1
                t_auto2 = asyncio.create_task(service.run_autonomous_tick())
                await asyncio.sleep(0)  # auto2 enqueued before the user message
                t_chat = asyncio.create_task(service.submit_message("hello"))
                await asyncio.sleep(0)
                gate.set()
                await asyncio.gather(t_auto1, t_auto2, t_chat)
            finally:
                await service.stop()
        # auto1 was already running; the user message must overtake auto2.
        self.assertEqual(order[0], "auto")
        self.assertIn("chat", order)
        self.assertLess(order.index("chat"), len(order) - 1 if order[-1] == "auto" else len(order))
        self.assertEqual(order[1], "chat")


class PreemptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_autonomous_episode_yields_to_interactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConscioService(load_config(_write_config(tmp)))
            order: list[str] = []
            gate = asyncio.Event()
            # Endless tool-calling stub: would run STEP ticks until budget without preemption.
            auto_responses = [
                _tool_call_response("note_progress", '{"note": "working"}', f"call-{i}")
                for i in range(1, 30)
            ]
            auto_llm = _GatedLLM("auto", order, gate, auto_responses)
            chat_llm = _GatedLLM("chat", order, gate, [{"content": "hi there"}])
            service.runtime.autonomous_strategy.llm = auto_llm
            service.runtime.chat_strategy.llm = chat_llm
            await service.start(acquire_lock=False, background=True)
            try:
                t_auto = asyncio.create_task(service.run_autonomous_tick())
                await auto_llm.started.wait()
                t_chat = asyncio.create_task(service.submit_message("hello"))
                await asyncio.sleep(0)  # _pending_interactive is now > 0
                gate.set()
                auto_result = await t_auto
                chat_result = await t_chat
            finally:
                await service.stop()
        assert auto_result is not None
        self.assertEqual(auto_result.selected_action, "wait")
        self.assertEqual(auto_result.outcome_reason, "preempted by interactive event")
        self.assertEqual(service.last_autonomous_action, "wait:preempted")
        self.assertEqual(chat_result.output, "hi there")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from conscio.config import load_config
from conscio.service import ConscioService, EpisodeCancelled


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


class _HangingLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def chat_async(self, messages: list[dict], **kwargs: object) -> dict:
        self.started.set()
        await asyncio.Event().wait()
        return {"content": ""}


def _preset_event() -> asyncio.Event:
    ev = asyncio.Event()
    ev.set()
    return ev


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


class PreemptionSkipsGoalReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_goal_review_skipped_while_user_waits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConscioService(load_config(_write_config(tmp)))
            service._goal_review_interval = 1  # review would fire on every tick
            order: list[str] = []
            gate = asyncio.Event()
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
                await t_chat
            finally:
                await service.stop()
        assert auto_result is not None
        self.assertEqual(auto_result.outcome_reason, "preempted by interactive event")
        # The goal review (an extra LLM call on the autonomous model) must not
        # run between the preempted episode and the waiting chat: the chat is
        # the very next LLM interaction after the single autonomous round.
        self.assertEqual(order[1], "chat")
        self.assertEqual(order.count("auto"), 1)


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


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_current_fails_future_and_worker_survives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConscioService(load_config(_write_config(tmp)))
            hang = _HangingLLM()
            service.runtime.chat_strategy.llm = hang
            await service.start(acquire_lock=False, background=True)
            try:
                t = asyncio.create_task(service.submit_message("will hang"))
                await hang.started.wait()
                info = service.cancel_current()
                self.assertTrue(info["cancelled"])
                with self.assertRaises(EpisodeCancelled):
                    await t
                self.assertFalse(service.paused)
                self.assertEqual(service.last_error, "episode_cancelled")
                # Worker survives: a fresh message still processes.
                service.runtime.chat_strategy.llm = _GatedLLM(
                    "chat", [], _preset_event(), [{"content": "still alive"}]
                )
                result = await asyncio.wait_for(service.submit_message("ping"), 10)
                self.assertEqual(result.output, "still alive")
            finally:
                await service.stop()

    async def test_episode_timeout_cancels_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[service]\n"
                f'home = "{tmp}"\n'
                'api_key = "k"\nweb_password = "p"\n'
                "autonomous = false\nepisode_timeout = 0.2\n",
                encoding="utf-8",
            )
            service = ConscioService(load_config(path))
            service.runtime.chat_strategy.llm = _HangingLLM()
            await service.start(acquire_lock=False, background=True)
            try:
                with self.assertRaises(EpisodeCancelled):
                    await asyncio.wait_for(service.submit_message("slow"), 10)
                self.assertEqual(service.last_error, "episode_timeout")
            finally:
                await service.stop()

    async def test_stop_during_running_episode_fails_future(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = ConscioService(load_config(_write_config(tmp)))
            hang = _HangingLLM()
            service.runtime.chat_strategy.llm = hang
            await service.start(acquire_lock=False, background=True)
            t = asyncio.create_task(service.submit_message("will hang"))
            await hang.started.wait()
            await service.stop()
            with self.assertRaises(RuntimeError):  # EpisodeCancelled is a RuntimeError too
                await t

    async def test_message_timeout_returns_504_and_episode_completes(self) -> None:
        import httpx

        from conscio.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[service]\n"
                f'home = "{tmp}"\n'
                'api_key = "k"\nweb_password = "p"\n'
                "autonomous = false\nmessage_timeout = 0.1\n",
                encoding="utf-8",
            )
            service = ConscioService(load_config(path))
            gate = asyncio.Event()
            slow = _GatedLLM("chat", [], gate, [{"content": "late answer"}])
            service.runtime.chat_strategy.llm = slow
            await service.start(acquire_lock=False, background=True)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    headers = {"Authorization": "Bearer k"}
                    resp = await client.post("/message", json={"content": "hi"}, headers=headers)
                    self.assertEqual(resp.status_code, 504)
                    gate.set()
                    await asyncio.sleep(0.2)  # let the episode finish
                    episodes = await client.get("/episodes", headers=headers)
                    self.assertEqual(episodes.status_code, 200)
                    self.assertTrue(
                        any("hi" in (e.get("input") or "") for e in episodes.json())
                    )
            finally:
                await service.stop()


if __name__ == "__main__":
    unittest.main()

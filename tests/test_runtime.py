from __future__ import annotations

import os
import tempfile
import unittest

from conscio.core.cognition import InputEvent
from conscio.core.runtime import CognitiveRuntime
from conscio.eval import run_eval_suite
from conscio.memory.store import MemoryStore


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
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


class EvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_eval_runs(self) -> None:
        rows = await run_eval_suite("smoke")

        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all(row.mode == "evented_full" for row in rows))


if __name__ == "__main__":
    unittest.main()

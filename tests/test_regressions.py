from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from conscio.core.agent import ConsciousAgent, compose_cycle_output
from conscio.core.identity import Identity
from conscio.core.monologue import Monologue
from conscio.core.workspace import Workspace
from conscio.memory.store import MemoryStore
from conscio.modules.executor import Executor
from conscio.tools import ToolRegistry


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_initialize_honors_db_path_and_does_not_hang(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "sessions.db")
            store = MemoryStore(db_path=db_path)
            await store.initialize()
            await store.create_session("s1", "test")
            sessions = await store.list_sessions()
            await store.close()

            self.assertTrue(os.path.exists(db_path))
            self.assertEqual(sessions[0]["id"], "s1")


class MonologueTests(unittest.TestCase):
    def test_chained_thought_depth_uses_effective_parent(self) -> None:
        monologue = Monologue()
        first = monologue.think("first", "answer")
        second = monologue.think("second", "answer")

        self.assertEqual(second.parent_id, first.id)
        self.assertEqual(second.depth, first.depth + 1)


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsafe_tool_is_disabled_by_default(self) -> None:
        registry = ToolRegistry()
        registry.load_builtins()
        executor = Executor(Workspace(), Monologue(), registry)

        with patch.dict(os.environ, {}, clear=True):
            results = await executor.execute([{"tool": "bash", "description": "echo hi"}])

        self.assertIn("disabled", results[0]["output"])

    async def test_string_args_are_mapped_to_tool_input(self) -> None:
        registry = ToolRegistry()
        registry.load_builtins()
        executor = Executor(Workspace(), Monologue(), registry)

        with patch.dict(os.environ, {"CONSCIO_ENABLE_UNSAFE_TOOLS": "1"}):
            results = await executor.execute([{"tool": "bash", "description": "echo hi"}])

        self.assertEqual(results[0]["output"], "hi")


class IdentityTests(unittest.TestCase):
    def test_agent_explicit_persona_updates_loaded_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity_path = os.path.join(tmp, "identity.json")
            with patch("conscio.core.identity._HOME_DIR", tmp), patch(
                "conscio.core.identity._IDENTITY_PATH",
                identity_path,
            ):
                Identity(name="Conscio", persona="old").save()
                ConsciousAgent(persona="new")

                reloaded = Identity.load_or_create()
                self.assertEqual(reloaded.persona, "new")


class AgentOutputTests(unittest.TestCase):
    def test_reasoning_action_replaces_original_reflection_output(self) -> None:
        output = compose_cycle_output(
            "original reflection",
            [{"tool": "reasoning", "output": "improved final answer"}],
        )

        self.assertEqual(output, "improved final answer")


if __name__ == "__main__":
    unittest.main()

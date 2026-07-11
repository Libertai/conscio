from __future__ import annotations

import os
import tempfile
import unittest

from conscio.config import AblationFlags
from conscio.core.cognition import InputEvent
from conscio.memory.store import MemoryStore
from conscio.tools import ToolRegistry
from conscio.v3.runtime import V3CognitiveRuntime


class RecordingLLM:
    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    async def chat_async(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append(messages)
        self.kwargs.append(dict(kwargs))
        return {"content": self.text}


class V3RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_episode_has_append_only_causal_trace_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(db_path=os.path.join(tmp, "v3.db"))
            runtime = V3CognitiveRuntime(llm=RecordingLLM(), memory=memory, cognitive_cycles=3)
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="hello", source="user"))
                persisted = await memory.cognitive_events(result.episode_id)
            finally:
                await runtime.close()

        types = [event["event_type"] for event in persisted]
        self.assertEqual(types[0], "message")
        self.assertEqual(types.count("broadcast"), 3)
        self.assertLess(types.index("action_proposal"), types.index("action_outcome"))
        self.assertEqual(types[-1], "checkpoint")
        self.assertEqual(result.causal_trace, persisted)
        self.assertTrue(result.checkpoint_reference.startswith("ckpt_"))
        self.assertEqual(len(result.affect_trajectory), 3)
        self.assertTrue(result.predictions)
        self.assertEqual(
            persisted[-1]["model_input"]["dynamic_context"], result.model_context
        )
        self.assertEqual(persisted[-1]["model_input"]["calls"], result.exact_model_inputs)
        self.assertEqual(result.exact_model_inputs[0]["messages"], runtime.executor.model_inputs[0]["messages"])

    async def test_checkpoint_restores_lineage_and_recurrent_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "restart.db")
            first = V3CognitiveRuntime(llm=RecordingLLM(), memory=MemoryStore(db_path=path))
            await first.initialize()
            one = await first.run_episode("one")
            state = first.recurrent_core.deterministic.copy()
            lineage = first.recurrent_core.lineage_id
            await first.close()

            second = V3CognitiveRuntime(llm=RecordingLLM(), memory=MemoryStore(db_path=path))
            await second.initialize()
            try:
                self.assertEqual(second.recurrent_core.lineage_id, lineage)
                self.assertEqual(second.recurrent_core.parent_checkpoint_id, one.checkpoint_reference)
                self.assertTrue((second.recurrent_core.deterministic == state).all())
                two = await second.run_episode("two")
                checkpoint = await second.memory.get_core_checkpoint(two.checkpoint_reference)
            finally:
                await second.close()

        self.assertEqual(checkpoint["parent_checkpoint_id"], one.checkpoint_reference)
        self.assertEqual(checkpoint["lineage_id"], lineage)

    async def test_memory_and_self_model_lesions_remove_access_and_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry()

            async def memory_tool() -> dict:
                return {"output": "secret"}

            tools.register(
                "search_memory",
                memory_tool,
                "read memory",
                capabilities={"memory_read"},
            )
            llm = RecordingLLM()
            runtime = V3CognitiveRuntime(
                llm=llm,
                tools=tools,
                memory=MemoryStore(db_path=os.path.join(tmp, "lesion.db")),
                ablation=AblationFlags(memory_retrieval=False, self_state_coupling=False),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode("inspect this")
            finally:
                await runtime.close()

        schemas = llm.kwargs[0].get("tools") or []
        names = {item["function"]["name"] for item in schemas}
        self.assertNotIn("search_memory", names)
        self.assertNotIn("self:", result.model_context)
        broadcasts = [event for event in result.causal_trace if event["event_type"] == "broadcast"]
        specialists = {
            candidate["specialist"]
            for event in broadcasts
            for candidate in event["payload"]["candidates"]
        }
        self.assertNotIn("memory", specialists)
        self.assertNotIn("self_model", specialists)

    async def test_safe_affect_control_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(db_path=os.path.join(tmp, "affect.db"))
            runtime = V3CognitiveRuntime(memory=memory)
            await runtime.initialize()
            try:
                state = await runtime.set_safe_affect_state(reason="operator recovery")
                rows = memory.fetchall("SELECT * FROM affect_interventions")
            finally:
                await runtime.close()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason"], "operator recovery")
        self.assertEqual(state.valence, 0.0)


if __name__ == "__main__":
    unittest.main()

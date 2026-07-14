from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from conscio.config import AblationFlags, load_config
from conscio.core.cognition import InputEvent
from conscio.eval.v3_experiments import validate_condition_blind_prompt
from conscio.memory.store import MemoryStore
from conscio.service import ConscioService
from conscio.tools import ToolRegistry
from conscio.v3.learning import AdapterState
from conscio.v3.recurrent_core import MODEL_VERSION
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


class ToolThenAnswerLLM(RecordingLLM):
    async def chat_async(self, messages: list[dict], **kwargs: object) -> dict:
        self.calls.append(messages)
        self.kwargs.append(dict(kwargs))
        if len(self.calls) == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "inspect", "arguments": "{}"},
                    }
                ],
            }
        return {"content": "done"}


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
        self.assertLess(types.index("intention_selected"), types.index("action_outcome"))
        self.assertIn("prediction_resolution", types)
        self.assertEqual(types[-1], "checkpoint")
        self.assertEqual(result.causal_trace, persisted)
        self.assertTrue(result.checkpoint_reference.startswith("ckpt_"))
        self.assertEqual(len(result.affect_trajectory), 3)
        self.assertTrue(result.predictions)
        self.assertTrue(all(item["resolved"] for item in result.predictions))
        self.assertEqual(
            persisted[-1]["model_input"]["dynamic_context"], result.model_context
        )
        self.assertEqual(persisted[-1]["model_input"]["calls"], result.exact_model_inputs)
        self.assertEqual(result.exact_model_inputs[0]["messages"], runtime.executor.model_inputs[0]["messages"])
        validate_condition_blind_prompt(result.exact_model_inputs[0]["messages"])

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

    async def test_sustained_affect_limit_triggers_causal_safe_state_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = V3CognitiveRuntime(
                llm=RecordingLLM(),
                memory=MemoryStore(db_path=os.path.join(tmp, "limit.db")),
                cognitive_cycles=2,
                affect_max_arousal=0.0,
                affect_exposure_cycles=2,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode("trigger bounded recovery")
                rows = runtime.memory.fetchall("SELECT * FROM affect_interventions")
            finally:
                await runtime.close()

        self.assertEqual(len(rows), 1)
        self.assertIn(
            "affect_intervention", [event["event_type"] for event in result.causal_trace]
        )

    async def test_llm_tool_call_is_authorized_as_proposal_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tools = ToolRegistry()
            calls = 0

            async def inspect() -> dict:
                nonlocal calls
                calls += 1
                return {"output": "observed"}

            tools.register("inspect", inspect, "inspect", capabilities={"local_read"})
            runtime = V3CognitiveRuntime(
                llm=ToolThenAnswerLLM(),
                tools=tools,
                memory=MemoryStore(db_path=os.path.join(tmp, "tool.db")),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode("inspect it")
            finally:
                await runtime.close()

        types = [event["event_type"] for event in result.causal_trace]
        self.assertEqual(calls, 1)
        self.assertLess(types.index("tool_proposal"), types.index("action_outcome"))
        authorization = next(
            event for event in result.causal_trace if event["event_type"] == "tool_authorization"
        )
        self.assertTrue(authorization["payload"]["allowed"])

    async def test_explicitly_promoted_prediction_adapter_restores_and_is_traced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adapter.db")
            memory = MemoryStore(db_path=path)
            await memory.initialize()
            state = AdapterState(
                base_model_version=MODEL_VERSION,
                revision=1,
                bias=0.2,
                validation_examples=8,
                validation_loss=0.1,
            )
            await memory.record_prediction_adapter_promotion(
                digest=state.digest(),
                base_model_version=state.base_model_version,
                revision=state.revision,
                payload=state.to_json(),
                approved_by="test_operator",
                validation_loss=state.validation_loss,
            )
            await memory.close()

            runtime = V3CognitiveRuntime(
                llm=RecordingLLM(), memory=MemoryStore(db_path=path)
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode("calibrate this")
            finally:
                await runtime.close()

        self.assertEqual(runtime.prediction_adapter, state)
        self.assertTrue(all(item["adapter_digest"] == state.digest() for item in result.predictions))
        self.assertTrue(
            any(item["raw_probability"] != item["probability"] for item in result.predictions)
        )

    async def test_service_persistence_trial_resumes_and_records_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                "[service]\n"
                f'home = "{tmp}"\n'
                'api_key = "test-key"\n'
                "autonomous = false\n"
                "[persistence_trial]\n"
                "enabled = true\n"
                'revision = "test-revision"\n'
                "max_heartbeat_gap = 3600\n",
                encoding="utf-8",
            )
            config = load_config(config_path)
            first = ConscioService(config)
            await first.start(background=False)
            try:
                await first.submit_message("first trial episode")
                first_trial_id = first.persistence_trial.identity.trial_id
            finally:
                await first.stop()

            second = ConscioService(config)
            await second.start(background=False)
            try:
                await second.submit_message("second trial episode")
                report = await second.persistence_trial_report()
                records = second.persistence_trial.records
            finally:
                await second.stop()

        self.assertEqual(second.persistence_trial.identity.trial_id, first_trial_id)
        self.assertIn("restart", [record.kind for record in records])
        self.assertEqual(report["identity"]["revision"], "test-revision")
        self.assertGreater(report["observed_elapsed_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()

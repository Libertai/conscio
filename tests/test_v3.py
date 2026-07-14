from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from conscio.config import AblationFlags, load_config
from conscio.core.cognition import InputEvent
from conscio.eval.v3_experiments import validate_condition_blind_prompt
from conscio.memory.store import MemoryStore
from conscio.service import ConscioService
from conscio.tools import ToolRegistry
from conscio.v3.curriculum import derive_curriculum_examples
from conscio.v3.learning import AdapterState
from conscio.v3.recurrent_core import MODEL_VERSION, HybridRecurrentCore
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
        self.assertEqual(len(result.affect_trajectory), 4)
        self.assertEqual(result.affect_trajectory[-1]["phase"], "action_outcome")
        self.assertLessEqual(
            result.affect_trajectory[-1]["after_need_pressure"],
            result.affect_trajectory[-1]["before_need_pressure"],
        )
        self.assertTrue(result.predictions)
        self.assertTrue(all(item["resolved"] for item in result.predictions))
        self.assertEqual(persisted[-1]["model_input"]["dynamic_context"], result.model_context)
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

    async def test_legacy_specialist_checkpoint_migrates_once_into_audited_lineage(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy-specialists.db")
            source = HybridRecurrentCore(seed=7).checkpoint().to_dict()
            source.pop("specialist_architecture_id")
            source["schema_version"] = 1
            source["specialist_states"] = {
                "perception": {"updates": 4, "last_digest": "p"},
                "memory": {"updates": 3, "last_digest": "m"},
                "world_model": {"updates": 2, "last_digest": "w"},
                "self_model": {
                    "updates": 2,
                    "last_digest": "s",
                    "uncertainty": 0.25,
                },
                "affect": {"updates": 2, "last_digest": "a"},
                "planning": {"updates": 2, "last_digest": "q"},
            }
            legacy_id = str(source["checkpoint_id"])
            legacy_lineage = str(source["lineage_id"])
            store = MemoryStore(db_path=path)
            await store.initialize()
            await store.save_core_checkpoint(source)
            await store.close()

            first = V3CognitiveRuntime(llm=RecordingLLM(), memory=MemoryStore(db_path=path))
            await first.initialize()
            try:
                migrated_id = first.recurrent_core.parent_checkpoint_id
                migrated_lineage = first.recurrent_core.lineage_id
                migrated = await first.memory.get_core_checkpoint(migrated_id or "")
                audit = await first.memory.cognitive_events(f"specialist_migration_{legacy_id}")
                migration_record = await first.memory.core_checkpoint_architecture_migration(legacy_id)
            finally:
                await first.close()

            self.assertIsNotNone(migrated)
            self.assertEqual(migrated["parent_checkpoint_id"], legacy_id)
            self.assertEqual(migrated["schema_version"], 2)
            self.assertEqual(
                migrated["specialist_architecture_id"],
                first.recurrent_core.specialist_architecture_id,
            )
            self.assertNotEqual(migrated_lineage, legacy_lineage)
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit[0]["event_type"], "checkpoint_lineage_migration")
            self.assertIsNotNone(migration_record)
            self.assertEqual(
                audit[0]["payload"]["migration_record_hash"],
                migration_record.record_hash,
            )
            self.assertEqual(
                migrated["specialist_states"]["autobiographical_memory"]["state"]["updates"],
                3,
            )

            second = V3CognitiveRuntime(
                llm=RecordingLLM(),
                memory=MemoryStore(db_path=path),
                restore_checkpoint_id=legacy_id,
            )
            await second.initialize()
            try:
                repeated_audit = await second.memory.cognitive_events(f"specialist_migration_{legacy_id}")
                self.assertEqual(second.recurrent_core.lineage_id, migrated_lineage)
            finally:
                await second.close()

        self.assertEqual(len(repeated_audit), 1)

    async def test_trained_legacy_checkpoint_requires_validated_architecture_migration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trained-legacy.db")
            source = HybridRecurrentCore(seed=9).checkpoint().to_dict()
            source.pop("specialist_architecture_id")
            source["schema_version"] = 1
            source["model_version"] = f"{MODEL_VERSION}.world-r1-legacy"
            source["specialist_states"] = {
                name: {"updates": 0, "last_digest": ""}
                for name in (
                    "perception",
                    "memory",
                    "world_model",
                    "self_model",
                    "affect",
                    "planning",
                )
            }
            store = MemoryStore(db_path=path)
            await store.initialize()
            await store.save_core_checkpoint(source)
            await store.close()

            runtime = V3CognitiveRuntime(memory=MemoryStore(db_path=path))
            migration_lookup = AsyncMock()
            runtime.memory.core_checkpoint_architecture_migration = migration_lookup  # type: ignore[method-assign]
            with self.assertRaisesRegex(ValueError, "trained legacy checkpoints"):
                await runtime.initialize()
            migration_lookup.assert_not_awaited()
            audit = await runtime.memory.cognitive_events(f"specialist_migration_{source['checkpoint_id']}")
            latest = await runtime.memory.latest_core_checkpoint()
            await runtime.memory.close()

        self.assertEqual(audit, [])
        self.assertEqual(latest["checkpoint_id"], source["checkpoint_id"])

    async def test_malformed_legacy_checkpoint_is_not_persisted_as_a_migration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "malformed-legacy.db")
            source = HybridRecurrentCore(seed=13).checkpoint().to_dict()
            source.pop("specialist_architecture_id")
            source["schema_version"] = 1
            source["deterministic_state"] = [0.0]
            source["specialist_states"] = {
                name: {"updates": 0, "last_digest": ""}
                for name in (
                    "perception",
                    "memory",
                    "world_model",
                    "self_model",
                    "affect",
                    "planning",
                )
            }
            store = MemoryStore(db_path=path)
            await store.initialize()
            await store.save_core_checkpoint(source)
            await store.close()

            runtime = V3CognitiveRuntime(memory=MemoryStore(db_path=path))
            with self.assertRaisesRegex(ValueError, "incompatible shape"):
                await runtime.initialize()
            checkpoint_count = runtime.memory.fetchone("SELECT COUNT(*) AS n FROM core_checkpoints")
            migration_count = runtime.memory.fetchone("SELECT COUNT(*) AS n FROM checkpoint_architecture_migrations")
            await runtime.memory.close()

        self.assertEqual(checkpoint_count["n"], 1)
        self.assertEqual(migration_count["n"], 0)

    async def test_invalid_legacy_affect_is_not_persisted_as_a_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "invalid-legacy-affect.db")
            source = HybridRecurrentCore(seed=17).checkpoint().to_dict()
            source.pop("specialist_architecture_id")
            source["schema_version"] = 1
            source["affect"]["need_errors"] = {"unexpected": 0.1}
            source["affect"]["valence"] = 42.0
            source["specialist_states"] = {
                name: {"updates": 0, "last_digest": ""}
                for name in (
                    "perception",
                    "memory",
                    "world_model",
                    "self_model",
                    "affect",
                    "planning",
                )
            }
            store = MemoryStore(db_path=path)
            await store.initialize()
            await store.save_core_checkpoint(source)
            await store.close()

            runtime = V3CognitiveRuntime(memory=MemoryStore(db_path=path))
            with self.assertRaisesRegex(ValueError, "need_errors"):
                await runtime.initialize()
            checkpoint_count = runtime.memory.fetchone("SELECT COUNT(*) AS n FROM core_checkpoints")
            migration_count = runtime.memory.fetchone("SELECT COUNT(*) AS n FROM checkpoint_architecture_migrations")
            await runtime.memory.close()

        self.assertEqual(checkpoint_count["n"], 1)
        self.assertEqual(migration_count["n"], 0)

    async def test_prediction_adapter_identity_includes_specialist_architecture(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old-adapter.db")
            store = MemoryStore(db_path=path)
            await store.initialize()
            old = AdapterState(base_model_version=MODEL_VERSION, revision=4, bias=0.8)
            await store.record_prediction_adapter_promotion(
                digest=old.digest(),
                base_model_version=old.base_model_version,
                revision=old.revision,
                payload=old.to_json(),
                approved_by="legacy-test",
                validation_loss=0.2,
            )
            await store.close()

            runtime = V3CognitiveRuntime(memory=MemoryStore(db_path=path))
            await runtime.initialize()
            try:
                self.assertEqual(runtime.prediction_adapter.revision, 0)
                self.assertEqual(
                    runtime.prediction_adapter.base_model_version,
                    runtime.recurrent_core.runtime_identity,
                )
            finally:
                await runtime.close()

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
        specialists = {candidate["specialist"] for event in broadcasts for candidate in event["payload"]["candidates"]}
        self.assertNotIn("memory", specialists)
        self.assertNotIn("self_model", specialists)

    async def test_self_model_lesion_never_reads_inherited_live_self_state(self) -> None:
        class ForbiddenSelfState:
            def __getattribute__(self, name: str) -> object:
                raise AssertionError(f"lesioned live self-state read: {name}")

            def __setattr__(self, name: str, value: object) -> None:
                raise AssertionError(f"lesioned live self-state mutation: {name}")

        with tempfile.TemporaryDirectory() as tmp:
            runtime = V3CognitiveRuntime(
                llm=RecordingLLM(),
                memory=MemoryStore(db_path=os.path.join(tmp, "self-lesion.db")),
                ablation=AblationFlags(self_state_coupling=False),
            )
            runtime.self_state = ForbiddenSelfState()  # type: ignore[assignment]
            await runtime.initialize()
            try:
                result = await runtime.run_episode("do not inspect live self state")
            finally:
                await runtime.close()

        self.assertEqual(result.self_state, {})
        self.assertTrue(result.tick_trace)
        self.assertTrue(all("self_state_delta" not in tick for tick in result.tick_trace))
        self.assertNotIn("self:", result.model_context)

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
        self.assertIn("affect_intervention", [event["event_type"] for event in result.causal_trace])

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
        authorization = next(event for event in result.causal_trace if event["event_type"] == "tool_authorization")
        self.assertTrue(authorization["payload"]["allowed"])
        curriculum = derive_curriculum_examples(result.causal_trace)
        families = {example.target_family for example in curriculum.examples}
        self.assertEqual(
            families,
            {
                "next_observation",
                "tool_outcome",
                "action_effect",
                "homeostatic_affect_change",
                "future_uncertainty",
            },
        )
        accepted, training, rejected = ConscioService._recorded_world_examples(curriculum.examples)
        self.assertEqual(rejected, 0)
        self.assertEqual(len(accepted), 5)
        self.assertEqual(len(training), 1)

    async def test_explicitly_promoted_prediction_adapter_restores_and_is_traced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "adapter.db")
            memory = MemoryStore(db_path=path)
            await memory.initialize()
            state = AdapterState(
                base_model_version=HybridRecurrentCore().runtime_identity,
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

            runtime = V3CognitiveRuntime(llm=RecordingLLM(), memory=MemoryStore(db_path=path))
            await runtime.initialize()
            try:
                result = await runtime.run_episode("calibrate this")
            finally:
                await runtime.close()

        self.assertEqual(runtime.prediction_adapter, state)
        self.assertTrue(all(item["adapter_digest"] == state.digest() for item in result.predictions))
        self.assertTrue(any(item["raw_probability"] != item["probability"] for item in result.predictions))

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

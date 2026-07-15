from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from conscio.config import load_config
from conscio.core.cognition import InputEvent
from conscio.memory.store import MemoryStore
from conscio.service import ConscioService
from conscio.tools import ToolRegistry
from conscio.v3.contracts import CognitiveEvent, ExecutionIntent
from conscio.v3.curriculum import derive_curriculum_examples
from conscio.v3.learning import derive_replay_samples
from conscio.v3.runtime import V3CognitiveRuntime
from conscio.v3.world_training import WorldCoreWeights


def _digest(character: str) -> str:
    return "sha256:" + (character * 64)


async def _seed_orphan(
    store: MemoryStore,
    *,
    runtime_identity: str,
    execution_character: str,
    episode_id: str,
) -> ExecutionIntent:
    intent = ExecutionIntent(
        execution_id="exec_" + (execution_character * 64),
        proposal_id=f"proposal-{execution_character}",
        action_digest=_digest("2"),
        context_digest=_digest("3"),
        runtime_identity=runtime_identity,
        competition_sequence=0,
        action_kind="tool",
        tool_manifest_digest=_digest("6"),
        tool="inspect",
        arguments_json='{"query":"status"}',
        capabilities=("local_read",),
    )
    await store.append_cognitive_event_idempotent(
        CognitiveEvent(
            event_type="execution_intent",
            source="runtime_executor",
            payload=intent.to_dict(),
            episode_id=episode_id,
            event_id=intent.event_id,
        )
    )
    return intent


class _ToolThenTextLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((list(messages), dict(kwargs)))
        if len(self.calls) == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-new",
                        "type": "function",
                        "function": {
                            "name": "inspect",
                            "arguments": '{"query":"new"}',
                        },
                    }
                ],
            }
        return {"content": "safe textual response"}


class _AskLLM:
    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del messages, kwargs
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-ask",
                    "type": "function",
                    "function": {
                        "name": "ask_user",
                        "arguments": '{"question":"Which file?"}',
                    },
                }
            ],
        }


class _UnknownOutcomeLLM:
    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del messages, kwargs
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-unknown",
                    "type": "function",
                    "function": {
                        "name": "inspect",
                        "arguments": '{"query":"remote"}',
                    },
                }
            ],
        }


class _AskUpstreamRuntime(V3CognitiveRuntime):
    """Make the recurrent proposal prefer ASK so the control path is deterministic."""

    def _select_proposal(self, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        return next(proposal for proposal in proposals if proposal["action"] == "ask")


class V3ExecutionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_restart_records_one_recovery_marker_across_repeated_initialization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "restart.db")
            bootstrap = V3CognitiveRuntime(memory=MemoryStore(db_path=path))
            await bootstrap.memory.initialize()
            intent = await _seed_orphan(
                bootstrap.memory,
                runtime_identity=bootstrap.recurrent_core.runtime_identity,
                execution_character="a",
                episode_id="crashed-episode",
            )
            await bootstrap.memory.close()

            first = V3CognitiveRuntime(memory=MemoryStore(db_path=path))
            await first.initialize()
            self.assertTrue(first.execution_safe_mode)
            self.assertEqual(first.unresolved_execution_ids, (intent.execution_id,))
            await first.close()

            second = V3CognitiveRuntime(memory=MemoryStore(db_path=path))
            await second.initialize()
            try:
                events = await second.memory.cognitive_events("crashed-episode")
                unresolved = await second.memory.unresolved_execution_intents()
            finally:
                await second.close()

        recoveries = [event for event in events if event["event_type"] == "execution_recovery"]
        self.assertEqual(len(recoveries), 1)
        self.assertEqual(recoveries[0]["payload"]["execution_id"], intent.execution_id)
        self.assertEqual(recoveries[0]["payload"]["intent_digest"], intent.intent_digest)
        self.assertIsNone(recoveries[0]["payload"].get("executed"))
        self.assertIsNone(recoveries[0]["payload"].get("succeeded"))
        self.assertEqual([row["event_id"] for row in unresolved], [intent.event_id])

    async def test_safe_mode_blocks_tools_without_redispatch_and_preserves_text(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "safe-mode.db")
            tools = ToolRegistry()
            executions = 0

            async def inspect(query: str) -> dict[str, Any]:
                nonlocal executions
                executions += 1
                return {"output": query}

            tools.register("inspect", inspect, "inspect", capabilities={"local_read"})
            llm = _ToolThenTextLLM()
            runtime = V3CognitiveRuntime(
                llm=llm,
                tools=tools,
                memory=MemoryStore(db_path=path),
            )
            await runtime.memory.initialize()
            intent = await _seed_orphan(
                runtime.memory,
                runtime_identity=runtime.recurrent_core.runtime_identity,
                execution_character="b",
                episode_id="crashed-tool-episode",
            )

            await runtime.initialize()
            self.assertEqual(executions, 0, "startup must never replay an orphaned dispatch")
            try:
                blocked = await runtime.run_episode(InputEvent(content="inspect the new status", source="autonomous"))
                text = await runtime.run_episode("respond without using a tool")
            finally:
                await runtime.close()

        self.assertEqual(executions, 0)
        self.assertEqual(blocked.metrics.tool_calls, 0)
        self.assertEqual(blocked.metrics.tool_proposals, 1)
        self.assertEqual(blocked.selected_action, "wait")
        self.assertEqual(text.output, "safe textual response")
        self.assertEqual(text.metrics.tool_calls, 0)
        authorization = next(
            event
            for event in blocked.causal_trace
            if event["event_type"] == "tool_authorization" and event["payload"]["call_id"] == "call-new"
        )
        self.assertFalse(authorization["payload"]["allowed"])
        self.assertFalse(
            any(
                event["event_type"] == "execution_intent" and event["payload"]["execution_id"] != intent.execution_id
                for event in blocked.causal_trace
            )
        )
        blocked_outcome = next(event for event in blocked.causal_trace if event["event_type"] == "action_outcome")
        self.assertFalse(blocked_outcome["payload"]["observed"])
        self.assertFalse(blocked_outcome["payload"]["learning_eligible"])
        blocked_curriculum = derive_curriculum_examples(blocked.causal_trace)
        self.assertFalse(any(example.target_family == "action_effect" for example in blocked_curriculum.examples))

    async def test_unknown_tool_execution_remains_unresolved_and_unlearned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "unknown-outcome.db")
            tools = ToolRegistry()
            executions = 0

            async def inspect(query: str) -> dict[str, Any]:
                nonlocal executions
                executions += 1
                return {
                    "output": f"connection lost after dispatch: {query}",
                    "error": True,
                    "execution_unknown": True,
                }

            tools.register("inspect", inspect, "inspect", capabilities={"network_read"})
            memory = MemoryStore(db_path=path)
            runtime = V3CognitiveRuntime(
                llm=_UnknownOutcomeLLM(),
                tools=tools,
                memory=memory,
                max_ticks=1,
                recurrent_weights=WorldCoreWeights.bootstrap(),
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="inspect remote state", source="autonomous"))
                unresolved = await memory.unresolved_execution_intents()
                safe_mode = runtime.execution_safe_mode
                unresolved_ids = runtime.unresolved_execution_ids
            finally:
                await runtime.close()

        self.assertEqual(executions, 1)
        self.assertEqual(result.metrics.tool_calls, 1)
        self.assertEqual(len(result.tool_results), 1)
        self.assertTrue(result.tool_results[0]["execution_unknown"])
        intent_event = next(event for event in result.causal_trace if event["event_type"] == "execution_intent")
        intent = ExecutionIntent.from_dict(intent_event["payload"])
        self.assertTrue(safe_mode)
        self.assertEqual(unresolved_ids, (intent.execution_id,))
        self.assertEqual([row["event_id"] for row in unresolved], [intent.event_id])

        event_types = [event["event_type"] for event in result.causal_trace]
        self.assertEqual(event_types.count("execution_uncertain"), 1)
        self.assertNotIn("execution_outcome", event_types)
        self.assertNotIn("tool_outcome", event_types)
        uncertain = next(event for event in result.causal_trace if event["event_type"] == "execution_uncertain")
        self.assertEqual(uncertain["payload"]["execution_id"], intent.execution_id)
        self.assertEqual(uncertain["payload"]["intent_digest"], intent.intent_digest)
        self.assertFalse(uncertain["payload"]["learning_eligible"])

        unobserved_targets = {"next_observation", "tool_outcome", "action_effect"}
        relevant_predictions = [
            prediction for prediction in result.predictions if prediction["target"] in unobserved_targets
        ]
        self.assertTrue(relevant_predictions)
        self.assertTrue(all(not prediction["resolved"] for prediction in relevant_predictions))
        self.assertTrue(all(prediction["error"] is None for prediction in relevant_predictions))
        relevant_resolutions = [
            event["payload"]
            for event in result.causal_trace
            if event["event_type"] == "prediction_resolution" and event["payload"].get("target") in unobserved_targets
        ]
        self.assertEqual(len(relevant_resolutions), len(relevant_predictions))
        self.assertTrue(all(resolution["observed"] is None for resolution in relevant_resolutions))
        self.assertTrue(all(resolution["error"] is None for resolution in relevant_resolutions))
        replay = derive_replay_samples(result.causal_trace)
        relevant_prediction_ids = {prediction["prediction_id"] for prediction in relevant_predictions}
        self.assertTrue(all(sample.prediction_id not in relevant_prediction_ids for sample in replay.samples))

        action_outcome = next(event for event in result.causal_trace if event["event_type"] == "action_outcome")
        self.assertFalse(action_outcome["payload"]["learning_eligible"])
        curriculum = derive_curriculum_examples(result.causal_trace)
        learned_families = {example.target_family for example in curriculum.examples}
        self.assertNotIn("tool_outcome", learned_families)
        self.assertNotIn("action_effect", learned_families)
        unobserved_affect = next(
            event
            for event in result.causal_trace
            if event["event_type"] == "affect" and event["payload"].get("phase") == "action_outcome"
        )
        self.assertFalse(unobserved_affect["payload"]["learning_eligible"])
        self.assertTrue(
            all(
                unobserved_affect["event_id"] not in example.provenance.source_event_ids
                for example in curriculum.examples
                if example.target_family == "homeostatic_affect_change"
            )
        )

    async def test_safe_mode_keeps_non_tool_control_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "safe-control.db")
            runtime = _AskUpstreamRuntime(
                llm=_AskLLM(),
                memory=MemoryStore(db_path=path),
            )
            await runtime.memory.initialize()
            intent = await _seed_orphan(
                runtime.memory,
                runtime_identity=runtime.recurrent_core.runtime_identity,
                execution_character="c",
                episode_id="crashed-control-episode",
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode("ambiguous request")
                unresolved = runtime.unresolved_execution_ids
            finally:
                await runtime.close()

        self.assertEqual(result.selected_action, "ask")
        self.assertEqual(result.output, "Which file?")
        self.assertEqual(unresolved, (intent.execution_id,))

    async def test_reconciliation_clears_only_named_orphan_without_outcome_claims(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reconcile.db")
            runtime = V3CognitiveRuntime(memory=MemoryStore(db_path=path))
            await runtime.memory.initialize()
            first = await _seed_orphan(
                runtime.memory,
                runtime_identity=runtime.recurrent_core.runtime_identity,
                execution_character="d",
                episode_id="crashed-first",
            )
            second = await _seed_orphan(
                runtime.memory,
                runtime_identity=runtime.recurrent_core.runtime_identity,
                execution_character="e",
                episode_id="crashed-second",
            )
            await runtime.initialize()
            try:
                payload = await runtime.reconcile_execution(
                    first.execution_id,
                    reason="verified side effects out of band",
                    operator="test_operator",
                )
                first_events = await runtime.memory.cognitive_events("crashed-first")
                history = await runtime.memory.cognitive_event_history()
                with self.assertRaises(KeyError):
                    await runtime.reconcile_execution(
                        "exec_" + ("f" * 64),
                        reason="not a startup orphan",
                        operator="test_operator",
                    )
            finally:
                await runtime.close()

        self.assertEqual(payload["execution_id"], first.execution_id)
        self.assertEqual(payload["resolution"], "operator_acknowledged_unknown")
        self.assertFalse(payload["learning_eligible"])
        self.assertTrue(runtime.execution_safe_mode)
        self.assertEqual(runtime.unresolved_execution_ids, (second.execution_id,))
        reconciliation = next(event for event in first_events if event["event_type"] == "execution_reconciliation")
        self.assertIsNone(reconciliation["payload"].get("executed"))
        self.assertIsNone(reconciliation["payload"].get("succeeded"))
        self.assertFalse(
            any(
                event["event_type"] in {"execution_outcome", "tool_outcome", "prediction_resolution"}
                for event in first_events
            )
        )
        curriculum = derive_curriculum_examples(history)
        self.assertFalse(any(example.target_family == "tool_outcome" for example in curriculum.examples))


class V3ExecutionRecoveryApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_bearer_status_and_reconciliation_surface(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("FastAPI/httpx are not installed")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "LIBERTAI_BASE_URL": "",
                    "LIBERTAI_API_KEY": "",
                    "LIBERTAI_MODEL": "",
                    "OPENAI_BASE_URL": "",
                    "OPENAI_API_KEY": "",
                },
                clear=False,
            ),
        ):
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                f'[service]\nhome = "{tmp}"\napi_key = "test-key"\nweb_password = "test-pass"\nautonomous = false\n',
                encoding="utf-8",
            )
            service = ConscioService(load_config(config_path))
            await service.memory.initialize()
            intent = await _seed_orphan(
                service.memory,
                runtime_identity=service.runtime.recurrent_core.runtime_identity,
                execution_character="1",
                episode_id="crashed-api-episode",
            )
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://test",
                ) as client:
                    denied_status = await client.get("/status")
                    denied_reconcile = await client.post(
                        f"/control/executions/{intent.execution_id}/reconcile",
                        json={"reason": "checked"},
                    )
                    headers = {"Authorization": "Bearer test-key"}
                    before = await client.get("/status", headers=headers)
                    reconciled = await client.post(
                        f"/control/executions/{intent.execution_id}/reconcile",
                        headers=headers,
                        json={"reason": "operator checked external state"},
                    )
                    after = await client.get("/status", headers=headers)
                    unknown = await client.post(
                        f"/control/executions/{'exec_' + ('f' * 64)}/reconcile",
                        headers=headers,
                        json={"reason": "unknown"},
                    )
            finally:
                await service.stop()

        self.assertEqual(denied_status.status_code, 401)
        self.assertEqual(denied_reconcile.status_code, 401)
        self.assertEqual(before.status_code, 200)
        self.assertTrue(before.json()["execution_safe_mode"])
        self.assertEqual(before.json()["unresolved_execution_count"], 1)
        self.assertEqual(before.json()["unresolved_execution_ids"], [intent.execution_id])
        self.assertEqual(reconciled.status_code, 200)
        self.assertEqual(reconciled.json()["resolution"], "operator_acknowledged_unknown")
        self.assertFalse(reconciled.json()["learning_eligible"])
        self.assertFalse(reconciled.json()["execution_safe_mode"])
        self.assertEqual(reconciled.json()["unresolved_execution_ids"], [])
        self.assertEqual(after.status_code, 200)
        self.assertFalse(after.json()["execution_safe_mode"])
        self.assertEqual(after.json()["unresolved_execution_count"], 0)
        self.assertEqual(unknown.status_code, 404)

    async def test_authenticated_service_can_reconcile_live_intent_runtime_rejects(
        self,
    ) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("FastAPI/httpx are not installed")

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "LIBERTAI_BASE_URL": "",
                    "LIBERTAI_API_KEY": "",
                    "LIBERTAI_MODEL": "",
                    "OPENAI_BASE_URL": "",
                    "OPENAI_API_KEY": "",
                },
                clear=False,
            ),
        ):
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                f'[service]\nhome = "{tmp}"\napi_key = "test-key"\nweb_password = "test-pass"\nautonomous = false\n',
                encoding="utf-8",
            )
            service = ConscioService(load_config(config_path))
            await service.start(background=False)
            try:
                intent = await _seed_orphan(
                    service.memory,
                    runtime_identity=service.runtime.recurrent_core.runtime_identity,
                    execution_character="2",
                    episode_id="live-api-episode",
                )
                with self.assertRaisesRegex(RuntimeError, "restart"):
                    await service.runtime.reconcile_execution(
                        intent.execution_id,
                        reason="direct live acknowledgement must fail",
                        operator="test_operator",
                    )
                self.assertTrue(service.runtime.execution_safe_mode)
                self.assertEqual(service.runtime.unresolved_execution_ids, (intent.execution_id,))

                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://test",
                ) as client:
                    reconciled = await client.post(
                        f"/control/executions/{intent.execution_id}/reconcile",
                        headers={"Authorization": "Bearer test-key"},
                        json={"reason": "authenticated operator checked external state"},
                    )
                    status = await client.get(
                        "/status",
                        headers={"Authorization": "Bearer test-key"},
                    )
            finally:
                await service.stop()

        self.assertEqual(reconciled.status_code, 200)
        self.assertEqual(reconciled.json()["execution_id"], intent.execution_id)
        self.assertFalse(reconciled.json()["learning_eligible"])
        self.assertFalse(reconciled.json()["execution_safe_mode"])
        self.assertEqual(reconciled.json()["unresolved_execution_ids"], [])
        self.assertEqual(status.status_code, 200)
        self.assertFalse(status.json()["execution_safe_mode"])
        self.assertEqual(status.json()["unresolved_execution_count"], 0)


if __name__ == "__main__":
    unittest.main()

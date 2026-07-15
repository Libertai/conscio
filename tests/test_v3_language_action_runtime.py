from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

from conscio.core.cognition import InputEvent
from conscio.memory.store import MemoryStore
from conscio.tools.registry import ToolRegistry
from conscio.v3.contracts import ExecutionIntent, ExecutionOutcome
from conscio.v3.curriculum import derive_curriculum_examples
from conscio.v3.runtime import V3CognitiveRuntime


class ParallelThenAnswerLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "risky-call",
                        "type": "function",
                        "function": {
                            "name": "risky_lookup",
                            "arguments": '{"slot":1}',
                        },
                    },
                    {
                        "id": "safe-call",
                        "type": "function",
                        "function": {
                            "name": "safe_lookup",
                            "arguments": '{"nested":{"x":"y"},"slot":2}',
                        },
                    },
                ],
            }
        return {"role": "assistant", "content": "done"}


class OneToolThenAnswerLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "late-denial",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"subject":"bounded"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "fallback response"}


class LateDenyRegistry(ToolRegistry):
    async def call(
        self,
        name: str,
        args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "output": "denied at the final policy gate",
            "error": True,
            "executed": False,
            "policy_denied": True,
        }


class V3LanguageActionRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_parallel_proposal_keeps_exact_identity_and_is_only_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calls: list[tuple[str, dict[str, Any]]] = []
            tools = ToolRegistry()

            async def risky_lookup(slot: int) -> dict[str, Any]:
                calls.append(("risky_lookup", {"slot": slot}))
                return {"output": "risky"}

            async def safe_lookup(slot: int, nested: dict[str, Any]) -> dict[str, Any]:
                calls.append(("safe_lookup", {"slot": slot, "nested": nested}))
                return {"output": "safe"}

            tools.register(
                "risky_lookup",
                risky_lookup,
                capabilities={"network_write"},
            )
            tools.register("safe_lookup", safe_lookup)
            memory = MemoryStore(db_path=os.path.join(tmp, "parallel.db"))
            runtime = V3CognitiveRuntime(
                llm=ParallelThenAnswerLLM(),
                tools=tools,
                memory=memory,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="inspect safely", source="autonomous"))
                unresolved = await memory.unresolved_execution_intents()
            finally:
                await runtime.close()

        expected_arguments = {"nested": {"x": "y"}, "slot": 2}
        self.assertEqual(calls, [("safe_lookup", expected_arguments)])
        self.assertEqual(result.metrics.tool_proposals, 2)
        self.assertEqual(result.metrics.tool_calls, 1)
        self.assertEqual(result.metrics.tool_rounds, 1)
        self.assertEqual(len(result.tool_results), 1)
        self.assertEqual(result.tool_results[0]["tool"], "safe_lookup")
        self.assertEqual(result.tool_results[0]["call_id"], "safe-call")
        self.assertEqual(unresolved, [])

        proposals = [event for event in result.causal_trace if event["event_type"] == "tool_proposal"]
        self.assertEqual(len(proposals), 2)
        safe_proposal = next(event for event in proposals if event["payload"]["call_id"] == "safe-call")
        self.assertEqual(safe_proposal["payload"]["candidate"]["arguments"], expected_arguments)
        intent_event = next(event for event in result.causal_trace if event["event_type"] == "execution_intent")
        outcome_event = next(event for event in result.causal_trace if event["event_type"] == "execution_outcome")
        intent = ExecutionIntent.from_dict(intent_event["payload"])
        outcome = ExecutionOutcome.from_dict(outcome_event["payload"])
        self.assertEqual(intent.arguments, expected_arguments)
        self.assertEqual(intent.tool, "safe_lookup")
        self.assertEqual(intent.proposal_id, safe_proposal["payload"]["proposal_id"])
        self.assertEqual(outcome.execution_id, intent.execution_id)
        self.assertEqual(outcome.intent_digest, intent.intent_digest)
        self.assertTrue(outcome.executed)
        self.assertTrue(outcome.succeeded)
        self.assertEqual(intent_event["event_id"], intent.event_id)
        self.assertEqual(outcome_event["event_id"], outcome.event_id)

        event_types = [event["event_type"] for event in result.causal_trace]
        self.assertLess(event_types.index("tool_proposal"), event_types.index("pre_action_forecast"))
        self.assertLess(event_types.index("pre_action_forecast"), event_types.index("action_competition"))
        self.assertLess(event_types.index("action_competition"), event_types.index("execution_intent"))
        self.assertLess(event_types.index("execution_intent"), event_types.index("execution_outcome"))
        self.assertLess(event_types.index("execution_outcome"), event_types.index("tool_outcome"))

    async def test_policy_denial_is_terminal_but_never_a_tool_learning_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            invoked = 0
            tools = LateDenyRegistry()

            async def lookup(subject: str) -> dict[str, Any]:
                nonlocal invoked
                invoked += 1
                return {"output": subject}

            tools.register("lookup", lookup)
            memory = MemoryStore(db_path=os.path.join(tmp, "denied.db"))
            runtime = V3CognitiveRuntime(
                llm=OneToolThenAnswerLLM(),
                tools=tools,
                memory=memory,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="look it up", source="autonomous"))
                unresolved = await memory.unresolved_execution_intents()
            finally:
                await runtime.close()

        self.assertEqual(invoked, 0)
        self.assertEqual(result.metrics.tool_proposals, 1)
        self.assertEqual(result.metrics.tool_calls, 0)
        self.assertEqual(result.metrics.prediction_errors, 0)
        self.assertEqual(unresolved, [])
        self.assertNotIn("Tool lookup returned", result.workspace_trace)
        event_types = [event["event_type"] for event in result.causal_trace]
        self.assertIn("execution_intent", event_types)
        self.assertIn("execution_outcome", event_types)
        self.assertNotIn("tool_outcome", event_types)
        outcome_event = next(event for event in result.causal_trace if event["event_type"] == "execution_outcome")
        outcome = ExecutionOutcome.from_dict(outcome_event["payload"])
        self.assertFalse(outcome.executed)
        self.assertIsNone(outcome.succeeded)
        self.assertEqual(outcome.disposition, "not_executed")
        self.assertEqual(outcome.reason_code, "policy_denied")

        tool_resolutions = [
            event["payload"]
            for event in result.causal_trace
            if event["event_type"] == "prediction_resolution" and event["payload"].get("target") == "tool_outcome"
        ]
        self.assertTrue(all(item["observed"] is None for item in tool_resolutions))
        action_affect = next(
            event["payload"]
            for event in reversed(result.causal_trace)
            if event["event_type"] == "affect" and event["payload"].get("phase") == "action_outcome"
        )
        self.assertTrue(action_affect["outcome_observed"])
        self.assertTrue(action_affect["succeeded"])
        curriculum = derive_curriculum_examples(result.causal_trace)
        self.assertNotIn(
            "tool_outcome",
            {example.target_family for example in curriculum.examples},
        )


if __name__ == "__main__":
    unittest.main()

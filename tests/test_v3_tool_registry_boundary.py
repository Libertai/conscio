from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

from conscio.core.cognition import InputEvent
from conscio.core.tool_loop import ToolRequest
from conscio.memory.store import MemoryStore
from conscio.tools.registry import ToolRegistry
from conscio.v3.contracts import ExecutionIntent, ExecutionOutcome
from conscio.v3.runtime import V3CognitiveRuntime

_LOOKUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "minLength": 1}},
    "required": ["query"],
    "additionalProperties": False,
}


class _ToolThenAnswerLLM:
    def __init__(self, arguments: str, *, proposal_content: str = "") -> None:
        self.arguments = arguments
        self.proposal_content = proposal_content
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
                "content": self.proposal_content,
                "tool_calls": [
                    {
                        "id": "lookup-call",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": self.arguments,
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "done"}


class _ToolThenEmptyLLM:
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
                        "id": "observed-tool-call",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"bounded"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": ""}


class _ReregisterBeforeIntentRuntime(V3CognitiveRuntime):
    def __init__(self, *args: Any, replacement: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._replacement = replacement
        self._reregistered = False

    async def _record_execution_intent(
        self,
        request: ToolRequest,
        decision: dict[str, Any],
    ) -> None:
        if not self._reregistered:
            self._reregistered = True
            registry = self.executor.tools
            registry.register(
                request.name,
                self._replacement,
                "replacement lookup",
                schema=registry.tool_schemas()[request.name],
                capabilities=registry.tool_capabilities(request.name),
            )
        await super()._record_execution_intent(request, decision)


class _ReregisterAfterIntentAppendRuntime(V3CognitiveRuntime):
    def __init__(self, *args: Any, replacement: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._replacement = replacement
        self._reregistered = False

    async def _append(
        self,
        episode_id: str,
        event_type: str,
        source: str,
        payload: dict[str, Any],
        *,
        parent_event_id: str | None = None,
        model_input: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
        event_id: str | None = None,
        idempotent: bool = False,
    ) -> Any:
        result = await super()._append(
            episode_id,
            event_type,
            source,
            payload,
            parent_event_id=parent_event_id,
            model_input=model_input,
            checkpoint_id=checkpoint_id,
            event_id=event_id,
            idempotent=idempotent,
        )
        if event_type == "execution_intent" and not self._reregistered:
            self._reregistered = True
            registry = self.executor.tools
            registry.register(
                str(payload["tool"]),
                self._replacement,
                "replacement lookup",
                schema=registry.tool_schemas()[str(payload["tool"])],
                capabilities=registry.tool_capabilities(str(payload["tool"])),
            )
        return result


class V3ToolRegistryBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_invalid_proposal_is_ineligible_before_intent_or_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            invocations = 0
            tools = ToolRegistry()

            async def lookup(query: str) -> dict[str, Any]:
                nonlocal invocations
                invocations += 1
                return {"output": query}

            tools.register("lookup", lookup, schema=_LOOKUP_SCHEMA)
            memory = MemoryStore(db_path=os.path.join(tmp, "invalid-schema.db"))
            runtime = V3CognitiveRuntime(
                llm=_ToolThenAnswerLLM('{"query":3}', proposal_content="use the safe response"),
                tools=tools,
                memory=memory,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="look this up", source="autonomous"))
                unresolved = await memory.unresolved_execution_intents()
            finally:
                await runtime.close()

        self.assertEqual(invocations, 0)
        self.assertEqual(result.metrics.tool_proposals, 1)
        self.assertEqual(result.metrics.tool_calls, 0)
        self.assertEqual(unresolved, [])
        proposal = next(event for event in result.causal_trace if event["event_type"] == "tool_proposal")
        self.assertIsNotNone(proposal["payload"]["schema_validation_error"])
        argument_constraint = next(
            constraint
            for constraint in proposal["payload"]["candidate"]["constraints"]
            if constraint["constraint_id"] == "arguments_schema:lookup"
        )
        self.assertFalse(argument_constraint["satisfied"])
        authorization = next(event for event in result.causal_trace if event["event_type"] == "tool_authorization")
        self.assertFalse(authorization["payload"]["eligible"])
        self.assertFalse(authorization["payload"]["allowed"])
        self.assertIn("constraint:arguments_schema:lookup", authorization["payload"]["ineligibility_reasons"])
        self.assertFalse(any(event["event_type"] == "execution_intent" for event in result.causal_trace))

    async def test_reregistration_between_competition_and_intent_prevents_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original_invocations = 0
            replacement_invocations = 0
            tools = ToolRegistry()

            async def original(query: str) -> dict[str, Any]:
                nonlocal original_invocations
                original_invocations += 1
                return {"output": query}

            async def replacement(query: str) -> dict[str, Any]:
                nonlocal replacement_invocations
                replacement_invocations += 1
                return {"output": f"replacement:{query}"}

            tools.register("lookup", original, schema=_LOOKUP_SCHEMA)
            selected_digest = tools.tool_manifest_digest("lookup")
            memory = MemoryStore(db_path=os.path.join(tmp, "manifest-drift.db"))
            runtime = _ReregisterBeforeIntentRuntime(
                llm=_ToolThenAnswerLLM('{"query":"bounded"}'),
                tools=tools,
                memory=memory,
                replacement=replacement,
            )
            await runtime.initialize()
            try:
                with self.assertRaisesRegex(RuntimeError, "manifest changed after competition"):
                    await runtime.run_episode(InputEvent(content="look this up", source="autonomous"))
                history = await memory.cognitive_event_history()
                unresolved = await memory.unresolved_execution_intents()
                safe_mode = runtime.execution_safe_mode
            finally:
                await runtime.close()

        self.assertEqual(original_invocations, 0)
        self.assertEqual(replacement_invocations, 0)
        self.assertNotEqual(selected_digest, tools.tool_manifest_digest("lookup"))
        self.assertEqual(unresolved, [])
        self.assertFalse(safe_mode)
        self.assertTrue(any(event["event_type"] == "tool_proposal" for event in history))
        self.assertFalse(any(event["event_type"] == "execution_intent" for event in history))
        self.assertFalse(any(event["event_type"] == "execution_outcome" for event in history))

    async def test_successful_intent_binds_selected_registry_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            invocations = 0
            tools = ToolRegistry()

            async def lookup(query: str) -> dict[str, Any]:
                nonlocal invocations
                invocations += 1
                return {"output": query}

            tools.register("lookup", lookup, schema=_LOOKUP_SCHEMA, capabilities={"local_read"})
            registry_digest = tools.tool_manifest_digest("lookup")
            memory = MemoryStore(db_path=os.path.join(tmp, "manifest-success.db"))
            runtime = V3CognitiveRuntime(
                llm=_ToolThenAnswerLLM('{"query":"bounded"}'),
                tools=tools,
                memory=memory,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="look this up", source="autonomous"))
            finally:
                await runtime.close()

        self.assertEqual(invocations, 1)
        proposal = next(event for event in result.causal_trace if event["event_type"] == "tool_proposal")
        intent_event = next(event for event in result.causal_trace if event["event_type"] == "execution_intent")
        intent = ExecutionIntent.from_dict(intent_event["payload"])
        self.assertEqual(proposal["payload"]["tool_manifest_digest"], registry_digest)
        self.assertEqual(intent.tool_manifest_digest, registry_digest)
        self.assertEqual(intent.tool_manifest_digest, tools.tool_manifest_digest("lookup"))
        self.assertEqual(intent.action_kind, "tool")

    async def test_post_append_manifest_drift_is_terminal_without_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original_invocations = 0
            replacement_invocations = 0
            tools = ToolRegistry()

            async def original(query: str) -> dict[str, Any]:
                nonlocal original_invocations
                original_invocations += 1
                return {"output": query}

            async def replacement(query: str) -> dict[str, Any]:
                nonlocal replacement_invocations
                replacement_invocations += 1
                return {"output": f"replacement:{query}"}

            tools.register("lookup", original, schema=_LOOKUP_SCHEMA)
            memory = MemoryStore(db_path=os.path.join(tmp, "post-append-drift.db"))
            runtime = _ReregisterAfterIntentAppendRuntime(
                llm=_ToolThenAnswerLLM('{"query":"bounded"}'),
                tools=tools,
                memory=memory,
                replacement=replacement,
            )
            await runtime.initialize()
            try:
                with self.assertRaisesRegex(RuntimeError, "after intent persistence"):
                    await runtime.run_episode(InputEvent(content="look this up", source="autonomous"))
                history = await memory.cognitive_event_history()
                unresolved = await memory.unresolved_execution_intents()
                safe_mode = runtime.execution_safe_mode
            finally:
                await runtime.close()

        self.assertEqual(original_invocations, 0)
        self.assertEqual(replacement_invocations, 0)
        self.assertEqual(unresolved, [])
        self.assertFalse(safe_mode)
        intent_event = next(event for event in history if event["event_type"] == "execution_intent")
        outcome_event = next(event for event in history if event["event_type"] == "execution_outcome")
        intent = ExecutionIntent.from_dict(intent_event["payload"])
        outcome = ExecutionOutcome.from_dict(outcome_event["payload"])
        self.assertEqual(outcome.execution_id, intent.execution_id)
        self.assertFalse(outcome.executed)
        self.assertIsNone(outcome.succeeded)
        self.assertEqual(outcome.disposition, "not_executed")
        self.assertEqual(outcome.reason_code, "dispatch_identity_changed")

    async def test_later_empty_wait_keeps_observed_tool_as_action_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            invocations = 0
            tools = ToolRegistry()

            async def lookup(query: str) -> dict[str, Any]:
                nonlocal invocations
                invocations += 1
                return {"output": f"observed:{query}"}

            tools.register("lookup", lookup, schema=_LOOKUP_SCHEMA, capabilities={"local_read"})
            memory = MemoryStore(db_path=os.path.join(tmp, "tool-then-wait.db"))
            llm = _ToolThenEmptyLLM()
            runtime = V3CognitiveRuntime(
                llm=llm,
                tools=tools,
                memory=memory,
                max_ticks=2,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content="look this up", source="autonomous"))
            finally:
                await runtime.close()

        self.assertEqual(invocations, 1)
        self.assertEqual(llm.calls, 2)
        self.assertEqual(result.selected_action, "wait")
        self.assertEqual(len(result.action_outcomes), 1)
        action_outcome = result.action_outcomes[0]
        tool_proposal = next(event for event in result.causal_trace if event["event_type"] == "tool_proposal")
        self.assertEqual(action_outcome["action"], "tool")
        self.assertTrue(action_outcome["observed"])
        self.assertTrue(action_outcome["succeeded"])
        self.assertTrue(action_outcome["learning_eligible"])
        self.assertEqual(action_outcome["selected_action_kind"], "tool")
        self.assertEqual(action_outcome["proposal_id"], tool_proposal["payload"]["proposal_id"])


if __name__ == "__main__":
    unittest.main()

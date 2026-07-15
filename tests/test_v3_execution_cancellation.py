from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from typing import Any

from conscio.core.cognition import InputEvent
from conscio.memory.store import CognitiveEventAppendResult, MemoryStore
from conscio.tools.registry import ToolRegistry
from conscio.v3.contracts import ExecutionIntent, ExecutionOutcome
from conscio.v3.runtime import V3CognitiveRuntime

LOOKUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"subject": {"type": "string"}},
    "required": ["subject"],
    "additionalProperties": False,
}


class CancelAfterDurableAppendStore(MemoryStore):
    """Raise one cancellation only after the selected record is committed."""

    def __init__(self, *, db_path: str, cancel_event_type: str) -> None:
        super().__init__(db_path=db_path)
        self.cancel_event_type = cancel_event_type
        self.cancel_count = 0

    async def append_cognitive_event_idempotent(
        self,
        event: Any,
    ) -> CognitiveEventAppendResult:
        result = await super().append_cognitive_event_idempotent(event)
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        if data.get("event_type") == self.cancel_event_type and self.cancel_count == 0:
            self.cancel_count += 1
            raise asyncio.CancelledError
        return result


class ToolThenAnswerLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del messages, kwargs
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "lookup-call",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"subject":"bounded"}',
                        },
                    }
                ],
            }
        return {"role": "assistant", "content": "done"}


class V3ExecutionCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_after_intent_commit_resyncs_orphan_and_blocks_dispatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = CancelAfterDurableAppendStore(
                db_path=os.path.join(tmp, "intent-cancel.db"),
                cancel_event_type="execution_intent",
            )
            tools = ToolRegistry()
            executions = 0

            async def lookup(subject: str) -> dict[str, Any]:
                nonlocal executions
                executions += 1
                return {"output": subject}

            tools.register("lookup", lookup, schema=LOOKUP_SCHEMA)
            runtime = V3CognitiveRuntime(
                llm=ToolThenAnswerLLM(),
                tools=tools,
                memory=memory,
            )
            await runtime.initialize()
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await runtime.run_episode(InputEvent(content="look it up", source="autonomous"))

                unresolved = await memory.unresolved_execution_intents()
                history = await memory.cognitive_event_history()
                safe_mode = runtime.execution_safe_mode
                unresolved_ids = runtime.unresolved_execution_ids
            finally:
                await runtime.close()

        journal = [event for event in history if event["event_type"].startswith("execution_")]
        self.assertEqual(memory.cancel_count, 1)
        self.assertEqual(executions, 0)
        self.assertTrue(safe_mode)
        self.assertEqual(len(unresolved), 1)
        intent = ExecutionIntent.from_dict(unresolved[0]["payload"])
        self.assertEqual(unresolved_ids, (intent.execution_id,))
        self.assertEqual(
            [event["event_type"] for event in journal],
            ["execution_intent"],
        )
        self.assertEqual(journal[0]["event_id"], intent.event_id)

    async def test_cancellation_after_outcome_commit_resyncs_terminal_without_false_orphan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = CancelAfterDurableAppendStore(
                db_path=os.path.join(tmp, "outcome-cancel.db"),
                cancel_event_type="execution_outcome",
            )
            tools = ToolRegistry()
            executions = 0

            async def lookup(subject: str) -> dict[str, Any]:
                nonlocal executions
                executions += 1
                return {"output": subject}

            tools.register("lookup", lookup, schema=LOOKUP_SCHEMA)
            llm = ToolThenAnswerLLM()
            runtime = V3CognitiveRuntime(llm=llm, tools=tools, memory=memory)
            await runtime.initialize()
            try:
                with self.assertRaises(asyncio.CancelledError):
                    await runtime.run_episode(InputEvent(content="look it up", source="autonomous"))

                unresolved_after_cancel = await memory.unresolved_execution_intents()
                safe_mode_after_cancel = runtime.execution_safe_mode
                unresolved_ids_after_cancel = runtime.unresolved_execution_ids
                history = await memory.cognitive_event_history()

                resumed = await runtime.run_episode("finish without another tool")
                unresolved_after_resume = await memory.unresolved_execution_intents()
            finally:
                await runtime.close()

        journal = [event for event in history if event["event_type"] in {"execution_intent", "execution_outcome"}]
        self.assertEqual(memory.cancel_count, 1)
        self.assertEqual(executions, 1)
        self.assertFalse(safe_mode_after_cancel)
        self.assertEqual(unresolved_ids_after_cancel, ())
        self.assertEqual(unresolved_after_cancel, [])
        self.assertEqual(unresolved_after_resume, [])
        self.assertEqual(resumed.output, "done")
        self.assertEqual(
            [event["event_type"] for event in journal],
            ["execution_intent", "execution_outcome"],
        )
        intent = ExecutionIntent.from_dict(journal[0]["payload"])
        outcome = ExecutionOutcome.from_dict(journal[1]["payload"])
        self.assertEqual(outcome.execution_id, intent.execution_id)
        self.assertEqual(outcome.intent_digest, intent.intent_digest)
        self.assertEqual(journal[1]["parent_event_id"], intent.event_id)


if __name__ == "__main__":
    unittest.main()

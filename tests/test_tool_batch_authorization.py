from __future__ import annotations

import unittest
from typing import Any

from conscio.core.tool_loop import ToolLoop, ToolLoopSession, ToolRequest
from conscio.core.workspace import Workspace
from conscio.tools.registry import PolicyToolRegistry, ToolRegistry


def _schemas(*names: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _call(call_id: str, name: str, arguments: str = "{}") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class ScriptedLLM:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((list(messages), dict(kwargs)))
        return self.responses.pop(0)


def _decision(
    action: str,
    *,
    selected_call_id: str | None = None,
    proposal_id: str | None = None,
    proposal_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    identifiers = proposal_ids if proposal_ids is not None else {"a": "proposal-a"}
    result: dict[str, Any] = {
        "action": action,
        "reason": f"selected {action}",
        "selected_call_id": selected_call_id,
        "proposal_ids": identifiers,
    }
    if selected_call_id is not None:
        result["execution_id"] = f"execution-{selected_call_id}"
        result["proposal_id"] = proposal_id or f"proposal-{selected_call_id}"
    return result


class BatchAuthorizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_batch_executes_at_most_one_and_completes_protocol(self) -> None:
        calls: list[str] = []
        tools = ToolRegistry()

        async def first() -> dict[str, Any]:
            calls.append("first")
            return {"output": "first-result"}

        async def second() -> dict[str, Any]:
            calls.append("second")
            return {"output": "second-result"}

        tools.register("first", first)
        tools.register("second", second)
        llm = ScriptedLLM(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("a", "first"), _call("b", "second")],
            },
            {"role": "assistant", "content": "finished"},
        )
        observations: list[tuple[str, dict[str, Any]]] = []
        outcomes: list[tuple[str, dict[str, Any]]] = []

        async def authorize(
            requests: tuple[tuple[str, ToolRequest], ...],
        ) -> dict[str, Any]:
            if not requests:
                return _decision("respond", proposal_ids={})
            self.assertEqual([call_id for call_id, _ in requests], ["a", "b"])
            return _decision(
                "tool",
                selected_call_id="b",
                proposal_ids={"a": "proposal-a", "b": "proposal-b"},
            )

        async def observe(request: ToolRequest, result: dict[str, Any]) -> None:
            observations.append((request.name, result))

        async def outcome(request: ToolRequest, result: dict[str, Any]) -> None:
            outcomes.append((request.name, result))

        session = ToolLoopSession(
            llm=llm,
            tools=tools,
            tool_schemas=_schemas("first", "second"),
            messages=[{"role": "user", "content": "go"}],
            pre_tool_batch_hook=authorize,
            on_tool_observation=observe,
            on_selected_action_outcome=outcome,
        )

        result = await session.step(Workspace(), max_rounds=2)

        self.assertEqual(result.kind, "final")
        self.assertEqual(calls, ["second"])
        self.assertEqual(session.tool_rounds, 1)
        self.assertEqual(len(session.tool_proposals), 2)
        self.assertEqual([item[0] for item in observations], ["second"])
        self.assertEqual([item[0] for item in outcomes], ["second"])
        assistant = next(message for message in session.messages if message.get("tool_calls"))
        replies = [message for message in session.messages if message.get("role") == "tool"]
        self.assertEqual(len(assistant["tool_calls"]), 2)
        self.assertEqual({reply["tool_call_id"] for reply in replies}, {"a", "b"})
        self.assertIn("rejected", next(reply["content"] for reply in replies if reply["tool_call_id"] == "a"))

    async def test_normalized_intent_hook_succeeds_before_dispatch(self) -> None:
        order: list[str] = []
        tools = ToolRegistry()

        async def lookup(value: int) -> dict[str, Any]:
            order.append(f"tool:{value}")
            return {"output": "ok"}

        tools.register("lookup", lookup)
        llm = ScriptedLLM(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("a", "lookup", '{"value":2}')],
            },
            {"role": "assistant", "content": "done"},
        )

        async def authorize(requests: tuple[tuple[str, ToolRequest], ...]) -> dict[str, Any]:
            if not requests:
                return _decision("respond", proposal_ids={})
            return _decision("tool", selected_call_id="a")

        async def intent(request: ToolRequest, decision: dict[str, Any]) -> None:
            self.assertEqual(request.args, {"value": 2})
            self.assertEqual(decision["proposal_ids"], {"a": "proposal-a"})
            order.append("intent")

        session = ToolLoopSession(
            llm=llm,
            tools=tools,
            tool_schemas=_schemas("lookup"),
            messages=[{"role": "user", "content": "go"}],
            pre_tool_batch_hook=authorize,
            on_selected_action_intent=intent,
        )
        await session.step(Workspace(), max_rounds=2)

        self.assertEqual(order, ["intent", "tool:2"])
        self.assertEqual([request.args for request in session.tool_requests], [{"value": 2}])
        self.assertEqual([request.args for request in session.tool_proposals], [{"value": 2}])

    async def test_failed_intent_hook_prevents_dispatch(self) -> None:
        executions = 0
        tools = ToolRegistry()

        async def lookup() -> dict[str, Any]:
            nonlocal executions
            executions += 1
            return {"output": "unexpected"}

        tools.register("lookup", lookup)
        llm = ScriptedLLM({"role": "assistant", "content": "", "tool_calls": [_call("a", "lookup")]})

        async def authorize(requests: tuple[tuple[str, ToolRequest], ...]) -> dict[str, Any]:
            if not requests:
                return _decision("respond", proposal_ids={})
            return _decision("tool", selected_call_id="a")

        async def fail_intent(_: ToolRequest, __: dict[str, Any]) -> None:
            raise RuntimeError("intent journal unavailable")

        session = ToolLoopSession(
            llm=llm,
            tools=tools,
            tool_schemas=_schemas("lookup"),
            messages=[{"role": "user", "content": "go"}],
            pre_tool_batch_hook=authorize,
            on_selected_action_intent=fail_intent,
        )
        with self.assertRaisesRegex(RuntimeError, "intent journal unavailable"):
            await session.step(Workspace())

        self.assertEqual(executions, 0)
        self.assertEqual(session.tool_requests, [])
        self.assertEqual([request.name for request in session.tool_proposals], ["lookup"])

    async def test_respond_emits_the_exact_scored_response(self) -> None:
        llm = ScriptedLLM(
            {
                "role": "assistant",
                "content": "answer without lookup",
                "tool_calls": [_call("a", "lookup")],
            },
        )
        tools = ToolRegistry()
        executions = 0

        async def lookup() -> dict[str, Any]:
            nonlocal executions
            executions += 1
            return {"output": "lookup"}

        tools.register("lookup", lookup)

        async def authorize(_: tuple[tuple[str, ToolRequest], ...]) -> dict[str, Any]:
            return _decision("respond")

        session = ToolLoopSession(
            llm=llm,
            tools=tools,
            tool_schemas=_schemas("lookup"),
            messages=[{"role": "user", "content": "go"}],
            pre_tool_batch_hook=authorize,
        )
        result = await session.step(Workspace(), max_rounds=2)

        self.assertEqual(result.kind, "final")
        self.assertEqual(result.text, "answer without lookup")
        self.assertEqual(executions, 0)
        self.assertEqual(session.tool_rounds, 1)
        self.assertEqual(len(session.tool_proposals), 1)
        self.assertIn("tools", llm.calls[0][1])
        self.assertEqual(len(llm.calls), 1)

    async def test_content_only_response_passes_through_action_competition(self) -> None:
        llm = ScriptedLLM({"role": "assistant", "content": "exact answer"})
        seen: list[tuple[tuple[str, ToolRequest], ...]] = []

        async def authorize(
            requests: tuple[tuple[str, ToolRequest], ...],
        ) -> dict[str, Any]:
            seen.append(requests)
            return _decision("respond", proposal_ids={})

        session = ToolLoopSession(
            llm=llm,
            tools=ToolRegistry(),
            tool_schemas=_schemas("lookup"),
            messages=[{"role": "user", "content": "go"}],
            pre_tool_batch_hook=authorize,
        )

        result = await session.step(Workspace())

        self.assertEqual(result.kind, "final")
        self.assertEqual(result.text, "exact answer")
        self.assertEqual(seen, [()])
        self.assertEqual(session.tool_rounds, 0)
        self.assertEqual(session.tool_proposals, [])

    async def test_wait_is_terminal_and_executes_nothing(self) -> None:
        llm = ScriptedLLM({"role": "assistant", "content": "", "tool_calls": [_call("a", "lookup")]})
        tools = ToolRegistry()
        executions = 0

        async def lookup() -> dict[str, Any]:
            nonlocal executions
            executions += 1
            return {"output": "lookup"}

        tools.register("lookup", lookup)

        async def authorize(_: tuple[tuple[str, ToolRequest], ...]) -> dict[str, Any]:
            return _decision("wait")

        session = ToolLoopSession(
            llm=llm,
            tools=tools,
            tool_schemas=_schemas("lookup"),
            messages=[{"role": "user", "content": "go"}],
            pre_tool_batch_hook=authorize,
        )
        result = await session.step(Workspace())

        self.assertEqual(result.kind, "wait")
        self.assertEqual(executions, 0)

    async def test_control_call_cannot_bypass_batch_authorization(self) -> None:
        seen: list[str] = []
        llm = ScriptedLLM(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("a", "ask_user", '{"question":"Which file?"}')],
            }
        )

        async def authorize(
            requests: tuple[tuple[str, ToolRequest], ...],
        ) -> dict[str, Any]:
            seen.extend(request.name for _, request in requests)
            return _decision("control", selected_call_id="a")

        session = ToolLoopSession(
            llm=llm,
            tools=ToolRegistry(),
            tool_schemas=_schemas("ask_user"),
            messages=[{"role": "user", "content": "go"}],
            pre_tool_batch_hook=authorize,
        )
        result = await session.step(Workspace())

        self.assertEqual(seen, ["ask_user"])
        self.assertEqual(result.kind, "control")
        self.assertEqual(result.control, "ask")
        self.assertEqual(result.text, "Which file?")

    async def test_missing_or_malformed_batch_decision_fails_closed(self) -> None:
        tools = ToolRegistry()
        executions = 0

        async def lookup() -> dict[str, Any]:
            nonlocal executions
            executions += 1
            return {"output": "lookup"}

        tools.register("lookup", lookup)
        for malformed in (None, True, {}, {"action": "tool", "reason": "yes"}):
            llm = ScriptedLLM({"role": "assistant", "content": "", "tool_calls": [_call("a", "lookup")]})

            async def authorize(
                _: tuple[tuple[str, ToolRequest], ...],
                value: Any = malformed,
            ) -> Any:
                return value

            session = ToolLoopSession(
                llm=llm,
                tools=tools,
                tool_schemas=_schemas("lookup"),
                messages=[{"role": "user", "content": "go"}],
                pre_tool_batch_hook=authorize,
            )
            with self.assertRaises((ValueError, AttributeError)):
                await session.step(Workspace())
        self.assertEqual(executions, 0)

    async def test_policy_denial_is_selected_but_unexecuted(self) -> None:
        tools = PolicyToolRegistry(denied_tools=["lookup"])
        executed = 0

        async def lookup() -> dict[str, Any]:
            nonlocal executed
            executed += 1
            return {"output": "lookup"}

        tools.register("lookup", lookup)
        llm = ScriptedLLM(
            {"role": "assistant", "content": "", "tool_calls": [_call("a", "lookup")]},
            {"role": "assistant", "content": "done"},
        )
        observed: list[dict[str, Any]] = []

        async def authorize(requests: tuple[tuple[str, ToolRequest], ...]) -> dict[str, Any]:
            if not requests:
                return _decision("respond", proposal_ids={})
            return _decision("tool", selected_call_id="a")

        async def observe(_: ToolRequest, result: dict[str, Any]) -> None:
            observed.append(result)

        session = ToolLoopSession(
            llm=llm,
            tools=tools,
            tool_schemas=_schemas("lookup"),
            messages=[{"role": "user", "content": "go"}],
            pre_tool_batch_hook=authorize,
            on_tool_observation=observe,
        )
        workspace = Workspace()
        result = await session.step(workspace, max_rounds=2)

        self.assertEqual(result.kind, "final")
        self.assertEqual(executed, 0)
        self.assertTrue(observed[0]["policy_denied"])
        self.assertFalse(observed[0]["executed"])
        self.assertEqual(workspace.read(), [])

    async def test_outcome_callback_failure_leaves_complete_protocol_without_reexecution(self) -> None:
        tools = ToolRegistry()
        executions = 0

        async def lookup() -> dict[str, Any]:
            nonlocal executions
            executions += 1
            return {"output": "once"}

        tools.register("lookup", lookup)
        llm = ScriptedLLM(
            {"role": "assistant", "content": "", "tool_calls": [_call("a", "lookup")]},
            {"role": "assistant", "content": "resumed"},
        )

        async def authorize(requests: tuple[tuple[str, ToolRequest], ...]) -> dict[str, Any]:
            if not requests:
                return _decision("respond", proposal_ids={})
            return _decision("tool", selected_call_id="a")

        async def fail_outcome(_: ToolRequest, __: dict[str, Any]) -> None:
            raise RuntimeError("journal unavailable")

        session = ToolLoopSession(
            llm=llm,
            tools=tools,
            tool_schemas=_schemas("lookup"),
            messages=[{"role": "user", "content": "go"}],
            pre_tool_batch_hook=authorize,
            on_selected_action_outcome=fail_outcome,
        )
        with self.assertRaisesRegex(RuntimeError, "journal unavailable"):
            await session.step(Workspace())

        assistant = next(message for message in session.messages if message.get("tool_calls"))
        replies = [message for message in session.messages if message.get("role") == "tool"]
        self.assertEqual([call["id"] for call in assistant["tool_calls"]], ["a"])
        self.assertEqual([reply["tool_call_id"] for reply in replies], ["a"])
        resumed = await session.step(Workspace())
        self.assertEqual(resumed.text, "resumed")
        self.assertEqual(executions, 1)

    async def test_native_unknown_malformed_and_nonfinite_calls_are_filtered(self) -> None:
        response = {
            "content": "",
            "tool_calls": [
                _call("unknown", "fabricated"),
                _call("malformed", "lookup", "not-json"),
                _call("nonfinite", "lookup", '{"value":NaN}'),
                _call("valid", "lookup", '{"value":2}'),
            ],
        }

        parsed = ToolLoop._tool_requests(response, {"lookup"})
        self.assertEqual(
            [(call_id, request.name, request.args) for call_id, request in parsed],
            [("valid", "lookup", {"value": 2})],
        )


if __name__ == "__main__":
    unittest.main()

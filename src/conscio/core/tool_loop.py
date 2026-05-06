"""LLM tool-use loop, factored out so it can drive both the user-chat and
autonomous-action paths."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from conscio.core.workspace import EntryType, Workspace, WorkspaceEntry


@dataclass
class ToolRequest:
    name: str
    args: dict[str, Any]


@dataclass
class ToolLoopResult:
    final_text: str | None
    tool_requests: list[ToolRequest] = field(default_factory=list)
    rounds: int = 0
    limit_reached: bool = False


ToolObservationCallback = Callable[[ToolRequest, dict[str, Any]], Awaitable[None]]


class ToolLoop:
    """Iteratively chat with the LLM, executing tool calls until a final answer or budget runs out."""

    DEFAULT_LIMIT_MESSAGE = (
        "Tool-use limit reached for this turn. Stop calling tools and provide a concise final answer "
        "using the tool observations above."
    )

    def __init__(
        self,
        *,
        llm: Any,
        tools: Any,
        max_rounds: int = 32,
        temperature: float = 0.4,
        max_tokens: int = 600,
        limit_message: str = DEFAULT_LIMIT_MESSAGE,
        on_tool_observation: ToolObservationCallback | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_rounds = max(1, int(max_rounds))
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.limit_message = limit_message
        self.on_tool_observation = on_tool_observation

    async def run(
        self,
        messages: list[dict[str, Any]],
        workspace: Workspace,
        tool_schemas: list[dict[str, Any]] | None,
    ) -> ToolLoopResult:
        if self.llm is None:
            return ToolLoopResult(final_text=None)
        if not tool_schemas:
            response = await self.llm.chat_async(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = str(response.get("content") or "").strip()
            return ToolLoopResult(final_text=content or None, rounds=1)

        tool_requests: list[ToolRequest] = []
        rounds = 0
        for _ in range(self.max_rounds):
            rounds += 1
            response = await self.llm.chat_async(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tool_schemas,
            )
            request = self._tool_request(response)
            if request is None:
                content = str(response.get("content") or "").strip()
                return ToolLoopResult(final_text=content or None, tool_requests=tool_requests, rounds=rounds)
            tool_requests.append(request)
            result = await self._execute_tool(request, workspace)
            if self.on_tool_observation is not None:
                await self.on_tool_observation(request, result)
            messages.append(self._assistant_tool_call_message(response, request))
            messages.append({
                "role": "tool",
                "tool_call_id": self._tool_call_id(response),
                "name": request.name,
                "content": str(result.get("output", ""))[:8000],
            })

        # Limit reached — ask for one final answer.
        messages.append({"role": "user", "content": self.limit_message})
        response = await self.llm.chat_async(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        rounds += 1
        content = str(response.get("content") or "").strip()
        if not content:
            observations = [
                entry.content
                for entry in workspace.read(limit=100, type_filter={EntryType.OBSERVATION})
                if entry.source == "tool"
            ]
            content = observations[-1] if observations else "Tool-use limit reached before a final answer."
        return ToolLoopResult(
            final_text=content or None,
            tool_requests=tool_requests,
            rounds=rounds,
            limit_reached=True,
        )

    async def _execute_tool(self, request: ToolRequest, workspace: Workspace) -> dict[str, Any]:
        if self.tools is None:
            return {"output": "Tool registry is unavailable.", "error": True}
        result = await self.tools.call(request.name, request.args)
        output = str(result.get("output", ""))
        workspace.write(
            f"Tool {request.name} returned: {output[:1000]}",
            source="tool",
            type=EntryType.OBSERVATION,
            priority=6,
            salience=0.72,
            confidence=0.9 if not result.get("error") else 0.35,
            novelty=0.75,
            urgency=0.45,
            metadata={"source": "tool", "event_type": "tool_result", "tool": request.name, "result": result},
        )
        return result

    @staticmethod
    def _tool_request(response: dict[str, Any]) -> ToolRequest | None:
        calls = response.get("tool_calls") or []
        if not calls:
            return None
        call = calls[0]
        function = call.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            return None
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"input": str(function.get("arguments") or "")}
        if not isinstance(args, dict):
            args = {"input": args}
        return ToolRequest(name=name, args=args)

    @staticmethod
    def _assistant_tool_call_message(response: dict[str, Any], request: ToolRequest) -> dict[str, Any]:
        calls = response.get("tool_calls") or []
        if calls:
            return {"role": "assistant", "content": response.get("content") or "", "tool_calls": calls}
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": request.name, "arguments": json.dumps(request.args)},
                }
            ],
        }

    @staticmethod
    def _tool_call_id(response: dict[str, Any]) -> str:
        calls = response.get("tool_calls") or []
        if calls and calls[0].get("id"):
            return str(calls[0]["id"])
        return "call-1"


def latest_tool_observation(workspace: Workspace, name: str) -> WorkspaceEntry | None:
    for entry in reversed(workspace.read(limit=100, type_filter={EntryType.OBSERVATION})):
        if entry.source == "tool" and entry.metadata.get("tool") == name:
            return entry
    return None

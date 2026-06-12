"""LLM tool-use loop, factored out so it can drive both the user-chat and
autonomous-action paths."""
from __future__ import annotations

import json
import re
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
        max_tokens: int = 2400,
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
        known_names = _schema_tool_names(tool_schemas)
        rounds = 0
        for _ in range(self.max_rounds):
            rounds += 1
            response = await self.llm.chat_async(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tool_schemas,
            )
            request = self._tool_request(response, known_names)
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
    def _tool_request(response: dict[str, Any], known_names: set[str] | None = None) -> ToolRequest | None:
        calls = response.get("tool_calls") or []
        if not calls:
            return _parse_dsml_tool_call(str(response.get("content") or ""), known_names=known_names)
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


# Recovers DeepSeek-style native tool-call markers that leak into the assistant
# `content` field instead of arriving as proper OpenAI `tool_calls`. Tokens
# observed: `<｜tool_calls_begin｜>`, `<｜tool_call_begin｜>`, `<｜tool_sep｜>`,
# `<｜tool_call_end｜>`, `<｜tool_calls_end｜>` (and a `<｜DSML｜tool_calls>` variant).
_DSML_SEP = re.compile(r"<\s*[｜|]\s*tool[_ ]?sep\s*[｜|]\s*>", re.IGNORECASE)
_DSML_CALL_BEGIN = re.compile(r"<\s*[｜|]\s*tool[_ ]?call[_ ]?begin\s*[｜|]\s*>", re.IGNORECASE)
_DSML_CALL_END = re.compile(r"<\s*[｜|]\s*tool[_ ]?call[_ ]?end\s*[｜|]\s*>", re.IGNORECASE)
_DSML_CALLS_BEGIN = re.compile(r"<\s*[｜|]\s*tool[_ ]?calls[_ ]?begin\s*[｜|]\s*>", re.IGNORECASE)
_DSML_CALLS_END = re.compile(r"<\s*[｜|]\s*tool[_ ]?calls[_ ]?end\s*[｜|]\s*>", re.IGNORECASE)
_DSML_HINT = re.compile(r"<\s*[｜|]\s*(?:dsml\s*[｜|]?\s*)?tool[_ ]?calls?", re.IGNORECASE)
_DSML_MARKER = re.compile(r"<\s*[｜|][^<>]*>")
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _schema_tool_names(tool_schemas: list[dict[str, Any]] | None) -> set[str] | None:
    """Tool names offered to the model this round — the only names a recovered
    DSML call may legitimately carry."""
    if not tool_schemas:
        return None
    names: set[str] = set()
    for schema in tool_schemas:
        function = schema.get("function") or {}
        name = str(function.get("name") or schema.get("name") or "").strip()
        if name:
            names.add(name)
    return names or None


def _parse_dsml_tool_call(content: str, known_names: set[str] | None = None) -> ToolRequest | None:
    if not content or not _DSML_HINT.search(content):
        return None
    # Restrict to the first call's region: prefer per-call markers, fall back
    # to the plural wrapper so prose before the leak never enters the parse.
    begin_match = _DSML_CALL_BEGIN.search(content) or _DSML_CALLS_BEGIN.search(content)
    region = content[begin_match.end():] if begin_match else content
    end_match = _DSML_CALL_END.search(region) or _DSML_CALLS_END.search(region)
    if end_match:
        region = region[: end_match.start()]
    sep_match = _DSML_SEP.search(region)
    if sep_match:
        # Strip a leading role token like "function" before <｜tool_sep｜>.
        name_and_args = region[sep_match.end():]
    else:
        name_and_args = region
    fence = _JSON_FENCE.search(name_and_args)
    if fence:
        prefix = name_and_args[: fence.start()].strip()
        args_raw = fence.group(1)
    else:
        brace = name_and_args.find("{")
        if brace == -1:
            return None
        prefix = name_and_args[:brace].strip()
        args_raw = _extract_balanced_json(name_and_args, brace)
        if args_raw is None:
            return None
    # Leftover marker tokens (e.g. a stray plural wrapper) are noise, not names.
    prefix = _DSML_MARKER.sub("\n", prefix)
    name = next(
        (line.strip().strip("`").strip() for line in prefix.splitlines() if line.strip()),
        "",
    )
    if not name:
        return None
    if known_names is not None and name not in known_names:
        # A final text answer that merely mentions the markers parses into
        # garbage here; never convert it into a phantom tool call.
        return None
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError:
        args = {"input": args_raw}
    if not isinstance(args, dict):
        args = {"input": args}
    return ToolRequest(name=name, args=args)


def _extract_balanced_json(text: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None

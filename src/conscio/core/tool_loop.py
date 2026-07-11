"""LLM tool-use loop, factored out so it can drive both the user-chat and
autonomous-action paths."""
from __future__ import annotations

import copy
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from conscio.core.workspace import EntryType, Workspace

DEFAULT_LIMIT_MESSAGE = (
    "Tool-use limit reached for this turn. Stop calling tools and provide a concise final answer "
    "using the tool observations above."
)


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


@dataclass
class StepResult:
    kind: Literal["tool", "final", "control", "empty", "exhausted"]
    text: str = ""
    tool_request: ToolRequest | None = None
    tool_result: dict[str, Any] | None = None
    control: str = ""  # "ask" | "refuse"
    rounds_used: int = 0
    limit_reached: bool = False


ToolObservationCallback = Callable[[ToolRequest, dict[str, Any]], Awaitable[None]]
PreToolHook = Callable[[ToolRequest], Awaitable[Any]]

_CONTROL_KINDS = {"ask_user": "ask", "refuse": "refuse"}

# Tools whose output is untrusted external data: spotlighted with explicit
# delimiters before it enters the workspace/prompt (quarantine defense).
WEB_CONTENT_TOOLS = frozenset({"web_fetch", "web_search"})

# Tools that can reach the network *without* going through the spotlighted web
# tools (curl/wget/python one-liners under unsafe_autonomy). Their calls are
# inspected so taint tracking cannot be bypassed by routing a fetch through a
# shell instead of web_fetch.
NETWORK_CAPABLE_TOOLS = frozenset({"bash", "execute_code"})
EXTERNAL_CONTENT_CAPABILITY = "external_content"
NETWORK_READ_CAPABILITY = "network_read"

UNTRUSTED_WEB_BEGIN = "<<UNTRUSTED_WEB_CONTENT url={url}>>"
UNTRUSTED_WEB_END = "<<END_UNTRUSTED>>"
_UNTRUSTED_BEGIN_PREFIX = "<<UNTRUSTED_WEB_CONTENT"

# Anything in *fetched* content that looks like one of our quarantine
# delimiters — `<<...UNTRUSTED_WEB_CONTENT...>>` / `<<...END_UNTRUSTED...>>` in
# any casing/spacing. A page must never be able to forge an early
# <<END_UNTRUSTED>> (escaping the quarantine block) or open a fake one.
_FORGED_DELIMITER_RE = re.compile(
    r"<<[^<>]*?(?:UNTRUSTED_WEB_CONTENT|END_UNTRUSTED)[^<>]*?>>",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s'\"<>`]+", re.IGNORECASE)
_NETWORK_CLIENT_RE = re.compile(
    r"\b(?:curl|wget|aria2c|httpie|nc|ncat|netcat|socat|sftp|scp|ssh|telnet)\b"
    r"|openssl\s+s_client|urllib|requests\.|httpx|aiohttp|http\.client|socket\.",
    re.IGNORECASE,
)


def web_request_url(request: ToolRequest) -> str:
    """Best-effort URL/query identifying what a web tool touched."""
    args = request.args or {}
    return str(args.get("url") or args.get("query") or args.get("input") or "").strip()


def web_taint_origin(request: ToolRequest, result: dict[str, Any] | None = None) -> str | None:
    return external_taint_origin(request, result)


def external_taint_origin(
    request: ToolRequest,
    result: dict[str, Any] | None = None,
    *,
    capabilities: set[str] | frozenset[str] | None = None,
) -> str | None:
    """Origin string when a tool call touched web content, else None.

    Covers the spotlighted web tools plus network-capable tools
    (bash/execute_code): a `curl`/`wget`/python fetch routed through a shell
    pulls untrusted bytes into the workspace just the same, so it must taint
    the episode too — otherwise the whole taint pipeline is bypassable.
    Conservative over-tainting is the accepted trade-off."""
    name = request.name
    caps = capabilities or frozenset()
    if name in WEB_CONTENT_TOOLS or EXTERNAL_CONTENT_CAPABILITY in caps:
        # MCP and other non-URL external tools have no url/query arg; fall back
        # to the tool name so the fact origin is still informative ("web:mcp__x__y").
        return web_request_url(request) or request.name
    if name in NETWORK_CAPABLE_TOOLS or NETWORK_READ_CAPABILITY in caps:
        args_blob = json.dumps(request.args or {}, ensure_ascii=False)
        output = str((result or {}).get("output", ""))
        url_match = _URL_RE.search(args_blob) or _URL_RE.search(output)
        if url_match:
            return url_match.group(0)
        if _NETWORK_CLIENT_RE.search(args_blob):
            return f"{name}:network"
    return None


def neutralize_untrusted_delimiters(text: str) -> str:
    """Strip forged quarantine delimiters from untrusted content. Substitution
    repeats until a fixpoint so removals can never reassemble a new token."""
    previous = None
    while previous != text:
        previous = text
        text = _FORGED_DELIMITER_RE.sub("[forged-delimiter-removed]", text)
    return text


def truncate_spotlighted(text: str, limit: int) -> str:
    """Truncate tool output without dropping the closing quarantine delimiter.

    A plain ``[:limit]`` slice on a spotlighted web result cuts inside the page
    body and discards the trailing <<END_UNTRUSTED>>, leaving the model with an
    opened-but-never-closed quarantine block (so later trusted content visually
    falls inside it)."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    if _UNTRUSTED_BEGIN_PREFIX in head:
        suffix = f"\n[truncated]\n{UNTRUSTED_WEB_END}"
        return text[: max(0, limit - len(suffix))] + suffix
    return head


def _spotlight_web_output(request: ToolRequest, result: dict[str, Any]) -> dict[str, Any]:
    """Wrap web tool output in explicit data-delimiters, after neutralizing any
    delimiter tokens the page itself contains (delimiter forgery). Spotlighting
    is probabilistic, not a guarantee — a sufficiently clever page could still
    influence reasoning within an episode; the taint/trust pipeline bounds the
    blast radius rather than eliminating it."""
    output = neutralize_untrusted_delimiters(str(result.get("output", "")))
    begin = UNTRUSTED_WEB_BEGIN.format(url=web_request_url(request) or request.name)
    return {**result, "output": f"{begin}\n{output}\n{UNTRUSTED_WEB_END}"}


async def _execute_tool(tools: Any, request: ToolRequest, workspace: Workspace) -> dict[str, Any]:
    if tools is None:
        return {"output": "Tool registry is unavailable.", "error": True}
    try:
        result = await tools.call(request.name, request.args)
    except Exception as exc:  # noqa: BLE001 — one misbehaving tool must not abort the episode
        result = {"output": f"Tool {request.name} raised {type(exc).__name__}: {exc}", "error": True}
    capabilities = _tool_capabilities(tools, request.name)
    if request.name in WEB_CONTENT_TOOLS or EXTERNAL_CONTENT_CAPABILITY in capabilities:
        result = _spotlight_web_output(request, result)
    output = str(result.get("output", ""))
    workspace.write(
        f"Tool {request.name} returned: {truncate_spotlighted(output, 1000)}",
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


def _tool_capabilities(tools: Any, name: str) -> frozenset[str]:
    getter = getattr(tools, "tool_capabilities", None)
    if not callable(getter):
        return frozenset()
    try:
        return frozenset(str(item) for item in getter(name))
    except Exception:  # noqa: BLE001 — capability metadata is advisory
        return frozenset()


class ToolLoopSession:
    """A steppable LLM tool-use session. The message list only ever grows
    (append-only `inject()`), keeping the prefix cache warm across steps."""

    def __init__(
        self,
        *,
        llm: Any,
        tools: Any,
        tool_schemas: list[dict[str, Any]] | None,
        messages: list[dict[str, Any]],
        temperature: float = 0.4,
        max_tokens: int = 2400,
        max_total_rounds: int = 32,
        control_tool_names: frozenset[str] = frozenset({"ask_user", "refuse"}),
        on_tool_observation: ToolObservationCallback | None = None,
        pre_tool_hook: PreToolHook | None = None,
        on_stream_event: Callable[[dict[str, Any]], None] | None = None,
        limit_message: str = DEFAULT_LIMIT_MESSAGE,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_schemas = tool_schemas or None
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_total_rounds = max(1, int(max_total_rounds))
        self.control_tool_names = control_tool_names
        self.on_tool_observation = on_tool_observation
        self.pre_tool_hook = pre_tool_hook
        self.on_stream_event = on_stream_event
        self.limit_message = limit_message
        self.tool_requests: list[ToolRequest] = []
        self.model_inputs: list[dict[str, Any]] = []
        self._known_names = _schema_tool_names(self.tool_schemas)
        self._rounds = 0
        self._closed = False
        self._streamed_this_round = 0

    @property
    def rounds_used(self) -> int:
        return self._rounds

    @property
    def exhausted(self) -> bool:
        return self._closed or self._rounds >= self.max_total_rounds

    @property
    def closed(self) -> bool:
        """True once the session can produce no further output (a forced
        final was already emitted). ``exhausted`` merely means the round
        budget is spent — one forced-final step may still be owed."""
        return self._closed

    def inject(self, content: str, role: str = "user") -> None:
        """Append-only context update: cache-safe, never rebuilds the list."""
        self.messages.append({"role": role, "content": content})

    async def _completion(self, *, tools: list[dict[str, Any]] | None, round_no: int) -> dict[str, Any]:
        """One LLM completion for the current message list. Streams when a
        token hook is set and the client supports it; otherwise identical to
        a plain ``chat_async`` call. The returned dict always has the
        non-streaming shape, so every downstream path (tool-call parsing,
        DSML recovery, message echo) is unchanged."""
        kwargs: dict[str, Any] = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            kwargs["tools"] = tools
        self.model_inputs.append({"messages": copy.deepcopy(self.messages), "kwargs": copy.deepcopy(kwargs)})
        self._streamed_this_round = 0
        stream_fn = getattr(self.llm, "chat_stream", None) if self.on_stream_event is not None else None
        if stream_fn is None:
            return await self.llm.chat_async(self.messages, **kwargs)
        gate = _StreamGate()
        done: dict[str, Any] = {}
        async for chunk in stream_fn(self.messages, **kwargs):
            kind = chunk.get("type")
            if kind == "content":
                self._emit_token(gate.feed(str(chunk.get("text") or "")), round_no)
            elif kind == "done":
                done = chunk
        self._emit_token(gate.finish(), round_no)
        response: dict[str, Any] = {"role": "assistant", "content": str(done.get("content") or "")}
        if done.get("tool_calls"):
            response["tool_calls"] = done["tool_calls"]
        return response

    def _emit_token(self, text: str, round_no: int) -> None:
        if not text or self.on_stream_event is None:
            return
        self._streamed_this_round += len(text)
        self.on_stream_event({"event": "token", "text": text, "round": round_no})

    def _emit_round_outcome(self, event: str) -> None:
        """`final` when streamed tokens were the answer; `discard` when the
        round turned out to be a tool call/control turn (clients drop the
        provisional text — the authoritative result follows out-of-band)."""
        if self.on_stream_event is not None and self._streamed_this_round:
            self.on_stream_event({"event": event, "round": self._rounds})
            self._streamed_this_round = 0

    async def step(
        self,
        workspace: Workspace,
        *,
        max_rounds: int = 1,
        should_stop: Callable[[], bool] | None = None,
    ) -> StepResult:
        if self.llm is None or self._closed:
            return StepResult(kind="exhausted" if self._closed else "empty", limit_reached=self._closed)
        rounds_this_step = 0
        last_request: ToolRequest | None = None
        last_result: dict[str, Any] | None = None
        for _ in range(max(1, int(max_rounds))):
            if self._rounds >= self.max_total_rounds:
                return await self._forced_final(workspace, rounds_this_step)
            self._rounds += 1
            rounds_this_step += 1
            response = await self._completion(tools=self.tool_schemas, round_no=self._rounds)
            requests = ToolLoop._tool_requests(response, self._known_names) if self.tool_schemas else []
            if not requests:
                content = str(response.get("content") or "").strip()
                if not content:
                    return StepResult(kind="empty", rounds_used=rounds_this_step)
                self.messages.append({"role": "assistant", "content": content})
                self._emit_round_outcome("final")
                return StepResult(kind="final", text=content, rounds_used=rounds_this_step)
            control = next(
                (req for _, req in requests if req.name in self.control_tool_names), None
            )
            if control is not None:
                self._emit_round_outcome("discard")
                text = self._control_text(control, response)
                self.messages.append({"role": "assistant", "content": text})
                return StepResult(
                    kind="control",
                    text=text,
                    tool_request=control,
                    control=_CONTROL_KINDS.get(control.name, control.name),
                    rounds_used=rounds_this_step,
                )
            # Execute every parallel call; the echoed assistant message then
            # carries exactly the executed calls, each followed by a matching
            # role:tool response — N tool_calls with fewer responses is a
            # protocol violation OpenAI-compatible backends reject with a 400.
            self._emit_round_outcome("discard")
            outcomes: list[tuple[str, ToolRequest, dict[str, Any]]] = []
            for call_id, request in requests:
                self.tool_requests.append(request)
                if self.pre_tool_hook is not None:
                    await self.pre_tool_hook(request)
                result = await _execute_tool(self.tools, request, workspace)
                if self.on_tool_observation is not None:
                    await self.on_tool_observation(request, result)
                outcomes.append((call_id, request, result))
            self.messages.append(
                ToolLoop._assistant_tool_call_message(
                    response, [(call_id, request) for call_id, request, _ in outcomes]
                )
            )
            for call_id, request, result in outcomes:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": request.name,
                    "content": truncate_spotlighted(str(result.get("output", "")), 8000),
                })
            _, last_request, last_result = outcomes[-1]
            # Round boundary: assistant echo + matching tool replies are already
            # appended, so stopping here keeps the message protocol consistent.
            if should_stop is not None and should_stop():
                break
        return StepResult(
            kind="tool",
            tool_request=last_request,
            tool_result=last_result,
            rounds_used=rounds_this_step,
        )

    async def _forced_final(self, workspace: Workspace, rounds_this_step: int) -> StepResult:
        """Total budget hit — append the limit message and force one final answer."""
        self.messages.append({"role": "user", "content": self.limit_message})
        response = await self._completion(tools=None, round_no=self._rounds + 1)
        self._rounds += 1
        rounds_this_step += 1
        self._closed = True
        content = str(response.get("content") or "").strip()
        if not content:
            observations = [
                entry.content
                for entry in workspace.read(limit=100, type_filter={EntryType.OBSERVATION})
                # Never promote spotlighted external content to the agent's own
                # final answer — that would hand the quarantined text authorship.
                if entry.source == "tool" and _UNTRUSTED_BEGIN_PREFIX not in entry.content
            ]
            # workspace.read sorts by (-priority, -timestamp), so [0] is the
            # highest-priority most-recent tool observation.
            content = observations[0] if observations else "Tool-use limit reached before a final answer."
        self.messages.append({"role": "assistant", "content": content})
        self._emit_round_outcome("final")
        return StepResult(kind="final", text=content, rounds_used=rounds_this_step, limit_reached=True)

    @staticmethod
    def _control_text(request: ToolRequest, response: dict[str, Any]) -> str:
        for key in ("question", "reason", "message", "text"):
            value = str(request.args.get(key) or "").strip()
            if value:
                return value
        return str(response.get("content") or "").strip()


class ToolLoop:
    """Iteratively chat with the LLM, executing tool calls until a final answer or budget runs out.

    Implemented as a run-to-completion wrapper over `ToolLoopSession`."""

    DEFAULT_LIMIT_MESSAGE = DEFAULT_LIMIT_MESSAGE

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
        session = ToolLoopSession(
            llm=self.llm,
            tools=self.tools,
            tool_schemas=tool_schemas,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_total_rounds=self.max_rounds,
            control_tool_names=frozenset(),  # v1 semantics: no control tools
            on_tool_observation=self.on_tool_observation,
            limit_message=self.limit_message,
        )
        while True:
            remaining = max(1, self.max_rounds - session.rounds_used)
            step = await session.step(workspace, max_rounds=remaining)
            if step.kind == "tool":
                continue  # still working — budget left
            final_text = step.text.strip() or None
            return ToolLoopResult(
                final_text=final_text,
                tool_requests=list(session.tool_requests),
                rounds=session.rounds_used,
                limit_reached=step.limit_reached,
            )

    @staticmethod
    def _tool_requests(
        response: dict[str, Any], known_names: set[str] | None = None
    ) -> list[tuple[str, ToolRequest]]:
        """All (call_id, request) pairs from the response — models may emit
        several parallel tool calls in one turn and each must be executed (or
        at least answered), not just calls[0]."""
        calls = response.get("tool_calls") or []
        if not calls:
            parsed = _parse_dsml_tool_call(
                str(response.get("content") or ""), known_names=known_names
            )
            return [("call-1", parsed)] if parsed is not None else []
        requests: list[tuple[str, ToolRequest]] = []
        for index, call in enumerate(calls):
            function = call.get("function") or {}
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"input": str(function.get("arguments") or "")}
            if not isinstance(args, dict):
                args = {"input": args}
            call_id = str(call.get("id") or f"call-{index + 1}")
            requests.append((call_id, ToolRequest(name=name, args=args)))
        return requests

    @staticmethod
    def _assistant_tool_call_message(
        response: dict[str, Any], executed: list[tuple[str, ToolRequest]]
    ) -> dict[str, Any]:
        """Assistant echo carrying exactly the executed calls, so every
        tool_call entry has a matching role:tool response in the transcript."""
        content = (response.get("content") or "") if response.get("tool_calls") else ""
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": request.name, "arguments": json.dumps(request.args)},
                }
                for call_id, request in executed
            ],
        }

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

# Streamed content may open a DSML marker mid-round ("<｜" / "<|"); everything
# from the first opener onward is withheld from the token stream.
_DSML_STREAM_OPENER = re.compile(r"<\s*[｜|]")
_DSML_TRAILING_OPEN = re.compile(r"<\s*$")


class _StreamGate:
    """Per-round token gate for streamed completions.

    Buffers the first ``SNIFF_CHARS`` characters to sniff leaked DSML
    tool-call markers (`_DSML_HINT`): a leaked call must never be streamed
    to clients — the recovery parser handles the assembled content after
    the round. Past the sniff window, content flows through except that
    anything from a marker opener onward is withheld, and a trailing
    ``<``(+spaces) run is carried so an opener split across chunks cannot
    slip through.
    """

    SNIFF_CHARS = 64

    def __init__(self) -> None:
        self._buffer = ""
        self._decided = False
        self._stopped = False
        self.emitted_chars = 0

    def feed(self, text: str) -> str:
        if self._stopped or not text:
            return ""
        self._buffer += text
        if not self._decided:
            if len(self._buffer) < self.SNIFF_CHARS:
                return ""
            self._decide()
            if self._stopped:
                return ""
        return self._drain(final=False)

    def finish(self) -> str:
        """Flush at end of stream (also decides short (<64 char) responses)."""
        if self._stopped:
            return ""
        if not self._decided:
            self._decide()
            if self._stopped:
                return ""
        return self._drain(final=True)

    def _decide(self) -> None:
        self._decided = True
        if _DSML_HINT.search(self._buffer):
            self._stopped = True
            self._buffer = ""

    def _drain(self, *, final: bool) -> str:
        match = _DSML_STREAM_OPENER.search(self._buffer)
        if match:
            out = self._buffer[: match.start()]
            self._buffer = ""
            self._stopped = True
        elif final:
            out, self._buffer = self._buffer, ""
        else:
            keep = 0
            tail = _DSML_TRAILING_OPEN.search(self._buffer)
            if tail:
                keep = len(self._buffer) - tail.start()
            cut = len(self._buffer) - keep
            out, self._buffer = self._buffer[:cut], self._buffer[cut:]
        self.emitted_chars += len(out)
        return out


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

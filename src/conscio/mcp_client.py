"""MCP client manager: connects configured servers, mirrors their tools into
the ToolRegistry as mcp__<server>__<tool>, and keeps them alive with backoff.

Lives outside conscio.tools/ on purpose: ToolRegistry.load_builtins() scans
that package with pkgutil and must never import this module."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from conscio.config import McpServerConfig
from conscio.core.tool_loop import EXTERNAL_CONTENT_CAPABILITY, NETWORK_READ_CAPABILITY
from conscio.tools.env import tool_env
from conscio.tools.registry import DEFAULT_TOOL_SCHEMA, ToolRegistry

logger = logging.getLogger(__name__)

MCP_CAPABILITY = "mcp"
_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")
_OPENAI_NAME_LIMIT = 64


def mcp_tool_name(server: str, tool: str) -> str:
    """`mcp__<server>__<tool>` sanitized to [A-Za-z0-9_-], capped to OpenAI's
    64-char function-name limit (truncate + 4-hex digest suffix on overflow)."""
    safe_server = _NAME_SAFE_RE.sub("_", server)
    safe_tool = _NAME_SAFE_RE.sub("_", tool)
    name = f"mcp__{safe_server}__{safe_tool}"
    if len(name) <= _OPENAI_NAME_LIMIT:
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:4]
    keep = _OPENAI_NAME_LIMIT - len(f"mcp__{safe_server}__") - 5
    if keep < 8:  # pathological server name; cap that instead
        return name[: _OPENAI_NAME_LIMIT - 5] + "_" + digest
    return f"mcp__{safe_server}__{safe_tool[:keep]}_{digest}"


@dataclass
class McpServerState:
    name: str
    status: str = "disconnected"  # connecting|connected|error|disabled|stopped
    tools: list[str] = field(default_factory=list)
    last_error: str = ""
    connected_at: float = 0.0
    reconnects: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "tools": list(self.tools),
            "last_error": self.last_error,
            "connected_at": self.connected_at,
            "reconnects": self.reconnects,
        }


def _fold_result(result: Any) -> dict[str, Any]:
    """Fold a CallToolResult into conscio's {"output": str, "error": bool} shape."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        block_type = getattr(block, "type", "")
        if block_type == "text":
            parts.append(str(getattr(block, "text", "")))
        elif block_type in {"image", "audio"}:
            parts.append(f"[{block_type} {getattr(block, 'mimeType', 'unknown')}]")
        elif block_type == "resource":
            resource = getattr(block, "resource", None)
            parts.append(f"[resource {getattr(resource, 'uri', 'unknown')}]")
        elif block_type == "resource_link":
            parts.append(f"[resource {getattr(block, 'uri', 'unknown')}]")
        else:
            parts.append(f"[{block_type or 'unknown'} content]")
    text = "\n".join(part for part in parts if part).strip()
    if getattr(result, "isError", False):
        return {"output": text or "MCP tool reported an error.", "error": True}
    if not text and getattr(result, "structuredContent", None) is not None:
        text = json.dumps(result.structuredContent, ensure_ascii=False)
    return {"output": text or "(no output)", "error": False}


class McpManager:
    """Owns one serve-task per configured server.

    DO NOT REFACTOR into an AsyncExitStack owned by start()/stop(): the MCP
    SDK's transports are anyio cancel scopes that MUST be entered and exited
    in the same task. Each _serve task opens and closes its own transport.
    Calling session.call_tool() from another task (the episode task) is safe —
    session I/O rides anyio memory streams to the session's receive loop."""

    BACKOFF_BASE = 2.0
    BACKOFF_CAP = 60.0

    def __init__(
        self,
        servers: list[McpServerConfig],
        registry: ToolRegistry,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._servers = list(servers)
        self._registry = registry
        self._on_event = on_event
        self._states: dict[str, McpServerState] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._reconnect_events: dict[str, asyncio.Event] = {}

    async def start(self) -> None:
        """Spawn serve tasks and return immediately; a down server never delays boot."""
        for cfg in self._servers:
            state = McpServerState(name=cfg.name)
            self._states[cfg.name] = state
            if not cfg.enabled:
                state.status = "disabled"
                continue
            self._stop_events[cfg.name] = asyncio.Event()
            self._reconnect_events[cfg.name] = asyncio.Event()
            self._tasks[cfg.name] = asyncio.create_task(self._serve(cfg))

    async def stop(self) -> None:
        for event in self._stop_events.values():
            event.set()
        for event in self._reconnect_events.values():
            event.set()  # wake any backoff sleep
        for name, task in list(self._tasks.items()):
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except (TimeoutError, asyncio.CancelledError):
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            state = self._states.get(name)
            if state is not None and state.status != "disabled":
                state.status = "stopped"
        self._tasks.clear()

    def status(self) -> list[dict[str, Any]]:
        return [self._states[cfg.name].as_dict() for cfg in self._servers if cfg.name in self._states]

    def request_reconnect(self, name: str, reason: str = "") -> None:
        state = self._states.get(name)
        if state is not None and reason:
            state.last_error = reason
        event = self._reconnect_events.get(name)
        if event is not None:
            event.set()

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event_type, data)
            except Exception:  # pragma: no cover - observer must never break serving
                logger.exception("MCP event emit failed")

    @asynccontextmanager
    async def _open_session(self, cfg: McpServerConfig):
        """Transport seam (tests override this). Yields an initialized ClientSession."""
        if cfg.transport == "stdio":
            params = StdioServerParameters(
                command=cfg.command,
                args=list(cfg.args),
                env={**tool_env(), **cfg.env},
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    async with asyncio.timeout(cfg.connect_timeout):
                        await session.initialize()
                    yield session
        else:
            async with streamablehttp_client(cfg.url, headers=cfg.headers or None) as (
                read_stream,
                write_stream,
                _get_session_id,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    async with asyncio.timeout(cfg.connect_timeout):
                        await session.initialize()
                    yield session

    async def _serve(self, cfg: McpServerConfig) -> None:
        state = self._states[cfg.name]
        stop = self._stop_events[cfg.name]
        attempt = 0
        while not stop.is_set():
            state.status = "connecting"
            registered: list[str] = []
            try:
                async with self._open_session(cfg) as session:
                    tools = await self._list_all_tools(session)
                    registered = self._register_tools(cfg, session, tools)
                    state.status = "connected"
                    state.tools = list(registered)
                    state.last_error = ""
                    state.connected_at = time.time()
                    attempt = 0
                    self._emit("mcp.server.connected", {"server": cfg.name, "tools": registered})
                    reconnect = self._reconnect_events[cfg.name]
                    reconnect.clear()
                    stop_wait = asyncio.create_task(stop.wait())
                    reconnect_wait = asyncio.create_task(reconnect.wait())
                    try:
                        await asyncio.wait({stop_wait, reconnect_wait}, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        stop_wait.cancel()
                        reconnect_wait.cancel()
                        await asyncio.gather(stop_wait, reconnect_wait, return_exceptions=True)
            except Exception as exc:
                state.status = "error"
                state.last_error = str(exc)
                self._emit("mcp.server.error", {"server": cfg.name, "error": str(exc)})
                logger.warning("MCP server %s failed: %s", cfg.name, exc)
            finally:
                for name in registered:
                    self._registry.unregister(name)
                if registered:
                    state.tools = []
                    self._emit("mcp.server.disconnected", {"server": cfg.name})
            if stop.is_set():
                break
            attempt += 1
            state.reconnects += 1
            delay = min(self.BACKOFF_BASE**attempt, self.BACKOFF_CAP) * (1 + random.random() * 0.5)
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def _list_all_tools(self, session: Any) -> list[Any]:
        tools: list[Any] = []
        cursor: str | None = None
        while True:
            result = await session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = getattr(result, "nextCursor", None)
            if not cursor:
                return tools

    def _register_tools(self, cfg: McpServerConfig, session: Any, tools: list[Any]) -> list[str]:
        registered: list[str] = []
        for tool in tools:
            if cfg.allowed and tool.name not in cfg.allowed:
                continue
            if tool.name in cfg.denied:
                continue
            name = mcp_tool_name(cfg.name, tool.name)
            if self._registry.tool_manifest(name) is not None:
                logger.warning("MCP tool %s collides with an existing tool; skipped", name)
                continue
            schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else DEFAULT_TOOL_SCHEMA
            capabilities = (
                {MCP_CAPABILITY}
                if cfg.trusted
                else {MCP_CAPABILITY, EXTERNAL_CONTENT_CAPABILITY, NETWORK_READ_CAPABILITY}
            )
            self._registry.register(
                name,
                self._make_wrapper(cfg, session, tool.name),
                description=f"[MCP:{cfg.name}] {tool.description or tool.name}",
                schema=schema,
                capabilities=capabilities,
            )
            registered.append(name)
        return registered

    def _make_wrapper(self, cfg: McpServerConfig, session: Any, tool_name: str):
        async def call_mcp_tool(**args: Any) -> dict[str, Any]:
            try:
                async with asyncio.timeout(cfg.call_timeout):
                    result = await session.call_tool(tool_name, args or None)
                return _fold_result(result)
            except TimeoutError:
                message = f"MCP tool {tool_name} timed out after {cfg.call_timeout:.0f}s."
                self.request_reconnect(cfg.name, message)
                return {
                    "output": message,
                    "error": True,
                    "execution_unknown": True,
                }
            except Exception as exc:
                self.request_reconnect(cfg.name, str(exc))
                return {
                    "output": f"MCP server '{cfg.name}' error: {exc}",
                    "error": True,
                    "execution_unknown": True,
                }

        return call_mcp_tool

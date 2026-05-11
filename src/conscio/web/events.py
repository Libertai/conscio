"""Server-Sent Events bridge from the cognitive workspace to web clients.

The broker exposes two distinct event sources:

1. ``WorkspaceEntry`` broadcasts from ``Workspace.subscribe`` (the existing
   internal blackboard), one SSE event per entry with the entry kind as the
   event name (observation, action, …).

2. Service-level signals (project status changes, episode commits, ticks,
   chat messages, status snapshots) emitted via ``service_emit``.

Each connected client owns an ``asyncio.Queue`` with bounded capacity. On
overflow the oldest event is dropped and a ``dropped`` counter is shipped
on the next emission so the client can detect it fell behind. Slow clients
must never stall the cognition loop, so all enqueueing is non-blocking.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from conscio.core.workspace import Workspace, WorkspaceEntry


CLIENT_QUEUE_SIZE = 256
HEARTBEAT_SECONDS = 15.0
STATUS_TICK_SECONDS = 2.0
BACKLOG_SIZE = 80


@dataclass
class _Client:
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=lambda: asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE))
    dropped: int = 0


class WorkspaceEventBroker:
    """Fan-out broker. One per ``ConscioService`` lifetime."""

    def __init__(self, workspace: Workspace, *, backlog: int = BACKLOG_SIZE) -> None:
        self._workspace = workspace
        self._clients: list[_Client] = []
        self._lock = asyncio.Lock()
        self._unsubscribe: Callable[[], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Ring buffer of recent events so a freshly-connected client doesn't
        # stare at an empty timeline while it waits for the next live signal.
        self._backlog: deque[dict[str, Any]] = deque(maxlen=max(0, backlog))

    # ── lifecycle ────────────────────────────────────────────────

    def attach(self) -> None:
        if self._unsubscribe is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._unsubscribe = self._workspace.subscribe(self._on_entry)

    def detach(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._clients.clear()

    # ── client registration ──────────────────────────────────────

    def register(self) -> _Client:
        client = _Client()
        # Pre-fill the queue with backlog so the client sees recent history
        # immediately on connect. Tag each replayed event so the UI can style
        # them differently if it wants to (faded, etc.).
        for past in list(self._backlog):
            try:
                client.queue.put_nowait({**past, "_replay": True})
            except asyncio.QueueFull:
                break
        self._clients.append(client)
        return client

    @property
    def backlog_size(self) -> int:
        return len(self._backlog)

    def unregister(self, client: _Client) -> None:
        try:
            self._clients.remove(client)
        except ValueError:
            pass

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ── publishing ───────────────────────────────────────────────

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Fan out a service-level event to every client. Safe from any thread
        (re-routes to the loop via ``call_soon_threadsafe`` if necessary)."""
        payload = {"type": event_type, "ts": time.time(), **data}
        self._record(payload)
        running_loop: asyncio.AbstractEventLoop | None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None and running_loop is self._loop:
            self._dispatch(payload)
        elif self._loop is not None:
            self._loop.call_soon_threadsafe(self._dispatch, payload)
        else:
            self._dispatch(payload)

    def _record(self, payload: dict[str, Any]) -> None:
        if self._backlog.maxlen:
            self._backlog.append(payload)

    def _dispatch(self, payload: dict[str, Any]) -> None:
        for client in self._clients:
            self._enqueue(client, payload)

    def _enqueue(self, client: _Client, payload: dict[str, Any]) -> None:
        try:
            client.queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                client.queue.get_nowait()
                client.dropped += 1
            except asyncio.QueueEmpty:
                pass
            try:
                client.queue.put_nowait({**payload, "_dropped": client.dropped})
            except asyncio.QueueFull:
                # Pathological: even after a drop we can't fit. Skip.
                pass

    # ── workspace handler (sync, called from cognition loop) ─────

    def _on_entry(self, entry: WorkspaceEntry) -> None:
        payload = {
            "type": f"workspace.{entry.type.value}",
            "ts": entry.timestamp,
            "source": entry.source,
            "content": entry.content,
            "priority": entry.priority,
            "salience": entry.salience,
            "confidence": entry.confidence,
            "novelty": entry.novelty,
            "urgency": entry.urgency,
            "metadata": entry.metadata,
            "kind": entry.type.value,
        }
        self._record(payload)
        # The workspace broadcast runs inside the cognition coroutine which
        # owns ``self._loop``, so a direct dispatch is fine. We still guard.
        if self._loop is None or self._loop.is_closed():
            self._dispatch(payload)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._dispatch(payload)
        else:
            self._loop.call_soon_threadsafe(self._dispatch, payload)


# ── SSE wire format helpers ────────────────────────────────────────


def encode_sse(payload: dict[str, Any], *, event: str | None = None, retry_ms: int | None = None) -> str:
    """Serialise one SSE record. Caller controls the event name explicitly so
    the broker's typed payloads can map onto SSE event names cleanly."""
    parts: list[str] = []
    if retry_ms is not None:
        parts.append(f"retry: {retry_ms}")
    name = event or payload.get("type", "message")
    parts.append(f"event: {name}")
    parts.append(f"data: {json.dumps(payload, separators=(',', ':'), default=str)}")
    parts.append("")
    parts.append("")
    return "\n".join(parts)


async def stream_events(
    broker: WorkspaceEventBroker,
    *,
    is_disconnected: Callable[[], "asyncio.Future[bool] | bool"] | None = None,
) -> Iterable[bytes]:
    """Async generator yielding SSE-encoded bytes for a single client.

    ``is_disconnected`` is an optional async callable (Starlette's
    ``request.is_disconnected``) used to break the loop early.
    """
    client = broker.register()
    try:
        yield encode_sse({"type": "stream.open", "ts": time.time()}, retry_ms=3000).encode("utf-8")
        while True:
            if is_disconnected is not None:
                disc = is_disconnected()
                if asyncio.iscoroutine(disc):
                    if await disc:
                        break
                elif disc:
                    break
            try:
                payload = await asyncio.wait_for(client.queue.get(), timeout=HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield b": ping\n\n"
                continue
            yield encode_sse(payload, event=payload.get("type")).encode("utf-8")
    finally:
        broker.unregister(client)

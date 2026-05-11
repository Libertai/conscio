from __future__ import annotations

import asyncio
import json

import pytest

from conscio.core.workspace import EntryType, Workspace, WorkspaceEntry
from conscio.web.events import (
    CLIENT_QUEUE_SIZE,
    WorkspaceEventBroker,
    encode_sse,
    stream_events,
)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_subscribe_and_unsubscribe_clean() -> None:
    async def scenario() -> None:
        ws = Workspace()
        broker = WorkspaceEventBroker(ws)
        broker.attach()
        try:
            assert len(ws._subscribers) == 1  # noqa: SLF001
        finally:
            broker.detach()
        assert len(ws._subscribers) == 0  # noqa: SLF001

    _run(scenario())


def test_workspace_entry_reaches_client() -> None:
    async def scenario() -> None:
        ws = Workspace()
        broker = WorkspaceEventBroker(ws)
        broker.attach()
        try:
            client = broker.register()
            entry = WorkspaceEntry(
                content="hello",
                source="test",
                type=EntryType.ACTION,
                priority=3,
            )
            ws.broadcast(entry)
            payload = await asyncio.wait_for(client.queue.get(), timeout=0.5)
            assert payload["type"] == "workspace.action"
            assert payload["content"] == "hello"
            assert payload["source"] == "test"
            assert payload["priority"] == 3
        finally:
            broker.detach()

    _run(scenario())


def test_emit_propagates_to_clients() -> None:
    async def scenario() -> None:
        ws = Workspace()
        broker = WorkspaceEventBroker(ws)
        broker.attach()
        try:
            client = broker.register()
            broker.emit("project.updated", {"project_id": "p1", "status": "paused"})
            payload = await asyncio.wait_for(client.queue.get(), timeout=0.5)
            assert payload["type"] == "project.updated"
            assert payload["project_id"] == "p1"
            assert payload["status"] == "paused"
        finally:
            broker.detach()

    _run(scenario())


def test_overflow_drops_oldest_and_reports_dropped() -> None:
    async def scenario() -> None:
        ws = Workspace()
        broker = WorkspaceEventBroker(ws)
        broker.attach()
        try:
            client = broker.register()
            # Saturate the queue beyond capacity.
            for i in range(CLIENT_QUEUE_SIZE + 5):
                broker.emit("noise", {"i": i})
            # Drain a few to inspect dropped marker.
            seen_dropped = False
            for _ in range(CLIENT_QUEUE_SIZE):
                payload = client.queue.get_nowait()
                if "_dropped" in payload:
                    seen_dropped = True
            assert seen_dropped
            assert client.dropped >= 5
        finally:
            broker.detach()

    _run(scenario())


def test_stream_events_yields_open_then_event() -> None:
    async def scenario() -> None:
        ws = Workspace()
        broker = WorkspaceEventBroker(ws)
        broker.attach()
        try:
            gen = stream_events(broker)
            first = await gen.__anext__()
            assert b"event: stream.open" in first
            broker.emit("hello", {"v": 1})
            second = await gen.__anext__()
            assert b"event: hello" in second
            assert b'"v":1' in second
            await gen.aclose()
        finally:
            broker.detach()

    _run(scenario())


def test_encode_sse_round_trip() -> None:
    out = encode_sse({"type": "x", "n": 3}, event="x", retry_ms=2500)
    assert out.startswith("retry: 2500\n")
    assert "\nevent: x\n" in out
    body_line = next(line for line in out.split("\n") if line.startswith("data:"))
    body = json.loads(body_line[len("data: "):])
    assert body == {"type": "x", "n": 3}

"""Integration tests for the new chat + events HTTP routes.

These exercise the cookie-auth /ui/api/* surface added in Phase 1 without
spinning up the full cognitive runtime — we use a stub service that exposes
just the attributes the router touches.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from conscio.core.workspace import Workspace
from conscio.memory.store import MemoryStore
from conscio.web.events import WorkspaceEventBroker
from conscio.webui import create_web_router


@dataclass
class _StubConfig:
    web_password: str = "letmein"
    web_secure_cookies: bool = False


@dataclass
class _StubServiceStatus:
    running: bool = True
    paused: bool = False


@dataclass
class _StubResult:
    output: str
    selected_action: str = "reply"


class _StubService:
    def __init__(self, db_path: Path) -> None:
        self.config = _StubConfig()
        self.memory = MemoryStore(db_path=str(db_path))
        self._workspace = Workspace()
        self.event_broker = WorkspaceEventBroker(self._workspace)
        self.event_broker.attach()
        self.last_message: str | None = None
        self.latest_model_context = ""

    # The router calls these on the snapshot path; minimal stubs are enough.
    async def status(self) -> _StubServiceStatus:
        return _StubServiceStatus()

    async def list_projects(self) -> list[dict[str, Any]]: return []
    async def list_influences(self) -> list[dict[str, Any]]: return []
    async def recent_episodes(self, limit: int) -> list[dict[str, Any]]: return []
    async def recent_trace(self) -> str: return ""
    async def recent_facts(self, limit: int) -> list[dict[str, Any]]: return []
    async def list_skills(self) -> list[dict[str, Any]]: return []

    @property
    def goals(self):
        async def list_goals(): return []
        return type("G", (), {"list_goals": staticmethod(list_goals)})()

    async def submit_message(self, content: str) -> _StubResult:
        self.last_message = content
        return _StubResult(output=f"echo: {content}")


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    svc = _StubService(tmp_path / "test.db")
    a = FastAPI()
    a.include_router(create_web_router(svc))  # type: ignore[arg-type]
    a.state.svc = svc
    return a


def _login(client: TestClient) -> None:
    r = client.post("/ui/login", json={"password": "letmein"})
    assert r.status_code == 200, r.text


def test_chat_round_trip(app: FastAPI) -> None:
    with TestClient(app) as client:
        _login(client)

        # Default session is auto-created on first list / first send.
        r = client.post(
            "/ui/api/chat/sessions/main/messages",
            json={"content": "ping"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["agent"] == "echo: ping"
        assert body["selected_action"] == "reply"

        msgs = client.get("/ui/api/chat/sessions/main/messages").json()
        assert [m["role"] for m in msgs] == ["user", "agent"]
        assert [m["content"] for m in msgs] == ["ping", "echo: ping"]


def test_chat_session_create_and_delete(app: FastAPI) -> None:
    with TestClient(app) as client:
        _login(client)
        r = client.post("/ui/api/chat/sessions", json={"title": "scratchpad"})
        assert r.status_code == 200, r.text
        sid = r.json()["id"]

        sessions = client.get("/ui/api/chat/sessions").json()
        assert any(s["id"] == sid for s in sessions)

        r = client.delete(f"/ui/api/chat/sessions/{sid}")
        assert r.status_code == 200

        # Default session cannot be deleted (400, not 500).
        r = client.delete("/ui/api/chat/sessions/main")
        assert r.status_code == 400


def test_routes_require_auth(app: FastAPI) -> None:
    with TestClient(app) as client:
        for path in [
            "/ui/api/chat/sessions",
            "/ui/api/chat/sessions/main/messages",
            "/ui/api/events",
        ]:
            r = client.get(path)
            assert r.status_code == 401, f"{path} should require auth"


def test_events_route_registered_with_auth(app: FastAPI) -> None:
    """The streaming behaviour itself is covered by tests/test_web_events.py
    against the broker directly. The Starlette TestClient's sync transport
    can't cleanly cancel a long-lived StreamingResponse without a real loop,
    so here we only verify the route exists and enforces the session cookie."""
    routes = {(r.path, list(r.methods)[0]) for r in app.routes if hasattr(r, "methods")}  # type: ignore[attr-defined]
    assert ("/ui/api/events", "GET") in routes
    with TestClient(app) as client:
        # Without a session cookie, the route refuses with 401 before the
        # streaming generator is ever instantiated.
        r = client.get("/ui/api/events")
        assert r.status_code == 401

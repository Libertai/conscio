from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Cookie, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from conscio.service import ConscioService
from conscio.web.auth import (
    LOGIN_FAILURE_LIMIT,
    SESSION_COOKIE,
    SESSION_TTL_SECONDS,
    check_password,
    login_failure_count,
    record_login_failure,
    require_web_auth,
    session_token,
    sweep_login_failures,
    sweep_sessions,
)
# Re-export for backwards compatibility with existing tests + callers.
from conscio.web.auth import (  # noqa: F401
    MAX_LOGIN_FAILURE_TRACKERS,
    MAX_SESSIONS,
    sweep_sessions as _sweep_sessions,
    sweep_login_failures as _sweep_login_failures,
)
from conscio.web.chat import ChatStore, DEFAULT_SESSION_ID  # noqa: F401
from conscio.web.events import stream_events


class LoginRequest(BaseModel):
    password: str


class TextRequest(BaseModel):
    content: str


class ChatMessageRequest(BaseModel):
    content: str
    session_id: str | None = None


class ChatSessionCreateRequest(BaseModel):
    title: str | None = None


class GoalCreateRequest(BaseModel):
    description: str
    priority: float = 0.5
    source: str = "user"


class GoalUpdateRequest(BaseModel):
    description: str | None = None
    status: str | None = None
    priority: float | None = None
    review_notes: str | None = None


# Auth helpers now live in conscio.web.auth. The leading-underscore alias
# below preserves the original private API used by the route handlers.
_require_web_auth = require_web_auth


def create_web_router(service: ConscioService) -> APIRouter:
    router = APIRouter()
    sessions: dict[str, float] = {}
    login_failures: dict[str, list[float]] = {}
    chat_store = ChatStore(service.memory)

    @router.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    @router.post("/ui/login", include_in_schema=False)
    async def login(req: LoginRequest, request: Request, response: Response) -> dict[str, bool]:
        if not service.config.web_password:
            raise HTTPException(status_code=500, detail="web_password is not configured")
        now = time.time()
        sweep_login_failures(login_failures, now)
        sweep_sessions(sessions, now)
        client = request.client.host if request.client else "unknown"
        if login_failure_count(login_failures, client, now) >= LOGIN_FAILURE_LIMIT:
            raise HTTPException(status_code=429, detail="too many login attempts")
        if not check_password(service, req.password):
            record_login_failure(login_failures, client, now)
            raise HTTPException(status_code=401, detail="invalid password")
        login_failures.pop(client, None)
        token = session_token()
        sessions[token] = now + SESSION_TTL_SECONDS
        response.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            secure=service.config.web_secure_cookies,
            samesite="lax",
            max_age=SESSION_TTL_SECONDS,
        )
        return {"ok": True}

    @router.post("/ui/logout", include_in_schema=False)
    async def logout(response: Response, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        if conscio_web_session:
            sessions.pop(conscio_web_session, None)
        response.delete_cookie(SESSION_COOKIE)
        return {"ok": True}

    @router.get("/ui/api/snapshot", include_in_schema=False)
    async def snapshot(conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        return {
            "status": (await service.status()).__dict__,
            "goals": await service.goals.list_goals(),
            "projects": await service.list_projects(),
            "influences": await service.list_influences(),
            "episodes": await service.recent_episodes(10),
            "trace": await service.recent_trace(),
            "model_context": service.latest_model_context,
            "facts": await service.recent_facts(10),
            "skills": await service.list_procedures(),
        }

    @router.post("/ui/api/message", include_in_schema=False)
    async def ui_message(req: TextRequest, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        result = await service.submit_message(req.content)
        return {"output": result.output, "selected_action": result.selected_action}

    @router.post("/ui/api/influence/goal", include_in_schema=False)
    async def ui_goal(req: TextRequest, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await service.submit_influence(req.content, kind="goal")

    @router.post("/ui/api/influence/constraint", include_in_schema=False)
    async def ui_constraint(req: TextRequest, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await service.submit_influence(req.content, kind="constraint")

    @router.post("/ui/api/control/{action}", include_in_schema=False)
    async def ui_control(action: str, conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        if action == "pause":
            service.pause()
            return {"paused": True}
        if action == "resume":
            service.resume()
            return {"paused": False}
        raise HTTPException(status_code=404, detail="unknown control action")

    @router.post("/ui/api/tick", include_in_schema=False)
    async def ui_tick(conscio_web_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        result = await service.run_autonomous_tick()
        if result is None:
            return {"output": "", "selected_action": "wait"}
        return {"output": result.output, "selected_action": result.selected_action}

    # ── Chat persistence (server-side, replaces localStorage) ────────────
    @router.get("/ui/api/chat/sessions", include_in_schema=False)
    async def ui_chat_sessions(
        conscio_web_session: str | None = Cookie(default=None),
    ) -> list[dict[str, Any]]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await chat_store.list_sessions()

    @router.post("/ui/api/chat/sessions", include_in_schema=False)
    async def ui_chat_session_create(
        req: ChatSessionCreateRequest,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await chat_store.create_session(req.title)

    @router.delete("/ui/api/chat/sessions/{session_id}", include_in_schema=False)
    async def ui_chat_session_delete(
        session_id: str,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, bool]:
        _require_web_auth(service, sessions, conscio_web_session)
        try:
            await chat_store.delete_session(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @router.get("/ui/api/chat/sessions/{session_id}/messages", include_in_schema=False)
    async def ui_chat_messages(
        session_id: str,
        limit: int = Query(default=200, ge=1, le=500),
        before_id: int | None = Query(default=None, ge=1),
        conscio_web_session: str | None = Cookie(default=None),
    ) -> list[dict[str, Any]]:
        _require_web_auth(service, sessions, conscio_web_session)
        await chat_store.ensure_default_session()
        if await chat_store.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="unknown chat session")
        return await chat_store.get_messages(session_id, limit=limit, before_id=before_id)

    @router.post("/ui/api/chat/sessions/{session_id}/messages", include_in_schema=False)
    async def ui_chat_post_message(
        session_id: str,
        req: TextRequest,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        await chat_store.ensure_default_session()
        if await chat_store.get_session(session_id) is None:
            # Without this check append_message's upsert silently resurrects
            # deleted sessions (or creates ghosts for mistyped ids).
            raise HTTPException(status_code=404, detail="unknown chat session")
        await chat_store.append_message(session_id, "user", req.content)
        try:
            result = await service.submit_message(req.content)
        except Exception as exc:  # noqa: BLE001 — surface as HTTP, keep the user message
            raise HTTPException(
                status_code=502,
                detail=f"agent failed to respond: {exc}",
            ) from exc
        agent_msg = await chat_store.append_message(
            session_id,
            "agent",
            result.output,
            selected_action=result.selected_action,
        )
        service.event_broker.emit(
            "chat.message",
            {
                "session_id": session_id,
                "user": req.content,
                "agent": result.output,
                "selected_action": result.selected_action,
                "agent_message_id": agent_msg["id"],
            },
        )
        return {
            "session_id": session_id,
            "user": req.content,
            "agent": result.output,
            "selected_action": result.selected_action,
        }

    # ── Projects ─────────────────────────────────────────────────────────
    @router.get("/ui/api/projects", include_in_schema=False)
    async def ui_projects(
        conscio_web_session: str | None = Cookie(default=None),
    ) -> list[dict[str, Any]]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await service.list_projects()

    @router.get("/ui/api/projects/{project_id}", include_in_schema=False)
    async def ui_project_detail(
        project_id: str,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        found = await service.get_project(project_id)
        if found is None:
            raise HTTPException(status_code=404, detail="project not found")
        return found

    @router.post("/ui/api/projects/{project_id}/pause", include_in_schema=False)
    async def ui_project_pause(
        project_id: str,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, str]:
        _require_web_auth(service, sessions, conscio_web_session)
        try:
            await service.set_project_status(project_id, "paused")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        return {"status": "paused"}

    @router.post("/ui/api/projects/{project_id}/resume", include_in_schema=False)
    async def ui_project_resume(
        project_id: str,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, str]:
        _require_web_auth(service, sessions, conscio_web_session)
        try:
            await service.set_project_status(project_id, "active")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="project not found") from exc
        return {"status": "active"}

    # ── Goals ────────────────────────────────────────────────────────────
    @router.get("/ui/api/goals", include_in_schema=False)
    async def ui_goals(
        status: str | None = Query(default=None),
        conscio_web_session: str | None = Cookie(default=None),
    ) -> list[dict[str, Any]]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await service.goals.list_goals(status=status)

    @router.post("/ui/api/goals", include_in_schema=False)
    async def ui_goal_create(
        req: GoalCreateRequest,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        goal = await service.goals.add_goal(
            req.description,
            source=req.source,
            priority=req.priority,
        )
        record = await service.goals.get_goal(goal.id)
        service.event_broker.emit("goal.changed", {"goal_id": goal.id, "action": "created"})
        return record or {"id": goal.id, "description": goal.description}

    @router.patch("/ui/api/goals/{goal_id}", include_in_schema=False)
    async def ui_goal_update(
        goal_id: str,
        req: GoalUpdateRequest,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        record = await service.goals.update_goal(
            goal_id,
            description=req.description,
            status=req.status,
            priority=req.priority,
            review_notes=req.review_notes,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="goal not found")
        service.event_broker.emit("goal.changed", {"goal_id": goal_id, "action": "updated"})
        return record

    @router.delete("/ui/api/goals/{goal_id}", include_in_schema=False)
    async def ui_goal_retire(
        goal_id: str,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        record = await service.goals.retire_goal(goal_id)
        if record is None:
            raise HTTPException(status_code=404, detail="goal not found")
        service.event_broker.emit("goal.changed", {"goal_id": goal_id, "action": "retired"})
        return record

    # ── Influences ───────────────────────────────────────────────────────
    @router.get("/ui/api/influences", include_in_schema=False)
    async def ui_influences(
        conscio_web_session: str | None = Cookie(default=None),
    ) -> list[dict[str, Any]]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await service.list_influences()

    @router.delete("/ui/api/influences/{influence_id}", include_in_schema=False)
    async def ui_influence_retire(
        influence_id: str,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        _require_web_auth(service, sessions, conscio_web_session)
        record = await service.goals.retire_influence(influence_id)
        if record is None:
            raise HTTPException(status_code=404, detail="influence not found")
        return record

    # ── Episodes (cursor pagination) ─────────────────────────────────────
    @router.get("/ui/api/episodes", include_in_schema=False)
    async def ui_episodes(
        limit: int = Query(default=20, ge=1, le=100),
        before: float | None = Query(default=None),
        conscio_web_session: str | None = Cookie(default=None),
    ) -> list[dict[str, Any]]:
        _require_web_auth(service, sessions, conscio_web_session)
        if before is None:
            return await service.recent_episodes(limit)
        return await service.episodes_before(before, limit)

    # ── Trace + Model Context ────────────────────────────────────────────
    @router.get("/ui/api/trace", include_in_schema=False)
    async def ui_trace(
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, str]:
        _require_web_auth(service, sessions, conscio_web_session)
        return {"trace": await service.recent_trace()}

    @router.get("/ui/api/model_context", include_in_schema=False)
    async def ui_model_context(
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, str]:
        _require_web_auth(service, sessions, conscio_web_session)
        return {"model_context": service.latest_model_context}

    # ── Memory search ────────────────────────────────────────────────────
    @router.get("/ui/api/memory/search", include_in_schema=False)
    async def ui_memory_search(
        q: str = Query(min_length=1),
        limit: int = Query(default=20, ge=1, le=100),
        conscio_web_session: str | None = Cookie(default=None),
    ) -> list[dict[str, Any]]:
        _require_web_auth(service, sessions, conscio_web_session)
        return await service.search_memory(q, limit)

    @router.get("/ui/api/memory/recent", include_in_schema=False)
    async def ui_memory_recent(
        limit: int = Query(default=20, ge=1, le=100),
        conscio_web_session: str | None = Cookie(default=None),
    ) -> dict[str, list[dict[str, Any]]]:
        _require_web_auth(service, sessions, conscio_web_session)
        return {
            "facts": await service.recent_facts(limit),
            "skills": await service.list_procedures(),
        }

    # ── Server-Sent Events stream ────────────────────────────────────────
    @router.get("/ui/api/events", include_in_schema=False)
    async def ui_events(
        request: Request,
        conscio_web_session: str | None = Cookie(default=None),
    ) -> StreamingResponse:
        _require_web_auth(service, sessions, conscio_web_session)
        broker = service.event_broker
        # Identity encoding defeats Caddy's gzip buffering without needing a
        # Caddyfile change. X-Accel-Buffering disables nginx buffering too.
        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        }
        return StreamingResponse(
            stream_events(broker, is_disconnected=request.is_disconnected),
            media_type="text/event-stream",
            headers=headers,
        )

    return router

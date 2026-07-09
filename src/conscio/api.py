from __future__ import annotations

import asyncio
import hmac
import logging
import math
import os
import signal
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from conscio import __version__
from conscio.config import ServiceConfig
from conscio.service import ConscioService, EpisodeCancelled
from conscio.web.events import SSEClientLimitError, encode_sse, stream_events
from conscio.webui import create_web_router

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# API clients may submit messages/influences as "user" (default, executable)
# or "system" (observation-only). They must not mint "autonomous" or "tool"
# provenance — that would let an API caller forge the agent's own actions.
_ALLOWED_API_SOURCES = frozenset({"user", "system"})


def _validated_source(source: str) -> str:
    return source if source in _ALLOWED_API_SOURCES else "user"


class MessageRequest(BaseModel):
    content: str = Field(max_length=64_000)
    source: str = "user"


class InfluenceRequest(BaseModel):
    content: str = Field(max_length=64_000)
    source: str = "user"


def create_app(service: ConscioService | None = None, config: ServiceConfig | None = None) -> FastAPI:
    svc = service or ConscioService(config)

    async def require_episode_budget() -> None:
        allowed, retry_after = svc.try_acquire_episode()
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="episode rate limit exceeded",
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        expected = svc.config.api_key
        if not expected:
            raise HTTPException(status_code=500, detail="API key is not configured")
        # Encode to bytes so a non-ASCII Authorization header raises a clean
        # 401 instead of a TypeError → 500 (matches web/auth.py hardening).
        provided = (authorization or "").encode("utf-8", "replace")
        wanted = f"Bearer {expected}".encode("utf-8", "replace")
        if not hmac.compare_digest(provided, wanted):
            logger.warning("API auth failed")
            raise HTTPException(status_code=401, detail="invalid API key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await svc.start()
        try:
            yield
        finally:
            await svc.stop()

    # Deliberately no CORS middleware: the API is bearer-token (CSRF-immune) and
    # the SPA is same-origin behind cookie auth with SameSite=lax. Adding CORS
    # would only widen the attack surface for zero current consumers.
    app = FastAPI(title="Conscio", version=__version__, lifespan=lifespan)

    if svc.config.max_request_bytes > 0:
        from conscio.web.limits import BodySizeLimitMiddleware  # noqa: PLC0415
        app.add_middleware(BodySizeLimitMiddleware, max_bytes=svc.config.max_request_bytes)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "running": svc.running, "version": __version__}

    @app.get("/status", dependencies=[Depends(require_auth)])
    async def status() -> dict[str, Any]:
        return (await svc.status()).__dict__

    @app.get("/metrics", dependencies=[Depends(require_auth)])
    async def metrics() -> dict[str, Any]:
        return await svc.metrics()

    @app.get("/metrics/prometheus", dependencies=[Depends(require_auth)])
    async def metrics_prometheus() -> Response:
        from conscio.web.prometheus import render_prometheus  # noqa: PLC0415

        body = render_prometheus(
            await svc.metrics(),
            {"sse_clients": svc.event_broker.client_count, "version": __version__},
        )
        return Response(content=body, media_type="text/plain; version=0.0.4")

    @app.get("/ready")
    async def ready() -> JSONResponse:
        """Readiness (vs /health liveness): running + database reachable.
        Unauthenticated: it leaks only a boolean, and proxies/orchestrators
        need to poll it without credentials."""
        if not svc.running:
            return JSONResponse({"ready": False, "reason": "service not running"}, status_code=503)
        try:
            svc.memory.fetchone("SELECT 1")
        except Exception as exc:  # noqa: BLE001 — any DB failure means not ready
            return JSONResponse({"ready": False, "reason": f"db probe failed: {exc}"}, status_code=503)
        return JSONResponse({"ready": True})

    def _message_blob(result: Any) -> dict[str, Any]:
        return {
            "output": result.output,
            "selected_action": result.selected_action,
            "session_id": result.session_id,
            "self_state": result.self_state,
            "attention_schema": result.attention_schema,
        }

    @app.post("/message", dependencies=[Depends(require_auth), Depends(require_episode_budget)])
    async def message(req: MessageRequest) -> dict[str, Any]:
        try:
            result = await asyncio.wait_for(
                svc.submit_message(req.content, source=_validated_source(req.source)),
                svc.config.message_timeout or None,
            )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="episode still running; poll /episodes or POST /control/cancel",
            ) from None
        except EpisodeCancelled as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _message_blob(result)

    @app.post("/message/stream", dependencies=[Depends(require_auth), Depends(require_episode_budget)])
    async def message_stream(req: MessageRequest, request: Request) -> StreamingResponse:
        """SSE variant of /message: chat.token / chat.discard events for this
        submission (matched by ref), terminated by message.result or
        message.error. Not subject to message_timeout — the caller sees live
        progress and can cancel instead."""
        ref = uuid.uuid4().hex
        broker = svc.event_broker
        try:
            client = broker.register()  # before enqueue: no token can be missed
        except SSEClientLimitError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc
        task = asyncio.create_task(
            svc.submit_message(req.content, source=_validated_source(req.source), ref=ref)
        )

        async def _gen():
            try:
                while True:
                    if task.done():
                        while not client.queue.empty():
                            payload = client.queue.get_nowait()
                            if payload.get("ref") == ref and payload.get("type") in ("chat.token", "chat.discard"):
                                yield encode_sse(payload, event=str(payload.get("type"))).encode("utf-8")
                        try:
                            blob = _message_blob(task.result())
                            yield encode_sse(
                                {"type": "message.result", "ref": ref, **blob}, event="message.result"
                            ).encode("utf-8")
                        except EpisodeCancelled as exc:
                            yield encode_sse(
                                {"type": "message.error", "ref": ref, "status": 409, "detail": str(exc)},
                                event="message.error",
                            ).encode("utf-8")
                        except Exception as exc:  # noqa: BLE001 — terminal SSE frame, never a 500 mid-stream
                            yield encode_sse(
                                {"type": "message.error", "ref": ref, "status": 500, "detail": str(exc)},
                                event="message.error",
                            ).encode("utf-8")
                        return
                    try:
                        payload = await asyncio.wait_for(client.queue.get(), timeout=0.25)
                    except TimeoutError:
                        # Abandoned streams must release their broker slot
                        # (MAX_SSE_CLIENTS); the episode itself keeps running.
                        if await request.is_disconnected():
                            return
                        continue
                    if payload.get("ref") == ref and payload.get("type") in ("chat.token", "chat.discard"):
                        yield encode_sse(payload, event=str(payload.get("type"))).encode("utf-8")
            finally:
                broker.unregister(client)

        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        }
        return StreamingResponse(_gen(), media_type="text/event-stream", headers=headers)

    @app.get("/events", dependencies=[Depends(require_auth)])
    async def events(request: Request) -> StreamingResponse:
        """Bearer-authed mirror of the operator console SSE stream."""
        try:
            client = svc.event_broker.register()
        except SSEClientLimitError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "10"}) from exc
        headers = {
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
        }
        return StreamingResponse(
            stream_events(svc.event_broker, client=client, is_disconnected=request.is_disconnected),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.post("/influence/goal", dependencies=[Depends(require_auth), Depends(require_episode_budget)])
    async def influence_goal(req: InfluenceRequest) -> dict[str, Any]:
        return await svc.submit_influence(req.content, kind="goal", source=_validated_source(req.source))

    @app.post("/influence/constraint", dependencies=[Depends(require_auth), Depends(require_episode_budget)])
    async def influence_constraint(req: InfluenceRequest) -> dict[str, Any]:
        return await svc.submit_influence(req.content, kind="constraint", source=_validated_source(req.source))

    @app.post("/control/pause", dependencies=[Depends(require_auth)])
    async def pause() -> dict[str, Any]:
        svc.pause()
        return {"paused": True}

    @app.post("/control/resume", dependencies=[Depends(require_auth)])
    async def resume() -> dict[str, Any]:
        svc.resume()
        return {"paused": False}

    @app.post("/control/cancel", dependencies=[Depends(require_auth)])
    async def cancel() -> dict[str, Any]:
        return svc.cancel_current()

    @app.post("/control/stop", dependencies=[Depends(require_auth)])
    async def stop(background_tasks: BackgroundTasks) -> dict[str, Any]:
        await svc.stop()
        background_tasks.add_task(_terminate_process)
        return {"running": False}

    @app.get("/goals", dependencies=[Depends(require_auth)])
    async def goals(status: str | None = None) -> list[dict[str, Any]]:
        return await svc.goals.list_goals(status=status)

    @app.get("/influences", dependencies=[Depends(require_auth)])
    async def influences() -> list[dict[str, Any]]:
        return await svc.list_influences()

    @app.get("/projects", dependencies=[Depends(require_auth)])
    async def projects() -> list[dict[str, Any]]:
        return await svc.list_projects()

    @app.get("/projects/{project_id}", dependencies=[Depends(require_auth)])
    async def project(project_id: str) -> dict[str, Any]:
        found = await svc.get_project(project_id)
        if found is None:
            raise HTTPException(status_code=404, detail="project not found")
        return found

    @app.post("/projects/{project_id}/pause", dependencies=[Depends(require_auth)])
    async def pause_project(project_id: str) -> dict[str, str]:
        try:
            await svc.set_project_status(project_id, "paused")
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None
        return {"status": "paused"}

    @app.post("/projects/{project_id}/resume", dependencies=[Depends(require_auth)])
    async def resume_project(project_id: str) -> dict[str, str]:
        try:
            await svc.set_project_status(project_id, "active")
        except KeyError:
            raise HTTPException(status_code=404, detail="project not found") from None
        return {"status": "active"}

    @app.post("/autonomy/tick", dependencies=[Depends(require_auth), Depends(require_episode_budget)])
    async def autonomy_tick() -> dict[str, Any]:
        result = await svc.run_autonomous_tick()
        if result is None:
            return {"output": "", "selected_action": "wait"}
        return {"output": result.output, "selected_action": result.selected_action}

    @app.get("/episodes", dependencies=[Depends(require_auth)])
    async def episodes(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
        return await svc.recent_episodes(limit)

    @app.get("/trace", dependencies=[Depends(require_auth)])
    async def trace() -> dict[str, str]:
        return {"trace": await svc.recent_trace()}

    @app.get("/memory/search", dependencies=[Depends(require_auth)])
    async def memory_search(q: str, limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
        return await svc.search_memory(q, limit)

    app.include_router(create_web_router(svc))

    # ── SPA host at /ui (Phase 4 swap; was previously dogfooded at /ui2) ──
    if (STATIC_DIR / "index.html").is_file():
        assets_dir = STATIC_DIR / "assets"
        if assets_dir.is_dir():
            app.mount(
                "/ui/assets",
                StaticFiles(directory=str(assets_dir)),
                name="ui-assets",
            )

        def _spa_shell_response() -> HTMLResponse:
            return HTMLResponse(
                (STATIC_DIR / "index.html").read_text(encoding="utf-8"),
                headers={"Cache-Control": "no-cache"},
            )

        @app.get("/ui", include_in_schema=False)
        @app.get("/ui/", include_in_schema=False)
        async def _ui_root() -> HTMLResponse:
            return _spa_shell_response()

        @app.get("/ui/{path:path}", include_in_schema=False)
        async def _ui_spa(path: str, request: Request) -> Response:
            # Don't shadow the API or the static-asset mount.
            if path.startswith("api/") or path.startswith("assets/"):
                raise HTTPException(status_code=404)
            return _spa_shell_response()

        # Bookmarks pointed at /ui2 during the Phase 0-3 dogfood era get
        # redirected so the operator's saved tab keeps working.
        @app.get("/ui2", include_in_schema=False)
        @app.get("/ui2/", include_in_schema=False)
        async def _ui2_redirect_root() -> RedirectResponse:
            return RedirectResponse("/ui/", status_code=301)

        @app.get("/ui2/{path:path}", include_in_schema=False)
        async def _ui2_redirect(path: str) -> RedirectResponse:
            return RedirectResponse(f"/ui/{path}", status_code=301)

    app.state.conscio_service = svc
    return app


async def _terminate_process() -> None:
    await asyncio.sleep(0.1)
    os.kill(os.getpid(), signal.SIGTERM)

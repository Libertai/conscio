from __future__ import annotations

import asyncio
import hmac
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from conscio.config import ServiceConfig
from conscio.service import ConscioService
from conscio.webui import create_web_router

STATIC_DIR = Path(__file__).resolve().parent / "static"

# API clients may submit messages/influences as "user" (default, executable)
# or "system" (observation-only). They must not mint "autonomous" or "tool"
# provenance — that would let an API caller forge the agent's own actions.
_ALLOWED_API_SOURCES = frozenset({"user", "system"})


def _validated_source(source: str) -> str:
    return source if source in _ALLOWED_API_SOURCES else "user"


class MessageRequest(BaseModel):
    content: str
    source: str = "user"


class InfluenceRequest(BaseModel):
    content: str
    source: str = "user"


def create_app(service: ConscioService | None = None, config: ServiceConfig | None = None) -> FastAPI:
    svc = service or ConscioService(config)

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        expected = svc.config.api_key
        if not expected:
            raise HTTPException(status_code=500, detail="API key is not configured")
        # Encode to bytes so a non-ASCII Authorization header raises a clean
        # 401 instead of a TypeError → 500 (matches web/auth.py hardening).
        provided = (authorization or "").encode("utf-8", "replace")
        wanted = f"Bearer {expected}".encode("utf-8", "replace")
        if not hmac.compare_digest(provided, wanted):
            raise HTTPException(status_code=401, detail="invalid API key")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await svc.start()
        try:
            yield
        finally:
            await svc.stop()

    app = FastAPI(title="Conscio", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "running": svc.running}

    @app.get("/status", dependencies=[Depends(require_auth)])
    async def status() -> dict[str, Any]:
        return (await svc.status()).__dict__

    @app.get("/metrics", dependencies=[Depends(require_auth)])
    async def metrics() -> dict[str, Any]:
        return await svc.metrics()

    @app.post("/message", dependencies=[Depends(require_auth)])
    async def message(req: MessageRequest) -> dict[str, Any]:
        result = await svc.submit_message(req.content, source=_validated_source(req.source))
        return {
            "output": result.output,
            "selected_action": result.selected_action,
            "session_id": result.session_id,
            "self_state": result.self_state,
            "attention_schema": result.attention_schema,
        }

    @app.post("/influence/goal", dependencies=[Depends(require_auth)])
    async def influence_goal(req: InfluenceRequest) -> dict[str, Any]:
        return await svc.submit_influence(req.content, kind="goal", source=_validated_source(req.source))

    @app.post("/influence/constraint", dependencies=[Depends(require_auth)])
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

    @app.post("/autonomy/tick", dependencies=[Depends(require_auth)])
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

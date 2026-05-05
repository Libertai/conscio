from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from conscio.config import ServiceConfig
from conscio.service import ConscioService
from conscio.webui import create_web_router


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
        if authorization != f"Bearer {expected}":
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

    @app.post("/message", dependencies=[Depends(require_auth)])
    async def message(req: MessageRequest) -> dict[str, Any]:
        result = await svc.submit_message(req.content, source=req.source)
        return {
            "output": result.output,
            "selected_action": result.selected_action,
            "session_id": result.session_id,
            "self_state": result.self_state,
            "attention_schema": result.attention_schema,
        }

    @app.post("/influence/goal", dependencies=[Depends(require_auth)])
    async def influence_goal(req: InfluenceRequest) -> dict[str, Any]:
        return await svc.submit_influence(req.content, kind="goal", source=req.source)

    @app.post("/influence/constraint", dependencies=[Depends(require_auth)])
    async def influence_constraint(req: InfluenceRequest) -> dict[str, Any]:
        return await svc.submit_influence(req.content, kind="constraint", source=req.source)

    @app.post("/control/pause", dependencies=[Depends(require_auth)])
    async def pause() -> dict[str, Any]:
        svc.pause()
        return {"paused": True}

    @app.post("/control/resume", dependencies=[Depends(require_auth)])
    async def resume() -> dict[str, Any]:
        svc.resume()
        return {"paused": False}

    @app.post("/control/stop", dependencies=[Depends(require_auth)])
    async def stop() -> dict[str, Any]:
        await svc.stop()
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
        await svc.set_project_status(project_id, "paused")
        return {"status": "paused"}

    @app.post("/projects/{project_id}/resume", dependencies=[Depends(require_auth)])
    async def resume_project(project_id: str) -> dict[str, str]:
        await svc.set_project_status(project_id, "active")
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
    app.state.conscio_service = svc
    return app

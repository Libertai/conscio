from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from conscio.config import ServiceConfig, load_config
from conscio.core.cognition import InputEvent
from conscio.core.runtime import CognitiveRuntime, EpisodeResult
from conscio.goals import GoalStore
from conscio.memory.store import MemoryStore
from conscio.tools import PolicyToolRegistry


@dataclass
class ServiceStatus:
    running: bool
    paused: bool
    session_id: str
    uptime: float
    autonomous: bool
    unsafe_autonomy: bool
    active_goal: dict[str, Any] | None = None
    episode_count: int = 0
    last_error: str = ""


@dataclass
class StoredEpisode:
    id: str
    source: str
    event_type: str
    input: str
    output: str
    selected_action: str
    created_at: float
    metrics: dict[str, Any] = field(default_factory=dict)


class ServiceLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(self.path, flags)
        except FileExistsError as exc:
            raise RuntimeError(f"Conscio service lock already exists: {self.path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "created_at": time.time()}))
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.acquired = False


class ConscioService:
    def __init__(self, config: ServiceConfig | None = None) -> None:
        self.config = config or load_config()
        self.config.ensure_layout()
        self.memory = MemoryStore(db_path=str(self.config.db_path))
        tools = PolicyToolRegistry(
            unsafe_autonomy=self.config.unsafe_autonomy,
            allowed_tools=self.config.allowed_tools,
            denied_tools=self.config.denied_tools,
            shell_timeout=self.config.shell_timeout,
        )
        tools.load_builtins()
        self.runtime = CognitiveRuntime(memory=self.memory, tools=tools)
        self.goals = GoalStore(self.memory)
        self.lock = ServiceLock(self.config.lock_path)
        self.queue: asyncio.Queue[InputEvent] = asyncio.Queue()
        self.running = False
        self.paused = False
        self.started_at = 0.0
        self.last_error = ""
        self._loop_task: asyncio.Task | None = None
        self._episodes: list[StoredEpisode] = []
        self._action_times: list[float] = []

    async def start(self, *, acquire_lock: bool = True, background: bool = True) -> None:
        if self.running:
            return
        if acquire_lock:
            self.lock.acquire()
        await self.goals.initialize()
        await self.runtime.initialize()
        self.running = True
        self.started_at = time.time()
        goal = await self.goals.active_goal()
        if goal:
            self.runtime.self_state.active_goal = goal.description
        if background and self.config.autonomous:
            self._loop_task = asyncio.create_task(self._autonomous_loop())

    async def stop(self) -> None:
        self.running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        await self.runtime.close()
        self.lock.release()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    async def submit_message(self, content: str, *, source: str = "user") -> EpisodeResult:
        return await self.run_event(InputEvent(content=content, source=source, event_type="message"))

    async def submit_influence(self, content: str, *, kind: str = "goal", source: str = "user") -> dict[str, Any]:
        influence = await self.goals.add_influence(content, kind=kind, source=source)
        await self.run_event(
            InputEvent(
                content=f"Influence received ({kind}): {content}",
                source=source,
                event_type=f"influence_{kind}",
                metadata={"influence_id": influence.id},
            )
        )
        return self.goals.as_dict(influence)

    async def run_autonomous_tick(self) -> EpisodeResult | None:
        if self.paused or not self.running:
            return None
        if not self._within_action_budget():
            return None
        goal = await self.goals.review()
        content = "Autonomous heartbeat: review my wants and choose the next useful action."
        if goal:
            content = f"Autonomous heartbeat: active goal is '{goal.description}'. Decide what I want to do next."
            self.runtime.self_state.active_goal = goal.description
        return await self.run_event(InputEvent(content=content, source="autonomous", event_type="heartbeat"))

    async def run_event(self, event: InputEvent) -> EpisodeResult:
        if not self.running:
            await self.start(background=False)
        self._action_times.append(time.time())
        try:
            result = await self.runtime.run_episode(event)
            self._store_episode(event, result)
            return result
        except Exception as exc:
            self.last_error = str(exc)
            if self.config.pause_on_error:
                self.pause()
            raise

    async def status(self) -> ServiceStatus:
        goal = await self.goals.active_goal()
        return ServiceStatus(
            running=self.running,
            paused=self.paused,
            session_id=self.runtime.session_id,
            uptime=time.time() - self.started_at if self.started_at else 0.0,
            autonomous=self.config.autonomous,
            unsafe_autonomy=self.config.unsafe_autonomy,
            active_goal=asdict(goal) if goal else None,
            episode_count=len(self._episodes),
            last_error=self.last_error,
        )

    async def recent_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        return [asdict(ep) for ep in self._episodes[-limit:]][::-1]

    async def recent_trace(self) -> str:
        return self.runtime.trace.format(limit=120)

    async def search_memory(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self.memory.search(query, limit)

    async def _autonomous_loop(self) -> None:
        while self.running:
            if not self.paused:
                await self.run_autonomous_tick()
            await asyncio.sleep(self.config.tick_interval)

    def _within_action_budget(self) -> bool:
        cutoff = time.time() - 3600
        self._action_times = [t for t in self._action_times if t >= cutoff]
        return len(self._action_times) < self.config.max_actions_per_hour

    def _store_episode(self, event: InputEvent, result: EpisodeResult) -> None:
        ep = StoredEpisode(
            id=f"{int(time.time() * 1000)}-{len(self._episodes) + 1}",
            source=event.source,
            event_type=event.event_type,
            input=event.content,
            output=result.output,
            selected_action=result.selected_action,
            created_at=time.time(),
            metrics=asdict(result.metrics),
        )
        self._episodes.append(ep)
        path = self.config.home / "events" / f"{ep.id}.json"
        path.write_text(json.dumps(asdict(ep), indent=2), encoding="utf-8")


def create_service(config_path: str | Path | None = None) -> ConscioService:
    return ConscioService(load_config(config_path))

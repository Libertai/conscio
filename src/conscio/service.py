from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from conscio.autonomy import AutonomyStore
from conscio.config import ServiceConfig, load_config
from conscio.core.context import ContextSettings
from conscio.core.cognition import InputEvent
from conscio.core.runtime import CognitiveRuntime, EpisodeResult
from conscio.goals import GoalStore
from conscio.llm.client import LLMClient
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
    queue_depth: int = 0
    current_event: str = ""
    current_project: dict[str, Any] | None = None
    current_task: dict[str, Any] | None = None
    last_autonomous_action: str = ""
    actions_last_hour: int = 0
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


@dataclass
class QueuedEvent:
    event: InputEvent
    future: asyncio.Future[EpisodeResult | None]
    autonomous: bool = False


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
            if self._remove_stale_lock():
                fd = os.open(self.path, flags)
            else:
                raise RuntimeError(f"Conscio service lock already exists: {self.path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "created_at": time.time()}))
        self.acquired = True

    def _remove_stale_lock(self) -> bool:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            self.path.unlink(missing_ok=True)
            return True
        except PermissionError:
            return False
        return False

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
            working_directory=self.config.working_directory,
        )
        tools.load_builtins()
        self._register_memory_tools(tools)
        llm = None
        if self.config.llm_base_url:
            llm = LLMClient(
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
                model=self.config.llm_model,
            )
        context_settings = ContextSettings(
            recent_episodes=self.config.context_recent_episodes,
            retrieved_memories=self.config.context_retrieved_memories,
            workspace_entries=self.config.context_workspace_entries,
            max_dynamic_chars=self.config.context_max_dynamic_chars,
            compaction_interval=self.config.context_compaction_interval,
            enable_semantic_compaction=self.config.context_enable_semantic_compaction,
        )
        self.runtime = CognitiveRuntime(
            memory=self.memory,
            tools=tools,
            llm=llm,
            context_settings=context_settings,
            context_provider=self._context_state,
            max_tool_rounds=self.config.model_tool_rounds,
        )
        self.goals = GoalStore(self.memory)
        self.autonomy = AutonomyStore(self.memory)
        self.lock = ServiceLock(self.config.lock_path)
        self.queue: asyncio.Queue[QueuedEvent] = asyncio.Queue()
        self.running = False
        self.paused = False
        self.started_at = 0.0
        self.last_error = ""
        self.current_event = ""
        self.last_autonomous_action = ""
        self.latest_model_context = ""
        self._loop_task: asyncio.Task | None = None
        self._event_task: asyncio.Task | None = None
        self._current_queue_item: QueuedEvent | None = None
        self._episodes: list[StoredEpisode] = []
        self._autonomous_action_times: list[float] = []
        self._event_lock = asyncio.Lock()

    def _register_memory_tools(self, tools: PolicyToolRegistry) -> None:
        async def remember_fact(
            fact: str | None = None,
            facts: list[str] | None = None,
            source: str = "agent",
            confidence: str = "HIGH",
            input: str | None = None,
        ) -> dict[str, Any]:
            candidates: list[str] = []
            if facts:
                candidates.extend(str(item) for item in facts)
            if fact:
                candidates.append(fact)
            if input:
                candidates.append(input)
            cleaned = [" ".join(item.strip().split()) for item in candidates if item and item.strip()]
            if not cleaned:
                return {"output": "No fact provided.", "error": True}
            for item in dict.fromkeys(cleaned):
                await self.memory.add_fact(item[:500], source=source, confidence=confidence)
            return {"output": f"Stored {len(dict.fromkeys(cleaned))} fact(s) in semantic memory.", "error": False}

        async def search_memory(
            query: str | None = None,
            limit: int = 10,
            input: str | None = None,
        ) -> dict[str, Any]:
            q = query if query is not None else input
            if not q:
                return {"output": "No memory query provided.", "error": True}
            rows = await self.memory.search_facts(q, limit)
            if not rows:
                return {"output": "No matching semantic memories found.", "error": False}
            lines = [f"- {row['fact']} ({row.get('confidence', '')})" for row in rows]
            return {"output": "\n".join(lines), "error": False, "results": rows}

        tools.register(
            "remember_fact",
            remember_fact,
            "Store one semantic fact in long-term memory.",
        )
        tools.register(
            "remember_facts",
            remember_fact,
            "Store one or more semantic facts in long-term memory.",
        )
        tools.register(
            "search_memory",
            search_memory,
            "Search semantic long-term memory facts.",
        )

    async def start(self, *, acquire_lock: bool = True, background: bool = True) -> None:
        if self.running:
            return
        if acquire_lock:
            self.lock.acquire()
        try:
            await self.goals.initialize()
            await self.autonomy.initialize()
            await self.runtime.initialize()
            self.running = True
            self.started_at = time.time()
            goal = await self.goals.active_goal()
            if goal:
                self.runtime.self_state.active_goal = goal.description
            if background:
                self._event_task = asyncio.create_task(self._event_worker())
            if background and self.config.autonomous:
                self._loop_task = asyncio.create_task(self._autonomous_loop())
        except Exception:
            self.lock.release()
            raise

    async def stop(self) -> None:
        self.running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        if self._event_task is not None:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
            self._event_task = None
        self._fail_pending_events(RuntimeError("Conscio service stopped."))
        await self.runtime.close()
        self.lock.release()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    async def submit_message(self, content: str, *, source: str = "user") -> EpisodeResult:
        result = await self._submit_event(InputEvent(content=content, source=source, event_type="message"))
        assert result is not None
        return result

    async def submit_influence(self, content: str, *, kind: str = "goal", source: str = "user") -> dict[str, Any]:
        influence = await self.goals.add_influence(content, kind=kind, source=source)
        await self._submit_event(
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
        if not self._within_action_budget(self._autonomous_action_times, self.config.max_actions_per_hour):
            return None
        self._autonomous_action_times.append(time.time())
        return await self._submit_event(
            InputEvent(
                content="Autonomous heartbeat: review my wants and choose the next useful action.",
                source="autonomous",
                event_type="heartbeat",
            ),
            autonomous=True,
        )

    async def run_event(self, event: InputEvent) -> EpisodeResult:
        result = await self._process_event(event)
        assert result is not None
        return result

    async def _submit_event(self, event: InputEvent, *, autonomous: bool = False) -> EpisodeResult | None:
        if not self.running:
            await self.start(background=False)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[EpisodeResult | None] = loop.create_future()
        if self._event_task is not None:
            await self.queue.put(QueuedEvent(event=event, future=future, autonomous=autonomous))
            return await future
        return await self._process_event(event, autonomous=autonomous)

    async def _event_worker(self) -> None:
        while self.running:
            item = await self.queue.get()
            self._current_queue_item = item
            try:
                result = await self._process_event(item.event, autonomous=item.autonomous)
                if not item.future.done():
                    item.future.set_result(result)
            except asyncio.CancelledError as exc:
                if not item.future.done():
                    item.future.set_exception(RuntimeError("Conscio service stopped."))
                raise exc
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            finally:
                if self._current_queue_item is item:
                    self._current_queue_item = None
                self.queue.task_done()

    def _fail_pending_events(self, exc: Exception) -> None:
        if self._current_queue_item is not None and not self._current_queue_item.future.done():
            self._current_queue_item.future.set_exception(exc)
            self._current_queue_item = None
        while True:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not item.future.done():
                item.future.set_exception(exc)
            self.queue.task_done()

    async def _process_event(self, event: InputEvent, *, autonomous: bool = False) -> EpisodeResult | None:
        async with self._event_lock:
            self.current_event = f"{event.source}:{event.event_type}"
            try:
                if autonomous:
                    return await self._plan_and_act(event)
                return await self._run_episode(event)
            finally:
                self.current_event = ""

    async def _run_episode(self, event: InputEvent) -> EpisodeResult:
        try:
            result = await self.runtime.run_episode(event)
            self.latest_model_context = result.model_context
            await self._store_episode(event, result)
            return result
        except Exception as exc:
            self.last_error = str(exc)
            if self.config.pause_on_error:
                self.pause()
            raise

    async def _plan_and_act(self, event: InputEvent) -> EpisodeResult:
        goal = await self.goals.review()
        if goal is None:
            return await self._run_episode(event)
        self.runtime.self_state.active_goal = goal.description
        project = await self.autonomy.get_or_create_project(goal.id, goal.description)
        if project is None:
            self.last_autonomous_action = "wait:project_paused"
            return await self._run_episode(
                InputEvent(
                    content=f"Autonomous heartbeat: active goal '{goal.description}' has a paused project, so I will not continue it.",
                    source="autonomous",
                    event_type="project_paused",
                )
            )
        constraints = await self.goals.active_constraints()
        recent_tool_actions = await self.autonomy.count_recent_actions("tool")
        if self.config.unsafe_autonomy and recent_tool_actions < self.config.max_actions_per_hour:
            note = (
                f"Conscio autonomous action for goal {goal.id}: "
                f"{goal.description[:80]} | constraints={len(constraints)}"
            )
            command = f"printf '%s\\n' {json.dumps(note)} | tee -a conscio_autonomy.log"
            task = await self.autonomy.ensure_next_task(
                project.id,
                "Write an autonomous progress note inside the configured VM working directory.",
                tool_name="bash",
                tool_args={"input": command},
            )
            await self.autonomy.update_task(task.id, status="active")
            tool_result = await self.runtime.tools.call(task.tool_name or "bash", task.tool_args)
            await self.autonomy.record_action("tool")
            status = "done" if not tool_result.get("error") and int(tool_result.get("exit_code", 0)) == 0 else "blocked"
            await self.autonomy.update_task(task.id, status=status, result=str(tool_result.get("output", ""))[:1000])
            self.last_autonomous_action = f"tool:{task.tool_name}:{status}"
            tool_event = InputEvent(
                content=f"Autonomous tool result for task {task.id}: {tool_result.get('output', '')}",
                source="tool",
                event_type="tool_result",
                metadata={"project_id": project.id, "task_id": task.id, "tool": task.tool_name},
            )
            return await self._run_episode(tool_event)
        task = await self.autonomy.ensure_next_task(
            project.id,
            "Reflect on the active goal and choose a concrete next step.",
        )
        await self.autonomy.update_task(task.id, status="active", result="Reflection episode queued.")
        self.last_autonomous_action = "reflect"
        content = (
            f"Autonomous heartbeat: active goal is '{goal.description}'. "
            f"Current project is '{project.title}'. Current task is '{task.description}'. "
            f"Constraints: {[c['content'] for c in constraints]}. Decide the next concrete action."
        )
        return await self._run_episode(InputEvent(content=content, source="autonomous", event_type="heartbeat"))

    async def status(self) -> ServiceStatus:
        goal = await self.goals.active_goal()
        current_project = await self.autonomy.active_project()
        current_task = await self.autonomy.active_task()
        return ServiceStatus(
            running=self.running,
            paused=self.paused,
            session_id=self.runtime.session_id,
            uptime=time.time() - self.started_at if self.started_at else 0.0,
            autonomous=self.config.autonomous,
            unsafe_autonomy=self.config.unsafe_autonomy,
            active_goal=asdict(goal) if goal else None,
            queue_depth=self.queue.qsize(),
            current_event=self.current_event,
            current_project=current_project,
            current_task=current_task,
            last_autonomous_action=self.last_autonomous_action,
            actions_last_hour=await self.autonomy.count_recent_actions("tool"),
            episode_count=len(await self.recent_episodes(1000)),
            last_error=self.last_error,
        )

    async def recent_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        stored = await self.autonomy.recent_episodes(limit)
        if stored:
            return stored
        return [asdict(ep) for ep in self._episodes[-limit:]][::-1]

    async def recent_trace(self) -> str:
        stored = await self.autonomy.recent_trace()
        return stored or self.runtime.trace.format(limit=120)

    async def search_memory(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self.memory.search(query, limit)

    async def recent_facts(self, limit: int = 10) -> list[dict[str, Any]]:
        return await self.memory.recent_facts(limit)

    async def list_skills(self) -> list[dict[str, Any]]:
        return await self.memory.list_skills()

    async def list_projects(self) -> list[dict[str, Any]]:
        return await self.autonomy.list_projects()

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        return await self.autonomy.get_project(project_id)

    async def set_project_status(self, project_id: str, status: str) -> None:
        changed = await self.autonomy.set_project_status(project_id, status)
        if not changed:
            raise KeyError(project_id)

    async def list_influences(self) -> list[dict[str, Any]]:
        return await self.goals.list_influences()

    async def _autonomous_loop(self) -> None:
        while self.running:
            if not self.paused:
                await self.run_autonomous_tick()
            await asyncio.sleep(self.config.tick_interval)

    def _recent_actions(self, items: list[float]) -> list[float]:
        cutoff = time.time() - 3600
        return [t for t in items if t >= cutoff]

    def _within_action_budget(self, items: list[float], limit: int) -> bool:
        recent = self._recent_actions(items)
        items[:] = recent
        return len(recent) < limit

    async def _store_episode(self, event: InputEvent, result: EpisodeResult) -> None:
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
        await self.autonomy.store_episode(
            episode_id=ep.id,
            source=ep.source,
            event_type=ep.event_type,
            input=ep.input,
            output=ep.output,
            selected_action=ep.selected_action,
            metrics=dict(ep.metrics),
            trace=result.cognitive_trace,
        )

    async def _context_state(self) -> dict[str, Any]:
        goal = await self.goals.active_goal()
        return {
            "active_goal": asdict(goal) if goal else None,
            "current_project": await self.autonomy.active_project(),
            "current_task": await self.autonomy.active_task(),
            "paused": self.paused,
            "autonomous": self.config.autonomous,
            "last_autonomous_action": self.last_autonomous_action,
        }


def create_service(config_path: str | Path | None = None) -> ConscioService:
    return ConscioService(load_config(config_path))

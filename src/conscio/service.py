from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from conscio.autonomy import AutonomyStore
from conscio.config import ServiceConfig, load_config
from conscio.core.cognition import InputEvent
from conscio.core.context import ContextSettings, provenance_marker
from conscio.core.runtime import CognitiveRuntime, EpisodeResult
from conscio.core.subagent import SubagentRunner, SubagentSpec
from conscio.core.tool_loop import external_taint_origin
from conscio.goals import GoalStore
from conscio.llm.router import LLMRouter
from conscio.memory.consolidation import ConsolidationEngine
from conscio.memory.embeddings import LibertAIEmbedder
from conscio.memory.lifecycle import create_home_backup, prune_backups
from conscio.memory.store import MemoryStore
from conscio.tools import PolicyToolRegistry, ScopedToolRegistry

logger = logging.getLogger(__name__)


# Tasks-state sentinel: "no pending task" is visible model state the agent must
# resolve itself (the v1 filler task is gone).
NO_PENDING_TASK_SENTINEL = (
    "NO_PENDING_TASK — you must add_task or set_task_status before acting"
)

# Self-management tools never count against the per-hour world-tool budget.
_SELF_MANAGEMENT_TOOLS = {
    "set_task_status",
    "add_task",
    "note_progress",
    "propose_subgoal",
    "learn_procedure",
}

# Consecutive add-only autonomous ticks before the task-discipline hard rule fires.
_ADD_ONLY_TICK_LIMIT = 3
_REDACT_KEYS = ("password", "token", "secret", "api_key", "apikey", "key")

# In-memory episode list and on-disk event-file caps to prevent unbounded
# growth in long-running services. The SQLite store is the source of truth;
# these are just fast-access caches / audit files.
_MAX_IN_MEMORY_EPISODES = 200
_MAX_EVENT_FILES = 500


@dataclass
class EpisodeTaint:
    """Per-episode taint tracker (quarantine defense): set when a web tool runs;
    consulted by remember_fact/remember_facts so web-derived knowledge is written
    with origin='web:<url>' and trust tier 1."""

    web: bool = False
    urls: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.web = False
        self.urls = []

    def note(self, url: str) -> None:
        self.web = True
        if url and url not in self.urls:
            self.urls.append(url)

    def to_dict(self) -> dict[str, Any]:
        return {"web": self.web, "urls": list(self.urls)}


@dataclass
class ServiceStatus:
    running: bool
    paused: bool
    session_id: str
    uptime: float
    autonomous: bool
    unsafe_autonomy: bool
    agent_profile: str = ""
    premises: str = ""
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


class EpisodeCancelled(RuntimeError):
    """Raised into waiters when the running episode is cancelled or times out."""


@dataclass
class QueuedEvent:
    event: InputEvent
    future: asyncio.Future[EpisodeResult | None]
    autonomous: bool = False
    ref: str = ""


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
        self._register_self_tools(tools)
        self.router = LLMRouter.from_config(self.config)
        llm = self.router.for_role("main") if self.router is not None else None
        self.llm_fast = self.router.for_role("fast") if self.router is not None else None
        if self.router is not None:
            embed_client = self.router.for_role("embeddings")
            self.memory.embedder = LibertAIEmbedder(embed_client, model=embed_client.model)
        context_settings = ContextSettings(
            recent_episodes=self.config.context_recent_episodes,
            retrieved_memories=self.config.context_retrieved_memories,
            workspace_entries=self.config.context_workspace_entries,
            max_dynamic_chars=self.config.context_max_dynamic_chars,
            compaction_interval=self.config.context_compaction_interval,
            enable_semantic_compaction=self.config.context_enable_semantic_compaction,
        )
        self.goals = GoalStore(self.memory, motivation=self.config.motivation)
        self.autonomy = AutonomyStore(
            self.memory,
            stale_flag_days=self.config.motivation.stale_flag_days,
            stale_block_days=self.config.motivation.stale_block_days,
        )
        self.runtime = CognitiveRuntime(
            memory=self.memory,
            tools=tools,
            llm=llm,
            context_settings=context_settings,
            context_provider=self._context_state,
            max_tool_rounds=self.config.model_tool_rounds,
            max_ticks=self.config.max_ticks,
            tool_rounds_per_tick=self.config.tool_rounds_per_tick,
            max_reflections=self.config.max_reflections,
            attention_broadcast_limit=self.config.attention_broadcast_limit,
            attention_char_budget=self.config.attention_char_budget,
            ablation=self.config.ablation,
            constraint_provider=self.goals.active_constraints,
            llm_fast=self.llm_fast,
            chat_temperature=self.config.chat_temperature,
            autonomous_temperature=self.config.autonomous_temperature,
            loop_max_tokens=(self.router.roles["main"].max_tokens if self.router is not None else None) or 2400,
            judge_max_tokens=self.config.judge_max_tokens,
            appraisal_max_tokens=self.config.appraisal_max_tokens,
        )
        # Strategy wiring through the public surface (no module-private pokes).
        self.runtime.autonomous_strategy.context_provider = self._autonomous_context_state
        self.runtime.autonomous_strategy.on_tool_observation = self._on_autonomous_tool_observation
        self.runtime.chat_strategy.on_tool_observation = self._on_chat_tool_observation
        self.runtime.executor.on_stream_event = self._on_stream_event
        self.episode_taint = EpisodeTaint()
        # Periodic budgeted consolidation (design §4.3): one persistent engine
        # so _last_cycle_ts carries the summarization window across cycles.
        self.consolidation = ConsolidationEngine(self.memory, llm=self.llm_fast or llm)
        self._consolidation_ticks = 0
        self._add_only_ticks = 0
        self._current_goal_id: str | None = None
        self._current_project_id: str | None = None
        self._tick_count = 0
        self._goal_review_interval = 10
        self.lock = ServiceLock(self.config.lock_path)
        # Interactive events (user/API) outrank autonomous heartbeats; seq keeps
        # FIFO order within a priority class (QueuedEvent itself is not comparable).
        self.queue: asyncio.PriorityQueue[tuple[int, int, QueuedEvent]] = asyncio.PriorityQueue()
        self._event_seq = itertools.count()
        self._pending_interactive = 0
        self.running = False
        self.paused = False
        self.started_at = 0.0
        self.last_error = ""
        self.current_event = ""
        self._current_ref = ""
        self.last_autonomous_action = ""
        self.latest_model_context = ""
        self._loop_task: asyncio.Task | None = None
        self._event_task: asyncio.Task | None = None
        self._current_queue_item: QueuedEvent | None = None
        self._current_episode_task: asyncio.Task | None = None
        self._episodes: list[StoredEpisode] = []
        self._backup_task: asyncio.Task | None = None
        self.last_backup_at: float = 0.0
        self.last_backup_error: str = ""
        self.rate_limited_total = 0
        # Lazily imported like the broker: conscio.web must not import at module load.
        from conscio.web.ratelimit import TokenBucket  # noqa: PLC0415
        self.episode_bucket: TokenBucket | None = (
            TokenBucket(
                capacity=float(self.config.episode_rate_burst),
                refill_per_second=self.config.episode_rate_per_minute / 60.0,
            )
            if self.config.episode_rate_per_minute > 0
            else None
        )
        self._autonomous_action_times: list[float] = []
        self._event_lock = asyncio.Lock()
        # SSE broker — attached on start(), detached on stop(). Imported lazily
        # so circular imports through conscio.web stay impossible.
        from conscio.web.events import WorkspaceEventBroker  # noqa: PLC0415
        self._event_broker: WorkspaceEventBroker = WorkspaceEventBroker(self.runtime.workspace)
        # MCP servers register their tools on the policy registry so allow/deny
        # and the action budget apply to them unchanged. Imported lazily to keep
        # the mcp dependency off the hot import path for CLI one-shots.
        from conscio.mcp_client import McpManager  # noqa: PLC0415
        self.mcp = McpManager(
            self.config.mcp_servers,
            self.runtime.tools,
            on_event=self._event_broker.emit,
        )

    def _register_self_tools(self, tools: PolicyToolRegistry) -> None:
        async def remember_fact(
            fact: str | None = None,
            facts: list[str] | None = None,
            source: str = "agent",
            confidence: str = "HIGH",
            input: str | None = None,
        ) -> dict[str, Any]:
            # Confabulated-provenance defense: the model must never mint
            # user-tier (trust 3) facts. Whatever `source` it passes, only
            # 'agent' is accepted from this tool.
            if source != "agent":
                source = "agent"
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
            episode_id = self.runtime.workspace.current_episode or None
            stored = len(dict.fromkeys(cleaned))
            if self.episode_taint.web:
                # Quarantine: this episode fetched web content, so any fact it
                # stores is web-derived (origin=web:<url>, trust tier 1).
                url = self.episode_taint.urls[-1] if self.episode_taint.urls else ""
                origin = f"web:{url}" if url else "web"
                for item in dict.fromkeys(cleaned):
                    await self.memory.add_fact(
                        item[:500],
                        origin=origin,
                        trust=1,
                        episode_id=episode_id,
                        confidence=confidence,
                    )
                return {
                    "output": (
                        f"Stored {stored} fact(s) in semantic memory "
                        f"(web-derived this episode: origin={origin}, trust=1)."
                    ),
                    "error": False,
                }
            for item in dict.fromkeys(cleaned):
                await self.memory.add_fact(
                    item[:500], source=source, episode_id=episode_id, confidence=confidence
                )
            return {"output": f"Stored {stored} fact(s) in semantic memory.", "error": False}

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
            # Same provenance marking as the prompt assemblers: web-derived
            # (trust 1) and user facts stay visibly labelled even when the
            # model pulls them explicitly through this tool.
            lines = [
                f"- {provenance_marker(row)}{row['fact']} ({row.get('confidence', '')})"
                for row in rows
            ]
            return {"output": "\n".join(lines), "error": False, "results": rows}

        async def set_task_status(
            task_id: str,
            status: str,
            result: str = "",
        ) -> dict[str, Any]:
            allowed = {"pending", "active", "done", "blocked"}
            if status not in allowed:
                return {"output": f"Invalid status '{status}'. Use one of: {sorted(allowed)}.", "error": True}
            task = await self.autonomy.get_task(task_id)
            if task is None:
                return {"output": f"Unknown task_id '{task_id}'.", "error": True}
            await self.autonomy.update_task(task_id, status=status, result=result[:1000])
            await self.autonomy.record_action("self_management")
            return {"output": f"Task {task_id} -> {status}.", "error": False}

        async def add_task(
            project_id: str,
            description: str,
            tool_name: str | None = None,
            tool_args: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            project = await self.autonomy.get_project(project_id)
            if project is None:
                return {"output": f"Unknown project_id '{project_id}'.", "error": True}
            task = await self.autonomy.add_task(
                project_id, description, tool_name=tool_name, tool_args=tool_args or {}
            )
            await self.autonomy.record_action("self_management")
            return {"output": f"Added task {task.id}: {description[:200]}.", "error": False, "task_id": task.id}

        async def note_progress(
            project_id: str | None = None,
            content: str = "",
        ) -> dict[str, Any]:
            text = content.strip()
            if not text:
                return {"output": "No content provided.", "error": True}
            note = f"[project={project_id or 'none'}] {text[:1500]}"
            await self.autonomy.note_progress(None, note)
            await self.autonomy.record_action("self_management")
            return {"output": "Progress note recorded.", "error": False}

        async def propose_subgoal(
            description: str,
            rationale: str = "",
        ) -> dict[str, Any]:
            text = description.strip()
            if not text:
                return {"output": "No description provided.", "error": True}
            if self.episode_taint.web:
                origin = self.episode_taint.urls[-1] if self.episode_taint.urls else "external content"
                influence = await self.goals.defer_influence(
                    text[:500],
                    kind="goal",
                    source="external_content",
                    reasoning=(
                        "Deferred because the proposal was made in an episode "
                        f"tainted by external content ({origin})."
                    ),
                    response="I deferred this proposed goal for later clean review.",
                )
                await self.autonomy.record_action("self_management")
                return {
                    "output": f"Deferred proposed goal as influence {influence.id}.",
                    "error": False,
                    "accepted": False,
                    "influence_id": influence.id,
                }
            proposed = await self.goals.propose_goal(
                text[:500],
                rationale=rationale[:500],
                source="self_proposed",
                priority=0.5,
                confidence=0.5,
                appraisal_weight=0.5,
            )
            await self.autonomy.record_action("self_management")
            if not proposed["accepted"]:
                return {
                    "output": str(proposed["reason"]),
                    "error": False,
                    "accepted": False,
                    "similar_goal_id": proposed.get("similar_goal_id"),
                }
            goal = proposed["goal"]
            return {
                "output": f"Proposed new goal {goal.id}: {text[:200]}.",
                "error": False,
                "accepted": True,
                "goal_id": goal.id,
            }

        async def learn_procedure(
            name: str,
            description: str,
            steps: str,
            trigger: str = "",
        ) -> dict[str, Any]:
            if self.episode_taint.web:
                return {
                    "output": (
                        "learn_procedure is disabled after external content or "
                        "network reads in this episode. Re-evaluate it in a clean episode."
                    ),
                    "error": True,
                }
            slug = " ".join(name.strip().split())
            if not slug or not description.strip() or not steps.strip():
                return {"output": "learn_procedure requires name, description and steps.", "error": True}
            await self.memory.upsert_procedure(
                slug[:80],
                description.strip()[:500],
                steps.strip()[:2000],
                trigger=trigger.strip()[:300],
            )
            await self.autonomy.record_action("self_management")
            return {"output": f"Recorded procedure '{slug[:80]}'.", "error": False}

        tools.register(
            "remember_fact",
            remember_fact,
            "Store one semantic fact in long-term memory.",
            capabilities={"memory_write"},
            schema={
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"], "default": "HIGH"},
                },
                "required": ["fact"],
                "additionalProperties": False,
            },
        )
        tools.register(
            "remember_facts",
            remember_fact,
            "Store one or more semantic facts in long-term memory.",
            capabilities={"memory_write"},
            schema={
                "type": "object",
                "properties": {
                    "facts": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"], "default": "HIGH"},
                },
                "required": ["facts"],
                "additionalProperties": False,
            },
        )
        tools.register(
            "search_memory",
            search_memory,
            "Search semantic long-term memory facts.",
            capabilities={"memory_read"},
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        tools.register(
            "set_task_status",
            set_task_status,
            "Mark a task pending/active/done/blocked. Use done when the task is "
            "complete; blocked only after attempting it.",
            capabilities={"self_management"},
            schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "active", "done", "blocked"]},
                    "result": {"type": "string", "description": "Short result/note to attach to the task."},
                },
                "required": ["task_id", "status"],
                "additionalProperties": False,
            },
        )
        tools.register(
            "add_task",
            add_task,
            "Add a new task to a project. Use to break work into concrete next steps.",
            capabilities={"self_management"},
            schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "description": {"type": "string"},
                    "tool_name": {"type": "string", "description": "Optional preferred tool to invoke."},
                    "tool_args": {"type": "object", "description": "Optional preferred args for tool_name."},
                },
                "required": ["project_id", "description"],
                "additionalProperties": False,
            },
        )
        tools.register(
            "note_progress",
            note_progress,
            "Record a free-form progress note in the autonomous trace and episodic memory.",
            capabilities={"self_management"},
            schema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        )
        tools.register(
            "propose_subgoal",
            propose_subgoal,
            "Propose a new self-authored goal. Will be reviewed in the next goal-review cycle.",
            capabilities={"self_modification"},
            schema={
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["description"],
                "additionalProperties": False,
            },
        )
        tools.register(
            "learn_procedure",
            learn_procedure,
            "Record a validated, reusable procedure you have confirmed works.",
            capabilities={"self_modification", "memory_write"},
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "steps": {"type": "string"},
                    "trigger": {"type": "string"},
                },
                "required": ["name", "description", "steps"],
                "additionalProperties": False,
            },
        )

        async def spawn_subagent(
            task: str,
            context: str = "",
            tools_allowed: list[str] | None = None,
            max_rounds: int | None = None,
        ) -> dict[str, Any]:
            text = (task or "").strip()
            if not text:
                return {"output": "No task provided.", "error": True}
            parent_id = self.runtime.workspace.current_episode or ""
            sub_id = uuid.uuid4().hex
            scoped = ScopedToolRegistry(
                self.runtime.tools,
                allowed=set(tools_allowed) if tools_allowed else None,
                denied_capabilities=frozenset(self.config.subagent_deny_capabilities),
            )
            spec = SubagentSpec(task=text, context=context or "", tools=tools_allowed, max_rounds=max_rounds)
            runner = SubagentRunner(
                llm=self._subagent_llm(),
                tools=scoped,
                on_tool_observation=self._subagent_observer(sub_id),
                emit=self._event_broker.emit,
                max_rounds=self.config.subagent_max_rounds,
                max_seconds=self.config.subagent_max_seconds,
            )
            outcome = await runner.run(spec, parent_episode_id=parent_id, subagent_id=sub_id)
            await self.memory.record_episode(
                episode_id=outcome.id,
                source="subagent",
                event_type="subagent_task",
                input=text,
                output=outcome.output or outcome.error,
                selected_action="answer" if not outcome.error else "error",
                tainted=self.episode_taint.web,
                metrics={
                    "rounds": outcome.rounds,
                    "tool_calls": len(outcome.tool_requests),
                    "limit_reached": outcome.limit_reached,
                },
                parent_episode_id=parent_id or None,
            )
            if outcome.error and not outcome.output:
                return {"output": outcome.error, "error": True, "subagent_id": outcome.id, "rounds": outcome.rounds}
            return {"output": outcome.output, "error": False, "subagent_id": outcome.id, "rounds": outcome.rounds}

        if self.config.subagents_enabled:
            tools.register(
                "spawn_subagent",
                spawn_subagent,
                "Delegate a bounded task to a focused sub-agent with its own tool "
                "loop and model. Returns the sub-agent's final result.",
                capabilities={"delegation"},
                schema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Self-contained task for the sub-agent."},
                        "context": {
                            "type": "string",
                            "description": "Optional extra context pasted into the sub-agent prompt.",
                        },
                        "tools_allowed": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional allowlist restricting the sub-agent's tools further.",
                        },
                        "max_rounds": {"type": "integer", "minimum": 1, "maximum": 32},
                    },
                    "required": ["task"],
                    "additionalProperties": False,
                },
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
            self._event_broker.attach()
            await self.mcp.start()
            goal = await self.goals.active_goal()
            if goal:
                self.runtime.self_state.active_goal = goal.description
            if background:
                self._event_task = asyncio.create_task(self._event_worker())
            if background and self.config.autonomous:
                self._loop_task = asyncio.create_task(self._autonomous_loop())
            if background and self.config.backup_interval_hours > 0:
                self._backup_task = asyncio.create_task(self._backup_loop())
        except Exception:
            self.lock.release()
            raise

    async def stop(self) -> None:
        self.running = False
        if self._current_episode_task is not None and not self._current_episode_task.done():
            self._current_episode_task.cancel()
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        if self._backup_task is not None:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
            self._backup_task = None
        if self._event_task is not None:
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
            self._event_task = None
        self._fail_pending_events(RuntimeError("Conscio service stopped."))
        await self.mcp.stop()
        self._event_broker.detach()
        await self.runtime.close()
        self.lock.release()

    @property
    def event_broker(self):
        """Public accessor for the SSE broker (consumed by the web router)."""
        return self._event_broker

    def pause(self) -> None:
        self.paused = True
        self._event_broker.emit("control.paused", {"paused": True})

    def resume(self) -> None:
        self.paused = False
        self._event_broker.emit("control.paused", {"paused": False})

    def cancel_current(self) -> dict[str, Any]:
        """Cancel the episode currently being processed, if any. The worker
        survives; the waiting caller receives EpisodeCancelled."""
        task = self._current_episode_task
        if task is None or task.done():
            return {"cancelled": False, "current_event": self.current_event}
        cancelled_event = self.current_event
        task.cancel()
        self._event_broker.emit("episode.cancelled", {"event": cancelled_event})
        return {"cancelled": True, "current_event": cancelled_event}

    def _on_stream_event(self, data: dict[str, Any]) -> None:
        """ToolLoopSession token hook → SSE broker. Sync and loop-affine (the
        session runs on the service loop), so emit() dispatches directly."""
        name = {"token": "chat.token", "discard": "chat.discard", "final": "chat.final"}.get(
            str(data.get("event") or "")
        )
        if name is None:
            return
        payload: dict[str, Any] = {
            "episode_id": self.runtime.workspace.current_episode,
            "source": self.current_event,
            "ref": self._current_ref,
            "round": data.get("round", 0),
        }
        if name == "chat.token":
            payload["text"] = str(data.get("text") or "")
        self._event_broker.emit(name, payload)

    def try_acquire_episode(self) -> tuple[bool, float]:
        """Global inbound rate check for episode-triggering endpoints."""
        if self.episode_bucket is None:
            return True, 0.0
        allowed, retry_after = self.episode_bucket.try_acquire()
        if not allowed:
            self.rate_limited_total += 1
            logger.warning("episode request rate-limited (retry in %.1fs)", retry_after)
        return allowed, retry_after

    async def submit_message(self, content: str, *, source: str = "user", ref: str = "") -> EpisodeResult:
        result = await self._submit_event(
            InputEvent(content=content, source=source, event_type="message"), ref=ref
        )
        assert result is not None
        return result

    async def submit_influence(self, content: str, *, kind: str = "goal", source: str = "user") -> dict[str, Any]:
        influence = await self.goals.add_influence(
            content, kind=kind, source=source, llm=self.llm_fast or self.runtime.autonomous_strategy.llm
        )
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
        persistent_tool_actions = await self.autonomy.count_recent_actions("tool")
        if persistent_tool_actions >= self.config.max_actions_per_hour:
            self.last_autonomous_action = "wait:budget_exhausted"
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

    async def _submit_event(
        self, event: InputEvent, *, autonomous: bool = False, ref: str = ""
    ) -> EpisodeResult | None:
        if not self.running:
            await self.start(background=False)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[EpisodeResult | None] = loop.create_future()
        if self._event_task is not None:
            if not autonomous:
                self._pending_interactive += 1
            priority = 1 if autonomous else 0
            await self.queue.put(
                (
                    priority,
                    next(self._event_seq),
                    QueuedEvent(event=event, future=future, autonomous=autonomous, ref=ref),
                )
            )
            return await future
        return await self._process_event(event, autonomous=autonomous, ref=ref)

    async def _event_worker(self) -> None:
        while self.running:
            _, _, item = await self.queue.get()
            self._current_queue_item = item
            if not item.autonomous and self._pending_interactive > 0:
                self._pending_interactive -= 1
            task = asyncio.create_task(self._process_event(item.event, autonomous=item.autonomous, ref=item.ref))
            self._current_episode_task = task
            try:
                result = await task
                if not item.future.done():
                    item.future.set_result(result)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    # The worker itself is being stopped (service shutdown).
                    if not item.future.done():
                        item.future.set_exception(RuntimeError("Conscio service stopped."))
                    task.cancel()
                    raise
                # Operator cancelled the episode task; the worker survives.
                if not item.future.done():
                    item.future.set_exception(EpisodeCancelled("episode cancelled by operator"))
                self.last_error = "episode_cancelled"
            except TimeoutError:
                if not item.future.done():
                    item.future.set_exception(
                        EpisodeCancelled(f"episode exceeded {self.config.episode_timeout:.0f}s")
                    )
                self.last_error = "episode_timeout"
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            finally:
                self._current_episode_task = None
                if self._current_queue_item is item:
                    self._current_queue_item = None
                self.queue.task_done()

    def _fail_pending_events(self, exc: Exception) -> None:
        if self._current_queue_item is not None and not self._current_queue_item.future.done():
            self._current_queue_item.future.set_exception(exc)
            self._current_queue_item = None
        while True:
            try:
                _, _, item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not item.future.done():
                item.future.set_exception(exc)
            self.queue.task_done()
        self._pending_interactive = 0

    async def _process_event(
        self, event: InputEvent, *, autonomous: bool = False, ref: str = ""
    ) -> EpisodeResult | None:
        async with self._event_lock:
            self.current_event = f"{event.source}:{event.event_type}"
            self._current_ref = ref
            self._current_goal_id = None
            self._current_project_id = None
            try:
                async with asyncio.timeout(self.config.episode_timeout or None):
                    if autonomous:
                        return await self._plan_and_act(
                            event, should_yield=lambda: self._pending_interactive > 0
                        )
                    return await self._run_episode(event)
            finally:
                self.current_event = ""
                self._current_ref = ""

    async def _run_episode(
        self, event: InputEvent, *, should_yield: Callable[[], bool] | None = None
    ) -> EpisodeResult:
        self.episode_taint.reset()
        try:
            result = await self.runtime.run_episode(event, should_yield=should_yield)
            self.latest_model_context = result.model_context
            await self._store_episode(event, result)
            return result
        except Exception as exc:
            self.last_error = str(exc)
            if self.config.pause_on_error:
                self.pause()
            raise

    async def _plan_and_act(
        self, event: InputEvent, *, should_yield: Callable[[], bool] | None = None
    ) -> EpisodeResult:
        await self._maybe_consolidate()  # periodic budgeted memory consolidation
        await self.goals.scheduler.decay_tick()  # drive homeostasis, once per tick
        goal = await self.goals.review()
        if goal is None:
            return await self._run_episode(event, should_yield=should_yield)
        self.runtime.self_state.active_goal = goal.description
        project = await self.autonomy.get_or_create_project(goal.id, goal.description)
        if project is None:
            self.last_autonomous_action = "wait:project_paused"
            return await self._run_episode(
                InputEvent(
                    content=(
                        f"Autonomous heartbeat: active goal '{goal.description}' has a "
                        f"paused project, so I will not continue it."
                    ),
                    source="autonomous",
                    event_type="project_paused",
                ),
                should_yield=should_yield,
            )
        # No filler task: "no pending task" is visible context state
        # (tasks.status NO_PENDING_TASK sentinel) the model must resolve.
        self._current_goal_id = goal.id
        self._current_project_id = project.id
        result = await self._run_episode(event, should_yield=should_yield)
        # This episode serviced the goal: bump its drive's satiation.
        await self.goals.scheduler.record_serviced(goal.id)
        requests = self.runtime.autonomous_strategy.last_tool_requests
        preempted = result.outcome_reason == "preempted by interactive event"
        if preempted:
            self.last_autonomous_action = "wait:preempted"
        elif requests:
            self.last_autonomous_action = f"tool:{requests[-1].name}"
        else:
            self.last_autonomous_action = "wait:no_action"
        self._update_task_discipline(requests)
        self._tick_count += 1
        # Skip the periodic goal review (an extra LLM call) while a user is
        # waiting: either this episode was preempted or an interactive event
        # arrived after it finished. The review re-arms on a later tick.
        user_waiting = preempted or (should_yield is not None and should_yield())
        if (
            not user_waiting
            and self.runtime.autonomous_strategy.llm is not None
            and self._goal_review_interval > 0
            and self._tick_count % self._goal_review_interval == 0
        ):
            try:
                await self.autonomy.record_action("goal_review_attempt")
                applied = await self.goals.review_with_llm(
                    self.llm_fast or self.runtime.autonomous_strategy.llm,
                    recent_episodes=await self.memory.recent_episodes(15),
                    recent_influences=await self.goals.list_influences(15),
                )
                if not applied:
                    await self.autonomy.record_action("goal_review_empty")
            except Exception as exc:  # noqa: BLE001 — review is best-effort
                try:
                    await self.autonomy.record_action("goal_review_error")
                except Exception:  # noqa: BLE001 — recording must not mask the original failure
                    pass
                self.last_error = f"goal_review_failed: {exc}"
        return result

    async def _maybe_consolidate(self) -> None:
        """Run ConsolidationEngine.consolidate_cycle on the autonomous cadence
        (every config.consolidation_interval ticks, separate from goal-review):
        budgeted LLM episode summarization into facts, the decay-to-archived
        pass, and the flag-gated contradiction sweep. Best-effort: failures are
        recorded in last_error and never block the tick."""
        interval = self.config.consolidation_interval
        if interval <= 0:
            return
        self._consolidation_ticks += 1
        if self._consolidation_ticks % interval != 0:
            return
        try:
            stats = await self.consolidation.consolidate_cycle(
                contradiction_judge=self._contradiction_judge(),
            )
            if stats.get("errors"):
                self.last_error = "consolidation: " + "; ".join(stats["errors"])
            self._event_broker.emit("memory.consolidated", stats)
        except Exception as exc:  # noqa: BLE001 — consolidation never blocks the tick
            logger.exception("consolidation cycle failed")
            self.last_error = f"consolidation_failed: {exc}"

    def _contradiction_judge(self) -> Any | None:
        """Flag-gated LLM yes/no judge for the budgeted contradiction sweep
        (config.enable_contradiction_check; design risk note: never on the
        write hot path, only inside the consolidation cycle)."""
        if not self.config.enable_contradiction_check:
            return None
        llm = self.llm_fast or self.runtime.autonomous_strategy.llm
        if llm is None:
            return None

        async def judge(fact_a: str, fact_b: str) -> bool:
            response = await llm.chat_async(
                [
                    {
                        "role": "system",
                        "content": (
                            "You judge whether two stored memory facts contradict "
                            "each other. Answer ONLY YES or NO."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"A: {fact_a}\nB: {fact_b}\nDo A and B contradict each other?",
                    },
                ],
                temperature=0.0,
                max_tokens=8,
            )
            return str(response.get("content") or "").strip().upper().startswith("YES")

        return judge

    def _update_task_discipline(self, requests: list[Any]) -> None:
        """Track the add_task-vs-set_task_status balance across autonomous ticks.
        Three consecutive add-only ticks arm the transition-nudge hard rule
        rendered by the autonomous assembler (task_discipline context key)."""
        names = {getattr(r, "name", "") for r in requests}
        if "set_task_status" in names:
            self._add_only_ticks = 0
        elif "add_task" in names:
            self._add_only_ticks += 1

    def _task_discipline_rule(self) -> str:
        if self._add_only_ticks < _ADD_ONLY_TICK_LIMIT:
            return ""
        return (
            f"You have added tasks on {self._add_only_ticks} consecutive ticks without "
            "progressing any. This tick you MUST set_task_status (done/blocked) on an "
            "existing task or make concrete progress — do not add a new task."
        )

    def _tool_capabilities(self, name: str) -> list[str]:
        getter = getattr(self.runtime.tools, "tool_capabilities", None)
        if not callable(getter):
            return []
        return sorted(str(item) for item in getter(name))

    def _redact_tool_args(self, value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                lowered = str(key).lower()
                if any(marker in lowered for marker in _REDACT_KEYS):
                    out[str(key)] = "[redacted]"
                else:
                    out[str(key)] = self._redact_tool_args(item)
            return out
        if isinstance(value, list):
            return [self._redact_tool_args(item) for item in value]
        return value

    def _record_tool_event(
        self,
        source: str,
        request: Any,
        result: dict[str, Any],
        *,
        episode_id: str | None = None,
    ) -> None:
        name = getattr(request, "name", "")
        capabilities = self._tool_capabilities(name)
        taint_origin = external_taint_origin(
            request, result, capabilities=frozenset(capabilities)
        ) or ""
        summary = " ".join(str(result.get("output") or result.get("error") or "").split())[:500]
        try:
            self.memory.record_tool_event(
                episode_id=episode_id if episode_id is not None else self.runtime.workspace.current_episode,
                tick=-1 if episode_id is not None else int(getattr(self.runtime.workspace, "_current_tick", -1)),
                source=source,
                tool=name,
                capabilities=capabilities,
                args=self._redact_tool_args(getattr(request, "args", {}) or {}),
                result_summary=summary,
                error=bool(result.get("error")),
                exit_code=result.get("exit_code"),
                taint_origin=taint_origin,
            )
        except Exception as exc:  # noqa: BLE001 — audit must not mask the tool result
            self.last_error = f"tool_audit_failed: {exc}"

    async def _autonomous_context_state(self) -> dict[str, Any]:
        goal_obj = await self.goals.active_goal()
        goal = asdict(goal_obj) if goal_obj else {}
        project = await self.autonomy.active_project() or {}
        active_task = await self.autonomy.active_task()
        # Stale-task watchdog: flag lingering tasks ([STALE] in context) and
        # auto-block the ones past the block threshold.
        stale = await self.autonomy.flag_stale_tasks()
        stale_ids = {t["id"] for t in stale["flagged"]}
        pending: list[dict[str, Any]] = []
        recently_completed: list[dict[str, Any]] = []
        if project:
            tasks = await self.autonomy.list_tasks(project["id"])
            pending = [
                {**t, "stale": t["id"] in stale_ids}
                for t in tasks
                if t["status"] == "pending"
            ]
            recently_completed = [t for t in tasks if t["status"] in {"done", "blocked"}][-3:]
        tasks_status = "ok" if (pending or active_task) else NO_PENDING_TASK_SENTINEL
        constraints = await self.goals.active_constraints()
        recent_episodes = await self.memory.recent_episodes(5)
        relevant_memory: list[dict[str, Any]] = []
        if goal:
            try:
                relevant_memory = await self.memory.search_facts(
                    str(goal.get("description", ""))[:120], limit=5
                )
            except Exception:  # noqa: BLE001 — search is best-effort
                relevant_memory = []
        recent_tool_actions = await self.autonomy.count_recent_actions("tool")
        budget_remaining = max(0, self.config.max_actions_per_hour - recent_tool_actions)
        return {
            "active_goal": goal,
            "current_project": project,
            "current_task": active_task,
            "tasks": {
                "pending": pending,
                "recently_completed": recently_completed,
                "status": tasks_status,
            },
            "recent_episodes": recent_episodes,
            "relevant_memory": relevant_memory,
            "constraints": constraints,
            "budget_remaining": budget_remaining,
            "budget_limit": self.config.max_actions_per_hour,
            "last_autonomous_action": self.last_autonomous_action,
            "goal_selection": self.goals.scheduler.last_selection or {},
            "drives": await self.goals.list_drives(),
            "task_discipline": self._task_discipline_rule(),
            "episode_taint": self.episode_taint.to_dict(),
        }

    def _note_web_taint(self, request: Any, result: dict[str, Any] | None = None) -> None:
        """Mark the episode tainted when a tool touches web content — the
        spotlighted web tools, or bash/execute_code reaching the network via
        curl/wget/python (otherwise a shell fetch bypasses the taint pipeline)."""
        capabilities = frozenset(self._tool_capabilities(getattr(request, "name", "")))
        origin = external_taint_origin(request, result, capabilities=capabilities)
        if origin is not None:
            self.episode_taint.note(origin)

    async def _on_chat_tool_observation(self, request: Any, result: dict[str, Any]) -> None:
        """Chat-path tool hook: taint tracking only (no autonomous budget)."""
        self._record_tool_event("chat", request, result)
        self._note_web_taint(request, result)

    async def _on_autonomous_tool_observation(self, request: Any, result: dict[str, Any]) -> None:
        """Hook fired by the autonomous tool loop after each tool call. Tracks
        web taint and records a tool action for the per-hour persistent budget.
        Self-management tools record themselves."""
        self._record_tool_event("autonomous", request, result)
        self._note_web_taint(request, result)
        name = getattr(request, "name", "")
        if name in _SELF_MANAGEMENT_TOOLS:
            return
        await self.autonomy.record_action("tool")

    def _subagent_llm(self) -> Any | None:
        """Model for sub-agent tool loops: the 'subagent' role when a router is
        configured (M2), else the same client the chat strategy uses."""
        if self.router is not None:
            try:
                return self.router.for_role("subagent")
            except Exception:  # noqa: BLE001 — fall back to the main client
                pass
        return self.runtime.chat_strategy.llm

    def _subagent_observer(self, subagent_id: str) -> Any:
        """Per-spawn observation hook. Taint notes go to the PARENT episode
        tracker on purpose: sub-agent output flows back into the parent
        context, so a sub-agent web fetch must quarantine the parent episode
        (otherwise delegation would bypass the injection defenses). Audit rows
        carry the SUB-episode id."""

        async def observe(request: Any, result: dict[str, Any]) -> None:
            self._record_tool_event("subagent", request, result, episode_id=subagent_id)
            self._note_web_taint(request, result)
            name = getattr(request, "name", "")
            if self.current_event.startswith("autonomous") and name not in _SELF_MANAGEMENT_TOOLS:
                await self.autonomy.record_action("tool")

        return observe

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
            agent_profile=self.config.agent.profile,
            premises=self.config.agent.premises,
            active_goal=asdict(goal) if goal else None,
            queue_depth=self.queue.qsize(),
            current_event=self.current_event,
            current_project=current_project,
            current_task=current_task,
            last_autonomous_action=self.last_autonomous_action,
            actions_last_hour=await self.autonomy.count_recent_actions("tool"),
            episode_count=await self.memory.count_episodes(),
            last_error=self.last_error,
        )

    async def recent_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        stored = await self.memory.recent_episodes(limit)
        if stored:
            return stored
        return [asdict(ep) for ep in self._episodes[-limit:]][::-1]

    async def episodes_before(self, cursor_ts: float, limit: int = 20) -> list[dict[str, Any]]:
        return await self.memory.episodes_before(cursor_ts, limit)

    async def recent_trace(self) -> str:
        episodes = await self.memory.recent_episodes(20)
        notes = await self.autonomy.recent_notes(20)
        items = [
            (float(e.get("created_at") or 0.0), e.get("trace") or "")
            for e in episodes
            if e.get("trace")
        ]
        items += [(float(n.get("created_at") or 0.0), n["content"]) for n in notes]
        items.sort(key=lambda item: item[0])
        stored = "\n\n".join(content for _, content in items[-20:])
        return stored or self.runtime.trace.format(limit=120)

    async def search_memory(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self.memory.search(query, limit)

    async def recent_facts(self, limit: int = 10) -> list[dict[str, Any]]:
        return await self.memory.recent_facts(limit)

    async def recent_tool_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self.memory.recent_tool_events(limit)

    async def metrics(self) -> dict[str, Any]:
        row = self.memory.fetchone("SELECT COUNT(*) AS n FROM tool_events")
        episode_metrics = [
            e.get("metrics") or {} for e in await self.memory.recent_episodes(200)
        ]
        return {
            "running": self.running,
            "paused": self.paused,
            "agent_profile": self.config.agent.profile,
            "premises": self.config.agent.premises,
            "external_side_effects": self.config.agent.external_side_effects,
            "unsafe_autonomy": self.config.unsafe_autonomy,
            "queue_depth": self.queue.qsize(),
            "actions_last_hour": await self.autonomy.count_recent_actions("tool"),
            "tool_events_total": int(row["n"]) if row else 0,
            "episode_count": await self.memory.count_episodes(),
            "llm_calls_recent": sum(int(m.get("llm_calls", 0) or 0) for m in episode_metrics),
            "tool_calls_recent": sum(int(m.get("tool_calls", 0) or 0) for m in episode_metrics),
            "latest_model_context_chars": len(self.latest_model_context or ""),
            "schema_version": self.memory.schema_version(),
            "db_path": str(self.config.db_path),
            "working_directory": str(self.config.working_directory),
            "mcp_servers": self.mcp.status(),
            "last_error": self.last_error,
            "last_backup_at": self.last_backup_at,
            "last_backup_error": self.last_backup_error,
            "rate_limited_total": self.rate_limited_total,
        }

    async def list_procedures(self) -> list[dict[str, Any]]:
        rows = await self.memory.list_procedures()
        # v1 UI compat: keep `skill`/`use_count` aliases alongside v2 columns.
        return [
            {**row, "skill": row["name"], "use_count": row["success_count"]}
            for row in rows
        ]

    async def list_projects(self) -> list[dict[str, Any]]:
        return await self.autonomy.list_projects()

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        return await self.autonomy.get_project(project_id)

    async def set_project_status(self, project_id: str, status: str) -> None:
        changed = await self.autonomy.set_project_status(project_id, status)
        if not changed:
            raise KeyError(project_id)
        self._event_broker.emit(
            "project.updated", {"project_id": project_id, "status": status}
        )

    async def list_influences(self) -> list[dict[str, Any]]:
        return await self.goals.list_influences()

    async def _backup_loop(self) -> None:
        """Periodic home backup + retention pruning. First run happens one full
        interval after start (a restart loop must not double backup I/O)."""
        interval = self.config.backup_interval_hours * 3600.0
        while self.running:
            await asyncio.sleep(interval)
            try:
                archive = await asyncio.to_thread(create_home_backup, self.config)
                await asyncio.to_thread(prune_backups, self.config, self.config.backup_retain)
                self.last_backup_at = time.time()
                self.last_backup_error = ""
                logger.info("scheduled backup written: %s", archive)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_backup_error = str(exc)
                logger.exception("scheduled backup failed")

    async def _autonomous_loop(self) -> None:
        backoff = 0.0
        while self.running:
            if not self.paused:
                try:
                    await self.run_autonomous_tick()
                    backoff = 0.0
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A failed episode must never kill the heartbeat: log it,
                    # back off, and keep ticking. _run_episode already recorded
                    # last_error (and paused, if pause_on_error is set).
                    logger.exception("autonomous tick failed")
                    backoff = min(max(backoff * 2, self.config.tick_interval), 300.0)
            await asyncio.sleep(backoff or self.config.tick_interval)

    def _recent_actions(self, items: list[float]) -> list[float]:
        cutoff = time.time() - 3600
        return [t for t in items if t >= cutoff]

    def _within_action_budget(self, items: list[float], limit: int) -> bool:
        recent = self._recent_actions(items)
        items[:] = recent
        return len(recent) < limit

    def _prune_event_files(self) -> None:
        """Delete oldest event JSON files when the events directory exceeds the cap."""
        events_dir = self.config.home / "events"
        try:
            files = sorted(events_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        if len(files) <= _MAX_EVENT_FILES:
            return
        for path in files[: len(files) - _MAX_EVENT_FILES]:
            try:
                path.unlink()
            except OSError:
                pass

    async def _store_episode(self, event: InputEvent, result: EpisodeResult) -> None:
        ep = StoredEpisode(
            id=result.episode_id or f"{int(time.time() * 1000)}-{len(self._episodes) + 1}",
            source=event.source,
            event_type=event.event_type,
            input=event.content,
            output=result.output,
            selected_action=result.selected_action,
            created_at=time.time(),
            metrics=asdict(result.metrics),
        )
        self._episodes.append(ep)
        if len(self._episodes) > _MAX_IN_MEMORY_EPISODES:
            del self._episodes[: len(self._episodes) - _MAX_IN_MEMORY_EPISODES]
        path = self.config.home / "events" / f"{ep.id}.json"
        path.write_text(json.dumps(asdict(ep), indent=2), encoding="utf-8")
        self._prune_event_files()
        await self.memory.record_episode(
            episode_id=ep.id,
            source=ep.source,
            event_type=ep.event_type,
            input=ep.input,
            output=ep.output,
            selected_action=ep.selected_action,
            tainted=self.episode_taint.web,
            web_origins=list(self.episode_taint.urls),
            metrics=dict(ep.metrics),
            trace=result.cognitive_trace,
            goal_id=self._current_goal_id,
            project_id=self._current_project_id,
        )
        self._event_broker.emit(
            "episode.created",
            {
                "id": ep.id,
                "source": ep.source,
                "event_type": ep.event_type,
                "selected_action": ep.selected_action,
                "input": ep.input[:280],
                "output": ep.output[:280],
                "metrics": dict(ep.metrics),
            },
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
            "episode_taint": self.episode_taint.to_dict(),
        }

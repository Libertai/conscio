from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from conscio.memory.store import MemoryStore

# Stale-task watchdog thresholds (days). Defaults mirror config.MotivationConfig;
# override per-deploy via the [motivation] TOML table.
STALE_FLAG_DAYS = 2.0
STALE_BLOCK_DAYS = 5.0


AUTONOMY_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    tool_name TEXT,
    tool_args TEXT NOT NULL DEFAULT '{}',
    result TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS progress_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS action_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


@dataclass
class Project:
    id: str
    goal_id: str
    title: str
    status: str
    created_at: float
    updated_at: float


@dataclass
class Task:
    id: str
    project_id: str
    description: str
    status: str
    tool_name: str | None
    tool_args: dict[str, Any]
    result: str
    created_at: float
    updated_at: float


class AutonomyStore:
    def __init__(
        self,
        memory: MemoryStore,
        *,
        stale_flag_days: float = STALE_FLAG_DAYS,
        stale_block_days: float = STALE_BLOCK_DAYS,
    ) -> None:
        self.memory = memory
        self.stale_flag_days = float(stale_flag_days)
        self.stale_block_days = float(stale_block_days)

    async def initialize(self) -> None:
        await self.memory.initialize()
        migrate_autonomy_schema(self.memory)

    async def get_or_create_project(self, goal_id: str, goal_description: str) -> Project | None:
        row = self.memory.fetchone(
            "SELECT * FROM projects WHERE goal_id = ? AND status = 'active' "
            "ORDER BY updated_at DESC LIMIT 1",
            (goal_id,),
        )
        if row:
            return Project(**row)
        paused = self.memory.fetchone(
            "SELECT * FROM projects WHERE goal_id = ? AND status = 'paused' ORDER BY updated_at DESC LIMIT 1",
            (goal_id,),
        )
        if paused:
            return None
        now = time.time()
        project = Project(
            id=uuid.uuid4().hex,
            goal_id=goal_id,
            title=f"Autonomous pursuit: {goal_description[:80]}",
            status="active",
            created_at=now,
            updated_at=now,
        )
        self.memory.execute(
            "INSERT INTO projects (id, goal_id, title, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project.id, project.goal_id, project.title, project.status, project.created_at, project.updated_at),
        )
        return project

    async def add_task(
        self,
        project_id: str,
        description: str,
        *,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        status: str = "pending",
    ) -> Task:
        now = time.time()
        task = Task(
            id=uuid.uuid4().hex,
            project_id=project_id,
            description=description,
            status=status,
            tool_name=tool_name,
            tool_args=tool_args or {},
            result="",
            created_at=now,
            updated_at=now,
        )
        self.memory.execute(
            "INSERT INTO tasks (id, project_id, description, status, tool_name, tool_args, result, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task.id,
                task.project_id,
                task.description,
                task.status,
                task.tool_name,
                json.dumps(task.tool_args),
                task.result,
                task.created_at,
                task.updated_at,
            ),
        )
        return task

    async def update_task(self, task_id: str, *, status: str, result: str = "") -> None:
        self.memory.execute(
            "UPDATE tasks SET status = ?, result = ?, updated_at = ? WHERE id = ?",
            (status, result, time.time(), task_id),
        )

    async def get_task(self, task_id: str) -> Task | None:
        row = self.memory.fetchone("SELECT * FROM tasks WHERE id = ?", (task_id,))
        return self._task_from_row(row) if row else None

    async def list_projects(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.memory.fetchall(
            "SELECT * FROM projects ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        project = self.memory.fetchone("SELECT * FROM projects WHERE id = ?", (project_id,))
        if not project:
            return None
        project["tasks"] = await self.list_tasks(project_id)
        return project

    async def list_tasks(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.memory.fetchall(
            "SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        )
        return [asdict(self._task_from_row(row)) for row in rows]

    async def active_project(self) -> dict[str, Any] | None:
        return self.memory.fetchone(
            "SELECT * FROM projects WHERE status = 'active' ORDER BY updated_at DESC LIMIT 1"
        )

    async def active_task(self) -> dict[str, Any] | None:
        row = self.memory.fetchone(
            "SELECT tasks.* FROM tasks "
            "JOIN projects ON projects.id = tasks.project_id "
            "WHERE tasks.status IN ('pending', 'active') AND projects.status = 'active' "
            "ORDER BY tasks.updated_at DESC LIMIT 1"
        )
        return asdict(self._task_from_row(row)) if row else None

    async def set_project_status(self, project_id: str, status: str) -> bool:
        # Use transaction to capture the rowcount under the lock.
        rowcount = self.memory.transaction([
            (
                "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), project_id),
            ),
        ])
        return rowcount > 0

    async def flag_stale_tasks(
        self,
        pending_days: float | None = None,
        block_days: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Stale-task watchdog (run per autonomous tick).

        Tasks pending/active longer than `pending_days` are returned as
        `flagged` (the context assembler renders them as [STALE]); tasks older
        than `block_days` are auto-transitioned to blocked with
        result="auto-blocked: stale" and an action_event('task_auto_blocked')
        per task, so the backlog cannot rot silently forever.
        """
        now = time.time()
        flag_days = self.stale_flag_days if pending_days is None else float(pending_days)
        blocking_days = self.stale_block_days if block_days is None else float(block_days)
        flag_cutoff = now - flag_days * 86400.0
        block_cutoff = now - blocking_days * 86400.0
        rows = self.memory.fetchall(
            "SELECT * FROM tasks WHERE status IN ('pending', 'active') AND updated_at < ? "
            "ORDER BY updated_at",
            (flag_cutoff,),
        )
        flagged: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for row in rows:
            task = asdict(self._task_from_row(row))
            if float(row["updated_at"]) < block_cutoff:
                block_ts = time.time()
                self.memory.transaction([
                    (
                        "UPDATE tasks SET status = ?, result = ?, updated_at = ? WHERE id = ?",
                        ("blocked", "auto-blocked: stale", block_ts, row["id"]),
                    ),
                    (
                        "INSERT INTO action_events (kind, created_at) VALUES (?, ?)",
                        ("task_auto_blocked", block_ts),
                    ),
                ])
                task["status"] = "blocked"
                task["result"] = "auto-blocked: stale"
                blocked.append(task)
            else:
                flagged.append(task)
        return {"flagged": flagged, "blocked": blocked}

    async def record_action(self, kind: str) -> None:
        self.memory.execute(
            "INSERT INTO action_events (kind, created_at) VALUES (?, ?)",
            (kind, time.time()),
        )

    async def count_recent_actions(self, kind: str, seconds: int = 3600) -> int:
        row = self.memory.fetchone(
            "SELECT COUNT(*) AS count FROM action_events WHERE kind = ? AND created_at >= ?",
            (kind, time.time() - seconds),
        )
        return int(row["count"]) if row else 0

    async def note_progress(self, episode_id: str | None, content: str) -> None:
        """Append a free-form progress note (used by the note_progress meta-tool)."""
        self.memory.execute(
            "INSERT INTO progress_notes (episode_id, content, created_at) VALUES (?, ?, ?)",
            (episode_id, content, time.time()),
        )

    async def recent_notes(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.memory.fetchall(
            "SELECT episode_id, content, created_at FROM progress_notes "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    def _task_from_row(self, row: dict | sqlite3.Row) -> Task:
        data = dict(row)
        data["tool_args"] = json.loads(data.get("tool_args") or "{}")
        return Task(**data)


def _table_columns(memory: MemoryStore, table: str) -> set[str]:
    try:
        return {str(row["name"]) for row in memory.fetchall(f"PRAGMA table_info({table})")}
    except sqlite3.OperationalError:
        return set()


def _ensure_column(memory: MemoryStore, table: str, name: str, definition: str) -> None:
    if name not in _table_columns(memory, table):
        memory.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def migrate_autonomy_schema(memory: MemoryStore) -> None:
    """Create current autonomy tables and add missing columns from older DBs."""
    memory.executescript(AUTONOMY_SCHEMA)
    for name, definition in {
        "status": "TEXT NOT NULL DEFAULT 'active'",
        "updated_at": "REAL NOT NULL DEFAULT 0",
    }.items():
        _ensure_column(memory, "projects", name, definition)
    for name, definition in {
        "status": "TEXT NOT NULL DEFAULT 'pending'",
        "tool_name": "TEXT",
        "tool_args": "TEXT NOT NULL DEFAULT '{}'",
        "result": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "REAL NOT NULL DEFAULT 0",
    }.items():
        _ensure_column(memory, "tasks", name, definition)
    for name, definition in {
        "episode_id": "TEXT",
    }.items():
        _ensure_column(memory, "progress_notes", name, definition)

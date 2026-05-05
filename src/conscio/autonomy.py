from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from conscio.memory.store import MemoryStore


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
CREATE TABLE IF NOT EXISTS service_episodes (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    event_type TEXT NOT NULL,
    input TEXT NOT NULL,
    output TEXT NOT NULL,
    selected_action TEXT NOT NULL,
    metrics TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS service_traces (
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
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def _conn(self) -> sqlite3.Connection:
        return self.memory._conn()

    async def initialize(self) -> None:
        await self.memory.initialize()
        self._conn().executescript(AUTONOMY_SCHEMA)
        self._conn().commit()

    async def get_or_create_project(self, goal_id: str, goal_description: str) -> Project | None:
        row = self._conn().execute(
            "SELECT * FROM projects WHERE goal_id = ? AND status = 'active' "
            "ORDER BY updated_at DESC LIMIT 1",
            (goal_id,),
        ).fetchone()
        if row:
            return Project(**dict(row))
        paused = self._conn().execute(
            "SELECT * FROM projects WHERE goal_id = ? AND status = 'paused' ORDER BY updated_at DESC LIMIT 1",
            (goal_id,),
        ).fetchone()
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
        self._conn().execute(
            "INSERT INTO projects (id, goal_id, title, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (project.id, project.goal_id, project.title, project.status, project.created_at, project.updated_at),
        )
        self._conn().commit()
        return project

    async def ensure_next_task(
        self,
        project_id: str,
        description: str,
        *,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> Task:
        row = self._conn().execute(
            "SELECT * FROM tasks WHERE project_id = ? AND status IN ('pending', 'active') "
            "ORDER BY created_at LIMIT 1",
            (project_id,),
        ).fetchone()
        if row:
            return self._task_from_row(row)
        now = time.time()
        task = Task(
            id=uuid.uuid4().hex,
            project_id=project_id,
            description=description,
            status="pending",
            tool_name=tool_name,
            tool_args=tool_args or {},
            result="",
            created_at=now,
            updated_at=now,
        )
        self._conn().execute(
            "INSERT INTO tasks (id, project_id, description, status, tool_name, tool_args, result, created_at, updated_at) "
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
        self._conn().commit()
        return task

    async def update_task(self, task_id: str, *, status: str, result: str = "") -> None:
        self._conn().execute(
            "UPDATE tasks SET status = ?, result = ?, updated_at = ? WHERE id = ?",
            (status, result, time.time(), task_id),
        )
        self._conn().commit()

    async def list_projects(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM projects ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        row = self._conn().execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            return None
        project = dict(row)
        project["tasks"] = await self.list_tasks(project_id)
        return project

    async def list_tasks(self, project_id: str) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        ).fetchall()
        return [asdict(self._task_from_row(row)) for row in rows]

    async def active_project(self) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM projects WHERE status = 'active' ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    async def active_task(self) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT tasks.* FROM tasks "
            "JOIN projects ON projects.id = tasks.project_id "
            "WHERE tasks.status IN ('pending', 'active') AND projects.status = 'active' "
            "ORDER BY tasks.updated_at DESC LIMIT 1"
        ).fetchone()
        return asdict(self._task_from_row(row)) if row else None

    async def set_project_status(self, project_id: str, status: str) -> bool:
        cursor = self._conn().execute(
            "UPDATE projects SET status = ?, updated_at = ? WHERE id = ?",
            (status, time.time(), project_id),
        )
        self._conn().commit()
        return cursor.rowcount > 0

    async def record_action(self, kind: str) -> None:
        self._conn().execute(
            "INSERT INTO action_events (kind, created_at) VALUES (?, ?)",
            (kind, time.time()),
        )
        self._conn().commit()

    async def count_recent_actions(self, kind: str, seconds: int = 3600) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) AS count FROM action_events WHERE kind = ? AND created_at >= ?",
            (kind, time.time() - seconds),
        ).fetchone()
        return int(row["count"]) if row else 0

    async def store_episode(
        self,
        *,
        episode_id: str,
        source: str,
        event_type: str,
        input: str,
        output: str,
        selected_action: str,
        metrics: dict[str, Any],
        trace: str,
    ) -> None:
        now = time.time()
        self._conn().execute(
            "INSERT OR REPLACE INTO service_episodes "
            "(id, source, event_type, input, output, selected_action, metrics, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (episode_id, source, event_type, input, output, selected_action, json.dumps(metrics), now),
        )
        self._conn().execute(
            "INSERT INTO service_traces (episode_id, content, created_at) VALUES (?, ?, ?)",
            (episode_id, trace, now),
        )
        self._conn().commit()

    async def recent_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM service_episodes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._episode_from_row(row) for row in rows]

    async def recent_trace(self, limit: int = 20) -> str:
        rows = self._conn().execute(
            "SELECT content FROM service_traces ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return "\n\n".join(row["content"] for row in reversed(rows))

    def _task_from_row(self, row: sqlite3.Row) -> Task:
        data = dict(row)
        data["tool_args"] = json.loads(data.get("tool_args") or "{}")
        return Task(**data)

    def _episode_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metrics"] = json.loads(data.get("metrics") or "{}")
        return data

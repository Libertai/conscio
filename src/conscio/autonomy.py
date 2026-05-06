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

    async def initialize(self) -> None:
        await self.memory.initialize()
        self.memory.executescript(AUTONOMY_SCHEMA)

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

    async def ensure_next_task(
        self,
        project_id: str,
        description: str,
        *,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
    ) -> Task:
        row = self.memory.fetchone(
            "SELECT * FROM tasks WHERE project_id = ? AND status IN ('pending', 'active') "
            "ORDER BY created_at LIMIT 1",
            (project_id,),
        )
        if row:
            return self._task_from_row(row)
        return await self.add_task(project_id, description, tool_name=tool_name, tool_args=tool_args)

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
        self.memory.transaction([
            (
                "INSERT OR REPLACE INTO service_episodes "
                "(id, source, event_type, input, output, selected_action, metrics, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (episode_id, source, event_type, input, output, selected_action, json.dumps(metrics), now),
            ),
            (
                "INSERT INTO service_traces (episode_id, content, created_at) VALUES (?, ?, ?)",
                (episode_id, trace, now),
            ),
        ])

    async def note_progress(self, episode_id: str | None, content: str) -> None:
        """Append a free-form progress note to service_traces (used by note_progress meta-tool)."""
        self.memory.execute(
            "INSERT INTO service_traces (episode_id, content, created_at) VALUES (?, ?, ?)",
            (episode_id, content, time.time()),
        )

    async def recent_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.memory.fetchall(
            "SELECT * FROM service_episodes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._episode_from_row(row) for row in rows]

    async def recent_trace(self, limit: int = 20) -> str:
        rows = self.memory.fetchall(
            "SELECT content FROM service_traces ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return "\n\n".join(row["content"] for row in reversed(rows))

    def _task_from_row(self, row: dict | sqlite3.Row) -> Task:
        data = dict(row)
        data["tool_args"] = json.loads(data.get("tool_args") or "{}")
        return Task(**data)

    def _episode_from_row(self, row: dict | sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["metrics"] = json.loads(data.get("metrics") or "{}")
        return data

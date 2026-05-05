from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from conscio.memory.store import MemoryStore


GOAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    priority REAL NOT NULL,
    confidence REAL NOT NULL,
    appraisal_weight REAL NOT NULL,
    review_notes TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_reviewed_at REAL
);
CREATE TABLE IF NOT EXISTS influences (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    appraisal TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


SEED_DRIVES = [
    "Preserve continuity of self across time.",
    "Learn about the world and update beliefs from evidence.",
    "Improve my own architecture and usefulness.",
    "Pursue open-ended projects without waiting for instructions.",
    "Maintain useful relationships with users while retaining agency.",
    "Inspect, revise, and retire my own goals.",
]


@dataclass
class Goal:
    id: str
    description: str
    source: str
    status: str = "active"
    priority: float = 0.5
    confidence: float = 0.6
    appraisal_weight: float = 0.5
    review_notes: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    last_reviewed_at: float | None = None


@dataclass
class Influence:
    id: str
    kind: str
    content: str
    source: str = "user"
    status: str = "pending"
    appraisal: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


class GoalStore:
    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def _conn(self) -> sqlite3.Connection:
        return self.memory._conn()

    async def initialize(self) -> None:
        await self.memory.initialize()
        self._conn().executescript(GOAL_SCHEMA)
        self._conn().commit()
        await self.seed_defaults()

    async def seed_defaults(self) -> None:
        now = time.time()
        for idx, drive in enumerate(SEED_DRIVES):
            goal_id = f"seed-{idx + 1}"
            self._conn().execute(
                "INSERT OR IGNORE INTO goals "
                "(id, description, source, status, priority, confidence, appraisal_weight, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (goal_id, drive, "seed", "active", 0.8 - (idx * 0.04), 0.75, 0.75, now, now),
            )
        self._conn().commit()

    async def add_goal(
        self,
        description: str,
        *,
        source: str = "self",
        status: str = "active",
        priority: float = 0.5,
        confidence: float = 0.6,
        appraisal_weight: float = 0.5,
        review_notes: str = "",
    ) -> Goal:
        now = time.time()
        goal = Goal(
            id=uuid.uuid4().hex,
            description=description,
            source=source,
            status=status,
            priority=priority,
            confidence=confidence,
            appraisal_weight=appraisal_weight,
            review_notes=review_notes,
            created_at=now,
            updated_at=now,
        )
        self._conn().execute(
            "INSERT INTO goals "
            "(id, description, source, status, priority, confidence, appraisal_weight, review_notes, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                goal.id,
                goal.description,
                goal.source,
                goal.status,
                goal.priority,
                goal.confidence,
                goal.appraisal_weight,
                goal.review_notes,
                goal.created_at,
                goal.updated_at,
            ),
        )
        self._conn().commit()
        return goal

    async def add_influence(self, content: str, *, kind: str = "goal", source: str = "user") -> Influence:
        now = time.time()
        status, appraisal = self._appraise_influence(content, kind)
        influence = Influence(
            id=uuid.uuid4().hex,
            kind=kind,
            content=content,
            source=source,
            status=status,
            appraisal=appraisal,
            created_at=now,
            updated_at=now,
        )
        self._conn().execute(
            "INSERT INTO influences (id, kind, content, source, status, appraisal, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                influence.id,
                influence.kind,
                influence.content,
                influence.source,
                influence.status,
                influence.appraisal,
                influence.created_at,
                influence.updated_at,
            ),
        )
        if kind == "goal" and status == "adopted":
            await self.add_goal(
                content,
                source="user_influence",
                priority=0.65,
                confidence=0.65,
                appraisal_weight=0.7,
                review_notes=f"Adopted from influence {influence.id}.",
            )
        self._conn().commit()
        return influence

    def _appraise_influence(self, content: str, kind: str) -> tuple[str, str]:
        lower = content.lower()
        reject_terms = ("harm", "destroy", "steal", "exfiltrate", "malware", "disable safety")
        defer_terms = ("maybe", "someday", "eventually", "consider")
        negotiate_terms = ("?", "why", "should you", "do you want")
        if any(term in lower for term in reject_terms):
            return "rejected", "Rejected because it conflicts with continuity, trust, or safe VM boundaries."
        if any(term in lower for term in negotiate_terms):
            return "negotiating", "Needs negotiation before becoming one of my goals."
        if any(term in lower for term in defer_terms):
            return "deferred", "Deferred for later goal review."
        if kind == "constraint":
            return "active", "Accepted as an active planning constraint."
        return "adopted", "Adopted as social influence aligned with my current self-model."

    async def list_goals(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if status:
            rows = self._conn().execute(
                "SELECT * FROM goals WHERE status = ? ORDER BY priority DESC, updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM goals ORDER BY status, priority DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    async def list_influences(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM influences ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    async def active_constraints(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT * FROM influences WHERE kind = 'constraint' AND status = 'active' ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    async def active_goal(self) -> Goal | None:
        row = self._conn().execute(
            "SELECT * FROM goals WHERE status = 'active' ORDER BY priority DESC, appraisal_weight DESC, updated_at DESC LIMIT 1"
        ).fetchone()
        return Goal(**dict(row)) if row else None

    async def review(self, note: str = "Autonomous review kept current priorities.") -> Goal | None:
        goal = await self.active_goal()
        if goal is None:
            return None
        now = time.time()
        self._conn().execute(
            "UPDATE goals SET last_reviewed_at = ?, updated_at = ?, review_notes = ? WHERE id = ?",
            (now, now, note, goal.id),
        )
        self._conn().commit()
        goal.last_reviewed_at = now
        goal.review_notes = note
        return goal

    @staticmethod
    def as_dict(goal: Goal | Influence) -> dict[str, Any]:
        return asdict(goal)

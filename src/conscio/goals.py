from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from conscio.memory.store import MemoryStore

logger = logging.getLogger(__name__)


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

    async def initialize(self) -> None:
        await self.memory.initialize()
        self.memory.executescript(GOAL_SCHEMA)
        await self.seed_defaults()

    async def seed_defaults(self) -> None:
        now = time.time()
        items: list[tuple[str, tuple]] = []
        for idx, drive in enumerate(SEED_DRIVES):
            goal_id = f"seed-{idx + 1}"
            items.append((
                "INSERT OR IGNORE INTO goals "
                "(id, description, source, status, priority, confidence, appraisal_weight, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (goal_id, drive, "seed", "active", 0.8 - (idx * 0.04), 0.75, 0.75, now, now),
            ))
        self.memory.transaction(items)

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
        self.memory.execute(
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
        self.memory.execute(
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
            return self.memory.fetchall(
                "SELECT * FROM goals WHERE status = ? ORDER BY priority DESC, updated_at DESC LIMIT ?",
                (status, limit),
            )
        return self.memory.fetchall(
            "SELECT * FROM goals ORDER BY status, priority DESC, updated_at DESC LIMIT ?",
            (limit,),
        )

    async def list_influences(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.memory.fetchall(
            "SELECT * FROM influences ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    async def active_constraints(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.memory.fetchall(
            "SELECT * FROM influences WHERE kind = 'constraint' AND status = 'active' ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    async def active_goal(self) -> Goal | None:
        row = self.memory.fetchone(
            "SELECT * FROM goals WHERE status = 'active' ORDER BY priority DESC, appraisal_weight DESC, updated_at DESC LIMIT 1"
        )
        return Goal(**row) if row else None

    async def review_with_llm(
        self,
        llm: Any,
        *,
        recent_episodes: list[dict[str, Any]] | None = None,
        recent_influences: list[dict[str, Any]] | None = None,
        max_decisions: int = 8,
    ) -> list[dict[str, Any]]:
        """Ask the LLM to review goals; apply keep/retire/reprioritize decisions transactionally.

        Returns the list of applied decisions. Best-effort: invalid decisions are skipped silently.
        """
        active_goals = await self.list_goals(status="active", limit=50)
        paused_goals = await self.list_goals(status="paused", limit=50)
        goals = active_goals + paused_goals
        if not goals:
            return []
        prompt = self._build_review_prompt(goals, recent_episodes or [], recent_influences or [])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Conscio reviewing your own goals. Output a JSON array of decisions, "
                    "one per goal you want to update. Each item must have: "
                    "{\"goal_id\": string, \"action\": one of [\"keep\", \"retire\", \"reprioritize\"], "
                    "\"new_priority\": optional number in [0, 1] required if action=reprioritize, "
                    "\"reason\": short string}. "
                    "Output ONLY the JSON array, no surrounding prose."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = await llm.chat_async(messages, temperature=0.2, max_tokens=800)
        raw = str(response.get("content") or "").strip()
        decisions = self._parse_review_decisions(raw)
        if not decisions:
            if raw:
                logger.warning("goal_review: parse miss, raw=%r", raw[:2000])
                await self.memory.add_fact(
                    f"Goal review parse miss; raw response head: {raw[:200]}",
                    source="goal_review",
                    confidence="LOW",
                )
            return []
        valid_ids = {g["id"] for g in goals}
        applied: list[dict[str, Any]] = []
        now = time.time()
        for decision in decisions[:max_decisions]:
            goal_id = str(decision.get("goal_id", ""))
            action = str(decision.get("action", "")).lower()
            reason = str(decision.get("reason", ""))[:500]
            if goal_id not in valid_ids:
                continue
            if action == "keep":
                self.memory.execute(
                    "UPDATE goals SET last_reviewed_at = ?, updated_at = ?, review_notes = ? WHERE id = ?",
                    (now, now, reason or "kept by self-review", goal_id),
                )
                applied.append({"goal_id": goal_id, "action": "keep", "reason": reason})
            elif action == "retire":
                self.memory.execute(
                    "UPDATE goals SET status = 'retired', last_reviewed_at = ?, updated_at = ?, "
                    "review_notes = ? WHERE id = ?",
                    (now, now, reason or "retired by self-review", goal_id),
                )
                applied.append({"goal_id": goal_id, "action": "retire", "reason": reason})
            elif action == "reprioritize":
                try:
                    new_priority = float(decision.get("new_priority", 0.5))
                except (TypeError, ValueError):
                    continue
                new_priority = max(0.0, min(1.0, new_priority))
                self.memory.execute(
                    "UPDATE goals SET priority = ?, last_reviewed_at = ?, updated_at = ?, "
                    "review_notes = ? WHERE id = ?",
                    (new_priority, now, now, reason or "reprioritized by self-review", goal_id),
                )
                applied.append({
                    "goal_id": goal_id,
                    "action": "reprioritize",
                    "new_priority": new_priority,
                    "reason": reason,
                })
        if applied:
            await self.memory.add_fact(
                f"Goal review applied {len(applied)} decision(s): "
                + ", ".join(f"{d['action']}:{d['goal_id'][:8]}" for d in applied),
                source="goal_review",
                confidence="MEDIUM",
            )
        return applied

    @staticmethod
    def _parse_review_decisions(raw: str) -> list[dict[str, Any]]:
        if not raw:
            return []
        # Find the first JSON array in the response.
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        candidate = match.group(0) if match else raw
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    @staticmethod
    def _build_review_prompt(
        goals: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        influences: list[dict[str, Any]],
    ) -> str:
        lines = ["GOALS:"]
        for g in goals:
            lines.append(
                f"  - id={g['id']} status={g['status']} priority={g.get('priority', 0):.2f} "
                f"description={g.get('description', '')[:200]}"
            )
        lines.append("")
        lines.append("RECENT_EPISODES:")
        if not episodes:
            lines.append("  none")
        for e in episodes[:10]:
            lines.append(
                f"  - source={e.get('source', '')} action={e.get('selected_action', '')} "
                f"output={(e.get('output') or '')[:160]}"
            )
        lines.append("")
        lines.append("RECENT_INFLUENCES:")
        if not influences:
            lines.append("  none")
        for inf in influences[:10]:
            lines.append(
                f"  - kind={inf.get('kind', '')} status={inf.get('status', '')} "
                f"content={(inf.get('content') or '')[:160]}"
            )
        return "\n".join(lines)

    async def review(self, note: str = "Autonomous review kept current priorities.") -> Goal | None:
        goal = await self.active_goal()
        if goal is None:
            return None
        now = time.time()
        self.memory.execute(
            "UPDATE goals SET last_reviewed_at = ?, updated_at = ?, review_notes = ? WHERE id = ?",
            (now, now, note, goal.id),
        )
        goal.last_reviewed_at = now
        goal.review_notes = note
        return goal

    @staticmethod
    def as_dict(goal: Goal | Influence) -> dict[str, Any]:
        return asdict(goal)

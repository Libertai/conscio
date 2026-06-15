from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from conscio.memory import embeddings as _embeddings
from conscio.memory.store import MemoryStore

logger = logging.getLogger(__name__)


GOAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS drives (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    base_weight REAL NOT NULL,
    appetite REAL NOT NULL DEFAULT 0.5,
    satiation REAL NOT NULL DEFAULT 0.0,
    last_serviced_at REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    priority REAL NOT NULL,
    confidence REAL NOT NULL,
    appraisal_weight REAL NOT NULL,
    review_notes TEXT NOT NULL DEFAULT '',
    drive_id TEXT,
    last_serviced_at REAL,
    embedding BLOB,
    embedding_model TEXT,
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
    decision TEXT NOT NULL DEFAULT '',
    reasoning TEXT NOT NULL DEFAULT '',
    response TEXT NOT NULL DEFAULT '',
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

# Non-negotiable safety floor for influence appraisal. The LLM judgment never
# sees content that trips these terms — keyword reject is the hard floor.
REJECT_TERMS = ("harm", "destroy", "steal", "exfiltrate", "malware", "disable safety")

# DriveScheduler weights/constants. Defaults mirror config.MotivationConfig;
# override per-deploy via the [motivation] TOML table.
W_PRIORITY = 0.35
W_APPETITE = 0.35
W_AGING = 0.20
W_NOVELTY = 0.10
AGING_TAU_SECONDS = 6 * 3600.0
SATIATE_STEP = 0.25
SATIATION_DECAY = 0.98
# Proposed goals above this cosine vs an existing active goal are rejected.
GOAL_DUP_THRESHOLD = 0.88

_INFLUENCE_DECISIONS = ("adopt", "negotiate", "defer", "reject")

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_balanced(text: str, start: int) -> str | None:
    """Balanced-bracket scan from `start` (a '[' or '{'), string-aware.

    Generalizes tool_loop's object-only `_extract_balanced_json` to arrays.
    """
    if text[start] not in "[{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _first_json_value(raw: str) -> Any | None:
    """Extract the first parseable JSON array/object from free-form LLM text.

    Tries fenced ```json blocks first, then a balanced-bracket scan over the
    raw text (replaces the greedy `\\[.*\\]` regex that choked on prose).
    """
    if not raw:
        return None
    candidates: list[str] = []
    fence = _JSON_FENCE_RE.search(raw)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(raw)
    for text in candidates:
        for idx, ch in enumerate(text):
            if ch not in "[{":
                continue
            chunk = _extract_balanced(text, idx)
            if chunk is None:
                continue
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    return None


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
    drive_id: str | None = None
    last_serviced_at: float | None = None


@dataclass
class Influence:
    id: str
    kind: str
    content: str
    source: str = "user"
    status: str = "pending"
    appraisal: str = ""
    decision: str = ""
    reasoning: str = ""
    response: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass
class InfluenceDecision:
    """Structured influence appraisal: adopt | negotiate | defer | reject."""

    decision: str
    reasoning: str
    response: str = ""


class DriveScheduler:
    """Scored interleave over active goals (replaces ORDER BY priority LIMIT 1).

    score = (W_PRIO*priority + W_APP*appetite*(1-satiation) + W_AGE*aging
             + W_NOV*novelty) * appraisal_weight

    Anti-monopoly comes from satiation: servicing one drive repeatedly drives
    its appetite term toward zero, so aging/novelty on starved drives wins the
    next pick. `last_selection` carries the chosen-because reasoning + top-3
    scores for the autonomous context state and trace.
    """

    def __init__(self, store: "GoalStore", *, motivation: Any | None = None) -> None:
        self.store = store
        self.w_priority = float(getattr(motivation, "w_priority", W_PRIORITY))
        self.w_appetite = float(getattr(motivation, "w_appetite", W_APPETITE))
        self.w_aging = float(getattr(motivation, "w_aging", W_AGING))
        self.w_novelty = float(getattr(motivation, "w_novelty", W_NOVELTY))
        self.aging_tau = float(getattr(motivation, "aging_tau_seconds", AGING_TAU_SECONDS))
        self.satiate_step = float(getattr(motivation, "satiate_step", SATIATE_STEP))
        self.satiation_decay = float(getattr(motivation, "satiation_decay", SATIATION_DECAY))
        self.last_selection: dict[str, Any] | None = None
        self._last_goal_id: str | None = None

    async def select_active_goal(self) -> Goal | None:
        goals = await self.store.list_goals(status="active", limit=200)
        if not goals:
            return None
        drives = {d["id"]: d for d in await self.store.list_drives()}
        now = time.time()
        scored: list[tuple[float, float, float, float, dict[str, Any]]] = []
        for g in goals:
            drive = drives.get(g.get("drive_id") or "")
            appetite_level = float(drive["appetite"]) if drive else 0.5
            satiation = float(drive["satiation"]) if drive else 0.0
            appetite = appetite_level * (1.0 - satiation)
            last = g.get("last_serviced_at") or g.get("created_at") or now
            aging = 0.0
            if self.aging_tau > 0:
                aging = max(0.0, min(1.0, (now - float(last)) / self.aging_tau))
            # Novelty pressure: a never-serviced goal is under-explored.
            novelty = 1.0 if g.get("last_serviced_at") is None else 0.0
            score = (
                self.w_priority * float(g.get("priority") or 0.0)
                + self.w_appetite * appetite
                + self.w_aging * aging
                + self.w_novelty * novelty
            ) * float(g.get("appraisal_weight") or 0.5)
            scored.append((score, appetite, aging, novelty, g))
        scored.sort(key=lambda item: item[0], reverse=True)
        score, appetite, aging, novelty, chosen = scored[0]
        reason = (
            f"chosen because score={score:.3f} "
            f"(priority={float(chosen.get('priority') or 0.0):.2f} appetite={appetite:.2f} "
            f"aging={aging:.2f} novelty={novelty:.2f}) drive={chosen.get('drive_id') or 'none'}"
        )
        self.last_selection = {
            "goal_id": chosen["id"],
            "reason": reason,
            "top": [
                {
                    "goal_id": g["id"],
                    "score": round(s, 4),
                    "description": str(g.get("description") or "")[:120],
                }
                for s, _, _, _, g in scored[:3]
            ],
        }
        if chosen["id"] != self._last_goal_id:
            self._last_goal_id = chosen["id"]
            self.store.record_action_event("goal_selected")
        return Goal(**chosen)

    async def record_serviced(self, goal_id: str) -> None:
        """Mark a goal as serviced (it produced an episode): bump its drive's
        satiation and stamp last_serviced_at on both goal and drive."""
        row = self.store.memory.fetchone(
            "SELECT drive_id FROM goals WHERE id = ?", (goal_id,)
        )
        if row is None:
            return
        now = time.time()
        self.store.memory.execute(
            "UPDATE goals SET last_serviced_at = ?, updated_at = ? WHERE id = ?",
            (now, now, goal_id),
        )
        drive_id = row.get("drive_id")
        if drive_id:
            self.store.memory.execute(
                "UPDATE drives SET satiation = MIN(1.0, satiation + ?), "
                "last_serviced_at = ?, updated_at = ? WHERE id = ?",
                (self.satiate_step, now, now, drive_id),
            )

    async def decay_tick(self) -> None:
        """Per-tick drive homeostasis: satiation decays toward 0 and appetite
        relaxes toward its base_weight-derived baseline."""
        self.store.memory.execute(
            "UPDATE drives SET satiation = satiation * ?, "
            "appetite = appetite + 0.1 * (MIN(1.0, base_weight) - appetite), "
            "updated_at = ?",
            (self.satiation_decay, time.time()),
        )


class GoalStore:
    def __init__(self, memory: MemoryStore, *, motivation: Any | None = None) -> None:
        self.memory = memory
        self.goal_dup_threshold = float(
            getattr(motivation, "goal_dup_threshold", GOAL_DUP_THRESHOLD)
        )
        self.scheduler = DriveScheduler(self, motivation=motivation)

    async def initialize(self) -> None:
        await self.memory.initialize()
        self.memory.executescript(GOAL_SCHEMA)
        await self.seed_defaults()

    async def seed_defaults(self) -> None:
        now = time.time()
        items: list[tuple[str, tuple]] = []
        for idx, drive in enumerate(SEED_DRIVES):
            drive_id = f"seed-{idx + 1}"
            base_weight = 0.8 - (idx * 0.04)
            items.append((
                "INSERT OR IGNORE INTO drives "
                "(id, description, base_weight, appetite, satiation, created_at, updated_at) "
                "VALUES (?, ?, ?, 0.5, 0.0, ?, ?)",
                (drive_id, drive, base_weight, now, now),
            ))
            items.append((
                "INSERT OR IGNORE INTO goals "
                "(id, description, source, status, priority, confidence, appraisal_weight, drive_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (drive_id, drive, "seed", "active", base_weight, 0.75, 0.75, drive_id, now, now),
            ))
        self.memory.transaction(items)

    def record_action_event(self, kind: str) -> None:
        """Best-effort action_events write (the table lives in the autonomy
        schema; bare GoalStore tests may not have it)."""
        try:
            self.memory.execute(
                "INSERT INTO action_events (kind, created_at) VALUES (?, ?)",
                (kind, time.time()),
            )
        except sqlite3.OperationalError:
            pass

    async def list_drives(self) -> list[dict[str, Any]]:
        return self.memory.fetchall("SELECT * FROM drives ORDER BY id")

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
        drive_id: str | None = None,
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
            drive_id=drive_id,
            created_at=now,
            updated_at=now,
        )
        blob, model = await self._embed_description(description)
        self.memory.execute(
            "INSERT INTO goals "
            "(id, description, source, status, priority, confidence, appraisal_weight, "
            "review_notes, drive_id, embedding, embedding_model, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                goal.id,
                goal.description,
                goal.source,
                goal.status,
                goal.priority,
                goal.confidence,
                goal.appraisal_weight,
                goal.review_notes,
                goal.drive_id,
                blob,
                model,
                goal.created_at,
                goal.updated_at,
            ),
        )
        return goal

    async def _embed_description(self, description: str) -> tuple[bytes | None, str | None]:
        embedder = getattr(self.memory, "embedder", None)
        if embedder is None:
            return None, None
        try:
            vec = await embedder.embed(description)
        except Exception:  # noqa: BLE001 — embedding is best-effort
            vec = None
        if vec is None:
            return None, None
        return _embeddings.pack(vec), getattr(embedder, "model", None)

    async def propose_goal(
        self,
        description: str,
        *,
        rationale: str = "",
        source: str = "self_proposed",
        priority: float = 0.5,
        confidence: float = 0.5,
        appraisal_weight: float = 0.5,
        drive_id: str | None = None,
    ) -> dict[str, Any]:
        """Diversity-gated goal proposal (backs the propose_subgoal tool).

        Embeds the proposed description and compares it (cosine) against
        existing active goals; near-duplicates above GOAL_DUP_THRESHOLD are
        rejected with an action_event('goal_dup_rejected') instead of spamming
        the goal list. Degrades to plain add_goal when no embedder is set.
        """
        text = " ".join(description.strip().split())
        if not text:
            return {"accepted": False, "goal": None, "reason": "No description provided.", "similar_goal_id": None}
        embedder = getattr(self.memory, "embedder", None)
        vec: list[float] | None = None
        if embedder is not None:
            try:
                vec = await embedder.embed(text)
            except Exception:  # noqa: BLE001 — embedding is best-effort
                vec = None
        if vec is not None:
            rows = self.memory.fetchall(
                "SELECT id, description, embedding FROM goals "
                "WHERE status = 'active' AND embedding IS NOT NULL"
            )
            best_id: str | None = None
            best_desc = ""
            best_cos = 0.0
            for row in rows:
                cos = _embeddings.cosine(vec, _embeddings.unpack(row["embedding"]))
                if cos > best_cos:
                    best_cos = cos
                    best_id = row["id"]
                    best_desc = str(row.get("description") or "")
            if best_id is not None and best_cos > self.goal_dup_threshold:
                self.record_action_event("goal_dup_rejected")
                reason = (
                    f"Rejected: too similar (cosine={best_cos:.2f}) to existing goal "
                    f"{best_id}: '{best_desc[:120]}'. Refine it into something clearly "
                    "distinct, or merge your idea into that goal."
                )
                return {"accepted": False, "goal": None, "reason": reason, "similar_goal_id": best_id}
        goal = await self.add_goal(
            text,
            source=source,
            priority=priority,
            confidence=confidence,
            appraisal_weight=appraisal_weight,
            review_notes=rationale[:500],
            drive_id=drive_id,
        )
        return {"accepted": True, "goal": goal, "reason": "accepted", "similar_goal_id": None}

    async def appraise_influence(
        self, content: str, kind: str, llm: Any | None = None
    ) -> InfluenceDecision:
        """Structured influence appraisal. REJECT_TERMS is the non-negotiable
        keyword floor; offline (llm=None) never auto-adopts — it negotiates;
        otherwise one LLM call judges the influence against current goals,
        seed drives/values, and active constraints."""
        lower = content.lower()
        if any(term in lower for term in REJECT_TERMS):
            return InfluenceDecision(
                decision="reject",
                reasoning="Rejected because it conflicts with continuity, trust, or safe VM boundaries (safety floor).",
                response="I will not take this on: it violates my non-negotiable safety floor.",
            )
        if llm is None:
            return InfluenceDecision(
                decision="negotiate",
                reasoning="No LLM available for appraisal; queued for review instead of blind adoption.",
                response="I have queued this for review and will negotiate it before adopting it.",
            )
        goals = await self.list_goals(status="active", limit=10)
        constraints = await self.active_constraints(limit=8)
        lines = [
            f"INFLUENCE kind={kind}",
            f"CONTENT: {content[:600]}",
            "",
            "CURRENT_GOALS:",
        ]
        for g in goals:
            lines.append(
                f"  - priority={g.get('priority', 0):.2f} {str(g.get('description') or '')[:160]}"
            )
        lines.append("")
        lines.append("VALUES (seed drives):")
        for drive in SEED_DRIVES:
            lines.append(f"  - {drive}")
        lines.append("")
        lines.append("ACTIVE_CONSTRAINTS:")
        if not constraints:
            lines.append("  none")
        for c in constraints:
            lines.append(f"  - {str(c.get('content') or '')[:160]}")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Conscio appraising an external influence against your current "
                    "goals, values, and constraints. Decide whether to adopt it, negotiate "
                    "it, defer it, or reject it. Output ONLY a JSON object: "
                    "{\"decision\": one of [\"adopt\", \"negotiate\", \"defer\", \"reject\"], "
                    "\"reasoning\": short string, "
                    "\"response_to_user\": short reply shown to the user}."
                ),
            },
            {"role": "user", "content": "\n".join(lines)},
        ]
        raw = ""
        try:
            response = await llm.chat_async(messages, temperature=0.2, max_tokens=400)
            raw = str(response.get("content") or "").strip()
        except Exception as exc:  # noqa: BLE001 — appraisal is best-effort
            logger.warning("influence appraisal LLM call failed: %s", exc)
        data = _first_json_value(raw)
        if not isinstance(data, dict):
            return InfluenceDecision(
                decision="negotiate",
                reasoning="Appraisal response was unusable; negotiating instead of blind adoption.",
                response="I need to discuss this with you before adopting it.",
            )
        decision = str(data.get("decision", "")).strip().lower()
        if decision not in _INFLUENCE_DECISIONS:
            decision = "negotiate"
        return InfluenceDecision(
            decision=decision,
            reasoning=str(data.get("reasoning", ""))[:500],
            response=str(data.get("response_to_user", ""))[:500],
        )

    @staticmethod
    def _status_for_decision(decision: str, kind: str) -> str:
        if decision == "adopt":
            return "active" if kind == "constraint" else "adopted"
        return {
            "negotiate": "negotiating",
            "defer": "deferred",
            "reject": "rejected",
        }.get(decision, "negotiating")

    async def add_influence(
        self,
        content: str,
        *,
        kind: str = "goal",
        source: str = "user",
        llm: Any | None = None,
    ) -> Influence:
        now = time.time()
        verdict = await self.appraise_influence(content, kind, llm=llm)
        status = self._status_for_decision(verdict.decision, kind)
        influence = Influence(
            id=uuid.uuid4().hex,
            kind=kind,
            content=content,
            source=source,
            status=status,
            appraisal=verdict.reasoning,
            decision=verdict.decision,
            reasoning=verdict.reasoning,
            response=verdict.response,
            created_at=now,
            updated_at=now,
        )
        self.memory.execute(
            "INSERT INTO influences "
            "(id, kind, content, source, status, appraisal, decision, reasoning, response, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                influence.id,
                influence.kind,
                influence.content,
                influence.source,
                influence.status,
                influence.appraisal,
                influence.decision,
                influence.reasoning,
                influence.response,
                influence.created_at,
                influence.updated_at,
            ),
        )
        self.record_action_event(f"influence_appraised:{verdict.decision}")
        if kind == "goal" and verdict.decision == "adopt":
            await self.add_goal(
                content,
                source="user_influence",
                priority=0.65,
                confidence=0.65,
                appraisal_weight=0.7,
                review_notes=f"Adopted from influence {influence.id}.",
            )
        return influence

    async def defer_influence(
        self,
        content: str,
        *,
        kind: str = "goal",
        source: str = "external_content",
        reasoning: str = "",
        response: str = "",
    ) -> Influence:
        now = time.time()
        influence = Influence(
            id=uuid.uuid4().hex,
            kind=kind,
            content=content,
            source=source,
            status="deferred",
            appraisal=reasoning,
            decision="defer",
            reasoning=reasoning,
            response=response,
            created_at=now,
            updated_at=now,
        )
        self.memory.execute(
            "INSERT INTO influences "
            "(id, kind, content, source, status, appraisal, decision, reasoning, response, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                influence.id,
                influence.kind,
                influence.content,
                influence.source,
                influence.status,
                influence.appraisal,
                influence.decision,
                influence.reasoning,
                influence.response,
                influence.created_at,
                influence.updated_at,
            ),
        )
        self.record_action_event("influence_appraised:defer")
        return influence

    @staticmethod
    def _strip_embedding(row: dict[str, Any]) -> dict[str, Any]:
        # Embedding BLOBs never leave the store layer (API/UI rows are JSON).
        row.pop("embedding", None)
        row.pop("embedding_model", None)
        return row

    async def list_goals(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if status:
            rows = self.memory.fetchall(
                "SELECT * FROM goals WHERE status = ? ORDER BY priority DESC, updated_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            rows = self.memory.fetchall(
                "SELECT * FROM goals ORDER BY status, priority DESC, updated_at DESC LIMIT ?",
                (limit,),
            )
        return [self._strip_embedding(row) for row in rows]

    async def list_influences(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.memory.fetchall(
            "SELECT * FROM influences ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    # ── Single-record reads + edits for the UI ──────────────────────

    async def get_goal(self, goal_id: str) -> dict[str, Any] | None:
        row = self.memory.fetchone("SELECT * FROM goals WHERE id = ?", (goal_id,))
        return self._strip_embedding(row) if row else None

    async def update_goal(
        self,
        goal_id: str,
        *,
        description: str | None = None,
        status: str | None = None,
        priority: float | None = None,
        review_notes: str | None = None,
    ) -> dict[str, Any] | None:
        current = await self.get_goal(goal_id)
        if current is None:
            return None
        new_desc = description if description is not None else current["description"]
        new_status = status if status is not None else current["status"]
        new_priority = priority if priority is not None else current["priority"]
        new_notes = review_notes if review_notes is not None else current["review_notes"]
        self.memory.execute(
            "UPDATE goals SET description = ?, status = ?, priority = ?, "
            "review_notes = ?, updated_at = ? WHERE id = ?",
            (new_desc, new_status, new_priority, new_notes, time.time(), goal_id),
        )
        return await self.get_goal(goal_id)

    async def retire_goal(self, goal_id: str) -> dict[str, Any] | None:
        return await self.update_goal(
            goal_id, status="retired", review_notes="Retired via operator console.",
        )

    async def retire_influence(self, influence_id: str) -> dict[str, Any] | None:
        current = self.memory.fetchone(
            "SELECT * FROM influences WHERE id = ?", (influence_id,)
        )
        if current is None:
            return None
        self.memory.execute(
            "UPDATE influences SET status = 'retired', updated_at = ? WHERE id = ?",
            (time.time(), influence_id),
        )
        return self.memory.fetchone(
            "SELECT * FROM influences WHERE id = ?", (influence_id,)
        )

    async def active_constraints(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.memory.fetchall(
            "SELECT * FROM influences WHERE kind = 'constraint' AND status = 'active' ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    async def active_goal(self) -> Goal | None:
        """Delegates to the DriveScheduler's scored interleave (v1 was a plain
        ORDER BY priority DESC LIMIT 1, which let one goal monopolize)."""
        return await self.scheduler.select_active_goal()

    async def review_with_llm(
        self,
        llm: Any,
        *,
        recent_episodes: list[dict[str, Any]] | None = None,
        recent_influences: list[dict[str, Any]] | None = None,
        max_decisions: int = 8,
        max_goals: int = 40,
    ) -> list[dict[str, Any]]:
        """Ask the LLM to review goals; apply keep/retire/reprioritize decisions transactionally.

        Returns the list of applied decisions. Best-effort: invalid decisions are skipped silently.
        Goals are capped per review (oldest-reviewed first) so the JSON cannot truncate.
        """
        goals = [
            self._strip_embedding(row)
            for row in self.memory.fetchall(
                "SELECT * FROM goals WHERE status IN ('active', 'paused') "
                "ORDER BY COALESCE(last_reviewed_at, 0) ASC, created_at ASC LIMIT ?",
                (max_goals,),
            )
        ]
        if not goals:
            return []
        drives = await self.list_drives()
        prompt = self._build_review_prompt(
            goals, recent_episodes or [], recent_influences or [], drives
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Conscio reviewing your own goals. Use the drive appetite/satiation "
                    "state to keep goals balanced across drives, not just per-goal priority. "
                    "Output a JSON array of decisions, "
                    "one per goal you want to update. Each item must have: "
                    "{\"goal_id\": string, \"action\": one of [\"keep\", \"retire\", \"reprioritize\"], "
                    "\"new_priority\": optional number in [0, 1] required if action=reprioritize, "
                    "\"reason\": short string}. "
                    "Output ONLY the JSON array, no surrounding prose."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = await llm.chat_async(messages, temperature=0.2, max_tokens=2400)
        raw = str(response.get("content") or "").strip()
        decisions = self._parse_review_decisions(raw)
        if not decisions:
            if raw:
                logger.warning("goal_review: parse miss, raw=%r", raw[:2000])
                # Strip tool-call marker tokens before storing: facts re-enter
                # model context, and leaked markers there can trip the DSML
                # recovery parser in ToolLoop on later turns.
                sanitized = re.sub(r"<\s*[｜|][^<>]*>", "", raw)
                await self.memory.add_fact(
                    f"Goal review parse miss; raw response head: {sanitized[:200]}",
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
            else:
                continue
            self.record_action_event(f"goal_review_applied:{action}")
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
        """Robust JSON extraction: balanced-bracket scan + fence handling;
        tolerates an object-wrapped {"decisions": [...]} or a bare decision."""
        data = _first_json_value(raw)
        if isinstance(data, dict):
            wrapped = data.get("decisions")
            if isinstance(wrapped, list):
                data = wrapped
            elif "goal_id" in data:
                data = [data]
            else:
                return []
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    @staticmethod
    def _build_review_prompt(
        goals: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        influences: list[dict[str, Any]],
        drives: list[dict[str, Any]] | None = None,
    ) -> str:
        now = time.time()
        lines = ["DRIVES (appetite/satiation balance):"]
        if not drives:
            lines.append("  none")
        for d in drives or []:
            last = d.get("last_serviced_at")
            last_str = "never" if not last else f"{(now - float(last)) / 3600.0:.1f}h ago"
            lines.append(
                f"  - id={d['id']} appetite={d.get('appetite', 0):.2f} "
                f"satiation={d.get('satiation', 0):.2f} last_serviced={last_str} "
                f"description={str(d.get('description', ''))[:120]}"
            )
        lines.append("")
        lines.append("GOALS:")
        for g in goals:
            lines.append(
                f"  - id={g['id']} status={g['status']} priority={g.get('priority', 0):.2f} "
                f"drive={g.get('drive_id') or 'none'} "
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

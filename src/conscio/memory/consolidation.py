from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

import numpy as np

from conscio.memory import embeddings as _embeddings
from conscio.memory.store import (
    CONTRADICTION_LOW,
    MERGE_THRESHOLD,
    MemoryStore,
)

# Periodic consolidation budgets: one LLM call per cycle, capped fact emissions,
# capped contradiction-judge pairs. The decay pass archives (never deletes).
DECAY_DAYS = 14
MAX_CYCLE_FACTS = 8
MAX_CYCLE_EPISODES = 20
MAX_CONTRADICTION_PAIRS = 4
SWEEP_SAMPLE = 30


class ConsolidationEngine:
    """Memory consolidation v2 — replaces the v1 junk-skill + compaction logic.

    Two entry points:
    - record_episode(): per-episode cheap path (no LLM) — writes the unified
      episodes row with a deterministic summary and taint provenance.
    - consolidate_cycle(): periodic budgeted path — LLM summarization into
      semantic facts (via add_fact, origin=consolidation, so dedup applies),
      a decay-to-archived pass, and a budgeted contradiction sweep. Tainted
      (web-derived) episodes are excluded from summarization: consolidation
      must never promote web content to trust-2 facts (quarantine invariant).
    Best-effort throughout: failures are recorded, never raised to the tick.
    """

    name = "consolidation"

    def __init__(
        self,
        store: MemoryStore,
        *,
        llm: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.embedder = embedder or store.embedder
        self._last_cycle_ts = 0.0

    # ── Per-episode cheap path (no LLM) ──────────────────────────────────

    async def record_episode(
        self,
        *,
        episode_id: str,
        source: str,
        event_type: str,
        input_text: str,
        output: str,
        selected_action: str = "",
        metrics: dict[str, Any] | None = None,
        trace: str = "",
        tainted: bool = False,
        web_origins: list[str] | None = None,
        goal_id: str | None = None,
        project_id: str | None = None,
    ) -> str:
        summary = (
            f"Input: {input_text[:120]} -> action={selected_action}; "
            f"output={output[:180]}"
        )
        await self.store.record_episode(
            episode_id=episode_id,
            source=source,
            event_type=event_type,
            input=input_text,
            output=output,
            selected_action=selected_action,
            summary=summary,
            tainted=tainted,
            web_origins=web_origins,
            metrics=metrics,
            trace=trace,
            goal_id=goal_id,
            project_id=project_id,
        )
        return episode_id

    # ── Periodic budgeted path ───────────────────────────────────────────

    async def consolidate_cycle(
        self,
        llm: Any | None = None,
        embedder: Any | None = None,  # noqa: ARG002 — add_fact embeds via the store's embedder
        *,
        max_facts: int = MAX_CYCLE_FACTS,
        contradiction_judge: Any | None = None,
        decay_days: int = DECAY_DAYS,
        now: float | None = None,
    ) -> dict[str, Any]:
        llm = llm or self.llm
        now = time.time() if now is None else now
        stats: dict[str, Any] = {
            "facts_written": 0,
            "archived": 0,
            "contradicted": 0,
            "errors": [],
        }
        if llm is not None:
            try:
                stats["facts_written"] = await self._summarize_recent(llm, max_facts, now)
            except Exception as exc:  # noqa: BLE001 — consolidation never blocks the tick
                stats["errors"].append(f"summarize: {exc}")
                self._record_error()
        try:
            stats["archived"] = self._decay_pass(now, decay_days)
        except Exception as exc:  # noqa: BLE001
            stats["errors"].append(f"decay: {exc}")
            self._record_error()
        if contradiction_judge is not None:
            try:
                stats["contradicted"] = await self._contradiction_sweep(contradiction_judge)
            except Exception as exc:  # noqa: BLE001
                stats["errors"].append(f"contradiction_sweep: {exc}")
                self._record_error()
        return stats

    async def _summarize_recent(self, llm: Any, max_facts: int, now: float) -> int:
        # Quarantine: tainted (web-derived) episodes are excluded so consolidation
        # cannot launder web content into trust-2 origin='consolidation' facts.
        # Web-derived knowledge only enters semantic memory via the per-episode
        # remember_fact quarantine path (origin='web:<url>', trust tier 1).
        episodes = self.store.fetchall(
            "SELECT summary, input, output FROM episodes "
            "WHERE created_at > ? AND tainted = 0 "
            "ORDER BY created_at DESC LIMIT ?",
            (self._last_cycle_ts, MAX_CYCLE_EPISODES),
        )
        if not episodes:
            self._last_cycle_ts = now
            return 0
        lines = []
        for episode in episodes:
            line = episode.get("summary") or f"{episode['input'][:120]} -> {episode['output'][:120]}"
            lines.append(f"- {line}")
        messages = [
            {
                "role": "system",
                "content": (
                    "You consolidate an agent's episodic memory. From the episodes below, "
                    f"extract at most {max_facts} genuinely reusable, durable semantic facts. "
                    "Skip routine actions and one-off details. Output ONLY a JSON array of "
                    "strings, no surrounding prose."
                ),
            },
            {"role": "user", "content": "EPISODES:\n" + "\n".join(lines)},
        ]
        response = await llm.chat_async(messages, temperature=0.2, max_tokens=2400)
        # Advance the window only after the LLM call succeeds: setting it before
        # would permanently skip this window's episodes on a transient failure.
        self._last_cycle_ts = now
        facts = _parse_fact_list(str(response.get("content") or ""))
        written = 0
        for fact in facts[:max_facts]:
            result = await self.store.add_fact(
                fact[:500], origin="consolidation", confidence="MEDIUM"
            )
            if result.action == "inserted":
                written += 1
        return written

    def _decay_pass(self, now: float, decay_days: int) -> int:
        cutoff = now - decay_days * 86400
        return self.store.transaction([
            (
                "UPDATE facts SET status = 'archived', updated_at = ? "
                "WHERE status = 'active' AND trust <= 1 AND access_count = 0 "
                "AND (last_accessed IS NULL OR last_accessed < ?) AND created_at < ?",
                (now, cutoff, cutoff),
            ),
        ])

    async def _contradiction_sweep(self, judge: Any) -> int:
        rows = self.store.fetchall(
            "SELECT id, fact, trust, embedding FROM facts "
            "WHERE status = 'active' AND embedding IS NOT NULL "
            "ORDER BY created_at DESC LIMIT ?",
            (SWEEP_SAMPLE,),
        )
        if len(rows) < 2:
            return 0
        matrix = np.vstack([_embeddings.unpack(row["embedding"]) for row in rows])
        pairs: list[tuple[float, int, int]] = []
        for i in range(len(rows)):
            sims = _embeddings.cosine_matrix(matrix[i], matrix[i + 1 :])
            for offset, cos in enumerate(sims):
                if CONTRADICTION_LOW <= float(cos) < MERGE_THRESHOLD:
                    pairs.append((float(cos), i, i + 1 + offset))
        pairs.sort(reverse=True)
        marked = 0
        seen: set[int] = set()
        for _, i, j in pairs[:MAX_CONTRADICTION_PAIRS]:
            if i in seen or j in seen:
                continue
            contradicts = False
            try:
                contradicts = bool(await judge(rows[i]["fact"], rows[j]["fact"]))
            except Exception:  # noqa: BLE001 — judge is best-effort
                contradicts = False
            if contradicts:
                losers = await self.store.mark_contradiction(
                    int(rows[i]["id"]), int(rows[j]["id"])
                )
                marked += len(losers)
                seen.update({i, j})
        return marked

    def _record_error(self) -> None:
        try:
            self.store.execute(
                "INSERT INTO action_events (kind, created_at) VALUES (?, ?)",
                ("consolidation_error", time.time()),
            )
        except sqlite3.OperationalError:
            pass  # action_events lives in the autonomy schema; absent in bare stores


def _parse_fact_list(raw: str) -> list[str]:
    """Extract a list of fact strings from an LLM response (robust parse:
    balanced-bracket array scan, JSON fences tolerated, object-wrapped
    {"facts": [...]} tolerated)."""
    text = raw.strip()
    if not text:
        return []
    candidate = _extract_balanced(text, "[", "]") or _extract_balanced(text, "{", "}")
    if not candidate:
        return []
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("facts") or data.get("items") or []
    if not isinstance(data, list):
        return []
    out: list[str] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            fact = str(item.get("fact") or "").strip()
            if fact:
                out.append(fact)
    return out


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> str:
    start = text.find(open_ch)
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return ""

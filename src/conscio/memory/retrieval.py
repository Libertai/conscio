from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from conscio.memory import embeddings as _embeddings
from conscio.memory.embeddings import Embedder

if TYPE_CHECKING:  # pragma: no cover — type hints only, no runtime cycle
    from conscio.memory.store import MemoryStore

# Hybrid scoring weights: final = 0.55*cosine + 0.25*bm25_norm + 0.20*trust_norm,
# minus a small recency/decay nudge. When the embedder (or a stored embedding)
# is unavailable, retrieval degrades gracefully to pure BM25 ordering.
PREFILTER_LIMIT = 50
WEIGHT_COSINE = 0.55
WEIGHT_BM25 = 0.25
WEIGHT_TRUST = 0.20
AGE_PENALTY = 0.05
AGE_PENALTY_DAYS = 30.0


@dataclass
class RetrievedFact:
    id: int
    fact: str
    origin: str
    trust: int
    confidence: str
    score: float
    web_derived: bool
    provenance: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fact": self.fact,
            "origin": self.origin,
            # v1 callers/templates read `source`; keep it as an alias.
            "source": self.origin,
            "trust": self.trust,
            "confidence": self.confidence,
            "score": self.score,
            "web_derived": self.web_derived,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }


def build_fts_query(text: str, *, mode: str = "and") -> str:
    """Build an FTS5 MATCH expression from free text.

    mode="and" joins the high-IDF terms with AND (precise prefilter);
    mode="or" is the recall fallback when AND returns nothing.
    """
    terms: list[str] = []
    for raw in text.replace('"', " ").replace("'", " ").split():
        term = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-"})
        if len(term) >= 3 and term not in terms:
            terms.append(term)
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(f'"{term}"' for term in terms[:8])


def _candidate_rows(store: MemoryStore, match: str, limit: int) -> list[dict]:
    if not match:
        return []
    try:
        return store.fetchall(
            "SELECT f.id, f.fact, f.origin, f.trust, f.confidence, f.embedding, "
            "f.created_at, bm25(memory_fts) AS bm FROM memory_fts "
            "JOIN facts f ON f.id = CAST(memory_fts.ref_id AS INTEGER) "
            "WHERE memory_fts MATCH ? AND f.status = 'active' AND f.trust > 0 "
            "ORDER BY bm25(memory_fts) LIMIT ?",
            (f"memory_type:fact AND ({match})", limit),
        )
    except sqlite3.OperationalError:
        return []


async def retrieve_facts(
    store: MemoryStore,
    query: str,
    *,
    limit: int = 5,
    include_web: bool = True,
    max_web: int = 2,
    embedder: Embedder | None = None,
) -> list[RetrievedFact]:
    """Hybrid fact retrieval: FTS BM25 prefilter -> cosine rerank -> provenance shaping.

    Excludes non-active (archived/contradicted/superseded) and quarantined
    (trust 0) facts; caps web-derived facts at ``max_web``; bumps access
    counters on returned facts (feeds decay).
    """
    if not query.strip():
        return []
    rows = _candidate_rows(store, build_fts_query(query, mode="and"), PREFILTER_LIMIT)
    if not rows:
        rows = _candidate_rows(store, build_fts_query(query, mode="or"), PREFILTER_LIMIT)
    if not rows:
        return []

    query_vec: np.ndarray | None = None
    if embedder is not None:
        try:
            raw = await embedder.embed(query)
            if raw is not None:
                query_vec = np.asarray(raw, dtype=np.float32)
        except Exception:  # noqa: BLE001 — embedding is best-effort
            query_vec = None

    bm_values = [float(row["bm"]) for row in rows]
    bm_lo, bm_hi = min(bm_values), max(bm_values)
    now = time.time()
    scored: list[tuple[float, dict]] = []
    for row, bm in zip(rows, bm_values, strict=False):
        # bm25() is lower-is-better; normalize to [0, 1] with 1 = best match.
        bm_norm = 1.0 if bm_hi == bm_lo else (bm_hi - bm) / (bm_hi - bm_lo)
        if query_vec is None:
            score = bm_norm  # graceful degradation: pure BM25 ordering
        else:
            cos = 0.0
            blob = row.get("embedding")
            if blob:
                try:
                    cos = _embeddings.cosine(query_vec, _embeddings.unpack(blob))
                except Exception:  # noqa: BLE001 — malformed blob never breaks retrieval
                    cos = 0.0
            trust_norm = max(0.0, min(1.0, int(row["trust"]) / 3.0))
            age_days = max(0.0, now - float(row["created_at"] or now)) / 86400.0
            score = (
                WEIGHT_COSINE * cos
                + WEIGHT_BM25 * bm_norm
                + WEIGHT_TRUST * trust_norm
                - AGE_PENALTY * min(1.0, age_days / AGE_PENALTY_DAYS)
            )
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    out: list[RetrievedFact] = []
    web_used = 0
    for score, row in scored:
        origin = str(row["origin"])
        web_derived = origin.startswith("web:") or origin == "web"
        if web_derived:
            if not include_web or web_used >= max_web:
                continue
            web_used += 1
        out.append(
            RetrievedFact(
                id=int(row["id"]),
                fact=str(row["fact"]),
                origin=origin,
                trust=int(row["trust"]),
                confidence=str(row["confidence"]),
                score=float(score),
                web_derived=web_derived,
                provenance="web" if web_derived else origin,
                created_at=float(row["created_at"] or 0.0),
            )
        )
        if len(out) >= limit:
            break

    if out:
        ids = [item.id for item in out]
        placeholders = ", ".join("?" for _ in ids)
        store.execute(
            f"UPDATE facts SET access_count = access_count + 1, last_accessed = ? "
            f"WHERE id IN ({placeholders})",
            (now, *ids),
        )
    return out

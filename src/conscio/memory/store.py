from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from conscio.memory import embeddings as _embeddings
from conscio.memory.embeddings import Embedder

if TYPE_CHECKING:
    from conscio.v3.checkpoint_migration import CheckpointArchitectureMigration

_HOME_DIR = Path.home() / ".conscio"
_DB_PATH = _HOME_DIR / "sessions.db"

# Near-duplicate facts above this cosine are merged into the existing row.
MERGE_THRESHOLD = 0.93
# Cosine band [CONTRADICTION_LOW, MERGE_THRESHOLD) is "same topic, maybe
# divergent claim" — only there do we consult the (flag-gated) LLM judge.
CONTRADICTION_LOW = 0.80

# Trust tiers by fact origin: 3=user, 2=agent/consolidation, 1=web, 0=quarantined.
_TRUST_BY_ORIGIN = {
    "user": 3,
    "agent": 2,
    "consolidation": 2,
    "goal_review": 2,
    "runtime": 2,
    "compaction": 2,
    "quarantined": 0,
}
_CONF_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
SCHEMA_VERSION = 7

# Schema v2 (fresh-start DB, no migration from v1):
# - unified `episodes` keyed by the runtime's per-episode uuid (canonical id),
#   replacing v1 `episodic` + `service_episodes`/`service_traces`;
# - `facts` with provenance/trust/embedding/decay/contradiction links,
#   replacing v1 `semantic`;
# - deliberate `procedures`, replacing v1 junk-skill `procedural`;
# - `memory_fts` mirrors facts + episodes by ref_id.
# Embeddings are float32 little-endian BLOBs reranked with brute-force cosine
# over FTS candidates — fine up to ~50k facts (see memory/embeddings.py).
_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    goal_id       TEXT,
    project_id    TEXT,
    parent_episode_id TEXT,
    input         TEXT NOT NULL,
    output        TEXT NOT NULL,
    selected_action TEXT NOT NULL DEFAULT '',
    summary       TEXT NOT NULL DEFAULT '',
    tainted       INTEGER NOT NULL DEFAULT 0,
    web_origins   TEXT NOT NULL DEFAULT '[]',
    metrics       TEXT NOT NULL DEFAULT '{}',
    trace         TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_goal ON episodes (goal_id, created_at DESC);
CREATE TABLE IF NOT EXISTS facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fact          TEXT NOT NULL,
    norm_hash     TEXT NOT NULL,
    origin        TEXT NOT NULL,
    trust         INTEGER NOT NULL,
    episode_id    TEXT,
    confidence    TEXT NOT NULL DEFAULT 'MEDIUM',
    status        TEXT NOT NULL DEFAULT 'active',
    supersedes    INTEGER,
    superseded_by INTEGER,
    embedding     BLOB,
    embedding_model TEXT,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed REAL,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_norm ON facts (norm_hash);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts (status, trust, last_accessed);
CREATE TABLE IF NOT EXISTS procedures (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT NOT NULL,
    steps         TEXT NOT NULL,
    trigger       TEXT NOT NULL DEFAULT '',
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    origin        TEXT NOT NULL DEFAULT 'agent',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_meta (
    key           TEXT PRIMARY KEY,
    value         TEXT NOT NULL,
    updated_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id    TEXT NOT NULL DEFAULT '',
    tick          INTEGER NOT NULL DEFAULT -1,
    source        TEXT NOT NULL DEFAULT '',
    tool          TEXT NOT NULL,
    capabilities  TEXT NOT NULL DEFAULT '[]',
    args          TEXT NOT NULL DEFAULT '{}',
    result_summary TEXT NOT NULL DEFAULT '',
    error         INTEGER NOT NULL DEFAULT 0,
    exit_code     INTEGER,
    taint_origin  TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_events_created ON tool_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_events_episode ON tool_events (episode_id, tick);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(content, memory_type, ref_id UNINDEXED);
CREATE TABLE IF NOT EXISTS thoughts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    parent_id TEXT,
    type TEXT NOT NULL,
    question TEXT,
    answer TEXT,
    created_at REAL NOT NULL,
    depth INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    selected_action TEXT,
    episode_id TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages (session_id, created_at);
CREATE TABLE IF NOT EXISTS cognitive_events (
    sequence      INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id      TEXT NOT NULL UNIQUE,
    episode_id    TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    source        TEXT NOT NULL,
    payload       TEXT NOT NULL,
    model_input   TEXT,
    checkpoint_id TEXT,
    parent_event_id TEXT,
    schema_version INTEGER NOT NULL,
    observed_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cognitive_events_episode
    ON cognitive_events (episode_id, sequence);
CREATE TABLE IF NOT EXISTS core_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    lineage_id    TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    model_version TEXT NOT NULL,
    payload       TEXT NOT NULL,
    event_sequence INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_core_checkpoints_lineage
    ON core_checkpoints (lineage_id, created_at DESC);
CREATE TABLE IF NOT EXISTS checkpoint_architecture_migrations (
    sequence      INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id     TEXT NOT NULL UNIQUE,
    source_checkpoint_id TEXT NOT NULL UNIQUE,
    target_checkpoint_id TEXT NOT NULL UNIQUE,
    previous_record_hash TEXT,
    record_hash   TEXT NOT NULL UNIQUE,
    payload       TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS affect_interventions (
    intervention_id TEXT PRIMARY KEY,
    episode_id    TEXT NOT NULL DEFAULT '',
    operator      TEXT NOT NULL,
    reason        TEXT NOT NULL,
    before_state  TEXT NOT NULL,
    after_state   TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_adapter_promotions (
    sequence      INTEGER PRIMARY KEY AUTOINCREMENT,
    digest        TEXT NOT NULL UNIQUE,
    base_model_version TEXT NOT NULL,
    revision      INTEGER NOT NULL,
    payload       TEXT NOT NULL,
    approved_by   TEXT NOT NULL,
    validation_loss REAL,
    created_at    REAL NOT NULL
);
"""


def trust_for_origin(origin: str) -> int:
    if origin.startswith("web:") or origin == "web":
        return 1
    return _TRUST_BY_ORIGIN.get(origin, 2)


def normalize_fact(text: str) -> str:
    return " ".join(str(text).split())


def norm_hash(text: str) -> str:
    return hashlib.sha1(normalize_fact(text).casefold().encode("utf-8")).hexdigest()


@dataclass
class FactWriteResult:
    """Outcome of MemoryStore.add_fact.

    action: "inserted" | "merged" | "contradiction" | "skipped".
    """

    action: str
    fact_id: int = 0
    merged_with: int | None = None
    contradicted: list[int] = field(default_factory=list)


# Cross-process writers (CLI export/backup against a live service) get 5s of
# grace instead of an immediate "database is locked" error.
BUSY_TIMEOUT_MS = 5000


def _get_conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.executescript(_SCHEMA)
    _repair_legacy_schema(conn)
    now = time.time()
    conn.execute(
        "INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        ("schema_version", str(SCHEMA_VERSION), now),
    )
    conn.commit()
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.OperationalError:
        return set()


def _repair_legacy_schema(conn: sqlite3.Connection) -> None:
    """Repair additive schema gaps that CREATE IF NOT EXISTS cannot change."""
    fts_columns = _table_columns(conn, "memory_fts")
    if fts_columns and "ref_id" not in fts_columns:
        conn.execute("DROP TABLE memory_fts")
        conn.execute("CREATE VIRTUAL TABLE memory_fts USING fts5(content, memory_type, ref_id UNINDEXED)")
        _rebuild_memory_fts(conn)
    episode_columns = _table_columns(conn, "episodes")
    if episode_columns and "parent_episode_id" not in episode_columns:
        conn.execute("ALTER TABLE episodes ADD COLUMN parent_episode_id TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_parent ON episodes (parent_episode_id, created_at DESC)")


def _rebuild_memory_fts(conn: sqlite3.Connection) -> None:
    fact_columns = _table_columns(conn, "facts")
    if {"id", "fact"}.issubset(fact_columns):
        for row in conn.execute("SELECT id, fact FROM facts").fetchall():
            conn.execute(
                "INSERT INTO memory_fts (content, memory_type, ref_id) VALUES (?, ?, ?)",
                (row["fact"], "fact", str(row["id"])),
            )
    episode_columns = _table_columns(conn, "episodes")
    if {"id", "summary", "input", "output"}.issubset(episode_columns):
        for row in conn.execute("SELECT id, summary, input, output FROM episodes").fetchall():
            content = (row["summary"] or f"{row['input'][:300]} -> {row['output'][:300]}").strip()
            if content:
                conn.execute(
                    "INSERT INTO memory_fts (content, memory_type, ref_id) VALUES (?, ?, ?)",
                    (content, "episode", row["id"]),
                )


class MemoryStore:
    """SQLite-backed persistent memory for the conscious agent.

    Exposes an async API while performing small SQLite operations synchronously.
    Each store owns its connection so tests and callers can isolate db_path.
    All SQLite access (including from goals/autonomy/retrieval/consolidation)
    routes through this class's RLock-guarded helpers — the locking invariant.
    """

    def __init__(self, db_path: str | None = None, embedder: Embedder | None = None) -> None:
        self._db_path = db_path or str(_DB_PATH)
        self._conn_obj: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self.embedder = embedder

    def _conn(self) -> sqlite3.Connection:
        if self._conn_obj is None:
            self._conn_obj = _get_conn(self._db_path)
        return self._conn_obj

    async def initialize(self) -> None:
        self._get_shared()

    def _get_shared(self) -> None:
        self._conn()

    async def close(self) -> None:
        if self._conn_obj is not None:
            conn = self._conn_obj
            self._conn_obj = None
            conn.close()

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            cursor = self._conn().execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]

    def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        rows = self.fetchall(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn().execute(sql, params)
            self._conn().commit()

    def schema_version(self) -> int:
        row = self.fetchone("SELECT value FROM schema_meta WHERE key = ?", ("schema_version",))
        if row is None:
            # Existing fresh-start v2 DBs before public-beta metadata had the
            # core tables but no schema_meta row.
            return 2
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return 0

    def migrate_schema(self) -> int:
        """Ensure additive public-beta schema pieces exist and stamp version."""
        with self._lock:
            conn = self._conn()
            conn.executescript(_SCHEMA)
            _repair_legacy_schema(conn)
            now = time.time()
            conn.execute(
                "INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                ("schema_version", str(SCHEMA_VERSION), now),
            )
            conn.commit()
        return SCHEMA_VERSION

    def record_tool_event(
        self,
        *,
        episode_id: str,
        tick: int,
        source: str = "",
        tool: str,
        capabilities: list[str] | tuple[str, ...] | None = None,
        args: dict[str, Any] | None = None,
        result_summary: str = "",
        error: bool = False,
        exit_code: int | None = None,
        taint_origin: str = "",
    ) -> int:
        now = time.time()
        with self._lock:
            conn = self._conn()
            cursor = conn.execute(
                "INSERT INTO tool_events "
                "(episode_id, tick, source, tool, capabilities, args, result_summary, "
                "error, exit_code, taint_origin, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    episode_id,
                    int(tick),
                    source,
                    tool,
                    json.dumps(list(capabilities or [])),
                    json.dumps(args or {}, ensure_ascii=False),
                    result_summary,
                    int(bool(error)),
                    exit_code,
                    taint_origin,
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    async def recent_tool_events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.fetchall(
            "SELECT * FROM tool_events ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        for row in rows:
            row["capabilities"] = json.loads(row.get("capabilities") or "[]")
            row["args"] = json.loads(row.get("args") or "{}")
            row["error"] = bool(row.get("error"))
        return rows

    def executescript(self, sql: str) -> None:
        with self._lock:
            self._conn().executescript(sql)
            self._conn().commit()

    def transaction(self, items: list[tuple[str, tuple]]) -> int:
        """Execute multiple statements in a single locked commit. Returns rowcount of last statement."""
        with self._lock:
            conn = self._conn()
            last_rowcount = 0
            for sql, params in items:
                cursor = conn.execute(sql, params)
                last_rowcount = cursor.rowcount
            conn.commit()
            return last_rowcount

    # Legacy aliases — internal callers in store.py used _-prefixed names.
    _fetchall = fetchall
    _execute = execute
    _execute_many = transaction

    # ── Sessions (v1 compat no-ops; the sessions table is gone in v2) ────

    async def create_session(self, session_id: str, name: str = "") -> None:
        return None

    async def end_session(self, session_id: str, summary: str = "") -> None:
        return None

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        return []

    # ── Episodes (unified: chat + autonomous + service) ──────────────────

    async def record_episode(
        self,
        *,
        episode_id: str,
        source: str,
        event_type: str,
        input: str,
        output: str,
        selected_action: str = "",
        summary: str = "",
        tainted: bool = False,
        web_origins: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        trace: str = "",
        goal_id: str | None = None,
        project_id: str | None = None,
        parent_episode_id: str | None = None,
    ) -> None:
        """Upsert one unified episode row keyed by the runtime's per-episode uuid.

        Both the per-episode consolidator (cheap path: summary/taint) and the
        service layer (trace/metrics) write the same row; non-default fields
        from an earlier write are preserved when the later write omits them.
        """
        now = time.time()
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO episodes (id, source, event_type, goal_id, project_id, parent_episode_id, input, output, "
                "selected_action, summary, tainted, web_origins, metrics, trace, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "source = excluded.source, "
                "event_type = excluded.event_type, "
                "goal_id = COALESCE(excluded.goal_id, episodes.goal_id), "
                "project_id = COALESCE(excluded.project_id, episodes.project_id), "
                "parent_episode_id = COALESCE(excluded.parent_episode_id, episodes.parent_episode_id), "
                "input = excluded.input, "
                "output = excluded.output, "
                "selected_action = CASE WHEN excluded.selected_action != '' "
                "THEN excluded.selected_action ELSE episodes.selected_action END, "
                "summary = CASE WHEN excluded.summary != '' THEN excluded.summary ELSE episodes.summary END, "
                "tainted = MAX(episodes.tainted, excluded.tainted), "
                "web_origins = CASE WHEN excluded.web_origins != '[]' "
                "THEN excluded.web_origins ELSE episodes.web_origins END, "
                "metrics = CASE WHEN excluded.metrics != '{}' THEN excluded.metrics ELSE episodes.metrics END, "
                "trace = CASE WHEN excluded.trace != '' THEN excluded.trace ELSE episodes.trace END",
                (
                    episode_id,
                    source,
                    event_type,
                    goal_id,
                    project_id,
                    parent_episode_id,
                    input,
                    output,
                    selected_action,
                    summary,
                    int(bool(tainted)),
                    json.dumps(list(web_origins or [])),
                    json.dumps(metrics or {}),
                    trace,
                    now,
                ),
            )
            row = conn.execute("SELECT summary, input, output FROM episodes WHERE id = ?", (episode_id,)).fetchone()
            fts_content = (row["summary"] or f"{row['input'][:300]} -> {row['output'][:300]}").strip()
            conn.execute(
                "DELETE FROM memory_fts WHERE memory_type = ? AND ref_id = ?",
                ("episode", episode_id),
            )
            conn.execute(
                "INSERT INTO memory_fts (content, memory_type, ref_id) VALUES (?, ?, ?)",
                (fts_content, "episode", episode_id),
            )
            conn.commit()

    async def recent_episodes(self, limit: int = 10) -> list[dict]:
        rows = self._fetchall(
            "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._episode_from_row(row) for row in rows]

    async def episodes_before(self, cursor_ts: float, limit: int = 20) -> list[dict]:
        """Cursor pagination — return episodes older than ``cursor_ts``."""
        rows = self._fetchall(
            "SELECT * FROM episodes WHERE created_at < ? ORDER BY created_at DESC LIMIT ?",
            (cursor_ts, limit),
        )
        return [self._episode_from_row(row) for row in rows]

    async def get_episode(self, episode_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT * FROM episodes WHERE id = ?", (episode_id,))
        return self._episode_from_row(row) if row else None

    async def count_episodes(self) -> int:
        rows = self._fetchall("SELECT COUNT(*) AS count FROM episodes")
        return int(rows[0]["count"]) if rows else 0

    def _episode_from_row(self, row: dict) -> dict:
        data = dict(row)
        data["metrics"] = json.loads(data.get("metrics") or "{}")
        data["web_origins"] = json.loads(data.get("web_origins") or "[]")
        data["tainted"] = bool(data.get("tainted"))
        return data

    # ── V3 append-only causal history and recurrent checkpoints ─────────

    async def append_cognitive_event(self, event: Any) -> int:
        """Append one typed event.  There is intentionally no update/delete API."""
        data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        with self._lock:
            conn = self._conn()
            cursor = conn.execute(
                "INSERT INTO cognitive_events "
                "(event_id, episode_id, event_type, source, payload, model_input, checkpoint_id, "
                "parent_event_id, schema_version, observed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data["event_id"],
                    data["episode_id"],
                    data["event_type"],
                    data["source"],
                    json.dumps(data.get("payload") or {}, sort_keys=True),
                    json.dumps(data["model_input"], sort_keys=True) if data.get("model_input") is not None else None,
                    data.get("checkpoint_id"),
                    data.get("parent_event_id"),
                    int(data.get("schema_version", 1)),
                    float(data.get("observed_at", time.time())),
                ),
            )
            conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("cognitive event insert did not return a sequence")
            return cursor.lastrowid

    async def cognitive_events(self, episode_id: str) -> list[dict[str, Any]]:
        rows = self._fetchall("SELECT * FROM cognitive_events WHERE episode_id = ? ORDER BY sequence", (episode_id,))
        for row in rows:
            row["payload"] = json.loads(row.get("payload") or "{}")
            row["model_input"] = json.loads(row["model_input"]) if row.get("model_input") is not None else None
        return rows

    async def save_core_checkpoint(self, checkpoint: Any) -> None:
        """Persist a versioned immutable checkpoint; duplicate ids are errors."""
        data = checkpoint.to_dict() if hasattr(checkpoint, "to_dict") else dict(checkpoint)
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO core_checkpoints "
                "(checkpoint_id, lineage_id, parent_checkpoint_id, model_version, payload, "
                "event_sequence, schema_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    data["checkpoint_id"],
                    data["lineage_id"],
                    data.get("parent_checkpoint_id"),
                    data["model_version"],
                    json.dumps(data, sort_keys=True),
                    int(data.get("event_sequence", 0)),
                    int(data.get("schema_version", 1)),
                    float(data.get("created_at", time.time())),
                ),
            )
            conn.commit()

    async def latest_core_checkpoint(self, lineage_id: str | None = None) -> dict[str, Any] | None:
        if lineage_id:
            row = self.fetchone(
                "SELECT payload FROM core_checkpoints WHERE lineage_id = ? ORDER BY created_at DESC LIMIT 1",
                (lineage_id,),
            )
        else:
            row = self.fetchone("SELECT payload FROM core_checkpoints ORDER BY created_at DESC LIMIT 1")
        return json.loads(row["payload"]) if row else None

    async def get_core_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        row = self.fetchone("SELECT payload FROM core_checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
        return json.loads(row["payload"]) if row else None

    async def core_checkpoint_lineage_head(
        self,
        *,
        root_checkpoint_id: str,
        lineage_id: str,
    ) -> dict[str, Any]:
        """Return the unique validated descendant head for a checkpoint lineage."""
        if not root_checkpoint_id or not lineage_id:
            raise ValueError("checkpoint root and lineage identifiers are required")
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT * FROM core_checkpoints WHERE lineage_id = ?",
                (lineage_id,),
            ).fetchall()
            checkpoints: dict[str, dict[str, Any]] = {}
            children: dict[str, list[str]] = {}
            for row in rows:
                payload = json.loads(row["payload"])
                checkpoint_id = str(payload.get("checkpoint_id") or "")
                if (
                    not checkpoint_id
                    or row["checkpoint_id"] != checkpoint_id
                    or row["lineage_id"] != payload.get("lineage_id")
                    or row["parent_checkpoint_id"] != payload.get("parent_checkpoint_id")
                    or row["model_version"] != payload.get("model_version")
                    or row["event_sequence"] != payload.get("event_sequence")
                    or row["schema_version"] != payload.get("schema_version")
                    or float(row["created_at"]) != float(payload.get("created_at"))
                ):
                    raise ValueError("checkpoint lineage index is inconsistent")
                if checkpoint_id in checkpoints:
                    raise ValueError("checkpoint lineage contains a duplicate checkpoint")
                checkpoints[checkpoint_id] = payload
                parent = payload.get("parent_checkpoint_id")
                if isinstance(parent, str):
                    children.setdefault(parent, []).append(checkpoint_id)

            if root_checkpoint_id not in checkpoints:
                raise ValueError("checkpoint lineage root is missing")
            root = checkpoints[root_checkpoint_id]
            current_id = root_checkpoint_id
            visited = {current_id}
            while True:
                next_ids = children.get(current_id, [])
                if len(next_ids) > 1:
                    raise ValueError("checkpoint lineage is branched")
                if not next_ids:
                    break
                next_id = next_ids[0]
                if next_id in visited:
                    raise ValueError("checkpoint lineage contains a cycle")
                current = checkpoints[current_id]
                successor = checkpoints[next_id]
                if (
                    successor.get("model_version") != root.get("model_version")
                    or successor.get("schema_version") != root.get("schema_version")
                    or successor.get("specialist_architecture_id") != root.get("specialist_architecture_id")
                ):
                    raise ValueError("checkpoint lineage runtime identity changed")
                if int(successor["event_sequence"]) < int(current["event_sequence"]):
                    raise ValueError("checkpoint lineage event sequence regressed")
                visited.add(next_id)
                current_id = next_id
            if visited != set(checkpoints):
                raise ValueError("checkpoint lineage contains disconnected checkpoints")
            return checkpoints[current_id]

    @staticmethod
    def _architecture_migration_records(
        conn: sqlite3.Connection,
    ) -> tuple[CheckpointArchitectureMigration, ...]:
        from conscio.v3.checkpoint_migration import (  # noqa: PLC0415
            CheckpointArchitectureMigration,
        )

        rows = conn.execute("SELECT * FROM checkpoint_architecture_migrations ORDER BY sequence").fetchall()
        records: list[CheckpointArchitectureMigration] = []
        previous: str | None = None
        for row in rows:
            payload = json.loads(row["payload"])
            record = CheckpointArchitectureMigration.from_dict(payload)
            if record.previous_record_hash != previous:
                raise ValueError("checkpoint architecture migration hash chain is broken")
            if (
                row["record_id"] != record.record_id
                or row["source_checkpoint_id"] != record.source_checkpoint_id
                or row["target_checkpoint_id"] != record.target_checkpoint_id
                or row["record_hash"] != record.record_hash
                or row["previous_record_hash"] != record.previous_record_hash
            ):
                raise ValueError("checkpoint architecture migration index is inconsistent")
            records.append(record)
            previous = record.record_hash
        return tuple(records)

    async def core_checkpoint_architecture_migration(
        self, source_checkpoint_id: str
    ) -> CheckpointArchitectureMigration | None:
        """Return a verified migration for one source checkpoint, if present."""
        from conscio.v3.checkpoint_migration import content_digest  # noqa: PLC0415

        with self._lock:
            conn = self._conn()
            records = self._architecture_migration_records(conn)
            record = next(
                (item for item in records if item.source_checkpoint_id == source_checkpoint_id),
                None,
            )
            if record is None:
                return None
            source = conn.execute(
                "SELECT payload FROM core_checkpoints WHERE checkpoint_id = ?",
                (record.source_checkpoint_id,),
            ).fetchone()
            if source is None:
                raise ValueError("checkpoint architecture migration source is missing")
            if content_digest(json.loads(source["payload"])) != record.source_checkpoint_digest:
                raise ValueError("checkpoint architecture migration source digest differs")
            target = conn.execute(
                "SELECT payload FROM core_checkpoints WHERE checkpoint_id = ?",
                (record.target_checkpoint_id,),
            ).fetchone()
            if target is None:
                raise ValueError("checkpoint architecture migration target is missing")
            if content_digest(json.loads(target["payload"])) != record.target_checkpoint_digest:
                raise ValueError("checkpoint architecture migration target digest differs")
            return record

    async def migrate_core_checkpoint_architecture(
        self,
        *,
        source_checkpoint_id: str,
        source_architecture_id: str,
        target_checkpoint: Any,
        runtime_identity: str,
        transform_digest: str,
        evidence_digest: str | None = None,
        migrator: str = "runtime_bootstrap_migrator",
        reason: str = "bootstrap specialist architecture migration",
    ) -> CheckpointArchitectureMigration:
        """Atomically save a migrated checkpoint and its hash-chained audit record."""
        from conscio.v3.checkpoint_migration import (  # noqa: PLC0415
            CheckpointArchitectureMigration,
        )

        target = target_checkpoint.to_dict() if hasattr(target_checkpoint, "to_dict") else dict(target_checkpoint)
        with self._lock:
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                records = self._architecture_migration_records(conn)
                existing = next(
                    (item for item in records if item.source_checkpoint_id == source_checkpoint_id),
                    None,
                )
                if existing is not None:
                    conn.commit()
                    return existing
                source_row = conn.execute(
                    "SELECT payload FROM core_checkpoints WHERE checkpoint_id = ?",
                    (source_checkpoint_id,),
                ).fetchone()
                if source_row is None:
                    raise ValueError("architecture migration source checkpoint is missing")
                source = json.loads(source_row["payload"])
                created_at = time.time()
                record = CheckpointArchitectureMigration.create(
                    source_checkpoint=source,
                    source_architecture_id=source_architecture_id,
                    target_checkpoint=target,
                    runtime_identity=runtime_identity,
                    transform_digest=transform_digest,
                    evidence_digest=evidence_digest,
                    migrator=migrator,
                    reason=reason,
                    previous_record_hash=(records[-1].record_hash if records else None),
                    created_at=created_at,
                )
                conn.execute(
                    "INSERT INTO core_checkpoints "
                    "(checkpoint_id, lineage_id, parent_checkpoint_id, model_version, payload, "
                    "event_sequence, schema_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        target["checkpoint_id"],
                        target["lineage_id"],
                        target.get("parent_checkpoint_id"),
                        target["model_version"],
                        json.dumps(target, sort_keys=True),
                        int(target.get("event_sequence", 0)),
                        int(target["schema_version"]),
                        float(target.get("created_at", created_at)),
                    ),
                )
                conn.execute(
                    "INSERT INTO checkpoint_architecture_migrations "
                    "(record_id, source_checkpoint_id, target_checkpoint_id, "
                    "previous_record_hash, record_hash, payload, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.record_id,
                        record.source_checkpoint_id,
                        record.target_checkpoint_id,
                        record.previous_record_hash,
                        record.record_hash,
                        json.dumps(record.to_dict(), sort_keys=True),
                        record.created_at,
                    ),
                )
                conn.commit()
                return record
            except Exception:
                conn.rollback()
                raise

    async def record_affect_intervention(
        self,
        *,
        intervention_id: str,
        episode_id: str,
        operator: str,
        reason: str,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
    ) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO affect_interventions "
                "(intervention_id, episode_id, operator, reason, before_state, after_state, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    intervention_id,
                    episode_id,
                    operator,
                    reason,
                    json.dumps(before_state, sort_keys=True),
                    json.dumps(after_state, sort_keys=True),
                    time.time(),
                ),
            )
            conn.commit()

    async def record_prediction_adapter_promotion(
        self,
        *,
        digest: str,
        base_model_version: str,
        revision: int,
        payload: str,
        approved_by: str,
        validation_loss: float | None,
    ) -> None:
        """Append an explicitly approved adapter promotion; history is immutable."""
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT INTO prediction_adapter_promotions "
                "(digest, base_model_version, revision, payload, approved_by, validation_loss, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    digest,
                    base_model_version,
                    revision,
                    payload,
                    approved_by,
                    validation_loss,
                    time.time(),
                ),
            )
            conn.commit()

    async def latest_prediction_adapter(self, base_model_version: str | None = None) -> dict[str, Any] | None:
        if base_model_version is None:
            row = self.fetchone("SELECT * FROM prediction_adapter_promotions ORDER BY sequence DESC LIMIT 1")
        else:
            row = self.fetchone(
                "SELECT * FROM prediction_adapter_promotions WHERE base_model_version = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (base_model_version,),
            )
        if row is not None:
            row["state"] = json.loads(row["payload"])
        return row

    async def cognitive_event_history(self, limit: int = 100_000) -> list[dict[str, Any]]:
        rows = self._fetchall("SELECT * FROM cognitive_events ORDER BY sequence DESC LIMIT ?", (max(1, limit),))
        rows.reverse()
        for row in rows:
            row["payload"] = json.loads(row.get("payload") or "{}")
            row["model_input"] = json.loads(row["model_input"]) if row.get("model_input") is not None else None
        return rows

    # ── Facts (semantic memory with provenance + embeddings) ─────────────

    async def add_fact(
        self,
        fact: str,
        source: str | None = None,
        confidence: str = "MEDIUM",
        *,
        origin: str | None = None,
        trust: int | None = None,
        episode_id: str | None = None,
        contradiction_judge: Any | None = None,
    ) -> FactWriteResult:
        """Write one fact with dedup/merge/contradiction semantics.

        Back-compat: positional ``source`` maps to ``origin``. Steps:
        exact-dup merge via norm_hash; embed (best-effort); near-dup cosine
        merge above MERGE_THRESHOLD; flag-gated contradiction judge on the
        ambiguous band; else insert (INSERT OR IGNORE on norm_hash races).

        Merge semantics: re-asserting resurrects archived/contradicted rows
        (status back to 'active'), and trust is never raised across the
        web/agent boundary — web/quarantined rows keep the tainted tier.
        """
        text = normalize_fact(fact)
        if not text:
            return FactWriteResult(action="skipped")
        resolved_origin = (origin or source or "").strip() or "agent"
        resolved_trust = trust_for_origin(resolved_origin) if trust is None else int(trust)
        nh = norm_hash(text)

        existing = self.fetchone("SELECT id, trust, confidence, origin, status FROM facts WHERE norm_hash = ?", (nh,))
        if existing:
            return self._merge_into(existing, trust=resolved_trust, confidence=confidence)

        vec: list[float] | None = None
        if self.embedder is not None:
            try:
                vec = await self.embedder.embed(text)
            except Exception:  # noqa: BLE001 — embedding is best-effort
                vec = None

        if vec is not None:
            candidates = self._embedded_fact_candidates(text, limit=20)
            best: dict | None = None
            best_cos = 0.0
            for candidate in candidates:
                cos = _embeddings.cosine(np.asarray(vec), _embeddings.unpack(candidate["embedding"]))
                if cos > best_cos:
                    best_cos = cos
                    best = candidate
            if best is not None and best_cos > MERGE_THRESHOLD:
                return self._merge_into(best, trust=resolved_trust, confidence=confidence)
            if best is not None and contradiction_judge is not None and CONTRADICTION_LOW <= best_cos < MERGE_THRESHOLD:
                contradicts = False
                try:
                    contradicts = bool(await contradiction_judge(text, best["fact"]))
                except Exception:  # noqa: BLE001 — judge is best-effort
                    contradicts = False
                if contradicts:
                    fact_id = self._insert_fact_row(
                        text, nh, resolved_origin, resolved_trust, episode_id, confidence, vec
                    )
                    if fact_id is None:
                        refetched = self.fetchone(
                            "SELECT id, trust, confidence, origin, status FROM facts WHERE norm_hash = ?",
                            (nh,),
                        )
                        if refetched:
                            return self._merge_into(refetched, trust=resolved_trust, confidence=confidence)
                        return FactWriteResult(action="skipped")
                    losers = await self.mark_contradiction(fact_id, int(best["id"]))
                    return FactWriteResult(action="contradiction", fact_id=fact_id, contradicted=losers)

        fact_id = self._insert_fact_row(text, nh, resolved_origin, resolved_trust, episode_id, confidence, vec)
        if fact_id is None:
            # norm_hash race or legitimate normalization collision: merge instead of failing.
            refetched = self.fetchone(
                "SELECT id, trust, confidence, origin, status FROM facts WHERE norm_hash = ?",
                (nh,),
            )
            if refetched:
                return self._merge_into(refetched, trust=resolved_trust, confidence=confidence)
            return FactWriteResult(action="skipped")
        return FactWriteResult(action="inserted", fact_id=fact_id)

    def _merge_into(self, row: dict, *, trust: int, confidence: str) -> FactWriteResult:
        now = time.time()
        old_conf = str(row.get("confidence") or "MEDIUM")
        new_conf = old_conf if _CONF_RANK.get(old_conf, 1) >= _CONF_RANK.get(confidence, 1) else confidence
        old_trust = int(row["trust"])
        origin = str(row.get("origin") or "")
        tainted = origin.startswith("web:") or origin in {"web", "quarantined"}
        # Never launder trust upward across the web/agent boundary: a re-asserted
        # web/quarantined fact keeps the tainted tier (min) instead of being
        # promoted to the asserting origin's tier. Untainted rows keep max(old, new).
        merged_trust = min(old_trust, trust) if tainted else max(old_trust, trust)
        # Re-asserting a fact resurrects archived/contradicted rows; otherwise the
        # merge target stays invisible to retrieval (status='active' filter) forever.
        old_status = str(row.get("status") or "active")
        new_status = "active" if old_status in {"archived", "contradicted"} else old_status
        self._execute(
            "UPDATE facts SET access_count = access_count + 1, updated_at = ?, "
            "last_accessed = ?, trust = ?, confidence = ?, status = ? WHERE id = ?",
            (now, now, merged_trust, new_conf, new_status, row["id"]),
        )
        return FactWriteResult(action="merged", fact_id=int(row["id"]), merged_with=int(row["id"]))

    def _insert_fact_row(
        self,
        text: str,
        nh: str,
        origin: str,
        trust: int,
        episode_id: str | None,
        confidence: str,
        vec: list[float] | None,
    ) -> int | None:
        """Insert a facts row + its FTS mirror in one locked commit.

        Returns the new fact id, or None when the norm_hash unique index
        ignored the insert (caller falls back to merge semantics).
        """
        now = time.time()
        blob = _embeddings.pack(vec) if vec is not None else None
        model = getattr(self.embedder, "model", None) if vec is not None else None
        with self._lock:
            conn = self._conn()
            cursor = conn.execute(
                "INSERT OR IGNORE INTO facts "
                "(fact, norm_hash, origin, trust, episode_id, confidence, status, "
                "embedding, embedding_model, access_count, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, 0, ?, ?)",
                (text, nh, origin, trust, episode_id, confidence, blob, model, now, now),
            )
            if cursor.rowcount == 0:
                conn.commit()
                return None
            fact_id = int(cursor.lastrowid or 0)
            conn.execute(
                "INSERT INTO memory_fts (content, memory_type, ref_id) VALUES (?, ?, ?)",
                (text, "fact", str(fact_id)),
            )
            conn.commit()
            return fact_id

    def _embedded_fact_candidates(self, text: str, limit: int = 20) -> list[dict]:
        from conscio.memory.retrieval import build_fts_query  # noqa: PLC0415

        match = build_fts_query(text, mode="or")
        if not match:
            return []
        try:
            return self.fetchall(
                "SELECT f.id, f.fact, f.trust, f.confidence, f.origin, f.status, f.embedding "
                "FROM memory_fts "
                "JOIN facts f ON f.id = CAST(memory_fts.ref_id AS INTEGER) "
                "WHERE memory_fts MATCH ? AND f.status = 'active' AND f.embedding IS NOT NULL "
                "ORDER BY bm25(memory_fts) LIMIT ?",
                (f"memory_type:fact AND ({match})", limit),
            )
        except sqlite3.OperationalError:
            return []

    async def mark_contradiction(self, fact_id_a: int, fact_id_b: int) -> list[int]:
        """Mark contradiction between two facts. Trust floor: the lower tier
        loses; equal tiers mark both. Never deletes."""
        rows = {
            int(r["id"]): r
            for r in self.fetchall("SELECT id, trust FROM facts WHERE id IN (?, ?)", (fact_id_a, fact_id_b))
        }
        a = rows.get(int(fact_id_a))
        b = rows.get(int(fact_id_b))
        if a is None or b is None:
            return []
        if int(a["trust"]) > int(b["trust"]):
            losers = [b]
        elif int(b["trust"]) > int(a["trust"]):
            losers = [a]
        else:
            losers = [a, b]
        now = time.time()
        self._execute_many(
            [
                (
                    "UPDATE facts SET status = 'contradicted', updated_at = ? WHERE id = ?",
                    (now, loser["id"]),
                )
                for loser in losers
            ]
        )
        try:
            self._execute(
                "INSERT INTO action_events (kind, created_at) VALUES (?, ?)",
                ("fact_contradiction", now),
            )
        except sqlite3.OperationalError:
            pass  # action_events lives in the autonomy schema; absent in bare stores
        return [int(loser["id"]) for loser in losers]

    async def retrieve_facts(
        self,
        query: str,
        *,
        limit: int = 5,
        include_web: bool = True,
        max_web: int = 2,
        embedder: Embedder | None = None,
    ) -> list[Any]:
        """Hybrid retrieval (FTS BM25 prefilter -> cosine rerank -> provenance
        shaping). The single retrieval surface; see memory/retrieval.py."""
        from conscio.memory.retrieval import retrieve_facts  # noqa: PLC0415

        return await retrieve_facts(
            self,
            query,
            limit=limit,
            include_web=include_web,
            max_web=max_web,
            embedder=embedder or self.embedder,
        )

    async def search_facts(self, query: str, limit: int = 10) -> list[dict]:
        results = await self.retrieve_facts(query, limit=limit)
        if results:
            return [r.to_dict() for r in results]
        # Fallback for substring queries FTS cannot tokenize.
        rows = self._fetchall(
            "SELECT id, fact, origin, trust, confidence, created_at, updated_at FROM facts "
            "WHERE status = 'active' AND trust > 0 AND fact LIKE ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        return [{**row, "source": row["origin"]} for row in rows]

    async def recent_facts(self, limit: int = 10) -> list[dict]:
        rows = self._fetchall(
            "SELECT id, fact, origin, trust, confidence, status, created_at, updated_at "
            "FROM facts WHERE status = 'active' AND trust > 0 ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [{**row, "source": row["origin"]} for row in rows]

    # ── Procedures (deliberate, validated; replaces v1 junk skills) ──────

    async def upsert_procedure(
        self,
        name: str,
        description: str,
        steps: str,
        trigger: str = "",
        origin: str = "agent",
    ) -> None:
        now = time.time()
        self._execute(
            "INSERT INTO procedures (name, description, steps, trigger, origin, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "description = excluded.description, steps = excluded.steps, "
            "trigger = excluded.trigger, updated_at = excluded.updated_at",
            (name, description, steps, trigger, origin, now, now),
        )

    async def list_procedures(self) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM procedures ORDER BY success_count DESC, updated_at DESC",
        )

    # ── Chat (operator console persistence) ──────────────────────

    async def list_chat_sessions(self, limit: int = 50) -> list[dict]:
        return self._fetchall(
            "SELECT id, title, created_at, updated_at FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    async def get_chat_session(self, session_id: str) -> dict | None:
        return self.fetchone(
            "SELECT id, title, created_at, updated_at FROM chat_sessions WHERE id = ?",
            (session_id,),
        )

    async def upsert_chat_session(self, session_id: str, title: str | None = None) -> None:
        now = time.time()
        self._execute(
            "INSERT INTO chat_sessions (id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "title = COALESCE(excluded.title, chat_sessions.title), "
            "updated_at = excluded.updated_at",
            (session_id, title, now, now),
        )

    async def append_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        selected_action: str | None = None,
        episode_id: str | None = None,
    ) -> int:
        now = time.time()
        with self._lock:
            conn = self._conn()
            cursor = conn.execute(
                "INSERT INTO chat_messages "
                "(session_id, role, content, selected_action, episode_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, selected_action, episode_id, now),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    async def get_chat_messages(self, session_id: str, limit: int = 200, before_id: int | None = None) -> list[dict]:
        if before_id is not None:
            return self._fetchall(
                "SELECT id, session_id, role, content, selected_action, episode_id, created_at "
                "FROM chat_messages WHERE session_id = ? AND id < ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, before_id, limit),
            )
        return self._fetchall(
            "SELECT id, session_id, role, content, selected_action, episode_id, created_at "
            "FROM chat_messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )

    async def delete_chat_session(self, session_id: str) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.commit()

    # ── Full-text search ─────────────────────────────────────────

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        from conscio.memory.retrieval import build_fts_query

        fts_query = build_fts_query(query, mode="or")
        if fts_query:
            try:
                return self._fetchall(
                    "SELECT content, memory_type, ref_id, rank FROM memory_fts "
                    "WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
                    (fts_query, limit),
                )
            except sqlite3.OperationalError:
                pass
        # Fallback for short/special queries FTS cannot tokenize.
        like = f"%{query}%"
        facts = self._fetchall(
            "SELECT fact AS content, 'fact' AS memory_type, "
            "CAST(id AS TEXT) AS ref_id, 0 AS rank "
            "FROM facts WHERE status = 'active' AND trust > 0 AND fact LIKE ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (like, limit),
        )
        if len(facts) >= limit:
            return facts
        remaining = limit - len(facts)
        episodes = self._fetchall(
            "SELECT COALESCE(summary, input) AS content, 'episode' AS memory_type, "
            "CAST(id AS TEXT) AS ref_id, 0 AS rank "
            "FROM episodes WHERE (summary LIKE ? OR input LIKE ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (like, like, remaining),
        )
        return facts + episodes

    # ── Context assembly ─────────────────────────────────────────

    async def format_context(self, limit: int = 5) -> str:
        episodes = await self.recent_episodes(limit)
        parts: list[str] = []
        if episodes:
            parts.append("RECENT EPISODES:")
            for e in episodes:
                summary = e.get("summary") or e.get("input", "")
                parts.append(f"  - {summary}")
        return "\n".join(parts)

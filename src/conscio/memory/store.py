from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_HOME_DIR = Path.home() / ".conscio"
_DB_PATH = _HOME_DIR / "sessions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT,
    created_at REAL NOT NULL,
    ended_at REAL,
    summary TEXT
);
CREATE TABLE IF NOT EXISTS episodic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    outcome TEXT,
    confidence TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS semantic (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL UNIQUE,
    source TEXT,
    confidence TEXT DEFAULT 'MEDIUM',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS procedural (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill TEXT NOT NULL UNIQUE,
    description TEXT,
    steps TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(content, memory_type, source);
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
"""


def _get_conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


class MemoryStore:
    """SQLite-backed persistent memory for the conscious agent.

    Exposes an async API while performing small SQLite operations synchronously.
    Each store owns its connection so tests and callers can isolate db_path.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(_DB_PATH)
        self._conn_obj: sqlite3.Connection | None = None
        self._lock = threading.RLock()

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

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            cursor = self._conn().execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn().execute(sql, params)
            self._conn().commit()

    def _execute_many(self, items: list[tuple[str, tuple]]) -> None:
        with self._lock:
            for sql, params in items:
                self._conn().execute(sql, params)
            self._conn().commit()

    # ── Sessions ─────────────────────────────────────────────────

    async def create_session(self, session_id: str, name: str = "") -> None:
        self._execute(
            "INSERT OR IGNORE INTO sessions (id, name, created_at) VALUES (?, ?, ?)",
            (session_id, name, time.time()),
        )

    async def end_session(self, session_id: str, summary: str = "") -> None:
        self._execute(
            "UPDATE sessions SET ended_at = ?, summary = ? WHERE id = ?",
            (time.time(), summary, session_id),
        )

    async def list_sessions(self, limit: int = 20) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    # ── Episodic Memory ──────────────────────────────────────────

    async def add_episode(
        self,
        session_id: str,
        summary: str,
        outcome: str = "",
        confidence: str = "",
    ) -> None:
        ops = [
            (
                "INSERT INTO episodic (session_id, summary, outcome, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, summary, outcome, confidence, time.time()),
            ),
            (
                "INSERT INTO memory_fts (content, memory_type, source) VALUES (?, ?, ?)",
                (summary, "episodic", session_id),
            ),
        ]
        self._execute_many(ops)

    async def recent_episodes(self, session_id: str, limit: int = 10) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM episodic WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )

    async def count_episodes(self, session_id: str) -> int:
        rows = self._fetchall(
            "SELECT COUNT(*) AS count FROM episodic WHERE session_id = ?",
            (session_id,),
        )
        return int(rows[0]["count"]) if rows else 0

    # ── Semantic Memory ──────────────────────────────────────────

    async def add_fact(self, fact: str, source: str = "", confidence: str = "MEDIUM") -> None:
        now = time.time()
        ops = [
            (
                "INSERT INTO semantic (fact, source, confidence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(fact) DO UPDATE SET updated_at = ?, confidence = ?",
                (fact, source, confidence, now, now, now, confidence),
            ),
            (
                "INSERT INTO memory_fts (content, memory_type, source) VALUES (?, ?, ?)",
                (fact, "semantic", source),
            ),
        ]
        self._execute_many(ops)

    async def search_facts(self, query: str, limit: int = 10) -> list[dict]:
        return self._fetchall(
            "SELECT fact, source, confidence, created_at FROM semantic "
            "WHERE fact LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (f"%{query}%", limit),
        )

    async def recent_facts(self, limit: int = 10) -> list[dict]:
        return self._fetchall(
            "SELECT fact, source, confidence, created_at, updated_at FROM semantic "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    # ── Procedural Memory ────────────────────────────────────────

    async def add_skill(self, skill: str, description: str = "", steps: str = "") -> None:
        now = time.time()
        self._execute(
            "INSERT INTO procedural (skill, description, steps, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(skill) DO UPDATE SET "
            "description = excluded.description, steps = excluded.steps, updated_at = ?",
            (skill, description, steps, now, now, now),
        )

    async def use_skill(self, skill: str) -> None:
        self._execute(
            "UPDATE procedural SET use_count = use_count + 1, updated_at = ? WHERE skill = ?",
            (time.time(), skill),
        )

    async def list_skills(self) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM procedural ORDER BY use_count DESC",
        )

    # ── Thoughts (inner monologue persistence) ───────────────────

    async def save_thoughts(self, session_id: str, thoughts: list[dict]) -> None:
        ops = []
        for t in thoughts:
            ops.append((
                "INSERT OR REPLACE INTO thoughts "
                "(id, session_id, parent_id, type, question, answer, created_at, depth) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    t["id"],
                    session_id,
                    t.get("parent_id"),
                    t["type"],
                    t.get("question", ""),
                    t.get("answer", ""),
                    t.get("timestamp", time.time()),
                    t.get("depth", 0),
                ),
            ))
        if ops:
            self._execute_many(ops)

    async def load_thoughts(self, session_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM thoughts WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )

    # ── Full-text search ─────────────────────────────────────────

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        return self._fetchall(
            "SELECT content, memory_type, source, rank FROM memory_fts "
            "WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        )

    # ── Context assembly ─────────────────────────────────────────

    async def format_context(self, session_id: str, limit: int = 5) -> str:
        episodes = await self.recent_episodes(session_id, limit)
        parts: list[str] = []
        if episodes:
            parts.append("RECENT EPISODES:")
            for e in episodes:
                parts.append(f"  - {e['summary']} (confidence: {e.get('confidence', 'N/A')})")
        return "\n".join(parts)

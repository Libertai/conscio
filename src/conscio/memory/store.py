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
                "DELETE FROM memory_fts WHERE content = ? AND memory_type = ? AND source = ?",
                (fact, "semantic", source),
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

    # ── Chat (operator console persistence) ──────────────────────

    async def list_chat_sessions(self, limit: int = 50) -> list[dict]:
        return self._fetchall(
            "SELECT id, title, created_at, updated_at FROM chat_sessions "
            "ORDER BY updated_at DESC LIMIT ?",
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

    async def get_chat_messages(
        self, session_id: str, limit: int = 200, before_id: int | None = None
    ) -> list[dict]:
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

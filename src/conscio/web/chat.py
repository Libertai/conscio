"""Server-side persistence for the operator chat console.

Provides a thin async facade over MemoryStore's chat_sessions / chat_messages
tables, with a default ``main`` session created on first use so the UI can
work without explicit session management.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from conscio.memory.store import MemoryStore


DEFAULT_SESSION_ID = "main"
DEFAULT_SESSION_TITLE = "operator console"


class ChatStore:
    """Async helpers for chat session + message persistence."""

    def __init__(self, memory: MemoryStore) -> None:
        self._memory = memory

    async def ensure_default_session(self) -> str:
        existing = await self._memory.get_chat_session(DEFAULT_SESSION_ID)
        if existing is None:
            await self._memory.upsert_chat_session(DEFAULT_SESSION_ID, DEFAULT_SESSION_TITLE)
        return DEFAULT_SESSION_ID

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self._memory.list_chat_sessions(limit=limit)
        if not rows:
            await self.ensure_default_session()
            rows = await self._memory.list_chat_sessions(limit=limit)
        return rows

    async def create_session(self, title: str | None = None) -> dict[str, Any]:
        session_id = secrets.token_urlsafe(8)
        await self._memory.upsert_chat_session(session_id, title or "untitled")
        record = await self._memory.get_chat_session(session_id)
        assert record is not None
        return record

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return await self._memory.get_chat_session(session_id)

    async def delete_session(self, session_id: str) -> None:
        if session_id == DEFAULT_SESSION_ID:
            raise ValueError("cannot delete the default session")
        await self._memory.delete_chat_session(session_id)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        selected_action: str | None = None,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        if role not in ("user", "agent"):
            raise ValueError(f"unknown chat role: {role!r}")
        await self._memory.upsert_chat_session(session_id)
        message_id = await self._memory.append_chat_message(
            session_id,
            role,
            content,
            selected_action=selected_action,
            episode_id=episode_id,
        )
        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "selected_action": selected_action,
            "episode_id": episode_id,
            "created_at": time.time(),
        }

    async def get_messages(
        self,
        session_id: str,
        limit: int = 200,
        before_id: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = await self._memory.get_chat_messages(
            session_id, limit=limit, before_id=before_id
        )
        # The store returns DESC for paging; we hand back ASC for natural rendering.
        return list(reversed(rows))

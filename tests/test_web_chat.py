from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from conscio.memory.store import MemoryStore
from conscio.web.chat import ChatStore, DEFAULT_SESSION_ID


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=str(tmp_path / "chat.db"))


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_default_session_is_created(store: MemoryStore) -> None:
    chat = ChatStore(store)
    sid = _run(chat.ensure_default_session())
    assert sid == DEFAULT_SESSION_ID
    sessions = _run(chat.list_sessions())
    assert any(s["id"] == DEFAULT_SESSION_ID for s in sessions)


def test_round_trip_messages(store: MemoryStore) -> None:
    chat = ChatStore(store)
    _run(chat.ensure_default_session())
    _run(chat.append_message(DEFAULT_SESSION_ID, "user", "hello"))
    _run(chat.append_message(
        DEFAULT_SESSION_ID, "agent", "hi there", selected_action="reply", episode_id="ep-1",
    ))
    msgs = _run(chat.get_messages(DEFAULT_SESSION_ID))
    assert [m["content"] for m in msgs] == ["hello", "hi there"]
    assert [m["role"] for m in msgs] == ["user", "agent"]
    assert msgs[1]["selected_action"] == "reply"
    assert msgs[1]["episode_id"] == "ep-1"


def test_create_and_delete_named_session(store: MemoryStore) -> None:
    chat = ChatStore(store)
    record = _run(chat.create_session("debug session"))
    assert record["title"] == "debug session"
    sid = record["id"]
    _run(chat.append_message(sid, "user", "ping"))
    msgs = _run(chat.get_messages(sid))
    assert len(msgs) == 1

    _run(chat.delete_session(sid))
    assert _run(chat.get_session(sid)) is None
    assert _run(chat.get_messages(sid)) == []


def test_default_session_cannot_be_deleted(store: MemoryStore) -> None:
    chat = ChatStore(store)
    _run(chat.ensure_default_session())
    with pytest.raises(ValueError):
        _run(chat.delete_session(DEFAULT_SESSION_ID))


def test_unknown_role_rejected(store: MemoryStore) -> None:
    chat = ChatStore(store)
    with pytest.raises(ValueError):
        _run(chat.append_message(DEFAULT_SESSION_ID, "system", "nope"))


def test_pagination_before_id(store: MemoryStore) -> None:
    chat = ChatStore(store)
    _run(chat.ensure_default_session())
    ids = []
    for i in range(5):
        m = _run(chat.append_message(DEFAULT_SESSION_ID, "user", f"msg-{i}"))
        ids.append(m["id"])
    # Page back from the third id, expecting the two earlier messages, ascending.
    page = _run(chat.get_messages(DEFAULT_SESSION_ID, limit=10, before_id=ids[2]))
    assert [m["content"] for m in page] == ["msg-0", "msg-1"]

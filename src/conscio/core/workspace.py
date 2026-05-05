from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EntryType(str, Enum):
    OBSERVATION = "observation"
    INTENTION = "intention"
    PLAN = "plan"
    ACTION = "action"
    RESULT = "result"
    REFLECTION = "reflection"
    MEMORY = "memory"
    SYSTEM = "system"


@dataclass
class WorkspaceEntry:
    content: str
    source: str
    type: EntryType = EntryType.OBSERVATION
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: WorkspaceEntry) -> bool:
        return self.timestamp < other.timestamp


BroadcastHandler = Callable[[WorkspaceEntry], None]


class Workspace:
    """Global Workspace — a shared blackboard that modules read/write.

    Inspired by Baars' Global Workspace Theory: specialist modules compete
    for access; winning content is broadcast to all modules.
    """

    def __init__(self, max_entries: int = 100) -> None:
        self._entries: list[WorkspaceEntry] = []
        self._subscribers: list[BroadcastHandler] = []
        self._max_entries = max_entries

    def write(
        self,
        content: str,
        source: str,
        type: EntryType = EntryType.OBSERVATION,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceEntry:
        entry = WorkspaceEntry(
            content=content,
            source=source,
            type=type,
            priority=priority,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries.pop(0)
        return entry

    def broadcast(self, entry: WorkspaceEntry) -> None:
        for handler in self._subscribers:
            handler(entry)

    def write_and_broadcast(
        self,
        content: str,
        source: str,
        type: EntryType = EntryType.OBSERVATION,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceEntry:
        entry = self.write(content, source, type, priority, metadata)
        self.broadcast(entry)
        return entry

    def read(
        self,
        limit: int = 20,
        min_priority: int = 0,
        type_filter: set[EntryType] | None = None,
    ) -> list[WorkspaceEntry]:
        filtered = (e for e in self._entries if e.priority >= min_priority)
        if type_filter:
            filtered = (e for e in filtered if e.type in type_filter)
        sorted_entries = sorted(filtered, key=lambda e: (-e.priority, -e.timestamp))
        return sorted_entries[:limit]

    def attend(self, query: str, limit: int = 10) -> list[WorkspaceEntry]:
        query_lower = query.lower()
        scored: list[tuple[float, WorkspaceEntry]] = []
        for entry in self._entries:
            score = 0.0
            if query_lower in entry.content.lower():
                score += entry.priority + 1
            if query_lower in entry.source.lower():
                score += 0.5
            score += entry.priority * 0.1
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:limit]]

    def subscribe(self, handler: BroadcastHandler) -> Callable[[], None]:
        self._subscribers.append(handler)

        def unsubscribe() -> None:
            if handler in self._subscribers:
                self._subscribers.remove(handler)

        return unsubscribe

    def clear(self) -> None:
        self._entries.clear()

    @property
    def recent(self) -> list[WorkspaceEntry]:
        return sorted(self._entries, key=lambda e: -e.timestamp)[:10]

    @property
    def size(self) -> int:
        return len(self._entries)

    def format_context(self, limit: int = 20) -> str:
        return "\n".join(
            f"[{e.timestamp:.1f}] {e.source} ({e.type.value}): {e.content[:200]}"
            for e in self.read(limit=limit)
        )

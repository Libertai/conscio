from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntryType(str, Enum):
    OBSERVATION = "observation"
    INTENTION = "intention"
    PLAN = "plan"
    ACTION = "action"
    RESULT = "result"
    REFLECTION = "reflection"
    MEMORY = "memory"
    SYSTEM = "system"
    CONFLICT = "conflict"
    SELF_STATE = "self_state"


class Visibility(str, Enum):
    LOCAL = "local"
    GLOBAL = "global"


@dataclass
class WorkspaceEntry:
    content: str
    source: str
    type: EntryType = EntryType.OBSERVATION
    priority: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    salience: float = 0.0
    confidence: float = 0.5
    novelty: float = 0.0
    urgency: float = 0.0
    evidence: list[str] = field(default_factory=list)
    visibility: Visibility = Visibility.LOCAL
    attended: bool = False
    broadcast_count: int = 0
    episode_id: str = ""
    tick: int = -1
    appraised: bool = False
    resolved: bool = True  # set False by conflict writers; True once reflection handles it

    def __lt__(self, other: WorkspaceEntry) -> bool:
        return self.timestamp < other.timestamp


BroadcastHandler = Callable[[WorkspaceEntry], None]


class Workspace:
    """Global Workspace — a shared blackboard that modules read/write.

    Inspired by Baars' Global Workspace Theory: specialist modules compete
    for access; winning content is broadcast to all modules.
    """

    def __init__(self, max_entries: int = 400) -> None:
        self._entries: list[WorkspaceEntry] = []
        self._subscribers: list[BroadcastHandler] = []
        self._max_entries = max_entries
        self._current_episode = ""
        self._current_tick = -1

    @property
    def current_episode(self) -> str:
        """Canonical id of the episode in progress (memory provenance)."""
        return self._current_episode

    def write(
        self,
        content: str,
        source: str,
        type: EntryType = EntryType.OBSERVATION,
        priority: int = 0,
        metadata: dict[str, Any] | None = None,
        salience: float | None = None,
        confidence: float = 0.5,
        novelty: float = 0.0,
        urgency: float = 0.0,
        evidence: list[str] | None = None,
        visibility: Visibility = Visibility.LOCAL,
    ) -> WorkspaceEntry:
        entry = WorkspaceEntry(
            content=content,
            source=source,
            type=type,
            priority=priority,
            metadata=metadata or {},
            salience=salience if salience is not None else max(0.0, min(1.0, priority / 10)),
            confidence=confidence,
            novelty=novelty,
            urgency=urgency,
            evidence=evidence or [],
            visibility=visibility,
            episode_id=self._current_episode,
            tick=self._current_tick,
        )
        self._entries.append(entry)
        self._evict_overflow()
        return entry

    def _evict_overflow(self) -> None:
        """Evict oldest LOCAL entries of past episodes first (then GLOBAL ones);
        never evict unresolved entries or current-episode entries."""
        overflow = len(self._entries) - self._max_entries
        if overflow <= 0:
            return
        for visibility in (Visibility.LOCAL, Visibility.GLOBAL):
            candidates = sorted(
                (
                    e
                    for e in self._entries
                    if e.visibility == visibility
                    and e.episode_id != self._current_episode
                    and e.resolved
                ),
                key=lambda e: e.timestamp,
            )
            for entry in candidates[:overflow]:
                self._entries.remove(entry)
            overflow = len(self._entries) - self._max_entries
            if overflow <= 0:
                return

    def begin_episode(self, episode_id: str) -> list[WorkspaceEntry]:
        """Set current episode. Carry over unresolved CONFLICT/REFLECTION entries
        from prior episodes: re-tag entry.episode_id, set metadata['carryover_from'],
        decay urgency *= 0.5. Returns carried entries. Does NOT re-broadcast (no
        duplicate SSE events)."""
        self._current_episode = episode_id
        self._current_tick = 0
        carried: list[WorkspaceEntry] = []
        for entry in self._entries:
            if entry.episode_id == episode_id or entry.resolved:
                continue
            if entry.type not in (EntryType.CONFLICT, EntryType.REFLECTION):
                continue
            entry.metadata["carryover_from"] = entry.episode_id
            entry.episode_id = episode_id
            entry.urgency *= 0.5
            carried.append(entry)
        return carried

    def view(self, episode_id: str | None = None) -> list[WorkspaceEntry]:
        """Entries of the given episode (default: current), including carryover."""
        target = self._current_episode if episode_id is None else episode_id
        return [e for e in self._entries if e.episode_id == target]

    def unappraised(self, episode_id: str | None = None) -> list[WorkspaceEntry]:
        return [e for e in self.view(episode_id) if not e.appraised]

    def unresolved_conflicts(self, episode_id: str | None = None) -> list[WorkspaceEntry]:
        return [
            e for e in self.view(episode_id) if e.type == EntryType.CONFLICT and not e.resolved
        ]

    def unattended_in_episode(
        self, episode_id: str | None = None, limit: int = 40
    ) -> list[WorkspaceEntry]:
        entries = [e for e in self.view(episode_id) if not e.attended]
        return sorted(entries, key=lambda e: -e.timestamp)[:limit]

    def broadcast(self, entry: WorkspaceEntry) -> None:
        entry.visibility = Visibility.GLOBAL
        entry.attended = True
        entry.broadcast_count += 1
        for handler in self._subscribers:
            handler(entry)

    def broadcast_selected(self, entries: list[WorkspaceEntry]) -> None:
        for entry in entries:
            self.broadcast(entry)

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
    def global_entries(self) -> list[WorkspaceEntry]:
        return [e for e in self._entries if e.visibility == Visibility.GLOBAL]

    @property
    def local_entries(self) -> list[WorkspaceEntry]:
        return [e for e in self._entries if e.visibility == Visibility.LOCAL]

    @property
    def size(self) -> int:
        return len(self._entries)

    def format_context(self, limit: int = 20) -> str:
        return "\n".join(
            f"[{e.timestamp:.1f}] {e.source} ({e.type.value}): {e.content[:200]}"
            for e in self.read(limit=limit)
        )

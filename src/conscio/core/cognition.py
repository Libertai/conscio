from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from conscio.core.workspace import EntryType, Workspace, WorkspaceEntry


@dataclass
class CognitiveTraceEvent:
    event: str
    source: str
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)


class CognitiveTrace:
    """Mechanistic trace of the control architecture, separate from self-report."""

    def __init__(self) -> None:
        self._events: list[CognitiveTraceEvent] = []

    def record(self, event: str, source: str, **data) -> None:
        self._events.append(CognitiveTraceEvent(event=event, source=source, data=data))

    @property
    def events(self) -> list[CognitiveTraceEvent]:
        return list(self._events)

    def format(self, limit: int = 30) -> str:
        lines: list[str] = []
        for event in self._events[-limit:]:
            details = ", ".join(f"{k}={v}" for k, v in event.data.items())
            suffix = f" ({details})" if details else ""
            lines.append(f"[{event.timestamp:.1f}] {event.source}: {event.event}{suffix}")
        return "\n".join(lines)


@dataclass
class SelfState:
    """Small, explicit self-model used by attention and action selection."""

    active_goal: str = ""
    uncertainty: float = 0.5
    conflict_level: float = 0.0
    cognitive_load: float = 0.0
    current_strategy: str = "observe-reflect-plan-act-review"
    last_error: str | None = None
    known_limitations: list[str] = field(default_factory=list)

    def update_from_confidence(self, confidence: str) -> None:
        mapping = {"LOW": 0.85, "MEDIUM": 0.5, "HIGH": 0.15}
        self.uncertainty = mapping.get(confidence.upper(), self.uncertainty)

    def to_workspace_content(self) -> str:
        return (
            f"goal={self.active_goal or 'none'}; "
            f"uncertainty={self.uncertainty:.2f}; "
            f"conflict={self.conflict_level:.2f}; "
            f"load={self.cognitive_load:.2f}; "
            f"strategy={self.current_strategy}"
        )


class CognitiveModule(Protocol):
    name: str

    async def tick(self, workspace: Workspace, state: SelfState) -> list[WorkspaceEntry]:
        ...


class AttentionController:
    """Selects entries for global broadcast using consciousness-inspired salience."""

    def __init__(self, broadcast_limit: int = 3) -> None:
        self.broadcast_limit = broadcast_limit

    def score(self, entry: WorkspaceEntry, state: SelfState) -> float:
        conflict_bonus = 0.2 if entry.type == EntryType.CONFLICT else 0.0
        uncertainty_bonus = state.uncertainty * 0.15
        return (
            entry.novelty * 0.25
            + entry.salience * 0.25
            + entry.urgency * 0.20
            + entry.confidence * 0.10
            + min(1.0, entry.priority / 10) * 0.10
            + conflict_bonus
            + uncertainty_bonus
        )

    def select(
        self,
        entries: list[WorkspaceEntry],
        state: SelfState,
        limit: int | None = None,
    ) -> list[WorkspaceEntry]:
        ranked = sorted(entries, key=lambda e: self.score(e, state), reverse=True)
        return ranked[: limit or self.broadcast_limit]

    def attend(self, workspace: Workspace, state: SelfState, trace: CognitiveTrace) -> list[WorkspaceEntry]:
        selected = self.select(workspace.unattended(), state)
        if selected:
            workspace.broadcast_selected(selected)
            trace.record(
                "attention_selected",
                "attention",
                selected=[f"{e.source}:{e.type.value}" for e in selected],
            )
        return selected


class ConflictMonitor:
    """Detects simple constraint and execution conflicts that should reach attention."""

    _ONE_WORD_RE = re.compile(r"\b(one word|single word|one-word)\b", re.I)

    def inspect_plan(self, user_input: str, plan_text: str, workspace: Workspace) -> list[WorkspaceEntry]:
        conflicts: list[WorkspaceEntry] = []
        if self._ONE_WORD_RE.search(user_input):
            args_match = re.search(r"args:\s*(.*)", plan_text, re.I | re.S)
            proposed = args_match.group(1).strip() if args_match else plan_text.strip()
            if len(proposed.split()) > 1:
                conflicts.append(
                    workspace.write(
                        "Plan may violate one-word response constraint.",
                        source="conflict_monitor",
                        type=EntryType.CONFLICT,
                        priority=9,
                        salience=0.95,
                        confidence=0.8,
                        novelty=0.7,
                        urgency=0.9,
                        evidence=[proposed[:200]],
                    )
                )
        return conflicts

    def inspect_tool_results(
        self,
        tool_results: list[dict],
        workspace: Workspace,
    ) -> list[WorkspaceEntry]:
        conflicts: list[WorkspaceEntry] = []
        for result in tool_results:
            output = str(result.get("output", ""))
            if "Error:" in output or "disabled" in output or result.get("error"):
                conflicts.append(
                    workspace.write(
                        f"Tool result requires attention: {output[:200]}",
                        source="conflict_monitor",
                        type=EntryType.CONFLICT,
                        priority=8,
                        salience=0.9,
                        confidence=0.9,
                        novelty=0.6,
                        urgency=0.8,
                        evidence=[result.get("tool", "unknown")],
                    )
                )
        return conflicts


@dataclass
class ActionDecision:
    action: str
    reason: str


class ActionSelector:
    """Chooses whether to act, reflect, verify, or ask based on self-state."""

    def decide(self, state: SelfState, confidence: str, has_conflict: bool) -> ActionDecision:
        if has_conflict:
            return ActionDecision("reflect", "conflict detected in workspace")
        if confidence == "LOW" or state.uncertainty >= 0.75:
            return ActionDecision("reflect", "low confidence or high uncertainty")
        return ActionDecision("act", "confidence and conflict levels allow action")

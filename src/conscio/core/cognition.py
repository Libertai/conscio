from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol

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
    attention_focus: str = ""
    current_intention: str = ""
    prediction_error: float = 0.0
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
            f"focus={self.attention_focus or 'none'}; "
            f"intention={self.current_intention or 'none'}; "
            f"prediction_error={self.prediction_error:.2f}; "
            f"strategy={self.current_strategy}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_goal": self.active_goal,
            "uncertainty": self.uncertainty,
            "conflict_level": self.conflict_level,
            "cognitive_load": self.cognitive_load,
            "current_strategy": self.current_strategy,
            "last_error": self.last_error,
            "attention_focus": self.attention_focus,
            "current_intention": self.current_intention,
            "prediction_error": self.prediction_error,
            "known_limitations": list(self.known_limitations),
        }


@dataclass
class AttentionSchema:
    """Simplified model of the system's own attention."""

    focus: str = ""
    focus_strength: float = 0.0
    reason_for_focus: str = ""
    ignored: list[str] = field(default_factory=list)
    interruptors: list[str] = field(default_factory=list)

    def update(self, selected: list[WorkspaceEntry], ignored: list[WorkspaceEntry]) -> None:
        if not selected:
            self.focus = ""
            self.focus_strength = 0.0
            self.reason_for_focus = "no candidates selected"
            self.ignored = [f"{e.source}:{e.type.value}" for e in ignored[:5]]
            return
        top = selected[0]
        self.focus = f"{top.source}:{top.type.value}"
        self.focus_strength = max(top.salience, top.urgency, min(1.0, top.priority / 10))
        reasons = []
        if top.type == EntryType.CONFLICT:
            reasons.append("conflict")
        if top.urgency >= 0.7:
            reasons.append("urgency")
        if top.novelty >= 0.7:
            reasons.append("novelty")
        if top.salience >= 0.7:
            reasons.append("salience")
        self.reason_for_focus = "+".join(reasons) or "ranked attention score"
        self.ignored = [f"{e.source}:{e.type.value}" for e in ignored[:5]]
        self.interruptors = [
            f"{e.source}:{e.type.value}"
            for e in ignored
            if e.urgency >= 0.8 or e.type == EntryType.CONFLICT
        ][:5]

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus": self.focus,
            "focus_strength": self.focus_strength,
            "reason_for_focus": self.reason_for_focus,
            "ignored": list(self.ignored),
            "interruptors": list(self.interruptors),
        }


@dataclass
class InputEvent:
    content: str
    source: str = "user"
    event_type: str = "message"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionKind(str, Enum):
    ANSWER = "answer"
    TOOL = "tool"
    ASK = "ask"
    REFLECT = "reflect"
    REFUSE = "refuse"
    WAIT = "wait"
    STOP = "stop"


PredicateKind = Literal[
    "answer_delivered",
    "tool_succeeded",
    "tool_output_contains",
    "task_status",
    "goal_proposed",
    "none",
]


@dataclass
class PredictionPredicate:
    """Typed expectation about the observation an Intention will produce.

    Replaces the legacy free-text expected_observation string, which produced
    spurious word-overlap mismatches on most episodes.
    """

    kind: PredicateKind = "none"
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Intention:
    kind: ActionKind
    content: str
    source: str
    confidence: float = 0.5
    expected_observation: PredictionPredicate | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    risk: float = 0.0
    urgency: float = 0.0
    expected_value: float = 0.5


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

    def attend(
        self,
        workspace: Workspace,
        state: SelfState,
        trace: CognitiveTrace,
        schema: AttentionSchema | None = None,
    ) -> list[WorkspaceEntry]:
        candidates = workspace.unattended()
        selected = self.select(candidates, state)
        if selected:
            workspace.broadcast_selected(selected)
            state.attention_focus = f"{selected[0].source}:{selected[0].type.value}"
            trace.record(
                "attention_selected",
                "attention",
                selected=[f"{e.source}:{e.type.value}" for e in selected],
            )
        if schema is not None:
            ignored = [e for e in candidates if e not in selected]
            schema.update(selected, ignored)
        return selected


class AppraisalSystem:
    """Computes salience variables used by attention and action selection."""

    def appraise(
        self,
        content: str,
        *,
        source: str,
        type: EntryType,
        goal: str = "",
    ) -> dict[str, float]:
        lower = content.lower()
        conflict_terms = ("error", "conflict", "contradict", "unsafe", "failed")
        urgent_terms = ("now", "urgent", "must", "immediately", "deadline")
        goal_terms = set(goal.lower().split())
        content_terms = set(lower.split())
        goal_overlap = len(goal_terms & content_terms) / max(1, len(goal_terms))
        conflict = 1.0 if any(term in lower for term in conflict_terms) else 0.0
        urgency = 0.8 if any(term in lower for term in urgent_terms) else conflict * 0.7
        novelty = 0.7 if source in {"user", "tool", "environment"} else 0.35
        salience = max(goal_overlap, conflict, urgency, 0.4 if type == EntryType.OBSERVATION else 0.2)
        return {
            "salience": min(1.0, salience),
            "novelty": min(1.0, novelty),
            "urgency": min(1.0, urgency),
            "risk": min(1.0, conflict),
        }


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

    def select_intention(
        self,
        intentions: list[Intention],
        state: SelfState,
    ) -> Intention:
        if not intentions:
            return Intention(
                kind=ActionKind.WAIT,
                content="No actionable intention selected.",
                source="action_selector",
                confidence=0.2,
            )
        if state.conflict_level >= 0.7 or state.prediction_error >= 0.7:
            reflective = [i for i in intentions if i.kind == ActionKind.REFLECT]
            if reflective:
                return max(reflective, key=lambda i: i.confidence)
        def score(i: Intention) -> float:
            return (
                i.confidence * 0.35
                + i.expected_value * 0.25
                + i.urgency * 0.20
                - i.risk * 0.20
                - state.uncertainty * 0.10
            )
        return max(intentions, key=score)


class PredictionEngine:
    """Evaluates structural prediction predicates and emits a CONFLICT entry on failure."""

    def evaluate(
        self,
        intention: Intention,
        observed: str,
        workspace: Workspace,
    ) -> WorkspaceEntry | None:
        predicate = intention.expected_observation
        if predicate is None or predicate.kind == "none":
            return None
        passed = self._check(predicate, observed, intention, workspace)
        if passed:
            return None
        summary = f"observed '{observed[:160]}'"
        return workspace.write(
            f"Prediction failed for {intention.kind.value} ({predicate.kind}): {summary}.",
            source="prediction_engine",
            type=EntryType.CONFLICT,
            priority=8,
            salience=0.9,
            confidence=0.8,
            novelty=0.8,
            urgency=0.7,
            metadata={"prediction_error": 1.0, "predicate": predicate.kind},
        )

    def _check(
        self,
        predicate: PredictionPredicate,
        observed: str,
        intention: Intention,
        workspace: Workspace,
    ) -> bool:
        kind = predicate.kind
        if kind == "answer_delivered":
            return bool(observed and observed.strip())
        if kind == "tool_output_contains":
            needle = str(predicate.args.get("needle", "")).lower()
            return bool(needle) and needle in observed.lower()
        if kind == "tool_succeeded":
            tool_name = str(predicate.args.get("tool", intention.tool_name or ""))
            for entry in reversed(workspace.read(limit=100, type_filter={EntryType.OBSERVATION})):
                if entry.source != "tool":
                    continue
                if tool_name and entry.metadata.get("tool") != tool_name:
                    continue
                result = entry.metadata.get("result")
                if not isinstance(result, dict):
                    continue
                error = bool(result.get("error"))
                exit_code = result.get("exit_code", 0)
                return not error and (exit_code in (0, None))
            return False
        if kind == "goal_proposed":
            return "propose_subgoal" in observed.lower() or "goal" in observed.lower()
        if kind == "task_status":
            target = str(predicate.args.get("status", "")).lower()
            return target in observed.lower() if target else True
        return True

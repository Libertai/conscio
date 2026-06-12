from __future__ import annotations

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
    """Small, explicit self-model used by attention and action selection.

    Every field is live: it has a documented writer → reader pair, so the
    self-report surface can be audited against real signals.

    | Field | Writer | Reader |
    |---|---|---|
    | ``active_goal`` | service (``start``/``_plan_and_act``) | AppraisalSystem goal-overlap, prompt CURRENT_STATE |
    | ``uncertainty`` | ``update_tick()`` each tick: ``0.45*prediction_error_ema + 0.35*tool_failure_rate + 0.20*(1 - attention_dispersion)``, blended as an EMA | ``AttentionController.score`` (uncertainty bonus), ActionSelector |
    | ``conflict_level`` | ``update_tick()``: ``min(1, 0.5*fresh + 0.25*unresolved)``; decays ``×0.5`` at episode start instead of reset-to-0 | ActionSelector reflect path |
    | ``cognitive_load`` | ``update_load(used_chars, budget)`` after each prompt assembly (context budget fraction) | AttentionController (raises min-score cutoff when > 0.8), self_state dict, prompt |
    | ``prediction_error`` | PredictionEngine EMA on each resolution (set via ``update_tick``) | ActionSelector, attention conflict bonus |
    | ``attention_focus`` | ``AttentionController.attend`` | prompt, api |
    | ``current_intention`` | ActionSelector | prompt, api |
    | ``current_strategy`` | ActionSelector — the per-tick TickDecision name (``"step"/"answer"/"reflect"/...``); no longer static | prompt, api |
    | ``last_error`` | executor on tool/LLM exception | prompt, api |
    | ``known_limitations`` | ``note_tool_failure(tool, err)``: appended when the same tool fails ≥3 times in a session (deduped, capped 8) | prompt CURRENT_STATE, self_state dict |
    | ``tool_failures`` | PredictionEngine resolutions (executor calls ``note_tool_failure`` on failed tool expectations) | ``note_tool_failure`` limitation threshold |
    """

    active_goal: str = ""
    uncertainty: float = 0.5
    conflict_level: float = 0.0
    cognitive_load: float = 0.0
    current_strategy: str = ""
    last_error: str | None = None
    attention_focus: str = ""
    current_intention: str = ""
    prediction_error: float = 0.0
    known_limitations: list[str] = field(default_factory=list)
    tool_failures: dict[str, int] = field(default_factory=dict)

    def update_tick(
        self,
        prediction_error: float,
        tool_failure_rate: float,
        attention_dispersion: float,
        unresolved_conflicts: int = 0,
        fresh_failures: int = 0,
    ) -> None:
        """Per-tick self-state update from real signals (see the writer table)."""
        self.prediction_error = _clamp(prediction_error)
        target = (
            0.45 * _clamp(prediction_error)
            + 0.35 * _clamp(tool_failure_rate)
            + 0.20 * (1.0 - _clamp(attention_dispersion))
        )
        self.uncertainty = _clamp(0.5 * self.uncertainty + 0.5 * target)
        self.conflict_level = min(
            1.0, 0.5 * max(0, fresh_failures) + 0.25 * max(0, unresolved_conflicts)
        )

    def update_load(self, used_chars: int, budget: int) -> None:
        """Cognitive load = fraction of the context budget consumed by the last prompt."""
        self.cognitive_load = _clamp(used_chars / max(1, budget))

    def note_tool_failure(self, tool: str, error: str) -> None:
        """Count a tool failure; promote to a known limitation after 3 failures."""
        count = self.tool_failures.get(tool, 0) + 1
        self.tool_failures[tool] = count
        if count < 3:
            return
        prefix = f"tool {tool} failing repeatedly"
        if any(limitation.startswith(prefix) for limitation in self.known_limitations):
            return
        detail = " ".join(str(error).split())[:120]
        self.known_limitations.append(f"{prefix}: {detail}" if detail else prefix)
        if len(self.known_limitations) > 8:
            del self.known_limitations[: len(self.known_limitations) - 8]

    def to_workspace_content(self) -> str:
        return (
            f"goal={self.active_goal or 'none'}; "
            f"uncertainty={self.uncertainty:.2f}; "
            f"conflict={self.conflict_level:.2f}; "
            f"load={self.cognitive_load:.2f}; "
            f"focus={self.attention_focus or 'none'}; "
            f"intention={self.current_intention or 'none'}; "
            f"prediction_error={self.prediction_error:.2f}; "
            f"strategy={self.current_strategy or 'none'}"
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
            "tool_failures": dict(self.tool_failures),
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


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
    STEP = "step"


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

    Legacy (v1) shape kept for the interim runtime loop; the v2 engine in
    ``core/prediction.py`` forms :class:`~conscio.core.prediction.Expectation`
    objects *before* execution instead.
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


@dataclass
class AttentionSelection:
    """Result of one attention pass.

    ``dispersion`` is the normalized spread of candidate scores
    (``(max - min) / max``): ~0 when candidates are indistinguishable (no
    clear winner → more uncertainty), ~1 when one candidate dominates.
    Feeds ``SelfState.update_tick``.
    """

    selected: list[WorkspaceEntry]
    ignored: list[WorkspaceEntry]
    scores: dict[int, float]  # id(entry) -> score
    dispersion: float


class AttentionController:
    """Selects entries for global broadcast using consciousness-inspired salience.

    v2: the selection *gates the model context* — winners become the WORKSPACE
    section of the prompt — so selection is budgeted (``broadcast_limit``
    entries AND ``char_budget`` content chars) instead of a bare top-k. The
    user-input entry is always force-included so stale attention pressure can
    never gate out the current request. ``coupling=False``
    (ablation ``self_state_coupling`` off) drops the SelfState terms from
    scoring and the high-load cutoff.
    """

    HIGH_LOAD_MIN_SCORE = 0.35  # min-score cutoff applied when cognitive_load > 0.8

    def __init__(
        self,
        broadcast_limit: int = 6,
        char_budget: int = 4000,
        coupling: bool = True,
    ) -> None:
        self.broadcast_limit = max(1, int(broadcast_limit))
        self.char_budget = max(1, int(char_budget))
        self.coupling = coupling

    def score(self, entry: WorkspaceEntry, state: SelfState) -> float:
        conflict_bonus = 0.2 if entry.type == EntryType.CONFLICT else 0.0
        base = (
            entry.novelty * 0.25
            + entry.salience * 0.25
            + entry.urgency * 0.20
            + entry.confidence * 0.10
            + min(1.0, entry.priority / 10) * 0.10
            + conflict_bonus
        )
        if not self.coupling:
            return base
        return base + state.uncertainty * 0.15

    def attend(
        self,
        workspace: Workspace,
        state: SelfState,
        trace: CognitiveTrace,
        schema: AttentionSchema | None = None,
        *,
        episode_id: str | None = None,
        tick: int = -1,
    ) -> AttentionSelection:
        candidates = workspace.unattended_in_episode(episode_id)
        scores = {id(entry): self.score(entry, state) for entry in candidates}
        ranked = sorted(candidates, key=lambda e: scores[id(e)], reverse=True)
        min_score = (
            self.HIGH_LOAD_MIN_SCORE
            if self.coupling and state.cognitive_load > 0.8
            else 0.0
        )
        selected: list[WorkspaceEntry] = []
        used_chars = 0
        for entry in ranked:
            if _is_event_input(entry):
                # Force-include the triggering input: never gated out by
                # budget, limit, or the high-load cutoff.
                selected.append(entry)
                used_chars += len(entry.content)
                continue
            if len(selected) >= self.broadcast_limit:
                continue
            if used_chars + len(entry.content) > self.char_budget:
                continue
            if scores[id(entry)] < min_score:
                continue
            selected.append(entry)
            used_chars += len(entry.content)
        selected.sort(key=lambda e: scores[id(e)], reverse=True)
        selected_ids = {id(entry) for entry in selected}
        ignored = [entry for entry in candidates if id(entry) not in selected_ids]
        if selected:
            workspace.broadcast_selected(selected)
            state.attention_focus = f"{selected[0].source}:{selected[0].type.value}"
            trace.record(
                "attention_selected",
                "attention",
                selected=[f"{e.source}:{e.type.value}" for e in selected],
                tick=tick,
            )
        if schema is not None:
            schema.update(selected, ignored)
        return AttentionSelection(
            selected=selected,
            ignored=ignored,
            scores=scores,
            dispersion=_dispersion(list(scores.values())),
        )


def _is_event_input(entry: WorkspaceEntry) -> bool:
    """The raw ingested event entry (user message / heartbeat) for this episode."""
    return entry.source == "input" and entry.type == EntryType.OBSERVATION


def _dispersion(scores: list[float]) -> float:
    """Normalized spread of candidate scores: (max - min) / max, in [0, 1]."""
    if len(scores) < 2:
        return 0.0
    high = max(scores)
    low = min(scores)
    if high <= 0:
        return 0.0
    return _clamp((high - low) / high)


class AppraisalSystem:
    """Computes salience variables used by attention and action selection.

    v2: appraisal is a centralized per-tick phase — modules write plain
    evidence entries and :meth:`appraise_entries` stamps the unappraised ones.
    With ``enabled=False`` (ablation ``appraisal`` off) neutral 0.5 constants
    are returned. The flag-gated batched LLM appraisal pass (``llm_appraisal``)
    is wired by the runtime tick loop and reuses these heuristics as fallback.
    """

    NEUTRAL = {"salience": 0.5, "novelty": 0.5, "urgency": 0.5, "risk": 0.5}

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def appraise(
        self,
        content: str,
        *,
        source: str,
        type: EntryType,
        goal: str = "",
    ) -> dict[str, float]:
        if not self.enabled:
            return dict(self.NEUTRAL)
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

    def appraise_entries(
        self,
        entries: list[WorkspaceEntry],
        state: SelfState,
        recent: list[WorkspaceEntry] | None = None,
    ) -> list[WorkspaceEntry]:
        """Stamp appraisal variables on unappraised entries (phase 2 of a tick).

        Scores only ever raise a writer's explicit floor (``max`` blend), and
        novelty is damped when the same content already appeared in ``recent``.
        """
        appraised: list[WorkspaceEntry] = []
        seen = {entry.content for entry in (recent or [])}
        for entry in entries:
            if entry.appraised:
                continue
            scores = self.appraise(
                entry.content,
                source=entry.source,
                type=entry.type,
                goal=state.active_goal,
            )
            novelty = scores["novelty"] * (0.3 if entry.content in seen else 1.0)
            entry.salience = max(entry.salience, scores["salience"])
            entry.novelty = max(entry.novelty, novelty)
            entry.urgency = max(entry.urgency, scores["urgency"])
            entry.appraised = True
            appraised.append(entry)
        return appraised


@dataclass
class TickDecision:
    kind: ActionKind
    reason: str


class ActionSelector:
    """Per-tick arbitration: STEP / ANSWER / ASK / REFLECT / REFUSE / WAIT."""

    def decide_tick(
        self,
        *,
        state: SelfState,
        last_step: Any = None,
        pending_answer: str | None = None,
        report: Any = None,
        reflections_done: int = 0,
        max_reflections: int = 2,
        ablation: Any = None,
        fresh_failure: bool = False,
        session_live: bool = True,
    ) -> TickDecision:
        """Decide what this tick resolves to.

        Thresholds (retuned so a single fresh prediction failure reaches
        reflect): control step → ASK/REFUSE; pending answer + report passed →
        ANSWER; pending answer + violations with reflection budget left →
        REFLECT, else ANSWER (violation logged in the result); fresh
        prediction failure or ``conflict_level >= 0.5`` with budget left →
        REFLECT; session live → STEP; else WAIT.
        """
        reflection_enabled = getattr(ablation, "reflection", True) if ablation is not None else True
        budget_left = reflections_done < max_reflections
        decision = self._decide(
            state,
            last_step,
            pending_answer,
            report,
            reflection_enabled and budget_left,
            fresh_failure,
            session_live,
        )
        state.current_strategy = decision.kind.value
        state.current_intention = decision.kind.value
        return decision

    def _decide(
        self,
        state: SelfState,
        last_step: Any,
        pending_answer: str | None,
        report: Any,
        can_reflect: bool,
        fresh_failure: bool,
        session_live: bool,
    ) -> TickDecision:
        if last_step is not None and getattr(last_step, "kind", "") == "control":
            control = getattr(last_step, "control", "")
            kind = ActionKind.ASK if control == "ask" else ActionKind.REFUSE
            return TickDecision(kind, f"model invoked control tool ({control})")
        if pending_answer is not None:
            if report is None or report.passed:
                return TickDecision(ActionKind.ANSWER, "answer satisfies active constraints")
            if can_reflect:
                return TickDecision(ActionKind.REFLECT, "answer violates constraints; revising")
            return TickDecision(
                ActionKind.ANSWER,
                "constraint violation, reflection unavailable; answering with violation logged",
            )
        if (fresh_failure or state.conflict_level >= 0.5) and can_reflect:
            return TickDecision(
                ActionKind.REFLECT,
                "fresh prediction failure" if fresh_failure else "conflict level high",
            )
        if session_live:
            return TickDecision(ActionKind.STEP, "session live; continue working")
        return TickDecision(ActionKind.WAIT, "nothing to do")

    def select_intention(
        self,
        intentions: list[Intention],
        state: SelfState,
    ) -> Intention:
        """Legacy candidate-intention arbitration used by the interim runtime loop."""
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


# The legacy (v1) post-hoc PredictionEngine is gone: the v2 engine in
# ``core/prediction.py`` forms expectations *before* execution and resolves
# them against real outcomes (tool result dicts, ConstraintReports).

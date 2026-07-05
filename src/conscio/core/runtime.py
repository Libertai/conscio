"""Cognitive runtime v2 — the per-tick control loop owns the agent.

v1 was "modules tick → one module secretly runs the whole agent → break".
v2 inverts this: ``run_episode`` runs an explicit SENSE → APPRAISE → ATTEND →
EXECUTE → VALIDATE → SELF-STATE → DECIDE loop each tick; the LLM/tool work is
an :class:`~conscio.core.executor.EpisodeExecutor` invoked *after* attention,
one bounded step at a time. One engine, two prompt strategies (chat /
autonomous).

Latency invariant: a simple chat message resolves in exactly ONE
``chat_async`` call (tick 1: sense/appraise/attend are pure Python + one FTS
query; the executor's first step returns a final answer; structural constraint
validation is deterministic; DECIDE answers and breaks). Pinned by
``tests/test_engine.py``.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from conscio.config import AblationFlags
from conscio.core.autonomy_module import AutonomousPromptAssembler
from conscio.core.cognition import (
    ActionKind,
    ActionSelector,
    AppraisalSystem,
    AttentionController,
    AttentionSchema,
    CognitiveModule,
    CognitiveTrace,
    InputEvent,
    Intention,
    SelfState,
    TickDecision,
)
from conscio.core.constraints import (
    ConstraintCheck,
    ConstraintReport,
    ConstraintValidator,
    ParsedConstraint,
)
from conscio.core.context import ContextSettings, PromptAssembler
from conscio.core.executor import AutonomousStrategy, ChatStrategy, EpisodeExecutor
from conscio.core.prediction import PredictionEngine
from conscio.core.tool_loop import StepResult
from conscio.core.workspace import EntryType, Workspace, WorkspaceEntry
from conscio.llm.structured import structured_json
from conscio.memory.consolidation import ConsolidationEngine
from conscio.memory.store import MemoryStore
from conscio.tools import ToolRegistry

EMPTY_RESPONSE_FALLBACK = (
    "I recorded your message, but my inference backend returned an empty response. "
    "I will retain the facts and continue."
)

INTERNAL_OBSERVATION_MESSAGE = (
    "Internal observation recorded; no user-facing response needed."
)


@dataclass
class EpisodeMetrics:
    ticks: int = 0
    duration: float = 0.0
    attention_selections: int = 0
    prediction_errors: int = 0
    tool_calls: int = 0
    global_broadcasts: int = 0
    # v2 additive fields:
    llm_calls: int = 0
    tool_rounds: int = 0
    reflections: int = 0
    constraint_violations: int = 0


@dataclass
class EpisodeResult:
    output: str
    selected_action: str
    session_id: str
    workspace_trace: str
    cognitive_trace: str
    self_state: dict[str, Any]
    attention_schema: dict[str, Any]
    metrics: EpisodeMetrics
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    model_context: str = ""
    episode_id: str = ""
    # v2 additive fields:
    tick_trace: list[dict[str, Any]] = field(default_factory=list)
    constraint_report: list[dict[str, Any]] = field(default_factory=list)
    outcome_reason: str = ""  # TickDecision.reason for the episode-ending decision


class PerceptionModule:
    """Surfaces unperceived input events as observation evidence.

    Does NOT mark the raw input entry attended — attention must still see it
    so the force-include guard (user input always reaches broadcast) holds.
    """

    name = "observer"

    async def tick(self, workspace: Workspace, state: SelfState) -> list[WorkspaceEntry]:
        entries = [
            e for e in workspace.view()
            if e.source == "input"
            and e.type == EntryType.OBSERVATION
            and not e.metadata.get("perceived")
        ]
        produced: list[WorkspaceEntry] = []
        for entry in entries:
            entry.metadata["perceived"] = True
            produced.append(
                workspace.write(
                    f"Perceived {entry.metadata.get('event_type', 'message')} "
                    f"from {entry.metadata.get('source', 'user')}: {entry.content}",
                    source=self.name,
                    type=EntryType.OBSERVATION,
                    priority=6,
                    salience=0.7,
                    confidence=0.9,
                    novelty=0.8,
                    urgency=entry.urgency,
                    evidence=[entry.content[:200]],
                )
            )
        return produced


class MemoryRetrievalModule:
    name = "memory"

    def __init__(self, memory: MemoryStore, session_id: str = "") -> None:
        # Reads the unified episodes table globally (no session_id filter):
        # prior-process episodes stay visible across restarts. session_id is
        # kept for constructor compatibility only.
        self.memory = memory
        self.session_id = session_id
        self._ran = False

    def reset(self) -> None:
        self._ran = False

    async def tick(self, workspace: Workspace, state: SelfState) -> list[WorkspaceEntry]:
        if self._ran:
            return []
        self._ran = True
        episodes = await self.memory.recent_episodes(3)
        produced: list[WorkspaceEntry] = []
        for episode in episodes:
            summary = episode.get("summary") or episode.get("input", "")
            produced.append(
                workspace.write(
                    f"Relevant recent episode: {summary}",
                    source=self.name,
                    type=EntryType.MEMORY,
                    priority=4,
                    salience=0.45,
                    confidence=0.65,
                    novelty=0.25,
                    evidence=[episode.get("output", "")],
                )
            )
        return produced


class ReflectionModule:
    """Surfaces unresolved conflicts of the current episode (including
    carryover from prior episodes) as REFLECTION candidates referencing the
    conflict's expectation id. Episode-scoped via the workspace view — the v1
    ``episode_start`` timestamp filter is gone."""

    name = "reflector"

    def __init__(self) -> None:
        self._surfaced: set[int] = set()

    def reset(self) -> None:
        self._surfaced.clear()

    async def tick(self, workspace: Workspace, state: SelfState) -> list[WorkspaceEntry]:
        produced: list[WorkspaceEntry] = []
        for conflict in workspace.unresolved_conflicts():
            key = id(conflict)
            if key in self._surfaced:
                continue
            self._surfaced.add(key)
            produced.append(
                workspace.write(
                    f"Reflect on conflict before acting: {conflict.content}",
                    source=self.name,
                    type=EntryType.REFLECTION,
                    priority=7,
                    salience=0.75,
                    confidence=0.75,
                    novelty=0.5,
                    urgency=conflict.urgency,
                    evidence=[conflict.content],
                    metadata={
                        "conflict_expectation_id": conflict.metadata.get("expectation_id", ""),
                        "carryover_from": conflict.metadata.get("carryover_from", ""),
                    },
                )
            )
        return produced


class MemoryConsolidator:
    """Thin adapter over memory.consolidation.ConsolidationEngine.

    Per-episode cheap path only: writes the unified episodes row (no LLM,
    no junk skills, no compaction facts). Periodic budgeted consolidation
    runs via ConsolidationEngine.consolidate_cycle on the service cadence.
    """

    name = "consolidator"

    def __init__(self, settings: ContextSettings | None = None) -> None:
        self.settings = settings or ContextSettings()

    async def consolidate(
        self,
        memory: MemoryStore,
        session_id: str,
        event: InputEvent,
        result: EpisodeResult,
    ) -> list[str]:
        engine = ConsolidationEngine(memory)
        episode_id = result.episode_id or uuid.uuid4().hex
        await engine.record_episode(
            episode_id=episode_id,
            source=event.source,
            event_type=event.event_type,
            input_text=event.content,
            output=result.output,
            selected_action=result.selected_action,
            metrics=asdict(result.metrics),
        )
        return [episode_id]


_LLM_APPRAISAL_SYSTEM_PROMPT = (
    "You score workspace entries for a cognitive architecture. For each entry "
    "return salience, novelty and urgency in [0,1]. Respond with ONLY a JSON "
    'array: [{"index": 0, "salience": 0.5, "novelty": 0.5, "urgency": 0.5}].'
)

_APPRAISAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "salience": {"type": "number"},
                    "novelty": {"type": "number"},
                    "urgency": {"type": "number"},
                },
                "required": ["index"],
            },
        }
    },
    "required": ["scores"],
}

# Event sources that are pure observations: ingested into the workspace but
# never executed against the LLM (no user to answer, no heartbeat to act on).
_OBSERVATION_ONLY_SOURCES = {"tool", "system"}


class CognitiveRuntime:
    """Event-driven consciousness-architecture harness (v2 tick loop)."""

    def __init__(
        self,
        *,
        llm: Any | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        session_id: str | None = None,
        modules: list[CognitiveModule] | None = None,
        max_ticks: int = 8,
        max_tool_rounds: int = 32,
        tool_rounds_per_tick: int = 4,
        max_reflections: int = 2,
        attention_broadcast_limit: int = 6,
        attention_char_budget: int = 4000,
        ablation: AblationFlags | None = None,
        constraint_provider: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
        context_settings: ContextSettings | None = None,
        context_provider: Any | None = None,
        llm_fast: Any | None = None,
        chat_temperature: float = 0.4,
        autonomous_temperature: float = 0.3,
        loop_max_tokens: int = 2400,
        judge_max_tokens: int = 200,
        appraisal_max_tokens: int = 400,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:16]
        self.ablation = ablation or AblationFlags()
        self.workspace = Workspace()
        self.trace = CognitiveTrace()
        self.self_state = SelfState()
        self.attention_schema = AttentionSchema()
        self.attention = AttentionController(
            broadcast_limit=attention_broadcast_limit,
            char_budget=attention_char_budget,
            coupling=self.ablation.self_state_coupling,
        )
        self.appraisal = AppraisalSystem(enabled=self.ablation.appraisal)
        self.action_selector = ActionSelector()
        self.prediction = PredictionEngine(enabled=self.ablation.prediction)
        self.memory = memory or MemoryStore()
        self.tools = tools or ToolRegistry()
        if tools is None:
            self.tools.load_builtins()
        self.max_ticks = max(1, int(max_ticks))
        self.max_reflections = max(0, int(max_reflections))
        self.constraint_provider = constraint_provider
        self.context_settings = context_settings or ContextSettings()
        self.prompt_assembler = PromptAssembler(self.context_settings)
        self.last_model_context = ""
        self.llm_fast = llm_fast
        self._appraisal_max_tokens = appraisal_max_tokens
        self.validator = ConstraintValidator(
            llm=llm_fast or llm,
            judge_enabled=self.ablation.constraint_judge,
            judge_max_tokens=judge_max_tokens,
        )
        self.chat_strategy = ChatStrategy(
            assembler=self.prompt_assembler,
            context_provider=context_provider,
            llm=llm,
        )
        self.autonomous_assembler = AutonomousPromptAssembler(
            max_dynamic_chars=self.context_settings.max_dynamic_chars,
        )
        self.autonomous_strategy = AutonomousStrategy(
            assembler=self.autonomous_assembler,
            memory=self.memory,
            context_provider=context_provider,
            llm=llm,
        )
        self.chat_strategy.temperature = chat_temperature
        self.autonomous_strategy.temperature = autonomous_temperature
        self.executor = EpisodeExecutor(
            tools=self.tools,
            memory=self.memory,
            session_id=self.session_id,
            chat=self.chat_strategy,
            autonomous=self.autonomous_strategy,
            max_total_rounds=max_tool_rounds,
            rounds_per_tick=tool_rounds_per_tick,
            max_tokens=loop_max_tokens,
            prediction=self.prediction,
        )
        self.modules = modules or self._default_modules()
        self.consolidator = MemoryConsolidator(self.context_settings)

    @property
    def _autonomous_module(self) -> AutonomousStrategy:
        """Compat shim: external consumers (service.py, eval/legacy.py) poke
        ``.llm`` (settable), ``.last_tool_requests``, ``.context_provider``
        and ``.on_tool_observation`` — all live on the AutonomousStrategy."""
        return self.autonomous_strategy

    def _default_modules(self) -> list[CognitiveModule]:
        modules: list[CognitiveModule] = [PerceptionModule()]
        if self.ablation.memory_retrieval:
            modules.append(MemoryRetrievalModule(self.memory, self.session_id))
        modules.append(ReflectionModule())
        return modules

    async def initialize(self) -> None:
        await self.memory.initialize()
        await self.memory.create_session(self.session_id, name="conscio cognitive runtime")

    async def close(self) -> None:
        await self.memory.end_session(self.session_id)
        await self.memory.close()

    async def run_episode(
        self,
        event: InputEvent | str,
        *,
        should_yield: Callable[[], bool] | None = None,
    ) -> EpisodeResult:
        if isinstance(event, str):
            event = InputEvent(content=event)
        episode_id = uuid.uuid4().hex  # canonical episode id (memory provenance)
        carried = self.workspace.begin_episode(episode_id)
        self._reset_modules_for_episode()
        # conflict_level carries over via workspace CONFLICT entries (unresolved
        # conflicts are re-tagged to the new episode by begin_episode), not via
        # the self-state field — update_tick recomputes it from fresh signals.
        self.prediction.reset_episode()
        self.executor.reset()
        start = time.time()
        self.last_model_context = ""
        metrics = EpisodeMetrics()
        constraints = await self._fetch_constraints(event)  # once per episode
        self.trace.record(
            "episode_started", "runtime", event_source=event.source, carryover=len(carried)
        )
        self._ingest_event(event)

        executable = event.source not in _OBSERVATION_ONLY_SOURCES
        tick_trace: list[dict[str, Any]] = []
        final_report: ConstraintReport | None = None
        pending_answer: str | None = None
        answer_expectation = None
        reflections_done = 0
        empty_steps = 0
        extra_llm_calls = 0
        output = ""
        outcome: TickDecision | None = None
        prev_failures = 0
        prev_llm_calls = 0
        prev_tool_rounds = 0
        prev_state = self.self_state.to_dict()

        for tick in range(1, self.max_ticks + 1):
            # Cooperative preemption: an interactive event is waiting and this
            # (autonomous) episode already made at least one full tick of progress.
            if should_yield is not None and tick > 1 and should_yield():
                outcome = TickDecision(ActionKind.WAIT, "preempted by interactive event")
                self.trace.record("episode_preempted", "runtime", tick=tick)
                break
            metrics.ticks += 1
            self.workspace._current_tick = tick  # designed seam: runtime stamps the tick

            # 1 SENSE — modules produce LOCAL evidence entries (no scores).
            for module in self.modules:
                module_entries = await module.tick(self.workspace, self.self_state)
                if module_entries:
                    self.trace.record("module_tick", module.name, produced=len(module_entries))

            # 2 APPRAISE — centralized stamping of unappraised entries.
            recent = [e for e in self.workspace.view() if e.appraised]
            appraised = self.appraisal.appraise_entries(
                self.workspace.unappraised(), self.self_state, recent
            )
            if self.ablation.llm_appraisal and appraised and self.chat_strategy.llm is not None:
                if await self._llm_appraise(appraised):
                    extra_llm_calls += 1

            # 3 ATTEND — broadcast winners gate the model context.
            selection = self.attention.attend(
                self.workspace,
                self.self_state,
                self.trace,
                self.attention_schema,
                episode_id=episode_id,
                tick=tick,
            )
            metrics.attention_selections += len(selection.selected)

            # 4 EXECUTE — one bounded executor step after attention.
            step: StepResult | None = None
            forced_decision: TickDecision | None = None
            if executable and pending_answer is None and not self.executor.exhausted:
                broadcast_new = (
                    list(selection.selected) if self.ablation.attention_gating else None
                )
                step = await self.executor.step(
                    event=event,
                    workspace=self.workspace,
                    broadcast_new=broadcast_new,
                    state=self.self_state,
                    should_stop=should_yield,
                )
                if step.kind == "final":
                    pending_answer = step.text
                    answer_expectation = self.prediction.expect_answer(
                        constraints=constraints, tick=tick
                    )
                    self._record_intention(ActionKind.ANSWER, step.text)
                elif step.kind == "control":
                    self._record_intention(
                        ActionKind.ASK if step.control == "ask" else ActionKind.REFUSE,
                        step.text,
                    )
                elif step.kind == "empty":
                    if step.text:
                        # Deterministic offline path (autonomous, no LLM):
                        # nothing to execute — resolve to WAIT with the reason.
                        output = step.text
                        forced_decision = TickDecision(
                            ActionKind.WAIT, "no LLM configured; nothing to execute"
                        )
                    else:
                        # Empty LLM response: a recorded prediction failure;
                        # retry once next tick, then fall back to WAIT.
                        empty_steps += 1
                        self._record_empty_failure(constraints, tick)
                        if empty_steps >= 2:
                            output = EMPTY_RESPONSE_FALLBACK
                            forced_decision = TickDecision(
                                ActionKind.WAIT, "empty LLM response after retry"
                            )
                        else:
                            forced_decision = TickDecision(
                                ActionKind.STEP, "empty LLM response; retrying once"
                            )

            # 5 VALIDATE — constraint report + answer expectation resolution.
            report: ConstraintReport | None = None
            if pending_answer is not None:
                report = await self.validator.validate(pending_answer, constraints)
                final_report = report
                metrics.constraint_violations += len(report.violations)
                if answer_expectation is not None:
                    self.prediction.resolve_answer(
                        answer_expectation, report, self.workspace, tick
                    )
                    answer_expectation = None

            # 6 SELF-STATE — update from real signals.
            fresh_failures = self.prediction.episode_failures - prev_failures
            prev_failures = self.prediction.episode_failures
            unresolved = len(self.workspace.unresolved_conflicts())
            self.self_state.update_tick(
                self.prediction.error_ema,
                self.prediction.failure_rate(),
                selection.dispersion,
                unresolved_conflicts=unresolved,
                fresh_failures=fresh_failures,
            )
            if self.executor.last_model_context:
                self.self_state.update_load(
                    len(self.executor.last_model_context),
                    self.context_settings.max_dynamic_chars,
                )
            if fresh_failures:
                self.trace.record(
                    "prediction_error", "prediction_engine", error=self.self_state.prediction_error
                )

            # 7 DECIDE — per-tick arbitration.
            decision = forced_decision or self.action_selector.decide_tick(
                state=self.self_state,
                last_step=step,
                pending_answer=pending_answer,
                report=report,
                reflections_done=reflections_done,
                max_reflections=self.max_reflections,
                ablation=self.ablation,
                fresh_failure=fresh_failures > 0,
                session_live=executable and not self.executor.exhausted,
            )
            self.trace.record("tick_decision", "action_selector", kind=decision.kind.value, tick=tick)
            state_now = self.self_state.to_dict()
            tick_trace.append(
                {
                    "tick": tick,
                    "decision": decision.kind.value,
                    "reason": decision.reason,
                    "broadcast": [f"{e.source}:{e.type.value}" for e in selection.selected],
                    "llm_calls": self.executor.llm_calls - prev_llm_calls,
                    "tool_rounds": len(self.executor.tool_requests) - prev_tool_rounds,
                    "prediction_events": fresh_failures,
                    "self_state_delta": _state_delta(prev_state, state_now),
                }
            )
            prev_llm_calls = self.executor.llm_calls
            prev_tool_rounds = len(self.executor.tool_requests)
            prev_state = state_now

            if decision.kind in (ActionKind.ANSWER, ActionKind.ASK, ActionKind.REFUSE):
                outcome = decision
                if decision.kind == ActionKind.ANSWER and pending_answer is not None:
                    output = pending_answer
                elif step:
                    output = step.text
                break
            if decision.kind == ActionKind.REFLECT:
                self.executor.inject_reflection(self._reflection_text(report))
                for conflict in self.workspace.unresolved_conflicts():
                    conflict.resolved = True
                reflections_done += 1
                metrics.reflections += 1
                pending_answer = None  # discard the violating answer; revise next tick
                self.trace.record("reflection_injected", "runtime", count=reflections_done)
                continue
            if decision.kind == ActionKind.WAIT:
                outcome = decision
                break
            # STEP — keep working next tick.

        if outcome is None:
            outcome = TickDecision(ActionKind.WAIT, "tick budget exhausted")
        if outcome.kind == ActionKind.WAIT and not output:
            output = (
                INTERNAL_OBSERVATION_MESSAGE
                if event.source in {"autonomous", "tool", "system"}
                else "No stable intention emerged."
            )
        selected_action = outcome.kind.value

        metrics.duration = time.time() - start
        metrics.global_broadcasts = len(self.workspace.global_entries)
        metrics.prediction_errors = self.prediction.episode_failures
        metrics.llm_calls = self.executor.llm_calls + extra_llm_calls
        metrics.tool_calls = len(self.executor.tool_requests)
        metrics.tool_rounds = len(self.executor.tool_requests)
        self._capture_model_context()
        result = EpisodeResult(
            output=output,
            selected_action=selected_action,
            session_id=self.session_id,
            workspace_trace=self.workspace.format_context(limit=30),
            cognitive_trace=self.trace.format(limit=60),
            self_state=self.self_state.to_dict(),
            attention_schema=self.attention_schema.to_dict(),
            metrics=metrics,
            tool_results=list(self.executor.tool_results),
            model_context=self.last_model_context,
            episode_id=episode_id,
            tick_trace=tick_trace,
            constraint_report=final_report.to_dicts() if final_report is not None else [],
            outcome_reason=outcome.reason,
        )
        result.memory_ids = await self.consolidator.consolidate(
            self.memory, self.session_id, event, result
        )
        self.trace.record("episode_completed", "runtime", action=result.selected_action)
        return result

    async def run_daemon(
        self,
        events: list[InputEvent],
        *,
        dry_run: bool = True,
    ) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        for event in events:
            if dry_run:
                self.trace.record("daemon_dry_run_event", "runtime", event_source=event.source)
            results.append(await self.run_episode(event))
        return results

    async def _fetch_constraints(self, event: InputEvent) -> list[ParsedConstraint]:
        """Constraint fetch happens ONCE per episode: persistent rows from the
        provider (one sqlite read) merged with episode constraints extracted
        from the triggering input."""
        rows: list[dict[str, Any]] = []
        if self.constraint_provider is not None:
            try:
                rows = await self.constraint_provider()
            except Exception:  # noqa: BLE001 — constraints are best-effort context
                rows = []
        constraints = self.validator.parse(rows)
        constraints.extend(self.validator.extract_episode_constraints(event.content))
        return constraints

    def _ingest_event(self, event: InputEvent) -> None:
        """Write the raw event entry; appraisal happens in phase 2 like everything else."""
        self.workspace.write(
            event.content,
            source="input",
            type=EntryType.OBSERVATION,
            priority=7,
            confidence=0.95,
            urgency=0.5,
            metadata={"source": event.source, "event_type": event.event_type, **event.metadata},
        )

    def _record_intention(self, kind: ActionKind, content: str) -> None:
        intention = Intention(
            kind=kind,
            content=content,
            source="executor",
            confidence=max(0.0, min(1.0, 1.0 - self.self_state.uncertainty)),
        )
        self.workspace.write(
            f"Candidate {kind.value}: {content[:200]}",
            source="executor",
            type=EntryType.INTENTION,
            priority=7,
            salience=0.7,
            confidence=intention.confidence,
            novelty=0.55,
            urgency=0.35,
            metadata={"intention": intention},
        )
        self.self_state.current_intention = kind.value
        self.trace.record(
            "intention_selected", "action_selector", kind=kind.value, intention_source="executor"
        )

    def _record_empty_failure(self, constraints: list[ParsedConstraint], tick: int) -> None:
        """An empty LLM response is a failed answer expectation, not a masked success."""
        expectation = self.prediction.expect_answer(constraints=constraints, tick=tick)
        failing = ConstraintReport(
            checks=[
                ConstraintCheck(
                    "answer:nonempty",
                    "answer must be non-empty",
                    "structural",
                    False,
                    "empty LLM response",
                )
            ]
        )
        self.prediction.resolve_answer(expectation, failing, self.workspace, tick)

    def _reflection_text(self, report: ConstraintReport | None) -> str:
        lines = ["REFLECT: problems detected with the current approach:"]
        for conflict in self.workspace.unresolved_conflicts()[-3:]:
            lines.append(f"- {conflict.content}")
        if report is not None:
            for violation in report.violations:
                lines.append(
                    f"- constraint violated: {violation.text} ({violation.detail})"
                )
        lines.append(
            "Revise your approach so the next answer satisfies every active constraint, "
            "then respond again."
        )
        return "\n".join(lines)

    async def _llm_appraise(self, entries: list[WorkspaceEntry]) -> bool:
        """Flag-gated batched LLM appraisal pass: one scoring call per tick.
        Scores only ever raise the heuristic floors; failures fall back silently."""
        payload = [
            {"index": index, "content": entry.content[:300]}
            for index, entry in enumerate(entries)
        ]
        messages = [
            {"role": "system", "content": _LLM_APPRAISAL_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            llm = self.llm_fast or self.chat_strategy.llm
            data = await structured_json(
                llm,
                messages,
                schema=_APPRAISAL_SCHEMA,
                schema_name="appraisal_scores",
                max_tokens=self._appraisal_max_tokens,
            )
        except Exception:  # noqa: BLE001 — heuristics already stamped
            return False
        if isinstance(data, dict):
            data = data.get("scores")
        if not isinstance(data, list):
            return True
        items = data
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                entry = entries[int(item.get("index", -1))]
            except (ValueError, TypeError, IndexError):
                continue
            for key in ("salience", "novelty", "urgency"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    setattr(entry, key, max(getattr(entry, key), min(1.0, max(0.0, float(value)))))
        return True

    def _collect_intentions(self) -> list[Intention]:
        """Candidate intentions of the current episode (episode_id filter, not timestamps)."""
        intentions: list[Intention] = []
        for entry in self.workspace.view():
            if entry.type != EntryType.INTENTION:
                continue
            candidate = entry.metadata.get("intention")
            if isinstance(candidate, Intention):
                intentions.append(candidate)
        return intentions

    def _capture_model_context(self) -> None:
        context = self.executor.last_model_context
        if context:
            self.last_model_context = context

    def _reset_modules_for_episode(self) -> None:
        for module in self.modules:
            reset = getattr(module, "reset", None)
            if callable(reset):
                reset()


def _state_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Keys of the self-state dict whose values changed this tick."""
    return {key: value for key, value in after.items() if before.get(key) != value}

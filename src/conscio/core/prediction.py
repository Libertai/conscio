"""Prediction engine v2.

Expectations are formed *before* execution (``expect_tool`` runs in the
executor's pre-tool hook; ``expect_answer`` before an answer is accepted) and
resolved against ground truth: the tool's returned result dict for
``tool_succeeded`` and the :class:`ConstraintReport` for
``answer_satisfies_constraints`` — no post-hoc workspace scan, no
``bool(output)`` tautology. Failures write unresolved CONFLICT entries that
carry over across ticks and episodes; ``error_ema`` feeds SelfState.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from conscio.core.constraints import ConstraintReport, ParsedConstraint
from conscio.core.tool_loop import ToolRequest
from conscio.core.workspace import EntryType, Workspace, WorkspaceEntry

ExpectationKind = Literal[
    "tool_succeeded",
    "tool_output_contains",
    "answer_satisfies_constraints",
    "answer_nonempty",
    "task_status",
]


@dataclass
class Expectation:
    id: str
    kind: ExpectationKind
    args: dict[str, Any]
    intention_source: str
    created_tick: int
    resolved: bool = False
    passed: bool | None = None


class PredictionEngine:
    """Registers expectations pre-execution and resolves them against real outcomes.

    ``error_ema`` is an exponential moving average of resolution outcomes
    (1.0 = failure, 0.0 = success) updated on every resolution; it persists
    across episodes (``reset_episode`` clears pending expectations and the
    per-episode failure counters, but keeps the EMA).
    """

    def __init__(self, *, enabled: bool = True, ema_alpha: float = 0.35) -> None:
        self.enabled = enabled
        self.ema_alpha = ema_alpha
        self.error_ema: float = 0.0
        self._pending: list[Expectation] = []
        self._resolved_count = 0
        self._failure_count = 0

    def expect_tool(self, request: ToolRequest, tick: int) -> Expectation:
        """Form a tool_succeeded expectation BEFORE the tool runs."""
        expectation = Expectation(
            id=uuid.uuid4().hex[:12],
            kind="tool_succeeded",
            args={"tool": request.name, "tool_args": dict(request.args)},
            intention_source=f"tool:{request.name}",
            created_tick=tick,
        )
        if self.enabled:
            self._pending.append(expectation)
        return expectation

    def expect_answer(self, *, constraints: list[ParsedConstraint], tick: int) -> Expectation:
        """Form the answer expectation BEFORE the answer is accepted:
        the answer satisfies the active constraints (and is non-empty —
        enforced upstream: empty step results never become ANSWER intentions)."""
        expectation = Expectation(
            id=uuid.uuid4().hex[:12],
            kind="answer_satisfies_constraints",
            args={"constraints": [c.constraint_id for c in constraints]},
            intention_source="executor:answer",
            created_tick=tick,
        )
        if self.enabled:
            self._pending.append(expectation)
        return expectation

    def resolve_tool(
        self,
        exp: Expectation,
        result: dict,
        workspace: Workspace,
        tick: int,
    ) -> WorkspaceEntry | None:
        """Resolve a tool expectation against the returned result dict
        (``error``/``exit_code``). Returns a CONFLICT entry on failure."""
        if not self.enabled or exp.resolved:
            return None
        error = result.get("error")
        exit_code = result.get("exit_code", 0)
        if exp.kind == "tool_output_contains":
            needle = str(exp.args.get("needle", "")).lower()
            passed = bool(needle) and needle in str(result.get("output", "")).lower()
            detail = f"output does not contain {needle!r}"
        else:
            passed = not error and exit_code in (0, None)
            detail = f"error={error!r}" if error else f"exit_code={exit_code!r}"
        self._mark(exp, passed)
        if passed:
            return None
        tool = str(exp.args.get("tool", ""))
        return self._write_conflict(
            workspace,
            f"Prediction failed: expected tool '{tool}' to succeed; got {detail}.",
            exp,
            evidence=[str(result.get("output", ""))[:200]],
        )

    def resolve_answer(
        self,
        exp: Expectation,
        report: ConstraintReport,
        workspace: Workspace,
        tick: int,
    ) -> WorkspaceEntry | None:
        """Resolve the answer expectation against the ConstraintReport
        (not ``bool(output)``). Returns a CONFLICT entry on violation."""
        if not self.enabled or exp.resolved:
            return None
        passed = report.passed
        self._mark(exp, passed)
        if passed:
            return None
        violated = "; ".join(
            f"{check.constraint_id}: {check.text[:80]} ({check.detail})"
            for check in report.violations
        )
        return self._write_conflict(
            workspace,
            f"Prediction failed: answer violates constraints: {violated}.",
            exp,
            evidence=[check.constraint_id for check in report.violations],
        )

    def pending(self) -> list[Expectation]:
        return [exp for exp in self._pending if not exp.resolved]

    @property
    def episode_failures(self) -> int:
        """Resolved failures this episode (reset by ``reset_episode``)."""
        return self._failure_count

    def failure_rate(self) -> float:
        """Resolved failures / resolved, this episode."""
        if not self._resolved_count:
            return 0.0
        return self._failure_count / self._resolved_count

    def reset_episode(self) -> None:
        """Clear pending expectations and per-episode counters; keep the EMA."""
        self._pending.clear()
        self._resolved_count = 0
        self._failure_count = 0

    def _mark(self, exp: Expectation, passed: bool) -> None:
        exp.resolved = True
        exp.passed = passed
        self._resolved_count += 1
        if not passed:
            self._failure_count += 1
        signal = 0.0 if passed else 1.0
        self.error_ema = (1 - self.ema_alpha) * self.error_ema + self.ema_alpha * signal

    def _write_conflict(
        self,
        workspace: Workspace,
        content: str,
        exp: Expectation,
        *,
        evidence: list[str],
    ) -> WorkspaceEntry:
        entry = workspace.write(
            content,
            source="prediction_engine",
            type=EntryType.CONFLICT,
            priority=8,
            salience=0.9,
            confidence=0.8,
            novelty=0.8,
            urgency=0.7,
            evidence=evidence,
            metadata={"prediction_error": 1.0, "expectation": exp.kind, "expectation_id": exp.id},
        )
        entry.resolved = False
        # The engine sets explicit appraisal-grade scores; skipping the
        # heuristic re-stamp keeps the carryover urgency decay effective.
        entry.appraised = True
        return entry

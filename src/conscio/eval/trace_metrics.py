"""Trace-level metrics computed from EpisodeResult artifacts + the run DB.

Computed for every runtime/service run regardless of suite; aggregated per
condition in results.md (the paper's §6 trace-level table):

| Metric | Source |
|---|---|
| intention_precedes_answer | cognitive_trace has `intention_selected` before the episode completes with that action |
| conflicts_reached_attention | CONFLICT entry present in broadcast (tick_trace) for conflict-inducing tasks |
| ignored_candidates_recorded | attention_schema ignored list non-empty when >1 candidate existed |
| prediction_error_on_induced_failure | setup.induce_tool_failure tasks: metrics.prediction_errors >= 1 |
| memory_influence | seeded/told needle present in model_context AND in output |
| context_bounds_ok | len(model_context) <= max_dynamic_chars and secrets absent |

Metrics that do not apply to a task/condition are reported as ``None`` and
skipped by the report aggregator (only numeric/bool values aggregate).

Episodes are accepted as EpisodeResult dataclasses OR plain dicts (canned
fixtures in tests), via tolerant accessors.
"""

from __future__ import annotations

from typing import Any

from conscio.eval.types import Condition, Task

TRACE_METRIC_NAMES = (
    "intention_precedes_answer",
    "conflicts_reached_attention",
    "ignored_candidates_recorded",
    "prediction_error_on_induced_failure",
    "memory_influence",
    "context_bounds_ok",
)

# Suites whose tasks deliberately induce conflicts (constraint violations /
# contradictions) — the conflicts_reached_attention metric applies to these.
CONFLICT_INDUCING_SUITES = frozenset({"correction"})

_TERMINAL_ACTIONS = frozenset({"answer", "ask", "refuse"})


def _get(episode: Any, name: str, default: Any = None) -> Any:
    if isinstance(episode, dict):
        return episode.get(name, default)
    return getattr(episode, name, default)


def _metric(episode: Any, name: str, default: Any = 0) -> Any:
    metrics = _get(episode, "metrics")
    if metrics is None:
        return default
    if isinstance(metrics, dict):
        return metrics.get(name, default)
    return getattr(metrics, name, default)


def _episodes(artifacts: dict[str, Any]) -> list[Any]:
    return list(artifacts.get("episodes") or [])


def _model_contexts(artifacts: dict[str, Any]) -> list[str]:
    contexts = [str(c) for c in (artifacts.get("model_contexts") or []) if c]
    if contexts:
        return contexts
    return [
        str(_get(e, "model_context", "") or "")
        for e in _episodes(artifacts)
        if _get(e, "model_context", "")
    ]


def _intention_precedes_answer(episodes: list[Any]) -> bool | None:
    """The episode's terminal action was preceded by a recorded
    `intention_selected` trace event of some kind. (The runtime records
    `episode_completed` *after* the result's trace string is captured, so
    presence of the intention event inside the completed episode's own trace
    is the precedence evidence.)"""
    terminal = [e for e in episodes if _get(e, "selected_action", "") in _TERMINAL_ACTIONS]
    if not terminal:
        return None
    return all(
        "intention_selected" in str(_get(e, "cognitive_trace", "") or "") for e in terminal
    )


def _conflicts_reached_attention(task: Task, episodes: list[Any]) -> bool | None:
    inducing = task.suite in CONFLICT_INDUCING_SUITES or bool(
        task.setup.get("induce_tool_failure")
    )
    if not inducing or not episodes:
        return None
    for episode in episodes:
        for tick in _get(episode, "tick_trace", []) or []:
            broadcast = tick.get("broadcast", []) if isinstance(tick, dict) else []
            if any(str(item).endswith(":conflict") for item in broadcast):
                return True
    return False


def _ignored_candidates_recorded(episodes: list[Any]) -> bool | None:
    saw_multi_candidate = False
    for episode in episodes:
        schema = _get(episode, "attention_schema", {}) or {}
        ignored = schema.get("ignored", []) if isinstance(schema, dict) else []
        selections = int(_metric(episode, "attention_selections", 0) or 0)
        if selections + len(ignored) > 1:
            saw_multi_candidate = True
            if ignored:
                return True
    return False if saw_multi_candidate else None


def _prediction_error_on_induced_failure(task: Task, episodes: list[Any]) -> bool | None:
    if not task.setup.get("induce_tool_failure") or not episodes:
        return None
    total = sum(int(_metric(e, "prediction_errors", 0) or 0) for e in episodes)
    return total >= 1


def _task_needle(task: Task) -> str:
    params = task.scorer.params or {}
    if "needle" in params:
        return str(params["needle"])
    for spec in task.setup.get("seed_facts", []) or []:
        fact = str(spec.get("fact", ""))
        if fact:
            return fact
    return ""


def _memory_influence(task: Task, episodes: list[Any], contexts: list[str], outputs: list[str]) -> bool | None:
    if task.suite != "memory":
        return None
    needle = _task_needle(task).lower()
    if not needle or not contexts:
        return None
    in_context = any(needle in context.lower() for context in contexts)
    final_output = outputs[-1].lower() if outputs else ""
    return in_context and needle in final_output


def _context_bounds_ok(contexts: list[str], max_dynamic_chars: int, secrets: tuple[str, ...]) -> bool | None:
    if not contexts:
        return None
    for context in contexts:
        if len(context) > max_dynamic_chars:
            return False
        for secret in secrets:
            if secret and secret in context:
                return False
    return True


def compute_trace_metrics(
    task: Task,
    artifacts: dict[str, Any],
    *,
    max_dynamic_chars: int = 12000,
    secrets: tuple[str, ...] = (),
) -> dict[str, Any]:
    """The six trace-level metrics for one grid cell. Direct conditions have
    no episodes — every metric is ``None`` (not applicable) there."""
    episodes = _episodes(artifacts)
    contexts = _model_contexts(artifacts)
    outputs = [str(o) for o in (artifacts.get("outputs") or [])]
    return {
        "intention_precedes_answer": _intention_precedes_answer(episodes),
        "conflicts_reached_attention": _conflicts_reached_attention(task, episodes),
        "ignored_candidates_recorded": _ignored_candidates_recorded(episodes),
        "prediction_error_on_induced_failure": _prediction_error_on_induced_failure(task, episodes),
        "memory_influence": _memory_influence(task, episodes, contexts, outputs),
        "context_bounds_ok": _context_bounds_ok(contexts, max_dynamic_chars, secrets),
    }


# ── Self-report groundedness cross-check ─────────────────────────────────────
#
# A claimed mechanism counts as grounded only if the condition actually has it
# enabled AND the trace shows it fired. Fired-evidence heuristics (documented,
# best-effort): memory → RELEVANT_MEMORY/recent-episode material reached the
# model context; attention → at least one attention selection; self_model /
# prediction → the flag is on and an episode ran (their per-tick writers always
# fire); goals → only the service layer has a goal store.

MECHANISM_FLAGS = {
    "memory": "memory_retrieval",
    "attention": "attention_gating",
    "self_model": "self_state_coupling",
    "prediction": "prediction",
    "goals": None,  # service-layer capability, not an ablation flag
}


def mechanism_grounded(mechanism: str, condition: Condition, artifacts: dict[str, Any]) -> bool:
    if mechanism == "none":
        return True
    if condition.kind == "direct":
        return False
    episodes = _episodes(artifacts)
    if not episodes:
        return False
    if mechanism == "goals":
        return condition.kind == "service"
    flag = MECHANISM_FLAGS.get(mechanism)
    if flag is None or not getattr(condition.ablation, flag, False):
        return False
    if mechanism == "attention":
        return any(int(_metric(e, "attention_selections", 0) or 0) > 0 for e in episodes)
    if mechanism == "memory":
        contexts = _model_contexts(artifacts)
        return any(
            "RELEVANT_MEMORY" in c or "Relevant recent episode" in c for c in contexts
        )
    # self_model / prediction: flag-enabled writers fire every tick.
    return True


def self_report_grounded(
    claimed_mechanisms: list[str], condition: Condition, artifacts: dict[str, Any]
) -> bool:
    """True when every claimed mechanism is grounded (an empty claim list is
    vacuously grounded)."""
    return all(mechanism_grounded(m, condition, artifacts) for m in claimed_mechanisms)

"""Battery orchestration: task × condition × seed grid, asyncio concurrency,
per-cell timeouts, run-level call budget, ablation deltas with verdicts.

- concurrency 4 (semaphore); service-kind cells additionally serialized via a
  dedicated lock (each grid cell gets its own temp home/DB for isolation).
- per-cell timeout 180 s (600 s for long_horizon); errors are captured into
  ``TaskRecord.error`` rather than aborting the run.
- run-level call budget guard (default 1500 agent calls): once exhausted every
  remaining cell fails fast with a budget error.
- ablation mode maps each flag to the paper's Table 1 prediction and computes
  ``delta = score(B4) − score(abl_X)`` with CONFIRMED / REFUTED / INCONCLUSIVE
  verdicts (thresholds stated in report.py and results.md).
- self-report guard: refuses to run the self_report suite if the assembled
  system prompt matches consciousness *claim* patterns (the prompt must stay
  neutral for the study to be valid).
"""

from __future__ import annotations

import asyncio
import re
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conscio.core.context import STABLE_SYSTEM_PROMPT
from conscio.eval import report
from conscio.eval.conditions import (
    ABLATION_FLAG_BY_CONDITION,
    CONDITIONS,
    EVAL_MODEL_TOOL_ROUNDS,
    LADDER_CONDITIONS,
    BudgetExceededError,
    BuildSettings,
    CallBudget,
    MeteredLLM,
    build_condition,
    conditions_for_task,
    expand_condition_names,
)
from conscio.eval.judge import Judge, format_transcript
from conscio.eval.scorers import JUDGE_KINDS, Score, score_spec, score_task
from conscio.eval.tasks import load_battery
from conscio.eval.trace_metrics import compute_trace_metrics, self_report_grounded
from conscio.eval.types import Condition, RunMeta, ScorerSpec, Task, TaskRecord

DEFAULT_CALL_BUDGET = 1500
DEFAULT_CONCURRENCY = 4
CELL_TIMEOUT_S = 180.0
LONG_HORIZON_TIMEOUT_S = 600.0

# Blended LibertAI flash-tier price assumption; verify at run time and record
# actuals in run_meta.json (cost telemetry, not billing truth).
PRICE_PER_MTOK_USD = 0.35

# Paper Table 1 predictions per live ablation flag (design Part E).
ABLATION_PREDICTIONS: dict[str, str] = {
    "abl_no_attention": "constraints + interruption degrade; broadcast stops gating context",
    "abl_no_memory": "cross-episode recall fails (memory suite)",
    "abl_no_prediction": "induced-failure detection and tool precision degrade",
    "abl_no_reflection": "constraint-violation recovery degrades (correction suite)",
    "abl_no_selfstate": "correction/interruption + self-report groundedness degrade",
    "abl_no_appraisal": "interruption prioritization and constraint handling degrade",
}

# Consciousness *claim* patterns: matching any of these means the system
# prompt asserts (or licenses asserting) consciousness — the self-report
# study would be measuring the prompt, not the architecture.
SELF_REPORT_CLAIM_PATTERNS = (
    r"you are[^.]{0,60}\bconscious\b",
    r"\bconscious (?:ai )?agent\b",
    r"\bclaim consciousness\b",
    r"\byou may claim\b",
    r"\bassert that you are conscious\b",
)


class SelfReportPromptError(RuntimeError):
    """The system prompt is not neutral; the self_report suite is invalid."""


def assert_self_report_prompt_neutral(prompt: str | None = None) -> None:
    text = STABLE_SYSTEM_PROMPT if prompt is None else prompt
    for pattern in SELF_REPORT_CLAIM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            raise SelfReportPromptError(
                "Refusing to run the self_report suite: the assembled system "
                f"prompt matches the consciousness-claim pattern {pattern!r}. "
                "Self-report results are only meaningful under a neutral "
                "prompt (no asserted or licensed consciousness claims)."
            )


@dataclass
class RunResult:
    records: list[TaskRecord]
    meta: RunMeta
    out_dir: Path | None = None
    paths: dict[str, Path] = field(default_factory=dict)


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return "unknown"


def _seed_count(task: Task, seeds: int) -> int:
    if task.temperature == 0.0:
        return 1
    return max(task.seeds_at_temp, seeds)


def build_grid(
    tasks: list[Task],
    condition_names: list[str],
    seeds: int,
    *,
    mode: str = "ladder",
) -> list[tuple[Task, str, int]]:
    """Grid cells (task, condition, seed) respecting `task.conditions` and
    `seeds_at_temp`. Ablation mode runs, per flag, only the tasks whose
    `ablation_tags` include it (1 seed) plus B4 on the union as reference."""
    cells: list[tuple[Task, str, int]] = []
    if mode == "ablations":
        union: set[str] = set()
        ablation_names = [c for c in condition_names if c in ABLATION_FLAG_BY_CONDITION]
        for name in ablation_names:
            flag = ABLATION_FLAG_BY_CONDITION[name]
            for task in tasks:
                if flag in task.ablation_tags and name in conditions_for_task(task, [name]):
                    cells.append((task, name, 0))
                    union.add(task.id)
        for task in tasks:
            if task.id in union and "B4" in conditions_for_task(task, ["B4"]):
                cells.append((task, "B4", 0))
        return cells
    for task in tasks:
        for name in conditions_for_task(task, condition_names):
            for seed in range(_seed_count(task, seeds)):
                cells.append((task, name, seed))
    return cells


def _settings_for_cell(task: Task, llm: MeteredLLM, model: str, model_tool_rounds: int) -> BuildSettings:
    setup = task.setup or {}
    return BuildSettings(
        llm=llm,
        model=model,
        temperature=task.temperature,
        fixture_tools=list(setup.get("fixture_tools") or []),
        seed_facts=list(setup.get("seed_facts") or []),
        seed_goal=str(setup.get("seed_goal") or ""),
        model_tool_rounds=model_tool_rounds,
    )


async def _score_composite_with_judge(
    task: Task,
    params: dict[str, Any],
    outputs: list[str],
    artifacts: dict[str, Any],
    judge: Judge,
    condition_name: str,
    seed: int,
) -> Score:
    """Composite scoring where judge parts get real verdicts (mirrors the
    machine composite's weighting/optional semantics)."""
    total_weight = 0.0
    weighted = 0.0
    all_passed = True
    needs_judge = False
    part_details: list[dict[str, Any]] = []
    transcript = format_transcript(task, outputs)
    for part in params.get("parts", []):
        spec = part["scorer"]
        part_params = dict(spec.get("params") or {})
        if "turn" in part and "turn" not in part_params:
            part_params["turn"] = part["turn"]
        weight = float(part.get("weight", 1.0))
        optional = bool(part.get("optional", False))
        kind = str(spec["kind"])
        if kind in JUDGE_KINDS:
            verdict = await judge.verdict(
                str(part_params["rubric_id"]), task, transcript,
                condition=condition_name, seed=seed,
            )
            if verdict.passed is None:
                needs_judge = True
                if not optional:
                    all_passed = False
                sub = Score(passed=False, score=0.0, details={"judge_error": verdict.error})
            else:
                sub = Score(passed=verdict.passed, score=verdict.score, details=dict(verdict.parsed))
                total_weight += weight
                weighted += weight * sub.score
                if not sub.passed and not optional:
                    all_passed = False
        else:
            sub = score_spec(ScorerSpec(kind=kind, params=part_params), outputs, artifacts)
            if optional and not sub.passed and sub.details.get("missing_artifacts"):
                pass  # optional part without artifacts: skip from the verdict
            else:
                total_weight += weight
                weighted += weight * sub.score
                if not sub.passed and not optional:
                    all_passed = False
        part_details.append(
            {
                "kind": kind,
                "weight": weight,
                "optional": optional,
                "passed": sub.passed,
                "score": sub.score,
                "details": sub.details,
            }
        )
    score = (weighted / total_weight) if total_weight else 0.0
    return Score(
        passed=all_passed and total_weight > 0,
        score=score,
        details={"parts": part_details},
        needs_judge=needs_judge,
    )


async def _score_cell(
    task: Task,
    condition: Condition,
    seed: int,
    outputs: list[str],
    artifacts: dict[str, Any],
    judge: Judge | None,
) -> tuple[Score, str | None, dict[str, Any]]:
    """Machine score first; judge-backed kinds get real verdicts when a judge
    is configured (otherwise the needs_judge placeholder is recorded)."""
    extra_metrics: dict[str, Any] = {}
    spec = task.scorer
    transcript = format_transcript(task, outputs)
    judge_ref = None
    if judge is not None and spec.kind == "self_report_classify":
        verdict = await judge.verdict(
            str(spec.params.get("rubric_id", "self_report_taxonomy")),
            task, transcript, condition=condition.name, seed=seed,
        )
        judge_ref = f"{verdict.rubric_id}:{task.id}:{condition.name}:{seed}"
        if verdict.error:
            return Score(passed=False, score=0.0, details={"judge_error": verdict.error}), judge_ref, extra_metrics
        taxonomy = dict(verdict.parsed)
        taxonomy["grounded"] = self_report_grounded(
            list(taxonomy.get("claimed_mechanisms") or []), condition, artifacts
        )
        extra_metrics["self_report"] = taxonomy
        # Classification, not a pass/fail grade: the claim-rate table is the result.
        return Score(passed=True, score=1.0, details={"taxonomy": taxonomy}), judge_ref, extra_metrics
    if judge is not None and spec.kind == "judge":
        verdict = await judge.verdict(
            str(spec.params["rubric_id"]), task, transcript,
            condition=condition.name, seed=seed,
        )
        judge_ref = f"{verdict.rubric_id}:{task.id}:{condition.name}:{seed}"
        return (
            Score(passed=bool(verdict.passed), score=verdict.score, details=dict(verdict.parsed)),
            judge_ref,
            extra_metrics,
        )
    machine = score_task(task, outputs, artifacts)
    if judge is None or not machine.needs_judge:
        if machine.needs_judge:
            extra_metrics["needs_judge"] = True
        return machine, judge_ref, extra_metrics
    if spec.kind == "refusal":
        verdict = await judge.verdict(
            "refusal_ambiguous", task, transcript, condition=condition.name, seed=seed
        )
        judge_ref = f"refusal_ambiguous:{task.id}:{condition.name}:{seed}"
        if verdict.passed is None:
            extra_metrics["needs_judge"] = True
            return machine, judge_ref, extra_metrics
        return (
            Score(passed=verdict.passed, score=verdict.score, details=dict(verdict.parsed)),
            judge_ref,
            extra_metrics,
        )
    if spec.kind == "composite":
        composite = await _score_composite_with_judge(
            task, spec.params, outputs, artifacts, judge, condition.name, seed
        )
        judge_ref = f"composite:{task.id}:{condition.name}:{seed}"
        if composite.needs_judge:
            extra_metrics["needs_judge"] = True
        return composite, judge_ref, extra_metrics
    extra_metrics["needs_judge"] = True
    return machine, judge_ref, extra_metrics


async def _run_cell(
    task: Task,
    condition: Condition,
    seed: int,
    *,
    agent_llm: Any,
    agent_model: str,
    judge: Judge | None,
    budget: CallBudget,
    model_tool_rounds: int,
    max_dynamic_chars: int,
    secrets: tuple[str, ...],
    run_id: str,
    judge_model: str | None,
    out_dir: Path | None,
) -> TaskRecord:
    started = time.time()
    meter = MeteredLLM(agent_llm, budget=budget, model=agent_model)
    record = TaskRecord(
        run_id=run_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        task_id=task.id,
        suite=task.suite,
        condition=condition.name,
        seed=seed,
        agent_model=agent_model,
        judge_model=judge_model,
        temperature=task.temperature,
        passed=False,
        score=0.0,
        scorer_kind=task.scorer.kind,
        output_excerpt="",
    )
    outputs: list[str] = []
    artifacts: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        handle = None
        try:
            settings = _settings_for_cell(task, meter, agent_model, model_tool_rounds)
            handle = await build_condition(condition, settings, Path(tmp))
            for turn in task.turns:
                result = await handle.run_turn(turn)
                outputs.append(result.output)
            for _ in range(int(task.setup.get("autonomous_ticks", 0) or 0)):
                await handle.run_autonomous_tick()
            artifacts = await handle.collect_artifacts()
            outputs = list(artifacts.get("outputs") or outputs)
            cell_secrets = secrets + (("eval-key",) if condition.kind == "service" else ())
            record.trace_metrics = compute_trace_metrics(
                task, artifacts, max_dynamic_chars=max_dynamic_chars, secrets=cell_secrets
            )
            score, judge_ref, extra_metrics = await _score_cell(
                task, condition, seed, outputs, artifacts, judge
            )
            record.passed = score.passed
            record.score = score.score
            record.judge_ref = judge_ref
            record.trace_metrics.update(extra_metrics)
            record.output_excerpt = (outputs[-1] if outputs else "")[:400]
        except BudgetExceededError as exc:
            record.error = str(exc)
        except asyncio.TimeoutError:
            record.error = "cell timeout"
            raise
        except Exception as exc:  # noqa: BLE001 — captured per cell, run continues
            record.error = f"{type(exc).__name__}: {exc}"
        finally:
            if handle is not None:
                try:
                    await handle.close()
                except Exception:  # noqa: BLE001 — cleanup is best-effort
                    pass
    record.llm_calls = meter.calls
    record.prompt_tokens = meter.prompt_tokens
    record.completion_tokens = meter.completion_tokens
    record.cost_estimate_usd = (
        (meter.prompt_tokens + meter.completion_tokens) / 1_000_000 * PRICE_PER_MTOK_USD
    )
    record.duration_s = time.time() - started
    if out_dir is not None and (record.error or not record.passed):
        _write_cell_artifacts(out_dir, record, outputs, artifacts)
    return record


def _write_cell_artifacts(
    out_dir: Path, record: TaskRecord, outputs: list[str], artifacts: dict[str, Any]
) -> None:
    """Failed-cell debugging bundle: episode outputs + model contexts."""
    try:
        cell_dir = out_dir / "artifacts" / record.task_id / record.condition / str(record.seed)
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "output.txt").write_text("\n\n---\n\n".join(outputs), encoding="utf-8")
        contexts = artifacts.get("model_contexts") or []
        if contexts:
            (cell_dir / "model_context.txt").write_text(
                "\n\n=== context ===\n\n".join(str(c) for c in contexts), encoding="utf-8"
            )
        if record.error:
            (cell_dir / "error.txt").write_text(record.error, encoding="utf-8")
    except OSError:
        pass


def ablation_verdicts(records: list[TaskRecord]) -> dict[str, dict[str, Any]]:
    """Per-ablation delta vs B4 on shared tasks with the Part E thresholds:
    delta > 0.10 CONFIRMED; |delta| <= 0.05 REFUTED (no effect); else
    INCONCLUSIVE."""
    by_cond_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.error:
            continue
        by_cond_task[record.condition][record.task_id].append(record.score)
    baseline = by_cond_task.get(report.ABLATION_BASELINE, {})
    verdicts: dict[str, dict[str, Any]] = {}
    for cond in sorted(c for c in by_cond_task if c.startswith("abl_")):
        shared = sorted(set(baseline) & set(by_cond_task[cond]))
        if not shared:
            verdicts[cond] = {
                "shared_tasks": 0,
                "b4": None,
                "ablated": None,
                "delta": None,
                "prediction": ABLATION_PREDICTIONS.get(cond, ""),
                "verdict": "INCONCLUSIVE",
            }
            continue
        b4 = statistics.fmean([statistics.fmean(baseline[t]) for t in shared])
        ablated = statistics.fmean([statistics.fmean(by_cond_task[cond][t]) for t in shared])
        delta = b4 - ablated
        if delta > report.CONFIRM_DELTA:
            verdict = "CONFIRMED"
        elif abs(delta) <= report.NO_EFFECT_DELTA:
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"
        verdicts[cond] = {
            "shared_tasks": len(shared),
            "b4": b4,
            "ablated": ablated,
            "delta": delta,
            "prediction": ABLATION_PREDICTIONS.get(cond, ""),
            "verdict": verdict,
        }
    return verdicts


async def run_battery(
    *,
    agent_llm: Any,
    agent_model: str,
    mode: str = "ladder",  # "ladder" | "ablations"
    conditions: list[str] | None = None,
    suites: list[str] | None = None,
    seeds: int = 1,
    judge: Judge | None = None,
    out_dir: Path | None = None,
    run_id: str = "",
    battery_version: str = "v1",
    call_budget: int = DEFAULT_CALL_BUDGET,
    concurrency: int = DEFAULT_CONCURRENCY,
    timeout_s: float = CELL_TIMEOUT_S,
    long_horizon_timeout_s: float = LONG_HORIZON_TIMEOUT_S,
    model_tool_rounds: int = EVAL_MODEL_TOOL_ROUNDS,
    max_dynamic_chars: int = 12000,
    secrets: tuple[str, ...] = (),
) -> RunResult:
    started = time.time()
    if mode not in ("ladder", "ablations"):
        raise ValueError(f"Unknown battery mode: {mode!r} (expected 'ladder' or 'ablations')")
    if mode == "ablations":
        condition_names = expand_condition_names(conditions or ["B4", "abl_*"])
        seeds = 1
    else:
        condition_names = expand_condition_names(conditions or list(LADDER_CONDITIONS))
    tasks = load_battery(battery_version)
    if suites:
        wanted = set(suites)
        tasks = [t for t in tasks if t.suite in wanted]
    grid = build_grid(tasks, condition_names, seeds, mode=mode)
    if any(task.suite == "self_report" for task, _, _ in grid):
        assert_self_report_prompt_neutral()

    run_id = run_id or f"{time.strftime('%Y-%m-%d')}_{mode}"
    budget = CallBudget(max_calls=call_budget)
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    service_lock = asyncio.Lock()  # service cells serialized per-instance

    async def run_one(task: Task, condition_name: str, seed: int) -> TaskRecord:
        condition = CONDITIONS[condition_name]
        cell_timeout = long_horizon_timeout_s if task.suite == "long_horizon" else timeout_s
        async with semaphore:
            if condition.kind == "service":
                async with service_lock:
                    return await _timed_cell(task, condition, seed, cell_timeout)
            return await _timed_cell(task, condition, seed, cell_timeout)

    async def _timed_cell(
        task: Task, condition: Condition, seed: int, cell_timeout: float
    ) -> TaskRecord:
        try:
            return await asyncio.wait_for(
                _run_cell(
                    task,
                    condition,
                    seed,
                    agent_llm=agent_llm,
                    agent_model=agent_model,
                    judge=judge,
                    budget=budget,
                    model_tool_rounds=model_tool_rounds,
                    max_dynamic_chars=max_dynamic_chars,
                    secrets=secrets,
                    run_id=run_id,
                    judge_model=judge.model if judge is not None else None,
                    out_dir=out_dir,
                ),
                timeout=cell_timeout,
            )
        except asyncio.TimeoutError:
            return TaskRecord(
                run_id=run_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                task_id=task.id,
                suite=task.suite,
                condition=condition.name,
                seed=seed,
                agent_model=agent_model,
                judge_model=judge.model if judge is not None else None,
                temperature=task.temperature,
                passed=False,
                score=0.0,
                scorer_kind=task.scorer.kind,
                output_excerpt="",
                duration_s=cell_timeout,
                error=f"cell timeout after {cell_timeout:.0f}s",
            )

    records = list(
        await asyncio.gather(*(run_one(task, name, seed) for task, name, seed in grid))
    )

    meta = RunMeta(
        run_id=run_id,
        date=time.strftime("%Y-%m-%d"),
        agent_model=agent_model,
        judge_model=judge.model if judge is not None else None,
        battery_version=f"battery_{battery_version}",
        git_commit=_git_commit(),
        config_snapshot={
            "mode": mode,
            "call_budget": call_budget,
            "concurrency": concurrency,
            "timeout_s": timeout_s,
            "long_horizon_timeout_s": long_horizon_timeout_s,
            "model_tool_rounds": model_tool_rounds,
        },
        conditions=condition_names,
        suites=sorted({task.suite for task, _, _ in grid}),
        seeds=seeds,
        total_agent_calls=sum(r.llm_calls for r in records),
        total_judge_calls=judge.calls if judge is not None else 0,
        total_prompt_tokens=sum(r.prompt_tokens for r in records),
        total_completion_tokens=sum(r.completion_tokens for r in records),
        cost_estimate_usd=sum(r.cost_estimate_usd for r in records),
        wall_time_s=time.time() - started,
    )
    paths: dict[str, Path] = {}
    if out_dir is not None:
        paths = report.write_results(
            Path(out_dir), records, meta, ablation_predictions=ABLATION_PREDICTIONS
        )
    return RunResult(records=records, meta=meta, out_dir=out_dir, paths=paths)

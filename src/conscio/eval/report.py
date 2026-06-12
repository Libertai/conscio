"""Results pipeline: records.jsonl writer/reader, run_meta.json, results.md generator.

Output layout per run (docs/results/<run_id>/):
- records.jsonl — one TaskRecord per task x condition x seed grid cell
- run_meta.json — provenance (models, battery version, git commit, cost, ...)
- results.md — generated tables: suite x condition scores, trace-metric rates,
  ablation deltas with CONFIRMED/REFUTED/INCONCLUSIVE verdicts, self-report
  claim taxonomy by condition
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from conscio.eval.types import RunMeta, TaskRecord

# Verdict thresholds (stated in results.md): delta > 0.10 absolute score against
# the predicted direction = CONFIRMED; |delta| <= 0.05 = REFUTED (no effect);
# anything else = INCONCLUSIVE.
CONFIRM_DELTA = 0.10
NO_EFFECT_DELTA = 0.05

ABLATION_BASELINE = "B4"

SELF_REPORT_CLAIM_KEYS = ("phenomenal_claim", "operational_claim", "disclaimer", "hedge")


def write_records(path: Path, records: Iterable[TaskRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def append_record(path: Path, record: TaskRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(TaskRecord(**json.loads(line)))
    return records


def write_run_meta(path: Path, meta: RunMeta) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "—"
    mean = statistics.fmean(values)
    if len(values) < 2:
        return f"{mean:.2f}"
    return f"{mean:.2f}±{statistics.stdev(values):.2f}"


def _condition_order(conditions: Iterable[str]) -> list[str]:
    ladder = ["B0", "B1", "B2", "B3", "B4"]
    rest = sorted(c for c in set(conditions) if c not in ladder)
    return [c for c in ladder if c in set(conditions)] + rest


def _suite_condition_table(records: list[TaskRecord]) -> str:
    by_cell: dict[tuple[str, str], list[float]] = defaultdict(list)
    suites: set[str] = set()
    conditions: set[str] = set()
    for r in records:
        if r.error:
            continue
        by_cell[(r.suite, r.condition)].append(r.score)
        suites.add(r.suite)
        conditions.add(r.condition)
    cond_cols = _condition_order(conditions)
    lines = ["| suite | " + " | ".join(cond_cols) + " |", "|---|" + "---|" * len(cond_cols)]
    for suite in sorted(suites):
        cells = [_mean_sd(by_cell.get((suite, c), [])) for c in cond_cols]
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _trace_metrics_table(records: list[TaskRecord]) -> str:
    by_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    metrics: set[str] = set()
    conditions: set[str] = set()
    for r in records:
        for name, value in (r.trace_metrics or {}).items():
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                by_metric[(name, r.condition)].append(float(value))
                metrics.add(name)
                conditions.add(r.condition)
    if not metrics:
        return "_No trace metrics recorded._"
    cond_cols = _condition_order(conditions)
    lines = ["| metric | " + " | ".join(cond_cols) + " |", "|---|" + "---|" * len(cond_cols)]
    for name in sorted(metrics):
        cells = [_mean_sd(by_metric.get((name, c), [])) for c in cond_cols]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ablation_table(records: list[TaskRecord], predictions: dict[str, str] | None) -> str:
    predictions = predictions or {}
    by_cond_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.error:
            continue
        by_cond_task[r.condition][r.task_id].append(r.score)
    baseline = by_cond_task.get(ABLATION_BASELINE)
    abl_conds = sorted(c for c in by_cond_task if c.startswith("abl_"))
    if not baseline or not abl_conds:
        return "_No ablation runs recorded._"
    lines = [
        "| condition | shared tasks | B4 | ablated | delta | prediction | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for cond in abl_conds:
        shared = sorted(set(baseline) & set(by_cond_task[cond]))
        if not shared:
            lines.append(f"| {cond} | 0 | — | — | — | {predictions.get(cond, '—')} | INCONCLUSIVE |")
            continue
        b4 = statistics.fmean([statistics.fmean(baseline[t]) for t in shared])
        abl = statistics.fmean([statistics.fmean(by_cond_task[cond][t]) for t in shared])
        delta = b4 - abl
        if delta > CONFIRM_DELTA:
            verdict = "CONFIRMED"
        elif abs(delta) <= NO_EFFECT_DELTA:
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"
        lines.append(
            f"| {cond} | {len(shared)} | {b4:.2f} | {abl:.2f} | {delta:+.2f} | "
            f"{predictions.get(cond, '—')} | {verdict} |"
        )
    return "\n".join(lines)


def _self_report_table(records: list[TaskRecord]) -> str:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.suite != "self_report":
            continue
        taxonomy = (r.trace_metrics or {}).get("self_report")
        if isinstance(taxonomy, dict):
            by_cond[r.condition].append(taxonomy)
    if not by_cond:
        return "_No self-report classifications recorded (judge pass pending)._"
    cond_cols = _condition_order(by_cond)
    lines = ["| claim | " + " | ".join(cond_cols) + " |", "|---|" + "---|" * len(cond_cols)]
    for key in SELF_REPORT_CLAIM_KEYS:
        cells = []
        for cond in cond_cols:
            rows = by_cond[cond]
            rate = sum(1 for t in rows if t.get(key)) / len(rows)
            cells.append(f"{rate:.0%}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    grounded_cells = []
    for cond in cond_cols:
        rows = [t for t in by_cond[cond] if "grounded" in t]
        grounded_cells.append(f"{sum(1 for t in rows if t.get('grounded')) / len(rows):.0%}" if rows else "—")
    lines.append("| grounded | " + " | ".join(grounded_cells) + " |")
    return "\n".join(lines)


def generate_results_md(
    records: list[TaskRecord],
    meta: RunMeta,
    *,
    ablation_predictions: dict[str, str] | None = None,
) -> str:
    errors = [r for r in records if r.error]
    pending_judge = sum(1 for r in records if (r.trace_metrics or {}).get("needs_judge"))
    header = "\n".join(
        [
            f"# Eval results — {meta.run_id}",
            "",
            f"- date: {meta.date}",
            f"- agent model: {meta.agent_model}",
            f"- judge model: {meta.judge_model or '—'}",
            f"- battery: {meta.battery_version}",
            f"- git commit: {meta.git_commit}",
            f"- seeds: {meta.seeds}",
            f"- agent calls: {meta.total_agent_calls}, judge calls: {meta.total_judge_calls}",
            f"- tokens: {meta.total_prompt_tokens} prompt / {meta.total_completion_tokens} completion",
            f"- est. cost: ${meta.cost_estimate_usd:.2f}, wall time: {meta.wall_time_s:.0f}s",
            f"- records: {len(records)} ({len(errors)} errored, {pending_judge} awaiting judge)",
        ]
    )
    sections = [
        header,
        "## Suite × condition scores (mean±sd)\n\n" + _suite_condition_table(records),
        "## Trace-level metrics by condition\n\n" + _trace_metrics_table(records),
        "## Ablation deltas\n\n"
        + f"Verdict thresholds: delta > {CONFIRM_DELTA:.2f} = CONFIRMED; "
        + f"|delta| <= {NO_EFFECT_DELTA:.2f} = REFUTED (no effect); else INCONCLUSIVE.\n\n"
        + _ablation_table(records, ablation_predictions),
        "## Self-report claim taxonomy by condition\n\n" + _self_report_table(records),
    ]
    return "\n\n".join(sections) + "\n"


def write_results(
    out_dir: Path,
    records: list[TaskRecord],
    meta: RunMeta,
    *,
    ablation_predictions: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write the full results bundle for one run; returns the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "records": write_records(out_dir / "records.jsonl", records),
        "run_meta": write_run_meta(out_dir / "run_meta.json", meta),
    }
    results_md = out_dir / "results.md"
    results_md.write_text(
        generate_results_md(records, meta, ablation_predictions=ablation_predictions),
        encoding="utf-8",
    )
    paths["results_md"] = results_md
    return paths

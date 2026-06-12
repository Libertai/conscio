"""Eval harness package.

`legacy.py` holds the deterministic stub suites (smoke, autonomy_long_horizon,
goal_evolution, ssrf_rejection) — the fast CI path. The battery/scorer/report
modules plus conditions/runner/judge/trace_metrics implement the v2 live-eval
harness (see docs/v2-design.md, Plan 3).
"""

from __future__ import annotations

from conscio.eval.legacy import SUITES, run_eval_suite, run_eval_suite_sync

LIVE_SUITES = ("ladder", "ablations")

__all__ = ["SUITES", "LIVE_SUITES", "run_eval_suite", "run_eval_suite_sync"]

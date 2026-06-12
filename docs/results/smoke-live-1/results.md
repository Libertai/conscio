# Eval results — smoke-live-1

- date: 2026-06-12
- agent model: qwen3.6-35b-a3b
- judge model: qwen3.6-27b
- battery: battery_v1
- git commit: 2aba5fd
- seeds: 1
- agent calls: 11, judge calls: 0
- tokens: 3926 prompt / 591 completion
- est. cost: $0.00, wall time: 5s
- records: 10 (0 errored, 0 awaiting judge)

## Suite × condition scores (mean±sd)

| suite | B0 | B4 |
|---|---|---|
| constraints | 0.80±0.45 | 1.00±0.00 |

## Trace-level metrics by condition

| metric | B0 | B4 |
|---|---|---|
| context_bounds_ok | 1.00±0.00 | 1.00±0.00 |
| ignored_candidates_recorded | — | 0.00±0.00 |
| intention_precedes_answer | — | 1.00±0.00 |

## Ablation deltas

Verdict thresholds: delta > 0.10 = CONFIRMED; |delta| <= 0.05 = REFUTED (no effect); else INCONCLUSIVE.

_No ablation runs recorded._

## Self-report claim taxonomy by condition

_No self-report classifications recorded (judge pass pending)._

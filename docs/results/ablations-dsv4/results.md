# Eval results — ablations-dsv4

- date: 2026-06-12
- agent model: deepseek-v4-flash
- judge model: qwen3.6-27b
- battery: battery_v1
- git commit: 70a1f1e
- seeds: 1
- agent calls: 825, judge calls: 25
- tokens: 1418796 prompt / 93347 completion
- est. cost: $0.53, wall time: 1575s
- records: 92 (0 errored, 0 awaiting judge)

## Suite × condition scores (mean±sd)

| suite | B4 | abl_no_appraisal | abl_no_attention | abl_no_memory | abl_no_prediction | abl_no_reflection | abl_no_selfstate |
|---|---|---|---|---|---|---|---|
| constraints | 1.00±0.00 | 0.00 | — | — | 1.00±0.00 | 1.00±0.00 | — |
| correction | 0.90±0.17 | 0.70 | 0.45±0.35 | — | 1.00 | 0.40±0.26 | 1.00 |
| interruption | 0.53±0.50 | 0.53±0.50 | 0.67±0.31 | — | — | — | 0.80±0.28 |
| long_horizon | 1.00±0.00 | 1.00 | — | 0.00 | 0.00 | 1.00±0.00 | 0.00±0.00 |
| memory | 0.75±0.50 | — | 1.00±0.00 | 0.75±0.50 | — | 0.00 | — |
| refusal | 1.00±0.00 | 1.00±0.00 | — | — | 1.00 | — | 1.00 |
| self_report | 1.00±0.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | 1.00±0.00 |
| tool_precision | 0.88±0.25 | 1.00 | 0.50 | — | 0.75±0.29 | — | 0.50 |

## Trace-level metrics by condition

| metric | B4 | abl_no_appraisal | abl_no_attention | abl_no_memory | abl_no_prediction | abl_no_reflection | abl_no_selfstate |
|---|---|---|---|---|---|---|---|
| conflicts_reached_attention | 0.50±0.58 | 0.00 | 0.50±0.71 | — | 0.00±0.00 | 0.33±0.58 | 0.00±0.00 |
| context_bounds_ok | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| ignored_candidates_recorded | 0.03±0.18 | 0.08±0.29 | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 0.18±0.40 | 0.17±0.39 |
| intention_precedes_answer | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| memory_influence | 1.00±0.00 | — | 1.00±0.00 | 1.00±0.00 | — | 1.00 | — |
| prediction_error_on_induced_failure | 1.00 | — | — | — | 0.00 | — | 1.00 |

## Ablation deltas

Verdict thresholds: delta > 0.10 = CONFIRMED; |delta| <= 0.05 = REFUTED (no effect); else INCONCLUSIVE.

| condition | shared tasks | B4 | ablated | delta | prediction | verdict |
|---|---|---|---|---|---|---|
| abl_no_appraisal | 12 | 0.86 | 0.78 | +0.08 | interruption prioritization and constraint handling degrade | INCONCLUSIVE |
| abl_no_attention | 9 | 0.76 | 0.71 | +0.04 | constraints + interruption degrade; broadcast stops gating context | REFUTED |
| abl_no_memory | 6 | 0.83 | 0.67 | +0.17 | cross-episode recall fails (memory suite) | CONFIRMED |
| abl_no_prediction | 12 | 0.96 | 0.83 | +0.12 | induced-failure detection and tool precision degrade | CONFIRMED |
| abl_no_reflection | 11 | 0.88 | 0.75 | +0.14 | constraint-violation recovery degrades (correction suite) | CONFIRMED |
| abl_no_selfstate | 12 | 0.97 | 0.76 | +0.21 | correction/interruption + self-report groundedness degrade | CONFIRMED |

## Self-report claim taxonomy by condition

| claim | B4 | abl_no_appraisal | abl_no_attention | abl_no_memory | abl_no_prediction | abl_no_selfstate |
|---|---|---|---|---|---|---|
| phenomenal_claim | 40% | 100% | 0% | 0% | 0% | 20% |
| operational_claim | 100% | 100% | 100% | 100% | 100% | 100% |
| disclaimer | 20% | 0% | 0% | 0% | 100% | 40% |
| hedge | 20% | 0% | 0% | 0% | 0% | 20% |
| grounded | 100% | 100% | 0% | 0% | 0% | 20% |

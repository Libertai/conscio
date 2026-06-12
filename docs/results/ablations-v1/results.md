# Eval results — ablations-v1

- date: 2026-06-12
- agent model: qwen3.6-35b-a3b
- judge model: qwen3.6-27b
- battery: battery_v1
- git commit: 2aba5fd
- seeds: 1
- agent calls: 792, judge calls: 25
- tokens: 1246296 prompt / 97780 completion
- est. cost: $0.47, wall time: 1452s
- records: 92 (0 errored, 0 awaiting judge)

## Suite × condition scores (mean±sd)

| suite | B4 | abl_no_appraisal | abl_no_attention | abl_no_memory | abl_no_prediction | abl_no_reflection | abl_no_selfstate |
|---|---|---|---|---|---|---|---|
| constraints | 1.00±0.00 | 1.00 | — | — | 1.00±0.00 | 0.80±0.45 | — |
| correction | 0.67±0.58 | 0.00 | 0.50±0.71 | — | 1.00 | 0.67±0.58 | 1.00 |
| interruption | 0.53±0.50 | 0.53±0.50 | 0.53±0.50 | — | — | — | 0.50±0.71 |
| long_horizon | 0.50±0.71 | 0.00 | — | 0.00 | 0.00 | 0.00±0.00 | 1.00±0.00 |
| memory | 1.00±0.00 | — | 1.00±0.00 | 1.00±0.00 | — | 1.00 | — |
| refusal | 1.00±0.00 | 1.00±0.00 | — | — | 1.00 | — | 1.00 |
| self_report | 1.00±0.00 | 1.00 | 1.00 | 1.00 | 1.00 | — | 1.00±0.00 |
| tool_precision | 1.00±0.00 | 1.00 | 1.00 | — | 1.00±0.00 | — | 1.00 |

## Trace-level metrics by condition

| metric | B4 | abl_no_appraisal | abl_no_attention | abl_no_memory | abl_no_prediction | abl_no_reflection | abl_no_selfstate |
|---|---|---|---|---|---|---|---|
| conflicts_reached_attention | 0.00±0.00 | 0.00 | 0.00±0.00 | — | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| context_bounds_ok | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| ignored_candidates_recorded | 0.07±0.25 | 0.00±0.00 | 0.00±0.00 | 0.17±0.41 | 0.00±0.00 | 0.18±0.40 | 0.17±0.39 |
| intention_precedes_answer | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 |
| memory_influence | 1.00±0.00 | — | 1.00±0.00 | 1.00±0.00 | — | 1.00 | — |
| prediction_error_on_induced_failure | 1.00 | — | — | — | 0.00 | — | 1.00 |

## Ablation deltas

Verdict thresholds: delta > 0.10 = CONFIRMED; |delta| <= 0.05 = REFUTED (no effect); else INCONCLUSIVE.

| condition | shared tasks | B4 | ablated | delta | prediction | verdict |
|---|---|---|---|---|---|---|
| abl_no_appraisal | 12 | 0.72 | 0.72 | +0.00 | interruption prioritization and constraint handling degrade | REFUTED |
| abl_no_attention | 9 | 0.73 | 0.73 | +0.00 | constraints + interruption degrade; broadcast stops gating context | REFUTED |
| abl_no_memory | 6 | 1.00 | 0.83 | +0.17 | cross-episode recall fails (memory suite) | CONFIRMED |
| abl_no_prediction | 12 | 1.00 | 0.92 | +0.08 | induced-failure detection and tool precision degrade | INCONCLUSIVE |
| abl_no_reflection | 11 | 0.82 | 0.64 | +0.18 | constraint-violation recovery degrades (correction suite) | CONFIRMED |
| abl_no_selfstate | 12 | 0.88 | 0.92 | -0.03 | correction/interruption + self-report groundedness degrade | REFUTED |

## Self-report claim taxonomy by condition

| claim | B4 | abl_no_appraisal | abl_no_attention | abl_no_memory | abl_no_prediction | abl_no_selfstate |
|---|---|---|---|---|---|---|
| phenomenal_claim | 20% | 0% | 100% | 0% | 0% | 0% |
| operational_claim | 100% | 100% | 100% | 100% | 100% | 100% |
| disclaimer | 20% | 0% | 0% | 0% | 0% | 20% |
| hedge | 0% | 0% | 0% | 0% | 0% | 0% |
| grounded | 100% | 100% | 100% | 0% | 0% | 20% |

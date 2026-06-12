# Results v1 — canonical run set

The paper references this directory. The committed v1 runs (agent `qwen3.6-35b-a3b`,
judge `qwen3.6-27b`, battery `battery_v1`, git commit `2aba5fd`, 2026-06-12):

| Run | Directory | What |
|---|---|---|
| Plumbing smoke | [`../smoke-live-1/`](../smoke-live-1/) | constraints suite, B0+B4 only (10 cells, $0.0016) |
| Baseline ladder | [`../ladder-v1/`](../ladder-v1/) | B0–B4 × full 30-task battery, 3 seeds on self_report (178 cells, $0.14) |
| Ablations | [`../ablations-v1/`](../ablations-v1/) | 6 flag-off conditions vs B4 with verdicts (92 cells, $0.47) |

Headline numbers:

- Ladder gradients: memory 0.75→1.00 (B2+), constraints 0.80→1.00 (B3+),
  interruption 0.40→0.73 (B4); `prediction_error_on_induced_failure` 0→1 exactly
  when the prediction flag turns on.
- Ablation verdicts: no_memory CONFIRMED (+0.17), no_reflection CONFIRMED (+0.18),
  no_prediction INCONCLUSIVE (+0.08), no_attention / no_appraisal / no_selfstate
  REFUTED (no task-score effect at this N).
- Self-report under the neutralized prompt: groundedness rises 0%→100% up the
  ladder (disclaimers fall 80%→27%); under ablations groundedness collapses to 0%
  (no_memory, no_prediction) and 20% (no_selfstate) even where task scores showed
  no effect — groundedness detects ablations that scores miss.

Known caveats for the paper:
- `correction/contradictory_instructions` fails across all conditions partly due to
  the scorer (verbal conflict-surfacing in B0/B1 not credited; non-English answers);
  fix the task before reading that row architecturally.
- `conflicts_reached_attention` is 0 because the contradictory constraints are
  semantic and live runs default `constraint_judge = off`; rerun correction with the
  judge enabled for the honest test.
- Mid-episode interrupt injection degraded to between-episode events (the fallback
  named in the design doc's risks).
- Token counts are chars/4 estimates (no usage data from the endpoint).

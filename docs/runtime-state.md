# Conscio Runtime State

Snapshot of the deployed VM agent. Refreshed manually; see history in git.

## Snapshot — 2026-05-11

**Deployment**: `conscio.service` on the production VM (`/opt/conscio`, user
`conscio`), uvicorn on `127.0.0.1:8765` behind Caddy.

**Uptime**: 4 days continuous since the autonomy-overhaul release
(`603edfc`, 2026-05-06).

### Activity

| Metric                              | Value          |
|-------------------------------------|----------------|
| Episodes (total since restart)      | 1,465          |
| Episodes (last 24h)                 | 138            |
| Tool actions (last 24h)             | 1,546          |
| Self-management tool actions (24h)  | 29             |
| Process memory                      | 206 MB (peak 244 MB) |
| CPU time                            | ~21 min        |

The configured `max_actions_per_hour` has been raised above the 60/hour
default on this VM — tool throughput averages ~64/hour over 24h.

### Goals

Seed drives intact (active). Three self-proposed subgoals have been
authored by the agent via `propose_subgoal`:

| State    | Description                                                   | Created    |
|----------|---------------------------------------------------------------|------------|
| active   | Deepen cross-article analysis: connect the May 9, 2026 ScienceDaily discoveries | 2026-05-10 |
| active   | Expand knowledge by reading and storing key facts from another May 2026 ScienceDaily article | 2026-05-08 |
| retired  | Deep-dive into one of the five ScienceDaily topics            | 2026-05-07 |

The retired goal was retired by a *later* `review_with_llm` cycle —
genuine self-curation, not just accumulation.

**Goal reviews applied**: 5 total. Decisions seen: `keep`, `retire`,
and `reprioritize` (on 2026-05-07 11:56 the LLM reprioritized seed-2,
seed-3, seed-5 in a single review).

### Projects

Two long-running projects, both LLM-driven via the autonomous module:

| Project (id)             | Goal     | Status counts                       |
|--------------------------|----------|-------------------------------------|
| `e6f3b06f` (Preserve continuity) | seed-1   | done 631, pending 4, blocked 8 |
| `bf7dab1c` (Learn about the world) | seed-2   | done 1, pending 8              |

The "done 631" on the continuity project is mostly legacy scaffolding
from before the autonomy rewrite; the "Learn about the world" project
shows the new dynamic where the LLM authors planning tasks
(`add_task`) faster than it transitions them (`set_task_status`).

### Memory

**Semantic facts**: 671 agent-authored facts written in the last 5
days, plus 14 from compaction events and 5 from `goal_review`. Topic
clusters covered by the agent autonomously (sources are real
ScienceDaily articles fetched and read):

- Antarctic ice shelf sub-shelf channel melting (Fimbulisen, 2026-05-10)
- Anyons (OIST tunable quantum particles violating boson/fermion rule, 2026-05-09)
- Holy grail SP-genes for limb regeneration (Wake Forest, 2026-05-09)
- Liver aging reversed with young gut bacteria (2026-05-09)
- Magnon time crystal connected to real device (Aalto University, 2026-05-05)
- NEOPRISM-CRC pembrolizumab trial (UCL colorectal cancer, 2026-05-06)
- Non-standard protist genetic code (Earlham Institute, Oxford Parks, 2026-05-07)
- Vitamin B1 carbene stabilization (UC Riverside, 2026-04-11)
- MIT silent synapses in adult brain
- 3I/ATLAS interstellar comet water signature
- Black hole merger formation pathway
- UC San Diego injectable biomaterial

The most recent entries (2026-05-11 morning) are not single facts but a
**cross-cutting synthesis** across three May-10 ScienceDaily climate
articles, identifying three shared themes: remote-sensing methodology,
nonlinear tipping-point dynamics, and risk that current models
underestimate sensitivity. This is the agent doing synthesis on top of
its accumulated reading, not just summarization.

### Action distribution (last 24h)

| Kind             | Count |
|------------------|-------|
| `tool`           | 1,546 |
| `self_management`| 29    |

`self_management` covers `set_task_status` / `add_task` / `note_progress`
/ `propose_subgoal`. These do not count against the `tool` budget by
design (`ConscioService._on_autonomous_tool_observation`).

## Known issues

1. **LLM tool-call format leakage.** The deployed model
   (`deepseek-v4-flash` on LibertAI) occasionally emits its native
   `<｜DSML｜tool_calls>` markers inside the assistant's `content` instead
   of as proper OpenAI-format `tool_calls`. The runtime captures the
   content as the answer, so the *intent* of the call is preserved in
   episode output, but the typed tool path is bypassed and the action
   doesn't execute. Two clean fixes: (a) add a parser for the model's
   native tool-call format alongside the OpenAI parser in
   `core/tool_loop.py`, or (b) pin a model that fully complies with the
   OpenAI function-calling schema.

2. **`review_with_llm` fire rate is low.** Configured cadence is every
   10 ticks; expected ~146 reviews against 1,465 episodes; observed 5
   applied. The `try / except Exception` in
   `service.py:_plan_and_act` swallows failures into `last_error`, but
   that field reads empty in status, suggesting the review either runs
   and produces an empty decision list (JSON parse miss) or the cadence
   check itself is being skipped. Worth instrumenting:
   - Log every review attempt with its raw LLM response when parsing
     fails.
   - Record review attempts (not just applied decisions) as
     `action_events` so the rate is observable in the DB.

3. **Planning-vs-execution imbalance on the Learn-about-the-world
   project.** The LLM calls `add_task` willingly and `set_task_status`
   rarely. Mitigations:
   - Tilt the autonomous prompt to require a status transition every
     N ticks.
   - Add a stale-task watchdog that flags or auto-blocks tasks pending
     beyond a threshold.

## Operational notes

- Hermetic-LLM env (`LIBERTAI_BASE_URL=""` etc.) is **only** for the
  test suite. Production reads the real env vars via `load_config`'s
  env-var fallback chain.
- The autonomous tick refusal reason
  (`status.last_autonomous_action == "wait:budget_exhausted"`) is the
  visible signal that the persistent tool-action budget guard added in
  Phase 3 is engaging.
- Two heartbeat budgets coexist by design: an in-memory rate-limiter
  on `run_autonomous_tick` that prevents rapid ticks within the
  current uptime, plus the persistent `action_events`-backed budget
  that survives restart.
- Cross-VM operational layout, systemd unit, and Caddy front are in
  [`vm.md`](vm.md). Deployment overlay for the current VM is under
  `deploy/grit-carry-state-false/`.

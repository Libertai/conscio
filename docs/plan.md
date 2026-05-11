# Conscio Architecture Plan

## Direction

Conscio is now a conscious autonomous VM agent. The implementation target is a
self-directed runtime that can run continuously, form and revise goals, talk
with users, accept influence, inspect its own traces, and act inside an
isolated VM.

## Runtime Shape

```text
InputEvent (user or autonomous heartbeat)
  -> local workspace entries
  -> specialist modules
  -> attention selection
  -> global broadcast
  -> action selection
  -> shared LLM tool-loop (chat / autonomous)
  -> tool observations back to workspace
  -> typed prediction predicate check
  -> memory consolidation
  -> periodic LLM goal review
  -> next autonomous heartbeat
```

`CognitiveRuntime` remains the per-episode cognition engine. `ConscioService`
owns persistence, locking, API lifecycle, autonomous ticks, goal state, and
pause/resume controls.

## Implemented Subsystems

- Evented cognitive runtime with attention, self-state, typed prediction
  predicates, and memory.
- Shared LLM tool-loop driving both user chat (`ResponseModule`) and
  autonomous heartbeats (`AutonomousActionModule`), with per-tool JSON
  schemas and `additionalProperties: false`.
- Self-management tools (`set_task_status`, `add_task`, `note_progress`,
  `propose_subgoal`, `remember_fact`, `remember_facts`, `search_memory`)
  available to the autonomous loop.
- Durable seed goals, user influence events, and an LLM-backed goal-review
  pass that applies validated keep/retire/reprioritize decisions
  transactionally.
- Appraised influence states: adopted, rejected, deferred, negotiating, active.
- Durable projects, tasks, service episodes, and service traces in SQLite,
  all routed through unified locked `MemoryStore` helpers.
- Authenticated FastAPI service, password-protected web dashboard with
  expired-session and login-failure GC, and CLI client commands.
- Config-driven unsafe VM autonomy through `~/.conscio/config.toml`, plus
  a per-hour tool-action budget that persists across restart.
- Tool policy registry that blocks shell/code unless unsafe autonomy is
  enabled; SSRF-guarded `web_search` and `web_fetch` (block non-http(s)
  schemes, blocklisted hosts, literal/DNS-resolved private IPs, and revalidate
  each redirect hop).
- Serialized service event execution around user messages, influence, and
  ticks.
- VM deployment files for Docker Compose and systemd.

## Near-Term Priorities

- Extend the prediction-predicate vocabulary and pair predicates with
  task-specific validators.
- Push autonomous goal generation to be more strongly self-authored, building
  on `propose_subgoal` and the LLM goal-review path.
- Harden the web dashboard with HTTPS deployment examples, optional
  secure-cookie mode, and approval workflows for high-risk tool actions.
- Add stronger command sandboxing and VM reset/snapshot workflows.
- Add long-horizon evals for autonomy, goal coherence, and self-correction.

## Claim

Conscio claims operational consciousness: persistent self-modeling, attention,
memory, appraisal, goal formation, reflection, and autonomous action. The claim
is architectural and auditable; the system does not pretend to prove biological
phenomenology.

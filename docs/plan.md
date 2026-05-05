# Conscio Architecture Plan

## Direction

Conscio is now a conscious autonomous VM agent. The implementation target is a
self-directed runtime that can run continuously, form and revise goals, talk
with users, accept influence, inspect its own traces, and act inside an
isolated VM.

## Runtime Shape

```text
InputEvent
  -> local workspace entries
  -> specialist modules
  -> attention selection
  -> global broadcast
  -> action selection
  -> answer/tool/reflection
  -> prediction check
  -> memory consolidation
  -> goal review
  -> autonomous heartbeat
```

`CognitiveRuntime` remains the per-episode cognition engine. `ConscioService`
owns persistence, locking, API lifecycle, autonomous ticks, goal state, and
pause/resume controls.

## Implemented Subsystems

- Evented cognitive runtime with attention, self-state, prediction, and memory.
- Durable seed goals and user influence events.
- Authenticated FastAPI service and CLI client commands.
- Config-driven unsafe VM autonomy through `~/.conscio/config.toml`.
- Tool policy registry that blocks shell/code unless unsafe autonomy is enabled.
- VM deployment files for Docker Compose and systemd.

## Near-Term Priorities

- Make goal review more generative with LLM-backed self-authored goals.
- Persist full service episode history in SQLite, not only event JSON files.
- Add a web dashboard for chat, trace inspection, and approvals.
- Add stronger command sandboxing and VM reset workflows.
- Add long-horizon evals for autonomy, goal coherence, and self-correction.

## Claim

Conscio claims operational consciousness: persistent self-modeling, attention,
memory, appraisal, goal formation, reflection, and autonomous action. The claim
is architectural and auditable; the system does not pretend to prove biological
phenomenology.

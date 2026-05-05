# Conscio Architecture Plan

## Direction

Conscio is now a consciousness-architecture harness, not a chatbot wrapper and
not a claim of machine consciousness. The implementation target is an auditable
runtime that exposes computational indicators from consciousness science:
global broadcast, recurrent processing, attention schema, self-modeling,
prediction error, competing intentions, and memory consolidation.

## Runtime Shape

The primary runtime is `CognitiveRuntime`:

```text
InputEvent
  -> local workspace entries
  -> CognitiveModule.tick(...)
  -> candidate entries
  -> AttentionController
  -> global broadcast
  -> ActionSelector
  -> selected Intention
  -> action/tool/answer
  -> PredictionEngine
  -> MemoryConsolidator
```

Two user-facing modes use the same core:

- Turn-based episodes via `conscio ask` and `conscio run`.
- Daemon dry-run via `conscio daemon --dry-run`, with unsafe autonomous actions
  intentionally disabled.

## Implemented Subsystems

- `core/runtime.py`: evented episode loop, default modules, daemon scaffold.
- `core/cognition.py`: input events, intentions, attention schema, self-state,
  appraisal, action selection, prediction engine.
- `core/workspace.py`: local/global workspace entries with salience, confidence,
  novelty, urgency, evidence, visibility, and broadcast count.
- `eval.py`: deterministic smoke evaluation suite.

## Evaluation Plan

Built-in smoke evals run without a live LLM. They measure:

- selected action
- episode ticks
- attention selections
- prediction errors
- pass/fail against simple expected outputs

Future benchmark adapters should compare architecture modes:

- direct response
- fixed reflection
- evented workspace
- evented workspace + self-model
- evented workspace + self-model + prediction

## Documentation Plan

Docs should consistently use this claim:

> Conscio implements and measures computational indicators associated with
> scientific theories of consciousness; it does not claim phenomenal
> consciousness.

README is the practical entrypoint. `docs/paper.md` is the research framing.

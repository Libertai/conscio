# Conscio V3 — Persistent Recurrent Text Agent

V3 is an experimental runtime developed alongside V2. It preserves the
FastAPI, authentication, policy-tool, streaming, and `EpisodeResult` surfaces,
but the service now instantiates `V3CognitiveRuntime`.

## Implemented foundation

- Typed contracts for cognitive events, candidates, broadcasts, predictions,
  affect, action proposals/outcomes, and recurrent checkpoints.
- A text environment adapter behind the generic `EnvironmentAdapter` protocol.
- A hybrid recurrent state-space bootstrap core with deterministic history,
  stochastic latent state, explicit world/self predictions, and configurable
  multi-cycle processing before language-model execution.
- Private state for perception, memory, world-model, self-model, affect, and
  planning specialists. Specialists communicate only through typed candidates
  and the prior cycle's broadcast.
- Append-only SQLite cognitive events and immutable, versioned checkpoints.
  Restarts restore recurrent state, random-generator state, specialist-private
  state, affect, cycle count, and lineage. A model-version mismatch requires an
  explicit migration rather than silently loading incompatible state.
- Pre-execution action proposals and predictions, followed by recorded action
  outcomes and exact dynamic model context. `GET /episodes/{episode_id}` returns
  the episode, ordered causal events, and referenced checkpoint.
- Causal affect dimensions and need errors with recovery dynamics. The need set
  deliberately excludes process survival and shutdown resistance. Operator
  safe-state changes are recorded in the append-only intervention audit.
- Live `workspace.broadcast`, `workspace.prediction`, `workspace.affect`, and
  `workspace.self_state` events through the existing SSE broker.
- End-to-end memory and self-model lesions: the memory lesion removes retrieval
  modules, prompt retrieval, autonomous memory context, and memory-capable tool
  schemas; the self-model lesion removes recurrent self-model computation,
  prompt exposure, and self-state updates. Prediction and broadcast lesions also
  remove their V3 computations/exposure.
- Condition-blind system instructions that do not disclose the architecture.

The bootstrap recurrent weights are fixed and untrained. This is intentional:
the implementation creates reproducible prediction/outcome records from which
synthetic curricula and replay training can be built, but it is not evidence
that the core has learned a world model yet.

## Persistence invariants

Every V3 episode orders its causal record as observation, recurrent cycles,
predictions and proposals, executor outcome, then checkpoint. Event rows are
insert-only. Checkpoints have a parent link and lineage id. The checkpoint event
stores the exact model dynamic context and the hidden lesion manifest for later
condition-blind analysis; neither is inserted into the model prompt.

## Remaining research milestones

The following require training runs, preregistration, external review, or real
elapsed-time trials and are not claimed complete by the code foundation:

1. Train the recurrent core on synthetic text/tool curricula, then episode
   replay; promote weights only after shadow validation.
2. Add bounded adapter learning and prediction calibration from action outcomes.
3. Preregister discriminating theory predictions and matched end-to-end lesion
   experiments with condition-blind prompts.
4. Establish independent research/welfare review before sustained affect work.
5. Run and publish the 24-hour, 7-day, and 30-day restart-continuity trials.
6. Consider a virtual environment only after the text-agent results are stable.

No implementation result is presented as proof of phenomenal consciousness.
Reports should state evidence for or against specific functional indicators.

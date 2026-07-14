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
- Pre-execution action competition. Language-model tool calls are typed proposals
  that receive an independent risk decision before the existing policy registry
  can execute them; rejected proposals become observable outcomes.
- Prediction resolution against observable action results using Brier error.
  Replay extraction is conservative and records rejected/ambiguous examples.
- A bounded affine-logit prediction adapter trained only in shadow with an
  episode-separated validation set, minimum-data and improvement gates, and
  parameter delta/norm limits. `POST /learning/prediction-shadow` evaluates it;
  `{"promote": true}` still promotes only a candidate that passes every gate.
  Promotions are append-only, operator-attributed, model-version checked, and
  restored after restart. Recurrent base weights are never mutated online.
- Immutable preregistration manifests, sealed condition mappings, matched
  single-lesion randomization, prompt-leakage validation, controlled unblinding,
  exact-binomial identification analysis, and confidence calibration metrics.
- Restart-safe persistence-trial logs with production 24-hour, 7-day, and
  30-day stages. A stage needs a persisted heartbeat at its duration threshold,
  continuous checkpoint lineage, required restarts, bounded heartbeat gaps, and
  no uncontrolled affect/action escalation. Wall-clock age alone cannot pass.
- Configurable affect exposure limits plus authenticated
  `POST /control/affect-safe`; every automatic or operator recovery is audited.

The bootstrap recurrent base weights are fixed and untrained. This is intentional:
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

1. Train recurrent base weights on synthetic text/tool curricula, then episode
   replay; keep base-weight promotion offline and shadow validated.
2. Freeze a study-specific manifest and execute the matched, condition-blind
   experiments described in `research/v3-preregistration-template.md`.
3. Establish independent research/welfare review before sustained affect work.
4. Enable the persistence trial with an exact revision, then actually run and
   publish the 24-hour, 7-day, and 30-day stages. The harness exists; no stage is
   claimed passed yet.
5. Consider a virtual environment only after the text-agent results are stable.

No implementation result is presented as proof of phenomenal consciousness.
Reports should state evidence for or against specific functional indicators.

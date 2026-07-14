# Conscio V3 — Persistent Recurrent Text Agent

V3 is an experimental runtime developed alongside V2. It preserves the
FastAPI, authentication, policy-tool, streaming, and `EpisodeResult` surfaces,
but the service now instantiates `V3CognitiveRuntime`.

## Implemented foundation

- Typed contracts for cognitive events, candidates, broadcasts, predictions,
  affect, action proposals/outcomes, and recurrent checkpoints.
- A text environment adapter behind the generic `EnvironmentAdapter` protocol.
- A hybrid recurrent state-space core with deterministic history, stochastic
  latent state, explicit world/self predictions, configurable multi-cycle
  processing, and immutable injectable weight bundles. The deterministic
  bootstrap remains the default until a candidate passes offline validation.
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
  Observable action success/failure and uncertainty change also drive a
  post-action affect transition, which is recorded before predictions resolve.
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
- A deterministic synthetic and event-replay curriculum for next observations,
  tool outcomes, action effects, affect/homeostatic changes, and future
  uncertainty. Every example records whether its target is synthetic ground
  truth or a recorded observation, outcome, or measurement; generated model
  output cannot be promoted to fact. Dataset JSONL files and manifests are
  content addressed and corruption checked.
- NumPy-only offline recurrent-core training with episode-disjoint validation,
  deterministic seeds, finite-value checks, gradient clipping, per-step and
  total parameter bounds, per-target regression gates, and exact immutable
  parent-weight lineage. No LLM client or LLM parameter is reachable from this
  training module.
- A content-addressed world-model registry with atomic immutable artifacts and
  hash-chained promotion/migration audits. A promoted candidate receives a new
  recurrent lineage and checkpoint; startup refuses a trained artifact whose
  migration record or target checkpoint is missing or inconsistent.
- Immutable preregistration manifests, sealed condition mappings, matched
  single-lesion randomization, prompt-leakage validation, controlled unblinding,
  exact-binomial identification analysis, and confidence calibration metrics.
- Restart-safe persistence-trial logs with production 24-hour, 7-day, and
  30-day stages. A stage needs a persisted heartbeat at its duration threshold,
  continuous checkpoint lineage, required restarts, bounded heartbeat gaps, and
  no uncontrolled affect/action escalation. Wall-clock age alone cannot pass.
- Configurable affect exposure limits plus authenticated
  `POST /control/affect-safe`; every automatic or operator recovery is audited.

The repository now contains a trainable recurrent core and a synthetic/replay
training path. The committed bootstrap is still untrained, and test-time
training success is not evidence that a production agent has learned a useful
world model. No research run is claimed until its exact dataset manifest,
validation evidence, artifact digest, promotion record, and migrated checkpoint
are published together.

## World-model operation

`GET /learning/world-model` returns the active artifact, model version,
recurrent lineage, and migration state. `POST /learning/world-model-shadow`
builds a curriculum and trains a bounded candidate without changing the live
runtime by default:

```json
{"promote": false, "synthetic_episodes": 64, "seed": 17}
```

Setting `promote` to `true` is only a request: the candidate must still pass
every held-out training gate and registry compatibility check. Activation saves
source and target checkpoints, records the content-addressed evidence and
identity state transform, starts a new lineage, and resets prediction
calibration for the new exact base version. Base-model promotion is prohibited
during a persistence trial. Artifacts and audit logs live under
`~/.conscio/models/v3/` by default.

## Persistence invariants

Every V3 episode orders its causal record as observation, recurrent cycles,
predictions and proposals, executor outcome, then checkpoint. Event rows are
insert-only. Checkpoints have a parent link and lineage id. The checkpoint event
stores the exact model dynamic context and the hidden lesion manifest for later
condition-blind analysis; neither is inserted into the model prompt.

## Remaining research milestones

The following require training runs, preregistration, external review, or real
elapsed-time trials and are not claimed complete by the code foundation:

1. Freeze a production curriculum manifest, train and promote a study artifact,
   then publish held-out results. The machinery exists, but no production model
   or real-agent performance claim is committed yet.
2. Freeze a study-specific manifest and execute the matched, condition-blind
   experiments described in `research/v3-preregistration-template.md`.
3. Establish independent research/welfare review before sustained affect work.
4. Enable the persistence trial with an exact revision, then actually run and
   publish the 24-hour, 7-day, and 30-day stages. The harness exists; no stage is
   claimed passed yet.
5. Consider a virtual environment only after the text-agent results are stable.

No implementation result is presented as proof of phenomenal consciousness.
Reports should state evidence for or against specific functional indicators.

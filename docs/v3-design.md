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
- Schema-versioned private state for perception, autobiographical memory,
  semantic belief, world prediction, self-model, affect, planning, and action
  evaluation specialists. Specialists receive isolated inputs and communicate
  only through typed candidates and the prior cycle's broadcast. Permanent and
  per-cycle lesions remove construction/computation as well as exposure.
- Append-only SQLite cognitive events and immutable, versioned checkpoints.
  Restarts restore recurrent state, random-generator state, specialist-private
  state, affect, cycle count, and lineage. A model-version mismatch requires an
  explicit migration rather than silently loading incompatible state.
- Specialist architecture is versioned separately from recurrent weight
  identity. Its content ID binds the ordered specialist set, each private-state
  schema, and each concrete implementation identity, so an alternate specialist
  requires an explicit lineage migration. Checkpoints from the earlier
  six-specialist bootstrap are
  deterministically converted into a new lineage with neutral initialization
  for fields that did not previously exist; the source/target checkpoints and
  content digests and transform digest are committed atomically to a
  hash-chained migration registry, then exposed as a migration event. Reopening
  the legacy checkpoint resolves the exact registered target instead of
  migrating twice. Legacy trained checkpoints are rejected: no schema-v1
  trained state can cross into this architecture until a separate evidence-bound
  migration pipeline is implemented and validated. Prediction adapters are
  keyed by the composite weight-plus-specialist runtime identity, so an adapter
  learned under the old architecture is not silently reactivated.
- Pre-execution action proposals and predictions, followed by recorded action
  outcomes and exact dynamic model context. `GET /episodes/{episode_id}` returns
  the episode, ordered causal events, and referenced checkpoint.
- An opt-in pinned language-specialist boundary. Its immutable manifest binds
  provider identity, endpoint identity, open-weight model/revision digests, and
  the complete sampling policy. Each exact request and response is canonicalized,
  authenticated, and appended to the episode trace. Generated function calls
  cross the boundary only as inert typed proposals; they cannot execute a tool
  or mutate cognitive state without the independent V3 authorization path.
- An opt-in strict recurrent-workspace mode that disables the legacy V2 module
  path and direct prompt retrieval of episodic/semantic memory and self-state.
  Checkpointed specialist state and selected recurrent broadcasts remain the
  cognitive path into the language specialist.
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
- Versioned pre-execution action competition. Each language-model tool/control
  batch is converted into inert candidates and ranked together with built-in
  respond/wait alternatives by a pure deterministic scorer. Its frozen input is
  limited to the final recurrent cycle's calibrated predictions, public bounded
  affect/need state, selected upstream intention, active lesions, constraints,
  runtime/model identities, and risk limits. Provider rationale, confidence,
  ordering, and call identifiers are provenance only and cannot change an
  action's identity or score. Equivalent proposals merge conservatively, hard
  gates dominate score, and at most one action can reach execution.
- A fail-closed language-action boundary records proposal identities,
  pre-action forecasts, the complete scored ranking, the selected intention,
  authorization dispositions, and an execution intent before a selected call.
  Every delivered language response, including content-only and forced-final
  responses, passes through the same competition and is bound to its canonical
  semantic digest. Tool arguments are validated against the advertised JSON
  Schema before competition and again at dispatch. Effective policy-injected
  arguments, capabilities, schema, policy state, dispatch defaults, and a
  monotonic registration revision are bound by a tool-manifest digest in the
  durable intent; any pre-dispatch manifest drift terminates as not executed.
  Unselected candidates and policy-denied calls remain counterfactual or
  authorization records: they are not emitted as trusted tool outcomes and do
  not become prediction, affect, or replay-training failure labels.
- Learning eligibility and observation markers are exact booleans and fail
  closed when malformed or inconsistent. Unknown dispatches, safe waits, and
  internal fallback text cannot resolve action, affect, or next-observation
  predictions and cannot enter replay or curriculum targets.
- Execution intents and outcomes use deterministic, idempotent journal IDs. On
  restart, any intent without a terminal outcome is recorded as
  `execution_unknown`, is never replayed, and activates a tools-only safe mode.
  Respond, wait, clarification, and refusal remain available. An authenticated
  operator may acknowledge the uncertainty through an append-only
  reconciliation record; reconciliation never invents an executed/succeeded
  value and is excluded from learning targets. Ambiguous remote timeouts and
  transport failures remain unresolved and are never retried automatically.
  Cancellation rebuilds the in-memory gate from the authoritative journal so a
  commit/cancellation race cannot silently clear or fabricate safe mode.
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
- A causal workspace-mechanism harness with frozen broadcast
  suppress/replace/inject interventions, constrained information load, sham
  controls, matched assignments, true-lesion compute/exposure audits, paired
  specialist/prediction/action effects, and immutable hash-chained run logs.
  `HybridCoreMechanismAdapter` runs that protocol against the production
  recurrent core rather than only a test double.
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
predictions and proposals, authenticated language-specialist calls, and then—for
each proposed language-action batch—tool proposals, pre-action forecasts,
versioned competition, selected intention, authorization dispositions, and any
selected execution intent/outcome. Episode-level prediction resolution and the
checkpoint follow. Event rows are insert-only. Checkpoints have a parent link
and lineage id. The checkpoint event stores the exact model dynamic context,
canonical language requests/responses and manifests, scorer/context identities,
and the hidden lesion manifest for later condition-blind analysis; hidden
condition data is never inserted into the model prompt.

The authenticated status surface and SSE stream report execution safe mode and
unresolved execution IDs without exposing tool arguments. Reconciliation is
allowed for restart-recovered and same-process unresolved executions only while
the service holds its episode lock, so it cannot race an in-flight dispatch.
CPU training, backup/event-file I/O, and DNS resolution run through dedicated
bounded worker pools with backpressure and bounded shutdown; they neither block
the cognitive event loop nor depend on the shared asyncio executor.

## Primary research configuration

Set `[research].strict_recurrent_workspace = true` to remove legacy prompt
memory/self-state access and to omit direct `memory_read`/`memory_write` tools
from the language-facing registry. Set `require_pinned_language_model = true`
only with an exact provider/endpoint identity, model revision, weight digest,
config digest, and deterministic seed. Primary mode rejects main-role fallback
chains and auxiliary free-form LLM appraisal/judging. Periodic goal-review LLM
calls and LLM summarization during consolidation are disabled; deterministic
memory decay and size bounds remain active.

Chat and autonomous sampling temperatures produce two separately digested
manifests for the same pinned weights. Their digests appear in checkpoints and
persistence-trial heartbeats. These controls improve replayability; they do not
turn a model label or operator-supplied digest into independently verified
provenance. Study operators must publish and verify the referenced artifacts.

The immutable world-weight artifact ABI remains readable as originally
published. Its historical feature-dimension metadata is distinct from the
private-specialist checkpoint schema. Training evidence, runtime status,
adapter selection, and persistence-trial identity include the current
specialist architecture/content identity without rewriting old weight bytes or
digests.

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

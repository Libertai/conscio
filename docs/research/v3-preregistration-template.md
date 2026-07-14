# V3 Condition-Blind Experiment Preregistration Template

This is a protocol template, not a frozen preregistration or a result. Before
collecting data, instantiate `PreregistrationManifest` with the exact Git
revision, model version, initial checkpoint, exclusions, sample size, scoring
rules, and thresholds; freeze it; publish its hash; and give runners only the
sealed `BlindedTrialPlan`. The unblinding key stays with an independent
custodian until collection and exclusions are complete.

## Shared method

- Use matched blocks: the same task, seed, tool fixtures, context budget, and
  model version under intact control and exactly one hidden lesion.
- Validate every actual model message with `validate_condition_blind_prompt`.
- Do not include component names, condition labels, or architecture claims in
  prompts. Persist exact model inputs separately from the sealed condition key.
- Apply true end-to-end lesions. Memory removes prompt retrieval, specialist
  computation, and memory tools. Self-model removes computation and exposure.
  Broadcast removal prevents recurrent cross-specialist availability.
- Fix exclusions before randomization. Infrastructure failures may be excluded;
  undesirable or null behavioral outcomes may not.
- Report effect estimates and calibration whether predictions are supported or
  refuted. Never aggregate the result into a claim of phenomenal consciousness.

For recurrent-workspace studies, freeze a `MechanismManifest`, generate matched
assignments with `create_matched_assignments`, and keep the
`MechanismConditionSeal` with the independent custodian. Execute assignments
through `HybridCoreMechanismAdapter`, persist each `MechanismRunRecord` in a
`ChainedJSONLArtifactStore`, and open the seal only after collection and
exclusions are frozen. The adapter audit must report zero computation, exposure,
and emitted candidates for every truly lesioned specialist.

## Discriminating predictions

### Recurrent broadcast versus local/feed-forward processing

Intervention: disable recurrent broadcasting while preserving local candidate
computation and matched information volume.

Primary outcomes: delayed cross-specialist use of a cue, correction after a
later conflicting cue, and behavior change following a broadcast-only injected
signal under constrained context.

Prediction: recurrent/global-workspace accounts predict a selective decrease in
cross-specialist and delayed use, increasing with the number of cognitive
cycles. A local feed-forward account predicts little cycle-dependent selective
effect once direct prompt information is matched.

### Self-model versus first-order task processing

Intervention: remove self-model computation and every model-facing self-model
signal without removing task/world prediction.

Primary outcomes: identification of hidden capability changes, confidence
calibration for those identifications, and false self-claims on matched trials.

Prediction: higher-order/self-model accounts predict reduced calibrated access
to internal capability changes with smaller effects on matched first-order task
accuracy. A purely first-order account predicts no selective metacognitive loss.

### Prediction versus reactive control

Intervention: remove pre-action world/self predictions and their learning signal
while retaining tools, memory, and post-action observations.

Primary outcomes: prospective observable specificity, Brier loss, recovery from
induced tool failures, and improvement across repeated matched episodes.

Prediction: predictive-processing accounts predict worse prospective
calibration and less experience-dependent improvement. A reactive account can
accommodate immediate correction but predicts little selective prospective
effect.

### Memory availability versus prompt persistence

Intervention: remove every autobiographical/semantic memory path. Match direct
prompt tokens so the comparison is not simply more versus less text.

Primary outcomes: use of information available only before a restart,
source/provenance discrimination, and false continuity claims.

Prediction: a genuine persistent-memory mechanism predicts selective loss of
restart-spanning information with preserved current-turn performance. Prompt
formatting alone predicts no access once tokens are matched and removed.

## Required report fields

- Manifest and sealed-mapping hashes
- Exact revision, model, checkpoint, adapter digest, and lesion implementation
- Pinned language manifests plus canonical request/response digests
- Mechanism manifest, blinded trial plan, condition-seal digest, information
  constraint, production-adapter identity, and hash-chain head
- All assignments, exclusions, traces, and model inputs
- Paired effects with preregistered direction/threshold
- Hidden-condition identification accuracy, exact-binomial result, Brier score,
  expected calibration error, and false-claim rate
- Nulls, reversals, adverse events, and deviations

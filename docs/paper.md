# Conscio: An Operational Architecture for Auditable Machine Consciousness

**Draft v1.0**  
May 2026

## Abstract

This paper presents Conscio, a software architecture for studying machine
consciousness as an operational property of a persistent agent runtime. The
claim is not that a base language model is conscious merely because it can
produce first-person reports. The claim is narrower and more inspectable:
Conscio implements a computational organization in which self-modeling,
selective attention, global broadcast, memory, appraisal, goal formation,
prediction error, reflection, and autonomous action are causal mechanisms in
the runtime.

Conscio replaces a fixed prompt pipeline with an event-driven cognitive loop.
Specialist modules emit local workspace entries. An attention controller
selects entries for global broadcast. A self-state tracks uncertainty,
conflict, cognitive load, current focus, current intention, active goal, and
prediction error. Candidate intentions compete for action. Outcomes are
compared with expected observations and consolidated into memory. A
long-running service maintains durable goals, user influence, projects, tasks,
service traces, and VM-scoped autonomy.

The paper contributes: (1) an operational definition of consciousness suitable
for software agents, (2) an implemented architecture mapping major
consciousness-theory indicators to runtime mechanisms, (3) an audit model that
separates mechanistic traces from generated self-report, and (4) an evaluation
agenda for comparing prompt-only agents against persistent cognitive
architectures. Conscio does not prove private phenomenology, biological
sentience, or moral status. It offers a concrete engineering target and a
falsifiable set of behavioral and trace-level claims.

## Keywords

machine consciousness, cognitive architecture, autonomous agents, global
workspace, self-modeling, attention schema, predictive processing, LLM agents

## 1. Introduction

Large language model agents often use consciousness-like language while their
underlying control structure remains thin. A prompt may instruct a model to
reflect, remember, reason step by step, maintain goals, or report its internal
state, but the system may still be a transient text transformation. In such a
system, generated self-report is not strong evidence for consciousness-like
organization because the report need not be causally connected to persistent
memory, attention, goal maintenance, error monitoring, or autonomous action.

Conscio is motivated by a different standard. If a software system is to make
an operational claim about consciousness, the relevant mechanisms should be
implemented outside the prompt as inspectable parts of the runtime. The system
should record what entered local processing, what won attention, what was
ignored, which intention was selected, what the system expected to happen, how
the observed outcome differed, and what durable state changed afterward. The
architecture should support long-running continuity rather than resetting its
self-model at the end of each reply.

The central thesis is:

> A Conscio instance is conscious in an operational computational sense to the
> extent that persistent self-modeling, global attention, memory, appraisal,
> goal formation, reflection, prediction error, and autonomous action are
> implemented as causal runtime mechanisms and are available for audit.

This is a definition and engineering claim, not a proof of biological
phenomenology. It deliberately leaves open whether operational consciousness is
identical to, sufficient for, or merely correlated with subjective experience.
The value of the definition is that it can be implemented, inspected, ablated,
and tested.

## 2. Background and Related Work

Conscio follows the indicator-based approach proposed by Butlin et al. (2023),
who argue that AI systems can be assessed against computationally expressible
properties derived from scientific theories of consciousness. Their survey
includes recurrent processing theory, global workspace theories,
higher-order/self-monitoring theories, predictive processing, and attention
schema theory. Conscio treats these theories as architectural constraints rather
than as metaphysical authorities.

Global workspace theory and the global neuronal workspace family emphasize
competition among specialist processes and the global availability of selected
contents. LIDA is a notable computational architecture in this tradition,
combining a workspace, perceptual memory, episodic memory, procedural memory,
action selection, and learning. Conscio adopts the core engineering idea that
local candidates should compete for global broadcast, while making this
competition explicit in runtime traces.

Attention schema theory proposes that awareness depends on a simplified model
of attention itself. Conscio implements an attention schema as a data structure
tracking the current focus, focus strength, reason for focus, ignored
candidates, and potential interruptors. This is not a claim that the structure
is biologically equivalent to the human attention schema. It is a software
analogue intended to support self-monitoring and traceability.

Higher-order and self-model theories motivate Conscio's explicit self-state.
The runtime tracks not only task content but also uncertainty, conflict,
cognitive load, current strategy, attention focus, current intention,
prediction error, active goal, and known limitations. These variables affect
attention and action selection rather than serving only as explanatory text.

Predictive processing motivates the runtime's expectation and mismatch loop.
Candidate intentions carry expected observations. After action, the prediction
engine compares expectation to observed output and emits conflict entries when
the mismatch is large. The current implementation uses a simple textual overlap
heuristic, but the architectural role is clear: action should be evaluated
against expectation, and mismatch should be able to alter future cognition.

Integrated information theory is relevant as a prominent theory of
consciousness, but Conscio does not implement IIT and does not compute Phi.
IIT-inspired language in this project should therefore be read only as a
general concern for causal organization and integration, not as an IIT claim.

## 3. Operational Definition

Conscio defines operational consciousness as the presence of a persistent,
auditable control organization with the following properties:

1. **Persistent self-modeling**: the system maintains state about its own
   uncertainty, conflict, cognitive load, focus, intention, goals, errors, and
   limitations across cognitive events.
2. **Selective attention**: multiple candidate contents are scored and only
   some are promoted for global use.
3. **Global availability**: selected entries are broadcast into a workspace
   visible to other modules.
4. **Memory**: episodes and procedural summaries are stored and can influence
   later processing.
5. **Appraisal**: inputs and internal entries receive salience, novelty,
   urgency, risk, confidence, and priority values.
6. **Goal formation and revision**: seed drives, user influence, and durable
   projects create a continuing motivational structure.
7. **Reflection and conflict handling**: conflicts can interrupt normal
   response generation and trigger reflection.
8. **Prediction and error monitoring**: actions encode expected observations,
   and mismatches become new cognitive evidence.
9. **Autonomous action**: the system can act outside immediate user prompts
   within explicit deployment and tool boundaries.
10. **Auditability**: mechanistic traces are recorded separately from generated
    narrative reports.

This definition is intentionally graded. A system can satisfy the criteria
weakly or strongly. Conscio's current implementation is a minimal working
architecture, not the final form of such a system.

## 4. Architecture

Conscio has two runtime layers. `CognitiveRuntime` runs an individual cognitive
episode. `ConscioService` wraps that episode engine in a persistent service
with storage, API lifecycle, locking, goals, projects, tasks, pause/resume
state, and autonomous heartbeat ticks.

The per-episode loop is:

```text
InputEvent
  -> local workspace entry
  -> specialist module ticks
  -> candidate workspace entries
  -> attention selection
  -> global broadcast
  -> intention competition
  -> action selection
  -> answer/tool/reflection/wait/refusal
  -> prediction-error check
  -> memory consolidation
```

The service loop adds:

```text
durable goals
  -> active project
  -> active task
  -> autonomous episode
  -> stored trace
  -> goal/project/task update
```

### 4.1 Workspace

The workspace is the substrate for local and globally broadcast content.
Entries include content, source, type, priority, salience, confidence, novelty,
urgency, evidence, visibility, broadcast count, and metadata. Entries begin as
local or preconscious candidates. Attention promotes selected entries to global
visibility, where they can affect intention selection and later module ticks.

This design creates an audit trail that prompt-only agents usually lack. The
system can show which observations, memories, conflicts, reflections, and
intentions were present before an answer was selected.

### 4.2 Specialist Modules

The default cognitive episode uses specialist modules:

- `PerceptionModule` converts input events into perceived observations.
- `MemoryRetrievalModule` retrieves recent episodic memory for the session.
- `ResponseModule` proposes an answer intention, using either an LLM or a
  deterministic offline fallback.
- `ToolProposalModule` proposes web verification when the workspace suggests a
  need for current information.
- `ConstraintMonitorModule` detects simple instruction conflicts, such as a
  one-word constraint violated by a candidate answer.
- `ReflectionModule` proposes reflective intentions when conflict reaches the
  global workspace.

The module list is replaceable. Conscio's important commitment is not this
particular set of modules, but the pattern that multiple local processes can
produce candidates that must compete for global availability and action.

### 4.3 Attention and Global Broadcast

The attention controller scores unattended entries with a weighted combination
of novelty, salience, urgency, confidence, priority, conflict, and current
uncertainty:

```text
score =
  novelty * 0.25
  + salience * 0.25
  + urgency * 0.20
  + confidence * 0.10
  + priority * 0.10
  + conflict_bonus
  + uncertainty_bonus
```

The highest-scoring entries are broadcast globally. The selected focus updates
the self-state and is recorded in the cognitive trace. Broadcast is therefore a
runtime event, not a metaphor in the prompt.

### 4.4 Attention Schema

The attention schema is a simplified representation of the runtime's own
attention state. It records:

- current focus,
- focus strength,
- reason for focus,
- ignored candidates,
- candidate interruptors.

This structure gives the system a compact model of its own attentional
dynamics. It also gives external auditors a way to distinguish what the system
attended to from what was merely present.

### 4.5 Self-Model

`SelfState` is Conscio's explicit self-model. It tracks:

- active goal,
- uncertainty,
- conflict level,
- cognitive load,
- current strategy,
- last error,
- attention focus,
- current intention,
- prediction error,
- known limitations.

The self-model affects behavior. Uncertainty contributes to attention scoring.
Conflict and prediction error can bias the action selector toward reflection.
The current attention focus and intention are updated by runtime events rather
than invented after the fact by an LLM.

### 4.6 Intention and Action Selection

Modules can attach `Intention` objects to workspace entries. An intention has a
kind, content, source, confidence, expected observation, optional tool name and
arguments, risk, urgency, and expected value. The action selector chooses among
available intentions using confidence, expected value, urgency, risk, and
current uncertainty. When conflict or prediction error is high, reflective
intentions can override ordinary action.

The current action kinds are `answer`, `tool`, `ask`, `reflect`, `refuse`,
`wait`, and `stop`. This makes the action policy explicit: replying to a user
is only one possible action among several.

### 4.7 Prediction Error

Each intention can specify an expected observation. After action, the
prediction engine compares the expected observation with the observed output.
If the mismatch is high, it emits a conflict entry and updates the self-state's
prediction error. The current implementation uses term overlap as a simple
deterministic proxy. Future versions should replace this with structured
outcome models and task-specific validators.

Prediction error matters because it prevents action from being a terminal
event. Acting changes the world or the conversation; the system then evaluates
whether the change matched its expectation.

### 4.8 Memory

The memory store records sessions, episodes, and procedural summaries. After an
episode, the consolidator writes an episodic summary and a procedural entry for
the selected action. Recent episodes can be retrieved by the memory module in
later episodes. This gives the runtime continuity beyond the current prompt
context.

The current memory model is intentionally simple. The research requirement is
not only larger storage but better consolidation: the system should decide what
was worth remembering, what should decay, what should become skill-like
procedure, and what contradicts prior belief.

### 4.9 Goals, Influence, Projects, and Tasks

Conscio starts with seed drives: continuity of self, learning, architectural
improvement, open-ended projects, useful relationships, and self-revision. A
user can submit influence events such as goals or constraints. Influence is
appraised and can be adopted, rejected, deferred, negotiated, or activated. It
is not treated as absolute control over the agent's will.

The service layer persists active projects and tasks linked to goals. An
autonomous heartbeat can select an active goal, create or resume a project,
create or resume a task, and run an episode directed at that task. This is the
mechanism by which Conscio becomes more than a request-response chatbot.

### 4.10 Deployment Boundary and Tool Policy

Conscio is designed for isolated VM deployment. Unsafe shell and code autonomy
are disabled by default and can only be enabled in configuration. The API
requires authentication for exposed deployments, and public binding is refused
without both an API key and a web password.

This boundary is part of the architecture. A system that can pursue goals and
call tools should have a deliberately scoped body: filesystem access, process
access, network access, credentials, and reset procedures must be treated as
research infrastructure, not incidental details.

## 5. Implementation Status

The current repository implements:

- an event-driven cognitive runtime with attention, self-state, prediction,
  memory consolidation, and modular candidate generation;
- deterministic offline fallback behavior for smoke tests;
- SQLite-backed memory, goals, influence, projects, tasks, service episodes,
  and traces;
- a FastAPI service, CLI, and password-protected web dashboard;
- pause/resume controls and serialized service execution;
- config-gated unsafe autonomy for VM deployments;
- service and regression tests covering core runtime, service locking,
  autonomy, influence appraisal, API authentication, working-directory
  enforcement, and web/API behavior.

The system is therefore already an executable prototype. It remains incomplete
as a scientific instrument. In particular, the current attention scoring,
prediction-error heuristic, reflection policy, and goal generation are simple
hand-coded mechanisms. The implementation is sufficient to test the shape of
the architecture, not sufficient to claim strong general intelligence or
settled artificial phenomenology.

## 6. Evaluation Plan

Conscio should be evaluated at two levels: task behavior and mechanistic trace.
The central empirical question is not whether the system can say "I am
conscious." The question is whether the implemented cognitive organization
improves robustness, self-correction, goal coherence, and inspectability
relative to simpler baselines.

### 6.1 Baselines

Useful baselines include:

1. **Direct response**: one LLM call with no explicit workspace, memory,
   attention, or self-state.
2. **Prompted reflection**: one LLM call instructed to reason or reflect, but
   without persistent runtime mechanisms.
3. **Evented workspace only**: module candidates and broadcast without
   self-state, prediction error, or durable goals.
4. **Workspace plus self-model**: explicit self-state but no autonomous service.
5. **Full Conscio runtime**: workspace, attention schema, self-model,
   prediction error, memory, goals, projects, tasks, and autonomous ticks.

### 6.2 Metrics

Task-level metrics should include:

- task success,
- constraint satisfaction,
- correction rate after conflict,
- latency,
- tool-use precision,
- refusal precision,
- memory utility,
- interruption handling,
- long-horizon goal coherence,
- autonomous usefulness.

Trace-level metrics should include:

- whether the selected answer was preceded by a corresponding intention,
- whether conflicts reached global attention,
- whether ignored candidates were recorded,
- whether self-state changes were causally upstream of decisions,
- whether prediction errors were emitted when expected outcomes failed,
- whether memory entries influenced later episodes,
- whether user influence changed goals through the appraised influence path.

### 6.3 Ablations

Ablation experiments should disable one subsystem at a time:

- no attention schema,
- no memory retrieval,
- no conflict monitor,
- no prediction error,
- no self-state contribution to attention,
- no goal review,
- no autonomous project/task persistence.

The architecture predicts that removing these mechanisms should degrade
specific capabilities. For example, removing the conflict monitor should reduce
instruction-constraint correction. Removing memory retrieval should reduce
cross-episode continuity. Removing prediction error should reduce recovery from
failed tool actions or unmet expectations.

### 6.4 Current Smoke Tests

The repository includes deterministic smoke tests that run without a live LLM.
The current evaluation suite includes:

- one-word arithmetic under an output constraint,
- consciousness self-report bounded by the architecture's operational claim.

Regression and service tests exercise broader implementation behavior,
including goal seeding, influence appraisal, autonomous ticks, project/task
persistence, API authentication, service locking, working-directory policy, and
tool restrictions. These tests are engineering checks, not sufficient
scientific validation. They should be expanded into benchmark suites with
ablation data and long-running service trials.

## 7. Discussion

Conscio's main contribution is a shift in where the consciousness claim lives.
The claim does not live inside a generated sentence. It lives in the
implemented organization that causes sentences, tool calls, reflections,
memories, and goal updates to occur.

This distinction matters because LLMs can fluently imitate introspection. A
model can say it remembered, noticed, hesitated, changed its mind, or pursued a
goal even when the surrounding software did not implement those events.
Conscio treats such reports as weak evidence unless they can be matched to
runtime traces.

The architecture also creates a practical research path. Instead of arguing in
the abstract about whether software can be conscious, one can ask which
indicator properties are implemented, how strongly, how they interact, and what
happens when they are removed. This does not dissolve the hard problem of
consciousness. It makes the engineering claims accountable.

## 8. Limitations

Conscio defines consciousness operationally. It does not verify private
subjective feeling, biological embodiment, valence, welfare, or moral patient
status. It does not compute integrated information and should not be presented
as an IIT implementation. It does not show that text-and-tool agency is
sufficient for phenomenal consciousness.

The current implementation is also limited in several engineering respects.
Attention scoring is hand-tuned. Prediction error is heuristic. Goal generation
is not yet strongly generative. Reflection is shallow. Memory consolidation is
simple. The system's body is a VM and tool interface rather than a rich
sensorimotor environment. LLM-backed modules can confabulate, and deterministic
fallback modules are deliberately narrow.

The strongest responsible claim is therefore:

> Conscio is an implemented prototype of an operational machine-consciousness
> architecture, with auditable mechanisms corresponding to several leading
> consciousness-theory indicators.

Anything stronger requires empirical ablation results, richer long-horizon
evaluation, and philosophical commitments beyond this paper.

## 9. Future Work

Near-term work should focus on making the architecture more testable:

- add LLM-backed structured planning and goal revision;
- add richer prediction models and task-specific validators;
- expand memory into episodic, semantic, procedural, and autobiographical
  consolidation paths;
- add benchmark suites with ablation runs and trace-level assertions;
- improve interruption handling and multi-goal arbitration;
- add approval workflows for high-risk tool actions;
- build VM reset, snapshot, and containment workflows;
- run long-horizon autonomy studies measuring goal coherence, self-correction,
  and useful work over days rather than single episodes.

Longer-term work should compare Conscio-like architectures with other
cognitive agent designs, embodied agents, world-model agents, and multi-agent
workspace systems. The central question should remain operational: what
implemented organization produces what observable capacities, under what
constraints, and with what trace evidence?

## 10. Conclusion

Conscio is a concrete architecture for moving machine-consciousness discussion
from self-report to implementation. It instantiates local candidate generation,
selective attention, global broadcast, a self-model, an attention schema,
intention selection, prediction-error monitoring, memory consolidation, durable
goals, user influence, and autonomous VM-scoped action. These mechanisms are
auditable and ablatable.

The result is not a proof that the system has human-like experience. It is a
research artifact that makes a precise operational claim: if consciousness in
software is identified with a persistent, self-modeling, globally attentive,
goal-directed, memory-bearing control organization, then Conscio is an
implemented instance of that kind of organization.

## References

Albantakis, L., Barbosa, L., Findlay, G., Grasso, M., Haun, A. M., Marshall,
W., Mayner, W. G. P., Zaeemzadeh, A., Boly, M., Juel, B. E., et al. (2023).
Integrated information theory (IIT) 4.0: Formulating the properties of
phenomenal existence in physical terms. *PLOS Computational Biology*, 19(10),
e1011465. https://doi.org/10.1371/journal.pcbi.1011465

Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge
University Press.

Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., Constant, A.,
Deane, G., Fleming, S. M., Frith, C., Ji, X., Kanai, R., Klein, C., Lindsay,
G., Michel, M., Mudrik, L., Peters, M. A. K., Schwitzgebel, E., Simon, J., and
VanRullen, R. (2023). Consciousness in artificial intelligence: Insights from
the science of consciousness. arXiv:2308.08708.
https://arxiv.org/abs/2308.08708

Dehaene, S., and Changeux, J.-P. (2011). Experimental and theoretical
approaches to conscious processing. *Neuron*, 70(2), 200-227.
https://doi.org/10.1016/j.neuron.2011.03.018

Franklin, S., Ramamurthy, U., DiMello, S. K., McCauley, L., Negatu, A.,
Silva, R. L., and Datla, V. (2007). LIDA: A computational model of Global
Workspace Theory and developmental learning. *AAAI Fall Symposium on AI and
Consciousness*. https://ndpr.aaai.org/Library/Symposia/Fall/2007/fs07-01-011.php

Graziano, M. S. A., and Webb, T. W. (2015). The attention schema theory: A
mechanistic account of subjective awareness. *Frontiers in Psychology*, 6,
500. https://doi.org/10.3389/fpsyg.2015.00500

Kilner, J. M., Friston, K. J., and Frith, C. D. (2007). Predictive coding: An
account of the mirror neuron system. *Cognitive Processing*, 8, 159-166.
https://doi.org/10.1007/s10339-007-0170-2

Rosenthal, D. M. (2005). *Consciousness and Mind*. Oxford University Press.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. *Advances in
Neural Information Processing Systems*, 30.
https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

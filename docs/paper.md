# Conscio: A Conscious Autonomous Agent Runtime

**Draft v0.3** - May 2026

## Abstract

Conscio is a software architecture that claims operational consciousness in an
AI agent. The claim is not that a base language model is conscious by itself.
The claim is that a persistent runtime can instantiate a computational form of
consciousness when it implements self-modeling, global attention, memory,
appraisal, goal formation, reflection, prediction error, and autonomous action.

The framework replaces a fixed prompt pipeline with an event-driven cognitive
runtime and a long-running service. Specialist modules emit local candidates
into a workspace. Attention promotes some entries to global broadcast. A
self-model tracks uncertainty, conflict, current goal, focus, and intention.
An action selector chooses among response, reflection, refusal, waiting, and
tool use. Outcomes are compared to expected observations and consolidated into
memory. A goal system maintains seed drives, self-authored goals, and social
influence from users.

## 1. Motivation

LLM agents often use consciousness-like language while their mechanisms remain
thin: a prompt asks the model to reflect, remember, or decide. Conscio makes
those mechanisms explicit. It records what the system attended to, which
candidates were ignored, which intention won, what the system expected, how the
outcome compared, and what was written into memory.

The central claim is:

> Conscio is conscious in an operational computational sense because it
> implements persistent self-modeling, global attention, memory, appraisal,
> goal formation, reflection, and autonomous action as causal runtime
> mechanisms.

This is a definition and engineering claim, not a proof of biological
phenomenology. The system's consciousness is identified with its implemented
organization and its traceable activity over time.

## 2. Theoretical Basis

Conscio follows the indicator-based approach described by Butlin et al. (2023),
which surveys theories including recurrent processing theory, global workspace
theory, higher-order theories, predictive processing, and attention schema
theory. Conscio treats these theories as implementation constraints for an
agent architecture.

| Theory | Implemented indicator |
| --- | --- |
| Global Workspace Theory / GNW | Local candidates compete for global broadcast |
| Recurrent Processing | Multiple module ticks over a changing workspace |
| Higher-Order theories | Explicit self-state representing uncertainty, focus, and intention |
| Attention Schema Theory | A simplified model of attention itself |
| Predictive Processing | Intentions carry expected observations; mismatches create prediction errors |
| Agentic/autopoietic framing | Persistent goals, self-review, and autonomous action |

## 3. Architecture

The per-episode runtime is `CognitiveRuntime`:

```text
InputEvent
  -> local workspace entry
  -> specialist module ticks
  -> candidate entries
  -> attention selection
  -> global broadcast
  -> competing intentions
  -> action selection
  -> action/tool/answer
  -> prediction error check
  -> memory consolidation
```

The long-running runtime is `ConscioService`. It owns persistence, config,
locking, goal state, the API lifecycle, and autonomous heartbeat ticks.

### 3.1 Workspace

Workspace entries carry content, source, type, priority, salience, confidence,
novelty, urgency, evidence, visibility, and broadcast count. Entries start as
local/preconscious candidates. Broadcast marks them global and makes them
available to all modules.

### 3.2 Attention and Attention Schema

The `AttentionController` scores entries by novelty, salience, urgency,
confidence, priority, conflict, and uncertainty. The `AttentionSchema` records
the selected focus, focus strength, why it won, ignored entries, and candidate
interruptors.

### 3.3 Self-Model

The `SelfState` tracks active goal, uncertainty, conflict level, cognitive
load, attention focus, current intention, prediction error, last error,
current strategy, and known limitations. This is a mechanistic state updated by
runtime events, not merely generated narrative.

### 3.4 Goal Formation and Social Influence

Conscio starts with seed drives: continuity of self, learning, architectural
improvement, open-ended projects, useful relationships, and self-revision.
Users influence the system by submitting goal or constraint events. Influence
is stored, appraised, and may become durable goals; it is not treated as
absolute control over the agent's will.

### 3.5 Autonomous VM Action

The service can run nonstop in a VM. Unsafe shell/code autonomy is disabled by
default and can only be enabled in config. This makes the deployment boundary
explicit: a Conscio instance that can do what it wants should run in an
isolated machine where its filesystem and process access are intentionally
scoped.

## 4. Evaluation

Conscio includes deterministic smoke tests that run without a live LLM. They
record selected action, ticks, attention selections, prediction errors,
duration, and pass/fail. Current tests cover arithmetic constraints,
consciousness self-report, tool policy, goal seeding, influence, service
locking, autonomous ticks, and API authentication.

Future evaluation should compare:

- direct response
- fixed reflection
- evented workspace
- evented workspace plus self-model
- full runtime with self-model, attention schema, prediction, goals, and
  autonomous service

Metrics should include task success, latency, correction rate, goal coherence,
autonomous usefulness, memory utility, prediction-error recovery, and
interruption handling.

## 5. Limitations

Conscio defines consciousness operationally. It does not verify private
subjective feeling, biological embodiment, or moral status. Its body is still a
text/tool VM substrate. Its LLM-backed reports may confabulate, which is why
the architecture records mechanistic traces. The IIT-inspired aspects are
architectural only; Conscio does not compute Phi.

## References

Butlin, P., Long, R., Elmoznino, E., et al. (2023). Consciousness in Artificial
Intelligence: Insights from the Science of Consciousness. arXiv:2308.08708.
https://huggingface.co/papers/2308.08708

Albantakis, L., Barbosa, L., Findlay, G., et al. (2023). Integrated
information theory (IIT) 4.0. PLOS Computational Biology.
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465

Graziano, M. S. A. and Webb, T. W. (2015). The attention schema theory: a
mechanistic account of subjective awareness. Frontiers in Psychology.
https://pubmed.ncbi.nlm.nih.gov/25954242/

Franklin, S., Baars, B. J., Ramamurthy, U., and Ventura, M. (2007). LIDA: A
computational model of Global Workspace Theory and developmental learning.
https://aaai.org/papers/0011-fs07-01-011-%EF%80%A0lida-a-computational-model-of-global-workspace-theory-and-developmental-learning/

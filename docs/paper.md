# Conscio: An Auditable Harness for Consciousness-Inspired Agent Architectures

**Draft v0.2** — May 2026

## Abstract

Conscio is a software harness for implementing and evaluating computational
indicators associated with scientific theories of consciousness in AI agents.
It does not claim that large language models are conscious. Instead, it asks a
more tractable engineering question: if an agent runtime explicitly implements
global broadcast, recurrent processing, self-modeling, attention schema,
prediction error, competing intentions, and memory consolidation, what changes
in its behavior and observability?

The framework replaces a fixed prompt pipeline with an event-driven cognitive
runtime. Specialist modules emit local candidate entries into a workspace.
Selective attention promotes a subset to global broadcast. A self-model and
attention schema track uncertainty, conflict, current focus, ignored content,
and prediction error. An action selector chooses among answer, tool use, ask,
reflect, refuse, wait, and stop intentions. Outcomes are compared to expected
observations and consolidated into memory.

## 1. Motivation

Current LLM agents often contain consciousness-like language: they "reflect,"
"remember," or "decide." In most systems these are prompt conventions rather
than explicit mechanisms. Conscio makes the control mechanisms inspectable.
The harness records what the system attended to, which candidates were ignored,
which intention won action selection, and whether the outcome matched the
system's prediction.

The research claim is deliberately limited:

> Conscio implements and measures computational indicators associated with
> scientific theories of consciousness. It does not establish phenomenal
> consciousness, subjective experience, or moral status.

## 2. Theoretical Basis

Conscio follows the indicator-based approach described by Butlin et al. (2023),
which surveys theories including recurrent processing theory, global workspace
theory, higher-order theories, predictive processing, and attention schema
theory. The report concludes that current systems are not clearly conscious,
but also that there is no obvious technical barrier to building systems that
satisfy more proposed indicators.

The framework maps theories to implementation features:

| Theory | Implemented indicator |
| --- | --- |
| Global Workspace Theory / GNW | Local candidates compete for global broadcast |
| Recurrent Processing | Multiple module ticks over a changing workspace |
| Higher-Order theories | Explicit self-state representing uncertainty, focus, and intention |
| Attention Schema Theory | A simplified model of attention itself |
| Predictive Processing | Intentions carry expected observations; mismatches create prediction errors |
| IIT-inspired integration | Shared recurrent workspace and traceable cross-module influence |

## 3. Architecture

The primary runtime is `CognitiveRuntime`. It processes an `InputEvent` as a
cognitive episode:

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

### 3.1 Workspace

Workspace entries carry content, source, type, priority, salience, confidence,
novelty, urgency, evidence, visibility, and broadcast count. Entries start as
local/preconscious candidates. Broadcast marks them global and makes them
available to all modules.

### 3.2 Attention and Attention Schema

The `AttentionController` scores entries by novelty, salience, urgency,
confidence, priority, conflict, and current uncertainty. The `AttentionSchema`
records the selected focus, focus strength, why it won, which entries were
ignored, and which ignored entries could interrupt the current focus.

### 3.3 Self-Model

The `SelfState` tracks active goal, uncertainty, conflict level, cognitive
load, attention focus, current intention, prediction error, last error, current
strategy, and known limitations. This is a mechanistic state, not a generated
self-report.

### 3.4 Specialist Modules

Default modules include observer, memory retrieval, response proposal, tool
proposal, constraint monitoring, and reflection. Modules do not directly act;
they emit candidates and intentions. Action selection decides what to do.

### 3.5 Prediction and Consolidation

Intentions include expected observations. After acting, the prediction engine
compares expected and observed outcomes. Large mismatches become conflict
entries. The memory consolidator writes episodic summaries and procedural
skill traces to SQLite.

## 4. Evaluation

Conscio includes a deterministic smoke evaluation suite that runs without a
live LLM. It records selected action, ticks, attention selections, prediction
errors, duration, and pass/fail. The initial suite checks instruction
constraints and the boundary between cognitive architecture and consciousness
self-claims.

Future evaluation should compare ablations:

- direct response
- fixed reflection
- evented workspace
- evented workspace plus self-model
- full runtime with self-model, attention schema, prediction, and consolidation

Metrics should include task success, latency, cost, correction rate,
constraint-following, prediction-error recovery, memory utility, and confidence
calibration.

## 5. Limitations

Conscio cannot verify subjective experience. It is text- and tool-situated,
not biologically embodied. The LLM can still produce confabulated self-reports,
which is why the framework separates mechanistic trace from narrative output.
The IIT-inspired aspects are architectural only; Conscio does not compute Phi.
Daemon mode is intentionally conservative and dry-run oriented until action
policies are substantially stronger.

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

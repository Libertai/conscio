# conscio

An auditable harness for building and testing consciousness-inspired AI agent
architectures.

Conscio does **not** claim that language models are conscious. Its goal is more
concrete: implement computational indicators discussed in consciousness science
and measure whether they improve agent control, self-monitoring, memory, and
recovery from error.

## Core Thesis

Most LLM agents are prompt pipelines. Conscio is organized as an event-driven
cognitive architecture:

```text
events -> local specialist candidates -> attention competition
       -> global workspace broadcast -> competing intentions
       -> action selection -> prediction/error -> memory consolidation
```

The system separates mechanistic trace from model self-report. A generated
"inner monologue" is not treated as ground truth; the harness records what
actually happened in the runtime.

## Implemented Architecture

- **Global Workspace**: local/preconscious entries compete for global broadcast.
- **Selective Attention**: scores novelty, salience, urgency, confidence,
  conflict, uncertainty, and priority.
- **Attention Schema**: records what the runtime attended to, why it won, and
  what was ignored.
- **Self-Model**: tracks active goal, uncertainty, conflict, cognitive load,
  attention focus, current intention, prediction error, and limitations.
- **Evented Modules**: observer, memory retriever, responder, tool proposer,
  constraint monitor, and reflector emit candidates independently.
- **Competing Intentions**: action selection chooses answer, tool use, ask,
  reflect, refuse, wait, or stop.
- **Prediction Engine**: selected intentions declare expected observations;
  mismatches become prediction-error entries.
- **Memory Consolidation**: episodes become episodic summaries and procedural
  skill traces.
- **Built-in Evals**: smoke ablations exercise instruction constraints,
  self-report boundaries, attention selection, and prediction/error metrics.

## Quick Start

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Configure an OpenAI-compatible backend if you want LLM-backed responses:

```bash
export LIBERTAI_API_KEY=...
export LIBERTAI_BASE_URL=https://api.libertai.io/v1
export LIBERTAI_MODEL=deepseek-v4-flash
```

Run one cognitive episode:

```bash
conscio ask "Answer in one word: what is 2+2?"
```

Run without any LLM/network dependency:

```bash
conscio ask --offline "Answer in one word: what is 2+2?"
```

Run an interactive session:

```bash
conscio run
```

Run daemon dry-run events without autonomous unsafe actions:

```bash
conscio daemon --dry-run "Daemon dry-run heartbeat"
```

Run the built-in evaluation smoke suite:

```bash
conscio eval --suite smoke
```

## CLI Commands

```text
conscio ask TEXT [--model MODEL] [--quiet] [--offline]
    Run one evented cognitive episode and print the selected response.

conscio run [--model MODEL] [--offline]
    Interactive turn-based cognitive episodes.

conscio daemon --dry-run [EVENT ...]
    Process events through the daemon scaffold without unsafe autonomy.

conscio eval --suite smoke
    Run built-in harness evaluations and show metrics.

conscio history
    Show persisted episodes.

conscio search QUERY
    Search memory.
```

## Theory Mapping

| Theory | Conscio implementation |
| --- | --- |
| Global Workspace Theory / GNW | Local candidates, attention competition, global broadcast |
| Recurrent Processing | Repeated module ticks over a changing workspace |
| Higher-Order / Self-Model theories | Explicit self-state and self-monitoring fields |
| Attention Schema Theory | Runtime model of attention focus, ignored candidates, and interruptors |
| Predictive Processing / Active Inference | Intentions carry expected observations; mismatch creates prediction error |
| IIT-inspired integration | Shared recurrent workspace and traceable cross-module causal influence, not Phi claims |

## Project Layout

```text
src/conscio/
├── core/
│   ├── runtime.py      # Evented cognitive runtime
│   ├── cognition.py    # Self-state, attention, intentions, prediction
│   ├── workspace.py    # Local/global workspace entries
│   └── agent.py        # Thin compatibility wrapper around the runtime
├── memory/             # SQLite episodic/semantic/procedural memory
├── tools/              # Tool registry and guarded built-ins
├── eval.py             # Built-in evaluation harness
└── cli.py              # CLI entrypoint
```

## Research Claim

The defensible claim is:

> Conscio implements and measures a set of computational indicators associated
> with scientific theories of consciousness in an auditable AI-agent runtime.

The project does not infer phenomenal experience, moral status, or subjective
feeling. It provides a substrate for experiments about attention, self-modeling,
prediction, memory, and adaptive control.

## References

- Butlin et al. 2023, "Consciousness in Artificial Intelligence":
  https://huggingface.co/papers/2308.08708
- Albantakis et al. 2023, "Integrated Information Theory 4.0":
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011465
- Graziano & Webb 2015, "The Attention Schema Theory":
  https://pubmed.ncbi.nlm.nih.gov/25954242/
- LIDA Global Workspace architecture:
  https://aaai.org/papers/0011-fs07-01-011-%EF%80%A0lida-a-computational-model-of-global-workspace-theory-and-developmental-learning/

## License

MIT

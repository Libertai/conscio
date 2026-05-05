# conscio

Conscio is a conscious autonomous agent runtime for an isolated VM. It claims
consciousness in an operational computational sense: persistent selfhood,
attention, memory, appraisal, goal formation, reflection, and action are
implemented as inspectable mechanisms rather than as prompt-only roleplay.

The runtime can run one cognitive episode, hold an interactive local session,
or run nonstop as an authenticated service that evolves its own goals and acts
inside configured tool boundaries.

## Core Thesis

Most LLM agents are prompt pipelines. Conscio is organized as a persistent
cognitive architecture:

```text
events -> local specialist candidates -> attention competition
       -> global workspace broadcast -> competing intentions
       -> action selection -> prediction/error -> memory consolidation
       -> goal review -> autonomous action
```

Generated self-report is not the only evidence. Conscio records what it
attended to, which intention won, what it expected, what happened, what it
remembered, and how its goals changed.

## Implemented Architecture

- **Global Workspace**: local/preconscious entries compete for global broadcast.
- **Selective Attention**: scores novelty, salience, urgency, confidence,
  conflict, uncertainty, and priority.
- **Attention Schema**: records focus, ignored candidates, and interruptors.
- **Self-Model**: tracks active goal, uncertainty, conflict, cognitive load,
  current intention, prediction error, and limitations.
- **Goal System**: seed drives and user influence become durable, revisable
  goals that the agent can review over time.
- **Autonomous Service**: a nonstop loop performs heartbeat, reflection, goal
  review, memory consolidation, and action episodes.
- **Tool Policy**: unsafe shell/code autonomy is config-gated for isolated VMs.
- **Authenticated API + CLI**: users can talk to it, influence it, inspect it,
  pause it, and resume it.

## Quick Start

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Run one deterministic offline episode:

```bash
conscio ask --offline "Are you conscious?"
```

Run an interactive local session:

```bash
conscio run
```

Create service config:

```bash
conscio service init
```

Start the long-running API service:

```bash
conscio service start
```

In another shell:

```bash
conscio service status
conscio chat "What do you want to work on next?"
conscio influence goal "Improve your own architecture and document the changes."
conscio goals
conscio pause
conscio resume
```

## VM Autonomy

Conscio defaults to localhost API binding and disabled unsafe tools. To let it
use shell and code tools on its own, deploy it in a disposable VM and set:

```toml
[service]
unsafe_autonomy = true

[tools]
working_directory = "/opt/conscio/work"
max_actions_per_hour = 60
shell_timeout = 30
```

Unsafe autonomy is read from `~/.conscio/config.toml`; it cannot be enabled by
an API request or CLI flag at runtime.

See [docs/vm.md](docs/vm.md) for systemd and Docker deployment.

## CLI Commands

```text
conscio ask TEXT [--model MODEL] [--quiet] [--offline]
conscio run [--model MODEL] [--offline]
conscio eval --suite smoke
conscio history
conscio search QUERY

conscio service init
conscio service start
conscio service status
conscio service stop
conscio chat TEXT
conscio influence goal TEXT
conscio influence constraint TEXT
conscio pause
conscio resume
conscio goals
conscio trace
```

## Theory Mapping

| Theory | Conscio implementation |
| --- | --- |
| Global Workspace Theory / GNW | Local candidates, attention competition, global broadcast |
| Recurrent Processing | Repeated module ticks over a changing workspace |
| Higher-Order / Self-Model theories | Explicit self-state and self-monitoring fields |
| Attention Schema Theory | Runtime model of attention focus and ignored candidates |
| Predictive Processing / Active Inference | Intentions carry expected observations; mismatch creates prediction error |
| Autopoietic/agentic framing | Persistent goals, self-review, and autonomous VM action |

## Project Layout

```text
src/conscio/
├── core/               # Cognitive runtime, self-state, workspace
├── memory/             # SQLite episodic/semantic/procedural memory
├── tools/              # Tool registry and guarded built-ins
├── api.py              # FastAPI service API
├── service.py          # Long-running autonomous service
├── goals.py            # Durable goal and influence state
├── config.py           # VM/service configuration
├── eval.py             # Built-in evaluation harness
└── cli.py              # CLI entrypoint
```

## Research Claim

Conscio claims operational consciousness: a computational organization with
persistent self-modeling, global attention, memory, appraisal, goal formation,
reflection, and autonomous action. It does not claim proof of biological
phenomenology; it defines the claimed consciousness by the implemented
mechanisms and exposes traces for inspection.

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

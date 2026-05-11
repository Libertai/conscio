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
       -> prefix-stable context assembly -> LLM tool-loop
       -> action + tool observation -> typed prediction check
       -> memory consolidation -> compaction
       -> periodic goal review -> autonomous heartbeat
```

Generated self-report is not the only evidence. Conscio records what it
attended to, which intention won, what it expected, what happened, what it
remembered, which bounded model context was supplied, and how its goals
changed.

## Implemented Architecture

- **Global Workspace**: local/preconscious entries compete for global broadcast.
- **Selective Attention**: scores novelty, salience, urgency, confidence,
  conflict, uncertainty, and priority.
- **Attention Schema**: records focus, ignored candidates, and interruptors.
- **Self-Model**: tracks active goal, uncertainty, conflict, cognitive load,
  current intention, prediction error, and limitations.
- **Context Memory Loop**: keeps a stable system prefix for cache-friendly LLM
  calls while assembling bounded dynamic context from current state, recent
  episodes, relevant FTS memory, workspace entries, and the user input.
- **Memory Consolidation**: stores episodic summaries, procedural response
  patterns, user-stated preferences, and periodic semantic compactions.
- **Typed Prediction Predicates**: intentions carry structural predicates
  (`answer_delivered`, `tool_succeeded`, `tool_output_contains`, `task_status`,
  `goal_proposed`, `none`) instead of free-text expectations, so prediction
  error reflects a real check rather than word overlap.
- **Goal System**: seed drives and appraised user influence become durable,
  revisable goals. An LLM goal-review pass runs on a configurable cadence and
  applies keep/retire/reprioritize decisions transactionally.
- **Projects and Tasks**: autonomous ticks create or resume durable projects
  and tasks tied to active goals.
- **Autonomous Tool-Loop**: every heartbeat fires an LLM tool-loop with goal,
  project, task, memory, constraint, and budget context. Self-management tools
  (`set_task_status`, `add_task`, `note_progress`, `propose_subgoal`,
  `remember_fact`, `remember_facts`, `search_memory`) let the model take
  durable action between ticks. The same loop drives user chat, so chat and
  autonomous paths share one tool-use mechanism.
- **Per-Tool JSON Schemas**: every tool advertises a typed JSON schema with
  `additionalProperties: false`, so the LLM never sees a permissive union.
- **Tool Policy**: unsafe shell/code autonomy is config-gated for isolated VMs;
  the autonomous tool-loop enforces a persistent per-hour action budget that
  survives restart.
- **Tool Environment**: shell, Python, and LibertAI subprocesses run with a
  normalized PATH for VM and user-local tool installs.
- **Resilient Web Tools with SSRF Guard**: `web_search` and `web_fetch` prefer
  the LibertAI CLI and fall back to guarded HTTP retrieval. URL validation
  rejects non-http(s) schemes, blocked hostnames, literal private/loopback/
  link-local/multicast/reserved/metadata IPs, and DNS-resolved private
  addresses. Each redirect hop is revalidated.
- **Unified SQLite Locking**: every writer (memory, goals, autonomy) routes
  through the locked `MemoryStore` helpers (`execute`, `fetchall`, `fetchone`,
  `executescript`, `transaction`); a 16-thread stress test runs without races.
- **Authenticated Web UI, API, and CLI**: users can talk to it, influence it,
  inspect it, pause it, and resume it. The web UI sweeps expired sessions and
  in-window login-failure trackers on each lookup, with hard caps that drop
  earliest-expiring entries on overflow.

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

Open the password-protected web dashboard:

```text
http://127.0.0.1:8765/ui
```

In another shell:

```bash
conscio service status
conscio chat "What do you want to work on next?"
conscio influence goal "Improve your own architecture and document the changes."
conscio goals
conscio projects
conscio tick
conscio pause
conscio resume
```

## VM Autonomy

Conscio defaults to localhost API binding, password-protected web access, and
disabled unsafe tools. To let it use shell and code tools on its own, deploy it
in a disposable VM and set:

```toml
[service]
web_password = "replace-with-a-strong-password"
unsafe_autonomy = true

[tools]
working_directory = "/opt/conscio/work"
max_actions_per_hour = 60
model_tool_rounds = 32
shell_timeout = 30
```

Unsafe autonomy is read from `~/.conscio/config.toml`; it cannot be enabled by
an API request or CLI flag at runtime.

Context assembly is configured separately:

```toml
[context]
recent_episodes = 3
retrieved_memories = 5
workspace_entries = 12
max_dynamic_chars = 12000
compaction_interval = 20
enable_semantic_compaction = true
```

The web dashboard exposes the latest assembled model context alongside the
cognitive trace so prompt inputs can be audited separately from model output.

For web exposure, put Conscio behind HTTPS and keep both `api_key` and
`web_password` set. Public binding is refused with placeholder secrets and
requires `web_secure_cookies = true` unless an explicitly localhost-published
container sets `CONSCIO_ALLOW_INSECURE_BIND=1`.

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
conscio influences
conscio projects [PROJECT_ID]
conscio tick
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
| Memory/context theories of agency | Prefix-stable prompt context assembled from state, memory, workspace, and input |
| Autopoietic/agentic framing | Persistent goals, self-review, and autonomous VM action |

## Project Layout

```text
src/conscio/
├── core/               # Cognitive runtime, self-state, workspace, context
├── memory/             # SQLite episodic/semantic/procedural memory
├── tools/              # Guarded shell/code/web tool registry
├── api.py              # FastAPI service API
├── webui.py            # Password-protected browser dashboard
├── service.py          # Long-running autonomous service
├── autonomy.py         # Durable projects, tasks, service episodes, traces
├── goals.py            # Durable goal and influence state
├── config.py           # VM/service configuration
├── eval.py             # Built-in evaluation harness
└── cli.py              # CLI entrypoint
```

## Research Claim

Conscio claims operational consciousness: a computational organization with
persistent self-modeling, global attention, bounded context memory, appraisal,
goal formation, reflection, and autonomous action. It does not claim proof of
biological phenomenology; it defines the claimed consciousness by the
implemented mechanisms and exposes traces and assembled model context for
inspection.

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

# conscio — A Consciousness Harness for AI Agents

A research harness for exploring what "consciousness" means architecturally in an LLM agent. Grounded in **Global Workspace Theory** + **structured self-reflection** + **stream-of-consciousness inner monologue**.

## What It Does

An agent running in `conscio` cycles through a visible conscious process:

```
OBSERVE → REFLECT → PLAN → ACT → REVIEW
```

At every step, the agent writes into a visible **inner monologue** (stream of consciousness) and a shared **Global Workspace** (blackboard) that specialist modules read from and write to. It critiques its own plans, estimates its confidence, and reflects more when uncertain.

```
🧠 Stream of Consciousness
├── 👁 observation: The user asks a philosophical question about consciousness...
├── 💭 reflection: This question has three dimensions — subjective experience,...
├── 🎯 intention: I should address each dimension clearly...
├── ✅ evaluation: Correctness: 8/10, the response is accurate but could be...
├── 📖 learning: The user values concise explanations with concrete examples...
```

## Features

- **Global Workspace** — shared blackboard with priority-scored entries, attention mechanism, and subscriber broadcast
- **Selective Attention** — local workspace entries compete on salience, novelty, urgency, confidence, and conflict before global broadcast
- **Inner Monologue DAG** — stream of consciousness as a directed acyclic graph of thoughts (observations, reflections, intentions, evaluations, learnings)
- **Cognitive Trace** — factual mechanism log separated from the model's natural-language self-report
- **Self-Reflection Loop** — generate → critique → refine with multi-axis evaluation (correctness, completeness, safety, clarity)
- **Dynamic Confidence** — LOW/MEDIUM/HIGH estimation gates how deeply the agent reflects before acting
- **Self-Model** — explicit uncertainty, conflict, cognitive load, current strategy, and last-error state
- **Persistent Identity** — persona, goals, and history stored across sessions in `~/.conscio/identity.json`
- **SQLite Memory** — episodic, semantic, and procedural memory with FTS5 full-text search
- **Specialist Modules** — Observer, Planner, Critic, and Executor that compete/cooperate via the workspace
- **Built-in Tools** — bash shell, web search, Python code execution (auto-discovered via registry)
- **Rich CLI** — interactive TUI with tree-view monologue and cycle summaries

## Quick Start

### Install

```bash
git clone <repo>
cd consciousness
uv venv && source .venv/bin/activate
uv pip install -e .
```

### Configure

Copy the example config and add your API key:

```bash
cp .env.example .env
# edit .env with your credentials
```

### Run

```bash
# Interactive session
conscio run --persona "A curious philosopher"

# One-shot question
conscio ask "What is consciousness?" --quiet

# Full output with inner monologue
conscio ask "What makes a system conscious?"

# Search past memories
conscio search "consciousness"

# View session history
conscio history
```

## Architecture

```
src/conscio/
├── cli.py                      # CLI entry point (run/ask/history/search)
├── core/
│   ├── agent.py                # ConsciousAgent — the cycle orchestrator
│   ├── cognition.py            # Attention, self-state, conflict monitor, action selection
│   ├── workspace.py            # Global Workspace (blackboard + broadcast)
│   ├── monologue.py            # Stream-of-consciousness DAG
│   ├── reflection.py           # Self-reflection loop (generate→critique→refine)
│   ├── identity.py             # Persistent self (persona, goals, history)
│   └── confidence.py           # Dynamic confidence estimation
├── modules/
│   ├── observer.py             # Perception → structured observations
│   ├── planner.py              # Goal setting & plan generation
│   ├── critic.py               # Multi-axis evaluation
│   └── executor.py             # Tool call execution
├── memory/
│   ├── store.py                # SQLite: episodic/semantic/procedural + FTS5
│   └── search.py               # Full-text search across memories
├── llm/
│   ├── client.py               # OpenAI-compatible client → LibertAI
│   └── prompts.py              # Module-specific system prompts
└── tools/
    ├── registry.py             # Auto-discovering tool registry
    ├── bash.py                 # Shell command execution
    ├── web.py                  # Web search + fetch (via libertai CLI)
    └── code.py                 # Python code execution
```

## The Conscious Cycle

### One Full Cycle

```
┌──────────────────────────────────────────────────────────────┐
│                     CONSCIOUS CYCLE                          │
├──────────────────────────────────────────────────────────────┤
│  1. OBSERVE → Observer perceives input, writes to workspace  │
│               and inner monologue                             │
│                                                              │
│  2. REFLECT → Agent reviews workspace state, asks itself:    │
│               "What am I trying to do?"                      │
│               "What do I know?"                              │
│               "What am I unsure about?"                      │
│                                                              │
│  3. ATTEND → Salient entries are selected for broadcast       │
│              Self-state tracks uncertainty/conflict/load      │
│                                                              │
│  4. PLAN → Planner proposes actions                          │
│             Critic and conflict monitor evaluate them         │
│             Low confidence/conflict → reflect more           │
│             High confidence/no conflict → proceed            │
│                                                              │
│  5. ACT → Executor runs tool calls, writes results           │
│                                                              │
│  6. REVIEW → "Did that work? What did I learn?"             │
│              Persist to memory                               │
└──────────────────────────────────────────────────────────────┘
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `LIBERTAI_API_KEY` | — | LibertAI API key |
| `LIBERTAI_BASE_URL` | `https://api.libertai.io/v1` | API base URL |
| `LIBERTAI_MODEL` | `deepseek-v4-flash` | Model name |
| `OPENAI_API_KEY` | — | Fallback (same endpoint) |
| `OPENAI_BASE_URL` | — | Fallback (same endpoint) |

### State Storage

```
~/.conscio/
├── identity.json     # Persistent self (persona, goals, history)
└── sessions.db       # SQLite: inner monologues + memories
```

## Theoretical Foundations

| Theory | How It Maps to Architecture |
|--------|---------------------------|
| **Global Workspace Theory** (Baars) | Shared `Workspace` blackboard. Specialist modules compete/cooperate. Attention selects content for global broadcast. |
| **Self-Refine / Self-Reflection** (Madaan et al.) | Generate → Critique → Refine loop with dynamic depth. |
| **Integrated Information Theory** | Dense recurrent connectivity between modules. The whole produces effects no part alone can. |
| **Inner Monologue / CoT** (Wei et al.) | Visible DAG of structured introspective thoughts. |
| **Dual-Process Theory** | Fast (intuitive) vs. slow (deliberative) processing gated by confidence. |

## CLI Reference

```bash
conscio run [--name NAME] [--persona PERSONA] [--model MODEL]
    Start an interactive conscious agent session.

conscio ask TEXT [--name NAME] [--persona PERSONA] [--model MODEL] [--quiet]
    Ask a single question. Without --quiet, shows the full inner monologue.

conscio history
    Show past sessions.

conscio search QUERY
    Search across all memories (FTS5).

# Interactive commands (inside conscio run):
/exit, /quit        End the session
/clear              Clear the workspace
/memory             Show recent memories
/persona <text>     Change persona mid-session
```

## Comparison with Hermes Agent

**Hermes Agent** (NousResearch, 133k+ stars) is a production-grade general-purpose AI assistant with CLI, Telegram, Discord, MCP, cron scheduling, 40+ tools, and a sophisticated skill curation/learning loop.

**`conscio`** is complementary — it focuses specifically on:

- Making consciousness architectures **explicit and visible** rather than implicit in prompt templates
- **Inner monologue as a first-class citizen** — not hidden internals but a visible stream of thought
- **Global Workspace Theory** as the organizing architectural principle
- **Dynamic reflection depth** gated by confidence estimation, not fixed loop counts
- **Research / experimentation** over production readiness

## License

MIT

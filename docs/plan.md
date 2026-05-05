# conscio — A Consciousness Harness for AI Agents

A research-inspired Python library + CLI for exploring what "consciousness" means
architecturally in an LLM agent. Grounded in **Global Workspace Theory** +
**structured self-reflection** + **stream-of-consciousness inner monologue**.

## Core Idea

An agent running in `conscio` cycles through:

```
OBSERVE → REFLECT → PLAN → ACT → REVIEW
```

At every step, the agent writes into a visible **inner monologue** (stream of
consciousness) and a shared **Global Workspace** (blackboard) that multiple
specialist sub-agents read from and write to.

## Theoretical Foundations

| Theory | How It Maps to Architecture |
|--------|---------------------------|
| **Global Workspace Theory** (Baars) | Shared `Workspace` blackboard. Specialist modules (Observer, Planner, Critic, Executor) compete/cooperate. An attention mechanism selects salient content for global broadcast. |
| **Self-Refine / Self-Reflection** (Madaan et al.) | Generate → Critique → Refine loop. Each cycle estimates confidence; low confidence triggers deeper reflection. |
| **Integrated Information Theory** | Dense recurrent connectivity between modules (not just feedforward). The whole produces effects no part alone can. |
| **Inner Monologue / CoT** (Wei et al.) | Visible DAG of structured introspective thoughts — observations, reflections, intentions, evaluations, learnings. |
| **Dual-Process Theory** | Fast (intuitive) vs. slow (deliberative) processing. Confidence estimates gate the switch. |

## Architecture

```
src/conscio/
├── __init__.py
├── cli.py                          # CLI entry point
├── core/
│   ├── __init__.py
│   ├── workspace.py                # Global Workspace (blackboard + broadcast)
│   ├── agent.py                    # ConsciousAgent lifecycle orchestrator
│   ├── monologue.py                # Stream-of-consciousness DAG
│   ├── reflection.py               # Self-reflection loop (generate→critique→refine)
│   ├── identity.py                 # Persistent self (persona, goals, history)
│   └── confidence.py               # Dynamic confidence estimation
├── modules/
│   ├── __init__.py
│   ├── observer.py                 # Perception/sensory input
│   ├── planner.py                  # Goal setting & planning
│   ├── critic.py                   # Self-critique against axes
│   └── executor.py                 # Action/tool execution
├── memory/
│   ├── __init__.py
│   ├── store.py                    # SQLite-backed persistent memory
│   └── search.py                   # FTS5 search across memories
├── llm/
│   ├── __init__.py
│   ├── client.py                   # OpenAI-compatible → LibertAI
│   └── prompts.py                  # Role-specific system prompts
└── tools/
    ├── __init__.py
    ├── bash.py                     # Shell execution
    ├── web.py                      # Web search/fetch
    └── code.py                     # Python code execution
```

## Key Components

### 1. LLM Client (`llm/client.py`)

Wraps OpenAI SDK pointed at LibertAI (`https://api.libertai.io/v1`).
Supports sync + streaming chat completions and tool-calling.

### 2. Global Workspace (`core/workspace.py`)

A shared blackboard that all modules can read from and write to.

- `WorkspaceEntry` — content + source + priority + timestamp + type
- `Workspace` — ordered list with `write()`, `read()`, `broadcast()`, `attend()`
- Modules `subscribe()` and receive broadcast notifications
- Attention mechanism filters by priority + recency + relevance

### 3. Inner Monologue (`core/monologue.py`)

A DAG of `ThoughtNode` objects representing the agent's stream of consciousness.

- Node types: `OBSERVATION`, `REFLECTION`, `INTENTION`, `EVALUATION`, `LEARNING`
- Each node: `id, parent_id, question, answer, timestamp, type`
- Persisted to SQLite across sessions
- Tree-walking: "how did I arrive at this conclusion?"

### 4. Self-Reflection (`core/reflection.py`)

The meta-cognition loop:

- `generate()` — produce candidate output
- `critique(output, axes)` — evaluate against correctness, completeness, safety, efficiency
- `refine(output, critique)` — improve based on feedback
- `estimate_confidence(output, critiques)` — LOW / MEDIUM / HIGH

### 5. Dynamic Confidence (`core/confidence.py`)

Gates the reflection depth:

- LOW → add more reflection cycles with sharper critique axes
- MEDIUM → one refinement then proceed
- HIGH → proceed directly to action

### 6. Persistent Identity (`core/identity.py`)

The agent's sense of self:

- `Identity(name, persona, goals, history)` — loaded from `~/.conscio/identity.json`
- Goal tiers: `core` (unchanging), `session` (this conversation), `ephemeral` (this turn)
- `evolve(outcome)` — identity updates goals based on outcomes

### 7. Memory (`memory/store.py`)

SQLite-backed long-term knowledge:

- `episodes` — cycle summaries (episodic memory)
- `semantic` — facts learned (semantic memory)
- `procedural` — how-to knowledge (procedural memory)
- FTS5 for full-text search across all memories

### 8. Specialist Modules (`modules/`)

- **Observer** — wraps user input and tool output into structured observations
- **Planner** — proposes courses of action based on workspace state
- **Critic** — evaluates plans/actions against specific axes
- **Executor** — carries out tool calls and returns results

## The Conscious Cycle

### One Full Cycle

```
┌──────────────────────────────────────────────────────────────┐
│                     CONSCIOUS CYCLE                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. OBSERVE ──→ Observer perceives input                     │
│       │         ├── writes observation to Workspace          │
│       │         └── records in Inner Monologue               │
│       │                                                      │
│  2. REFLECT ──→ Agent reviews own state from workspace       │
│       │         ├── "What am I trying to do?" (self-model)   │
│       │         ├── "What do I know?" (knowledge audit)      │
│       │         ├── "What am I unsure about?" (confidence)   │
│       │         └── All written to Inner Monologue           │
│       │                                                      │
│  3. PLAN ──→ Planner proposes course of action               │
│       │         ├── writes plan to Workspace                 │
│       │         ├── Critic evaluates plan against axes       │
│       │         │   ├── LOW confidence → reflect more        │
│       │         │   ├── MEDIUM → one refine then proceed     │
│       │         │   └── HIGH → proceed directly              │
│       │         └── Intentions written to Inner Monologue    │
│       │                                                      │
│  4. ACT ──→ Executor carries out plan (tool calls)           │
│       │         ├── writes results to Workspace              │
│       │         └── records outcomes in Inner Monologue      │
│       │                                                      │
│  5. REVIEW ──→ Agent reflects on outcome                    │
│             ├── "Did that work?" (outcome assessment)        │
│             ├── "What did I learn?" (knowledge consolidation)│
│             ├── "Should I update my approach?" (self-mod)    │
│             └── Write to persistent memory                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Dynamic Depth

The agent does NOT always complete all 5 stages. After each stage, it
estimates confidence:

- **HIGH confidence** → skip to ACT immediately (fast/intuitive path)
- **MEDIUM confidence** → proceed through all stages
- **LOW confidence** → loop back to REFLECT with sharper questions

This mirrors the dual-process theory of human cognition: routine tasks
follow the fast path; novel/complex tasks trigger deep deliberation.

## Implementation Order

| Step | What | Files |
|------|------|-------|
| 1 | Project scaffold + LLM client | `pyproject.toml`, `llm/client.py`, `llm/prompts.py` |
| 2 | Global Workspace | `core/workspace.py` |
| 3 | Inner Monologue | `core/monologue.py` |
| 4 | Persistent Identity | `core/identity.py` |
| 5 | Self-Reflection + Confidence | `core/reflection.py`, `core/confidence.py` |
| 6 | SQLite Memory | `memory/store.py`, `memory/search.py` |
| 7 | Specialist Modules | `modules/observer.py`, `planner.py`, `critic.py`, `executor.py` |
| 8 | Tools | `tools/bash.py`, `tools/web.py`, `tools/code.py` |
| 9 | Agent Orchestrator | `core/agent.py` |
| 10 | CLI + Demo | `cli.py`, demo script |

## Technology Choices

| Concern | Choice |
|---------|--------|
| Language | Python 3.11+ |
| LLM Backend | LibertAI (OpenAI-compatible, `https://api.libertai.io/v1`) |
| Model | `deepseek-v4-flash` |
| LLM SDK | `openai >= 1.0` |
| Persistence | `aiosqlite` with WAL mode |
| CLI | `rich` + `argparse` |
| Async | `asyncio` throughout |
| Package manager | `uv` |

## Example Usage

### CLI

```bash
# Interactive session (one question at a time)
conscio run --persona "curious philosopher"

# One-shot question
conscio ask "What is consciousness?" --persona "scientist"

# Show past sessions
conscio history

# Search memories
conscio search "what did I learn about attention?"
```

### Programmatic

```python
from conscio import ConsciousAgent

agent = ConsciousAgent(name="Socrates")
agent.observe("User asks: what makes a system conscious?")
response = await agent.cycle()  # Full conscious cycle
print(response.inner_monologue)  # Visible stream of thought
print(response.final_answer)     # The answer
```

## State Storage

```
~/.conscio/
├── identity.json          # Persistent self (persona, goals, history)
├── sessions.db            # SQLite: inner monologues + memories
└── config.toml            # User config (model, provider, etc.)
```

## Relationship to Hermes Agent

Hermes Agent (NousResearch) is a production-grade general-purpose coding/life
assistant with CLI, Telegram, Discord, MCP, cron scheduling, and 40+ tools.

`conscio` is complementary — it focuses specifically on:

- **Making consciousness architecturally explicit** rather than implicit in prompts
- **Visible inner monologue** as a first-class citizen, not hidden internal state
- **Global Workspace Theory** as the organizing principle, not just a tool loop
- **Dynamic reflection depth** gated by confidence estimation
- **Research / experimentation** over production readiness

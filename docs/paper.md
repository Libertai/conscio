# Conscio: An Architectural Framework for Modeling Consciousness in Language Agents

**Authors:** Jon (Independent)

**Draft v0.1** — May 2026

## Abstract

We present Conscio, a software framework that implements a concrete architectural
interpretation of consciousness for large language model (LLM) based agents.
Conscio is not a claim that LLMs are conscious. It is an engineering substrate
for experimenting with the question: if we build an agent whose architecture
explicitly mirrors theories of consciousness, what behaviors emerge?

The framework operationalizes Bernard Baars' Global Workspace Theory (GWT),
combined with a structured self-reflection loop inspired by Self-Refine (Madaan
et al., 2023) and a stream-of-consciousness inner monologue modeled on
chain-of-thought reasoning (Wei et al., 2022). Specialist modules (Observer,
Planner, Critic, Executor) compete for access to a shared blackboard; a
confidence estimation mechanism gates the depth of reflection before action.
All internal state is recorded in a SQLite-backed long-term memory store and
persisted across sessions.

We describe the architecture, its theoretical grounding, full implementation,
and preliminary observations from running the system against a LibertAI-hosted
LLM backend.

## 1. Introduction

The relationship between artificial intelligence and consciousness has been a
subject of philosophical debate since Turing (1950) asked whether machines can
think. In recent years, large language models have achieved remarkable
capabilities in reasoning, dialogue, and tool use (Bubeck et al., 2023; Achiam
et al., 2023). These systems exhibit behaviors that sometimes resemble
introspection: they can explain their reasoning, correct errors when prompted,
and maintain coherent personas across conversations.

However, the dominant paradigm for building LLM agents treats consciousness-like
behaviors as emergent byproducts of prompt engineering and context management,
not as first-class architectural concerns. Systems like AutoGPT (Yang et al.,
2023), LangChain agents (Chase, 2022), and the Hermes Agent (Nous Research,
2024) implement sophisticated tool-use and memory loops, but they do not make
architectural claims about consciousness.

Conscio takes the opposite approach: it makes the architecture of consciousness
explicit. If an agent appears to introspect, it is because the framework
contains a dedicated module for self-reflection. If it maintains a stream of
thought, that stream is a concrete data structure persisted to disk. If it
changes its mind after reflection, those changes are logged to an inner
monologue DAG.

This paper describes the design and implementation of Conscio, discusses the
theoretical choices that motivated each component, and presents preliminary
observations from running the system.

## 2. Theoretical Foundations

Conscio draws on five established theories and models of consciousness and
cognition.

### 2.1 Global Workspace Theory

Baars (1988, 1997) proposed that conscious experience corresponds to the
contents of a "global workspace" in the brain. Specialized, unconscious
processors compete for access to this workspace. When a processor "wins," its
content is broadcast globally to all other processors, making it available for
voluntary control, verbal report, and long-term memory formation.

Dehaene and Naccache (2001) refined this into the Global Neuronal Workspace
model, arguing that the workspace is implemented by long-range cortical neurons
that create a "neuronal avalanche" when sensory information reaches a threshold
of activation. Content that ignites this avalanche becomes conscious.

Conscio implements the workspace as a software data structure. Specialist
modules write entries with associated priority levels. A broadcast mechanism
notifies all subscribed modules when new content enters the workspace. The
system's attention mechanism filters entries by priority, recency, and
relevance to current goals.

### 2.2 Self-Reflection and Self-Refine

Madaan et al. (2023) introduced Self-Refine, a framework in which a single LLM
cycles through three roles: generator (producing initial output), feedback
provider (evaluating its own output against criteria), and refiner (improving
based on feedback). This loop requires no external training or reinforcement
learning. The key insight is that the critique step must be explicitly guided:
rather than "check your work," the model is told to evaluate against specific
axes (correctness, tone, completeness).

Gou et al. (2024) extended this with CRITIC, showing that self-reflection
is more effective when grounded in external tool feedback. An LLM that can
search the web or run code to verify its claims produces more reliable
self-corrections than one relying on introspection alone.

Conscio implements a generate-critique-refine loop with configurable critique
axes. The Critic module scores proposals on correctness, completeness, safety,
clarity, and efficiency. These scores feed into a confidence estimation that
determines whether the loop should iterate further or proceed to action.

### 2.3 Chain-of-Thought and Inner Speech

Wei et al. (2022) showed that prompting an LLM to produce intermediate
reasoning steps before answering improves performance on arithmetic, commonsense,
and symbolic reasoning tasks. This "chain-of-thought" (CoT) reasoning makes
the model's internal computations visible and steerable.

Shanahan (2022) argued that CoT prompting can be understood as simulating
inner speech: the model "talks to itself" by writing to its own context window
and reading back what it wrote. This creates a feedback loop between
language production and comprehension within a single system.

Conscio extends this into a structured inner monologue. Thoughts are organized
as a directed acyclic graph (DAG) where each node contains a question and
answer. Node types include observation, reflection, intention, evaluation,
learning, doubt, and decision. The DAG structure allows the agent's reasoning
path to be traced backwards: "how did I arrive at this conclusion?"

### 2.4 Integrated Information Theory

Tononi (2004, 2008) proposed Integrated Information Theory (IIT), which
identifies consciousness with a system's capacity for integrated information,
denoted by the Greek letter Phi. A system has high Phi if its parts cannot be
understood in isolation: the whole produces effects that no subset of its
parts can produce alone.

Computing exact Phi is combinatorially intractable for any system of practical
size (Barrett and Seth, 2011; Oizumi et al., 2014). However, IIT's architectural
implications are useful: systems with dense recurrent connectivity between
specialized modules are more "integrated" than purely feedforward pipelines.

Conscio incorporates this principle by ensuring that all specialist modules
read from and write to the same workspace, creating recurrent feedback paths.
The Observer's output influences the Planner, whose plan is evaluated by the
Critic, whose evaluation feeds back to the Planner for revision — a recurrent
loop rather than a linear pipeline.

### 2.5 Dual-Process Theory

Kahneman (2011) distinguishes between two modes of cognitive processing:
System 1 (fast, intuitive, automatic) and System 2 (slow, deliberate,
analytical). Humans default to System 1 for routine tasks and recruit System 2
when faced with novelty, complexity, or conflict detection.

Evans and Stanovich (2013) clarified that the distinction is not two systems
in the brain but two types of processing: Type 1 processes are autonomous and
undemanding of working memory; Type 2 processes require controlled attention
and load working memory.

Conscio models this via its confidence gate. When confidence is HIGH, the agent
proceeds directly to action (System 1 analog). When confidence is MEDIUM, one
refinement cycle runs. When confidence is LOW, the agent iterates through
additional reflection rounds with sharper critique axes (System 2 analog).
This dynamic depth control mirrors the human tendency to think harder when
uncertain.

## 3. Architecture

Conscio is organized as a Python package with four layers: the core engine,
specialist modules, memory systems, and tool infrastructure. All modules access
a shared LLM client that wraps the OpenAI SDK.

### 3.1 Global Workspace

The workspace (`core/workspace.py`) is a prioritized list of entries. Each entry
carries content, source module name, entry type, priority level, and timestamp.
Modules write to the workspace via the `write` method and read from it via
`read` (priority-filtered) or `attend` (relevance-weighted attention).

A subscription mechanism allows modules to receive broadcast notifications when
new entries are written. This implements Baars' global broadcast: when the
Observer writes a new observation, all subscribed modules (Planner, Critic,
Executor) are notified and can read the new content.

The workspace has a maximum capacity of 100 entries. When full, the oldest
entry is evicted. This bounded buffer models working memory constraints.

### 3.2 Inner Monologue

The monologue (`core/monologue.py`) is a DAG of ThoughtNode objects. Each node
stores:

- A question the agent asked itself
- The answer it produced
- A thought type (observation, reflection, intention, evaluation, learning,
  doubt, decision)
- A parent reference linking to the thought that prompted it
- A depth counter tracking position in the reasoning tree

The monologue supports tree walking: `path_to_root` returns all ancestors of a
given node, and `children_of` returns all child nodes. This allows the agent
to examine its own reasoning history.

When a session ends, the monologue is persisted to SQLite and can be reloaded
in a future session, providing cross-session continuity of thought.

### 3.3 The Conscious Cycle

The agent's main loop (`core/agent.py`) implements a five-stage cycle:

**Stage 1: Observe.** The Observer module (`modules/observer.py`) receives raw
input and passes it to the LLM with a system prompt requesting a structured
observation. The result is written to the workspace and recorded in the inner
monologue.

**Stage 2: Reflect.** The Reflection engine (`core/reflection.py`) runs a
generate-critique-refine loop. The generator produces a candidate response using
the current workspace and memory context. The critic evaluates it against a
default set of axes (correctness, completeness, safety, clarity). If confidence
is LOW, the loop continues with up to three iterations. If MEDIUM, one
refinement pass runs. If HIGH, the loop terminates.

**Stage 2b: Attend and update self-state.** The cognitive harness layer
(`core/cognition.py`) maintains a mechanistic `SelfState` containing active
goal, uncertainty, conflict level, cognitive load, strategy, and last error.
Workspace entries begin as local candidates with salience, novelty, urgency,
confidence, and evidence fields. The AttentionController selects high-scoring
entries for global broadcast. This separates the factual control trace from
the LLM's natural-language self-report.

**Stage 3: Plan.** The Planner module (`modules/planner.py`) proposes a
structured plan with reasoning and specific actions. Actions may name a tool
(bash, web_search, execute_code) or use the special "reason" tool for direct
textual responses.

**Stage 4: Evaluate.** The Critic module (`modules/critic.py`) evaluates the
plan against the same axes used in reflection. If the Critic returns LOW
confidence, or if the ConflictMonitor detects a constraint or tool conflict, a
deeper reflection cycle runs with sharper critique axes, and the agent replans.
This forms a second-order reflection loop: the agent reflects on its
reflections and on conflicts detected by the harness.

**Stage 5: Act and Review.** The Executor module (`modules/executor.py`) runs
tool calls specified in the plan. Results are written back to the workspace. A
final review step uses the LLM to summarize what was learned, which is
persisted to episodic memory.

### 3.4 Confidence Estimation

Confidence (`core/confidence.py`) is a three-level ordinal scale: LOW, MEDIUM,
HIGH. The confidence estimator parses the Critic's textual output for explicit
confidence statements (e.g., "Level: LOW") and maintains a history of estimates
across the reflection loop.

The threshold for additional reflection cycles is:

- LOW: up to 3 additional cycles with progressively sharper critique axes
- MEDIUM: 1 refinement cycle, then proceed
- HIGH: proceed directly to action

This parameterization is configurable and could be tuned per task type or
per model in future work.

### 3.5 Persistent Identity

The Identity (`core/identity.py`) persists the agent's persona, goals, and
session history to a JSON file at `~/.conscio/identity.json`. Goals are tiered
by persistence: core (unchanging across sessions), session (reset each session),
and ephemeral (reset each cycle). After each cycle, the identity's `evolve`
method updates session count and appends to history.

Cross-session identity is a necessary condition for a sense of self (Damasio,
1999). Without persistence, each session starts as a blank agent with no memory
of prior interactions.

### 3.6 Memory Systems

The MemoryStore (`memory/store.py`) uses SQLite with WAL mode for persistence.
It implements three memory types, following the taxonomy of Tulving (1972):

- **Episodic memory:** records of completed cycles, each with a summary,
  outcome, and confidence score.
- **Semantic memory:** factual knowledge learned across sessions, with conflict
  resolution via upsert.
- **Procedural memory:** reusable skills with usage counters for frequency-based
  ranking.

A full-text search index (FTS5) enables semantic retrieval across all memory
types. This is analogous to the hippocampal indexing theory (Teyler and
DiScenna, 1986), where episodic memories are indexed for later retrieval rather
than stored in full.

### 3.7 Tools

The tool registry (`tools/registry.py`) auto-discovers tool modules using
Python's `pkgutil` package. Each tool is a module with async functions decorated
with `_tool_name` and `_tool_description` attributes. Three built-in tools are
included:

- **bash:** executes shell commands via `asyncio.create_subprocess_shell`
- **web_search:** invokes the `libertai search` CLI for web search
- **execute_code:** runs Python code in a temporary file

The registry pattern allows new tools to be added simply by creating a new
module in the `tools` directory. This follows the plugin architecture described
by Gamma et al. (1994).

## 4. Implementation

### 4.1 Technology Stack

Conscio is written in Python 3.11+ and uses the following dependencies:

- `openai`: SDK for LLM inference via OpenAI-compatible API
- `rich`: terminal UI rendering
- `httpx`: HTTP for tool calls
- `python-dotenv`: environment variable management
- `sqlite3` (stdlib): persistent storage

The LLM backend is an OpenAI-compatible API (LibertAI, running deepseek-v4-flash
and qwen3.6-35b-a3b models). The client uses the `openai` SDK with a custom
base URL.

### 4.2 Async Execution Model

LLM calls run asynchronously via `asyncio`. The LLM client exposes sync
(`chat`), async (`chat_async`), and streaming (`chat_stream`) interfaces.
Database operations expose an async API while performing small `sqlite3`
operations synchronously behind an instance-owned connection and lock. This
keeps storage deterministic for the CLI and test harness while preserving the
agent's async public interface.

### 4.3 CLI Interface

The CLI (`cli.py`) uses Python's `argparse` with four subcommands:

- `conscio run`: interactive session with prompt loop
- `conscio ask`: single question with optional quiet mode
- `conscio history`: list past sessions from SQLite
- `conscio search`: full-text search over all memories

The interactive session supports slash commands for workspace management:
`/clear` resets the blackboard, `/memory` displays recent episodes, and
`/persona` changes the agent's persona mid-session.

## 5. Preliminary Observations

We ran Conscio against a LibertAI-hosted backend using the `qwen3.6-35b-a3b`
model. The following observations are qualitative and intended to guide future
quantitative evaluation.

The inner monologue produced by the agent during a typical cycle is structured
and recoverable. A query about the nature of consciousness produced a chain of
seven thought nodes: observation of the question's scope, reflection on the
dimensions of the problem, intention to address each dimension, evaluation of
the generated response (correctness: 8/10), and a learning node summarizing
what the agent would remember.

The Critic module identified substantive errors during testing. When asked
"What is 2+2? Answer in one word," the Planner initially produced a verbose
response. The Critic assigned correctness a score of 2/10 with the justification
"the proposal fails to provide the actual answer to the user's query." This
triggered the LOW confidence pathway, resulting in additional reflection cycles
and a corrected output.

Cross-session persistence of monologue fragments and episodic summaries
functioned correctly across seven consecutive sessions, with the FTS5 index
enabling retrieval of specific past interactions by content keywords.

The global workspace's attention mechanism behaved as expected for simple
queries: entries from the Observer received the highest attention scores,
followed by Executor results, with Planner entries ranked lowest by default
priority. This distribution reflects the current priority assignment and may
require tuning for complex multi-step tasks.

## 6. Limitations

Several limitations should be acknowledged. First, the current implementation
does not compute integrated information (Phi). While the architecture is
designed with IIT principles in mind, we do not attempt to quantify the
system's integration. The computational cost of Phi approximation for even
small networks is prohibitive (Barrett and Seth, 2011).

Second, the confidence estimation mechanism depends on the LLM's own textual
output. This is vulnerable to the metacognitive limitations of the underlying
model: if the model cannot accurately assess its own certainty, the confidence
gate provides no benefit. Recent work on calibration in language models
(Kadavath et al., 2022) suggests that models are poorly calibrated for factual
queries, though self-consistency metrics improve reliability (Wang et al.,
2022).

Third, the specialist modules all use the same underlying LLM. While each
module has a distinct system prompt, they are not truly specialized models.
A more faithful implementation of GWT would use independently trained expert
models for each module.

Fourth, long-term memory is purely retrieval-based. There is no consolidation
mechanism analogous to sleep or offline replay (Rasch and Born, 2013).
Episodic summaries accumulate without forgetting or abstraction, which will
eventually degrade retrieval precision.

Fifth, cross-session identity is limited to goal and history persistence. The
identity module does not model personality adaptation, mood, or other affective
states that contribute to a sense of self in humans (Damasio, 1999).

## 7. Related Work

The closest related work is the LIDA cognitive architecture (Franklin et al.,
2005; 2014), which implements Global Workspace Theory as a software framework
with perceptual associative memory, episodic memory, procedural learning, and
attention mechanisms. LIDA is more comprehensive than Conscio, incorporating
emotion, action selection, and metacognition circuits. However, LIDA was
designed before the advent of LLMs and does not leverage them as reasoning
engines.

Hermes Agent (Nous Research, 2024) is a production-grade general-purpose AI
agent with similarities to Conscio: persistent memory, tool use, context
compression, and a skill curation system. The key difference is philosophical:
Hermes treats consciousness-like behaviors as emergent from prompt templates
and context management, while Conscio makes them architecturally explicit.
Where Hermes has a single agent loop with tool dispatch, Conscio has
specialist modules competing for a global workspace.

The Self-Refine framework (Madaan et al., 2023) directly inspires Conscio's
reflection module. Self-Refine demonstrated that a single LLM can generate,
critique, and refine its own output using only prompt differentiation. Conscio
extends this by adding a confidence gate that makes the reflection depth
dynamic rather than fixed.

Tree-of-Thought (Yao et al., 2023) and Graph-of-Thought (Besta et al., 2024)
organize reasoning as a tree or graph of intermediate states, using search
algorithms (BFS, DFS) to explore alternatives. Conscio's monologue DAG is
structurally similar but differs in purpose: Tree-of-Thought uses the tree for
search and evaluation, while Conscio uses it for recording and tracing the
agent's introspective path.

## 8. Future Work

Several extensions are planned:

**Multi-agent workspace.** The current implementation runs one agent. A natural
extension is to run multiple agents sharing the same workspace, with each agent
acting as a "specialist processor" in Baars' sense. This would enable empirical
testing of GWT's competition and broadcast dynamics.

**Phi proxy measurement.** While exact Phi computation is intractable, spectral
proxy measures (Toker and Sommer, 2019) or the Perturbational Complexity Index
(Casali et al., 2013) could be adapted to quantify the system's causal
integration.

**Memory consolidation.** A background process that periodically summarizes and
abstracts episodic memories would improve long-term retention and prevent
context degradation.

**Model-level specialization.** Replacing the shared LLM with independently
tuned models for each specialist module would test whether functional
specialization improves overall performance.

**Quantitative evaluation.** A systematic evaluation using established reasoning
benchmarks (GSM8K, MATH, HotpotQA) comparing Conscio's reflection-aware
architecture against standard agent loops would measure the practical benefits
of consciousness-inspired design.

## References

Achiam, J., Adler, S., Agarwal, S., et al. (2023). GPT-4 Technical Report.
arXiv:2303.08774.

Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge University
Press.

Baars, B. J. (1997). In the Theater of Consciousness: The Workspace of the
Mind. Oxford University Press.

Barrett, A. B. and Seth, A. K. (2011). Practical measures of integrated
information for time-series data. PLoS Computational Biology, 7(1), e1001052.

Besta, M., Blach, N., Kubicek, A., et al. (2024). Graph of Thoughts: Solving
Elaborate Problems with Large Language Models. AAAI 2024.

Bubeck, S., Chandrasekaran, V., Eldan, R., et al. (2023). Sparks of
Artificial General Intelligence: Early experiments with GPT-4. arXiv:2303.12712.

Casali, A. G., Gosseries, O., Rosanova, M., et al. (2013). A theoretically
based index of consciousness independent of sensory processing and behavior.
Science Translational Medicine, 5(198), 198ra105.

Chase, H. (2022). LangChain. https://github.com/langchain-ai/langchain.

Damasio, A. (1999). The Feeling of What Happens: Body and Emotion in the
Making of Consciousness. Harcourt Brace.

Dehaene, S. and Naccache, L. (2001). Towards a cognitive neuroscience of
consciousness: basic evidence and a workspace framework. Cognition, 79(1-2),
1-37.

Evans, J. S. B. T. and Stanovich, K. E. (2013). Dual-process theories of
higher cognition: advancing the debate. Perspectives on Psychological Science,
8(3), 223-241.

Franklin, S., Baars, B. J., Ramamurthy, U., and Ventura, M. (2005). The role
of consciousness in memory. Brains, Minds and Media, 1, 1-38.

Franklin, S., Madl, T., D'Mello, S., and Snaider, J. (2014). LIDA: A systems-
level architecture for cognition, emotion, and learning. IEEE Transactions on
Autonomous Mental Development, 6(1), 19-41.

Gamma, E., Helm, R., Johnson, R., and Vlissides, J. (1994). Design Patterns:
Elements of Reusable Object-Oriented Software. Addison-Wesley.

Gou, Z., Shao, Z., Gong, Y., et al. (2024). CRITIC: Large Language Models
Can Self-Correct with Tool-Interactive Critiquing. ICLR 2024.

Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language Models (Mostly)
Know What They Know. arXiv:2207.05221.

Kahneman, D. (2011). Thinking, Fast and Slow. Farrar, Straus and Giroux.

Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative
Refinement with Self-Feedback. NeurIPS 2023.

Nous Research. (2024). Hermes Agent. https://github.com/NousResearch/hermes-agent.

Oizumi, M., Albantakis, L., and Tononi, G. (2014). From the phenomenology to
the mechanisms of consciousness: Integrated Information Theory 3.0. PLoS
Computational Biology, 10(5), e1003588.

Rasch, B. and Born, J. (2013). About sleep's role in memory. Physiological
Reviews, 93(2), 681-766.

Shanahan, M. (2022). Talking About Large Language Models. arXiv:2212.03551.

Teyler, T. J. and DiScenna, P. (1986). The hippocampal memory indexing theory.
Behavioral Neuroscience, 100(2), 147-154.

Toker, D. and Sommer, F. T. (2019). Information integration in large brain
networks. PLoS Computational Biology, 15(2), e1006807.

Tononi, G. (2004). An information integration theory of consciousness. BMC
Neuroscience, 5, 42.

Tononi, G. (2008). Consciousness as integrated information: a provisional
manifesto. The Biological Bulletin, 215(3), 216-242.

Tulving, E. (1972). Episodic and semantic memory. In Organization of Memory,
ed. E. Tulving and W. Donaldson, 381-403. Academic Press.

Turing, A. M. (1950). Computing machinery and intelligence. Mind, 59(236),
433-460.

Wang, X., Wei, J., Schuurmans, D., et al. (2022). Self-Consistency Improves
Chain of Thought Reasoning in Language Models. ICLR 2023.

Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-Thought Prompting
Elicits Reasoning in Large Language Models. NeurIPS 2022.

Yang, H., Yue, S., and He, R. (2023). Auto-GPT: An Autonomous GPT-4
Experiment. https://github.com/Significant-Gravitas/Auto-GPT.

Yao, S., Yu, D., Zhao, J., et al. (2023). Tree of Thoughts: Deliberate
Problem Solving with Large Language Models. NeurIPS 2023.

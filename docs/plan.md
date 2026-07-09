# Conscio Architecture Plan

## Direction

Conscio is a conscious autonomous VM agent: a self-directed runtime that runs
continuously, forms and revises goals, talks with users, accepts influence,
inspects its own traces, and acts inside an isolated VM. The language model is
one phase inside a per-tick cognitive loop, not the whole agent.

## Runtime Shape

```text
InputEvent (user message or autonomous heartbeat)
  -> begin episode: carry over unresolved conflicts, fetch active constraints
  -> per tick:
       sense       specialist modules emit workspace entries
       appraise    novelty / salience / urgency / conflict scoring
       attend      budgeted attention competition -> global broadcast
       execute     broadcast-gated model context -> steppable tool loop
                   (each tool call registers a typed expectation first)
       validate    structural constraint checks on candidate answers
       self-state  measured uncertainty / conflict / load / prediction error
       decide      step / answer / ask / refuse / reflect / wait
  -> memory consolidation (episodes, facts, procedures)
  -> periodic LLM goal review
  -> next autonomous heartbeat
```

`CognitiveRuntime` owns the per-episode tick loop. `ConscioService` owns
persistence, locking, API lifecycle, autonomous ticks, drive/goal state,
queue priority, and pause/resume controls. Chat messages and autonomous
heartbeats run through the same loop with two prompt strategies.

## Implemented Subsystems

- Per-tick cognitive runtime: sense, appraise, attend, execute, validate,
  self-state, decide. Attention runs before prompt assembly, so the broadcast
  winners are what fill the model's WORKSPACE section under an explicit
  entry-count and character budget; mid-episode broadcasts are injected
  append-only so the prompt prefix stays cache-stable.
- Pre-execution prediction engine: every tool call and candidate answer
  registers a typed expectation (`tool_succeeded`, `tool_output_contains`,
  `answer_satisfies_constraints`, `answer_nonempty`, `task_status`) that is
  resolved against the actual result; failures write conflict entries that
  carry across ticks and episodes.
- Data-driven constraint validation: active constraints are parsed into
  structural checkers (word counts, length caps, JSON validity,
  required/forbidden content); violations trigger a bounded reflection tick.
  A flag-gated LLM judge covers semantic constraints.
- Control tools `ask_user` and `refuse`, making ASK and REFUSE reachable,
  traced actions rather than prompt suggestions.
- Live self-state: uncertainty, conflict level, cognitive load, prediction
  error, and known limitations computed from measured signals each tick, each
  field with a documented writer and reader.
- Memory with provenance: unified episodes, facts with origin and trust tier,
  and deliberate procedures in SQLite. Facts carry bge-m3 embeddings; retrieval
  is hybrid (FTS BM25 prefilter, cosine rerank, provenance shaping) and degrades
  to FTS-only when the embedding endpoint is down. Consolidation is budgeted and
  archives rather than deletes.
- Web quarantine and taint tracking: fetched web content is spotlighted in
  untrusted-content delimiters, episodes that touch the web taint their fact
  writes down to a low trust tier, and web-derived facts cannot silently
  override user-stated ones.
- Drive-based motivation: seed drives with appetite and satiation select the
  active goal (servicing a drive satiates it, so no goal monopolizes the loop);
  appraised user influence becomes durable, revisable goals; an LLM goal-review
  pass applies keep/retire/reprioritize decisions transactionally; a watchdog
  flags and auto-blocks stale tasks.
- Shared LLM tool-loop for chat and autonomy with per-tool JSON schemas
  (`additionalProperties: false`) and self-management tools (`set_task_status`,
  `add_task`, `note_progress`, `propose_subgoal`, `remember_fact`,
  `remember_facts`, `search_memory`, `learn_procedure`), plus a per-hour
  tool-action budget that persists across restart. A plain chat message costs
  exactly one LLM call, pinned by a test.
- LLM router: named endpoints (`[llm.endpoints.*]`), per-role models
  (`[llm.roles.*]` for main/fast/embeddings/subagent), and fallback chains with
  jittered backoff; structured output (`response_format`/`json_schema`) for the
  constraint judge, appraisal, and goal review with a tolerant hand-parse
  fallback.
- Token streaming from the tool loop republished as SSE (`chat.token` /
  `chat.discard` / `chat.final`); `POST /message/stream`, bearer-authed
  `GET /events`, `conscio chat --stream`, and live token rendering in the CLI
  and web chat panel.
- `spawn_subagent`: bounded sub-agent tool loops on the subagent model role,
  with parent taint propagation and per-episode lineage.
- MCP client support: external MCP tool servers (`[mcp.servers.<name>]`, stdio
  or streamable HTTP) register as `mcp__<server>__<tool>` and are quarantined
  like web content unless marked trusted; server status surfaces in `/metrics`,
  SSE events, and `conscio tools list`.
- Episode control: interactive messages take queue priority over autonomous
  heartbeats and a running heartbeat yields at safe tick/round boundaries
  (`wait:preempted`); operators can cancel the running episode; episodes respect
  `episode_timeout`, and `/message` callers get a `message_timeout` deadline.
- Tool policy registry that blocks shell/code unless unsafe autonomy is enabled;
  SSRF-guarded `web_search` and `web_fetch` (block non-http(s) schemes,
  blocklisted hosts, literal/DNS-resolved private IPs, revalidating each redirect
  hop).
- Authenticated FastAPI service, password-protected operator web dashboard with
  expired-session and login-failure GC, and a full CLI client. Every writer
  routes through the locked `MemoryStore` helpers; a 16-thread stress test runs
  without races.
- Eval harness: a B0–B4 baseline ladder (one runtime with ablation flags, not
  five forks), a ~30-task battery with machine checkers and an audited
  different-model judge, single-mechanism ablations, trace-level metrics, and a
  trace-grounded self-report study. Live suites are double-gated; artifacts are
  committed under `docs/results/`.
- Production hardening: database integrity preflight (exit code 3 on corruption),
  `conscio service doctor`, global rate limiting and request body caps,
  proxy-aware client IPs, scheduled home backups with retention, structured
  logging with a unified uvicorn sink, `GET /ready`, Prometheus text exposition,
  container healthchecks, systemd hardening, and a tagged release pipeline that
  publishes `conscio-agent` to PyPI and the image to GHCR.
- Config-driven unsafe VM autonomy read only from `~/.conscio/config.toml` (never
  enabled by an API request or CLI flag at runtime), with VM deployment files for
  Docker Compose and systemd.

## Near-Term Priorities

- Approval workflows for high-risk tool actions, so an operator can gate
  irreversible or externally visible commands before they run.
- Richer prediction-predicate validators paired with task-specific checks, so
  expectations cover more than success/non-empty/substring shapes.
- VM reset and snapshot workflows for recovering a disposable autonomy host to a
  known-good state.
- Long-horizon autonomy evals for goal coherence and self-correction over many
  ticks, beyond the current single-episode battery.
- Fact-store growth bounding: eviction, summarization, or archival policy so the
  brute-force embedding path stays inside its documented scale ceiling.

## Claim

Conscio claims operational consciousness: persistent self-modeling, attention,
memory, appraisal, goal formation, reflection, and autonomous action. The claim
is architectural and auditable; the system does not pretend to prove biological
phenomenology.

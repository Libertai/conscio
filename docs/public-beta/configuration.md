# Configuration

The service reads TOML from `CONSCIO_CONFIG` or `~/.conscio/config.toml`.
`conscio service init` writes a complete default config with generated secrets.

## Service

```toml
[service]
home = "/home/conscio/.conscio"
host = "127.0.0.1"
port = 8765
client_url = ""
api_key = "replace-with-a-long-random-token"
web_password = "replace-with-a-long-random-password"
web_secure_cookies = false
allow_insecure_public_bind = false
autonomous = true
tick_interval = 30
consolidation_interval = 20
enable_contradiction_check = false
unsafe_autonomy = false
pause_on_error = true
backup_interval_hours = 24
backup_retain = 14
trusted_proxies = []
max_request_bytes = 262144
episode_rate_per_minute = 30
episode_rate_burst = 10
log_level = "INFO"
log_format = "text"
log_file = ""
http_access_log = false
episode_timeout = 600
message_timeout = 300
```

Operational notes:

- `home` contains `state.db`, `service.lock`, logs, sessions, events, and
  approvals.
- `api_key` protects `/status`, `/message`, `/trace`, and the other service API
  endpoints with `Authorization: Bearer ...`.
- `web_password` protects `/ui` and `/ui/api/...`.
- `unsafe_autonomy` gates the `bash` and `execute_code` tools.
- `pause_on_error` pauses autonomous action after service-level processing
  errors.
- `episode_timeout` hard wall-clock cap per episode in seconds (0 disables).
- `message_timeout` how long HTTP callers wait for `/message` before a 504 (0 disables).
- `backup_interval_hours`/`backup_retain` schedule home backups and retention; `conscio db prune` prunes manually.
- `trusted_proxies` enables proxy-header (X-Forwarded-For) client IPs; set to `["127.0.0.1"]` behind Caddy so login throttling sees real client IPs. Env override `CONSCIO_TRUSTED_PROXIES` (comma-separated).
- `max_request_bytes` caps HTTP bodies (413); `episode_rate_per_minute`/`episode_rate_burst` rate-limit episode-triggering endpoints (429, in-process, resets on restart).
- `log_format = "json"` emits one JSON object per line for log shipping; `log_file` adds a rotating file sink; `http_access_log` enables uvicorn access logs (off by default — the reverse proxy is the edge log). Env override `CONSCIO_LOG_LEVEL`.

## Agent Profile

```toml
[agent]
profile = "research"
premises = ""
external_side_effects = "policy"
```

The `[agent]` table sets the operating posture for a deployment.

- `profile` selects a named posture: `research` (the historic conservative
  defaults) or `autonomous_vm` (a dedicated premises VM where broad local
  agency is normal). Choosing `autonomous_vm` derives several defaults —
  `premises = "dedicated_vm"`, `external_side_effects = "mostly_free"`,
  `unsafe_autonomy = true`, and a `working_directory` under `/opt/conscio/work`
  — but explicit TOML values still override every derived default.
- `premises` is a free-text label the agent reads as part of its self-model
  context (empty by default under `research`).
- `external_side_effects` sets the policy stance for off-model side effects:
  `policy` (conservative) or `mostly_free` (permissive, paired with the
  `autonomous_vm` profile).

## Model Backend

```toml
[llm]
base_url = "https://api.libertai.io/v1"
api_key = ""
model = "deepseek-v4-flash"
timeout = 120
max_retries = 2
retry_backoff = 0.5
embedding_model = "bge-m3"
```

Environment fallbacks are `LIBERTAI_BASE_URL`, `LIBERTAI_API_KEY`,
`LIBERTAI_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.
`timeout` bounds each model call in seconds; `max_retries` is the per-call retry budget
the OpenAI-compatible SDK applies to transport-level failures. `retry_backoff` is the
base (seconds) for jittered exponential backoff between fallback attempts.
`embedding_model` selects the model used by memory embedding retrieval.

### Named endpoints and roles

For multi-provider or multi-model setups, define named endpoints under
`[llm.endpoints.<name>]` and assign roles under `[llm.roles.<role>]`. Endpoint
keys are `base_url`, `api_key`, `timeout`, `max_retries`, `response_format`, and
`tool_choice`; role keys are `endpoint`, `model`, `max_tokens`, and `fallback`.

```toml
[llm.endpoints.primary]
base_url = "https://api.libertai.io/v1"
api_key = ""
timeout = 120
max_retries = 2
response_format = "auto"   # auto | none | json_object | json_schema
tool_choice = true

[llm.endpoints.local]
base_url = "http://127.0.0.1:8080/v1"
api_key = ""
timeout = 60
max_retries = 1
response_format = "none"
tool_choice = true

[llm.roles.main]
endpoint = "primary"
model = "deepseek-v4-flash"
max_tokens = 2400
fallback = [{ endpoint = "local", model = "qwen3.6-27b" }]

[llm.roles.fast]
endpoint = "local"
model = "qwen3.6-27b"
```

`fallback` is an ordered list of `{ endpoint, model }` pairs tried after the
primary target on transport-class failures (connection, timeout, 429, 5xx); the
router walks the chain with jittered exponential backoff.

Roles select a model for a purpose: `main` drives the tool loop; `fast` handles
the constraint judge, LLM appraisal, consolidation, and goal review; `embeddings`
serves memory retrieval; `subagent` is reserved for spawned sub-tasks. Any role
left unset falls back to `main`.

## V3 Research Controls

These settings are opt-in and are intended for reproducible primary research,
not ordinary deployments:

```toml
[research]
strict_recurrent_workspace = true
require_pinned_language_model = true
language_provider = "local-openai-compatible"
language_endpoint_id = "research-node-a"
language_model_revision = "exact-upstream-revision"
language_weight_digest = "<lowercase-sha256>"
language_config_digest = "<lowercase-sha256>"
language_seed = 17
```

`strict_recurrent_workspace` removes legacy V2 modules, direct prompt access to
stored episodes, semantic retrieval, and V2 self-state, and language-facing
tools with `memory_read` or `memory_write` capabilities. The recurrent
specialists and their selected broadcasts remain available. Pinned language
mode also requires a configured endpoint, rejects a fallback chain on the
`main` role, and requires `constraint_judge`, `llm_appraisal`, and
`enable_contradiction_check` to remain disabled. It records canonical exact
requests/responses and immutable chat/autonomous manifests in the V3 causal
trace. Goal-review LLM calls and consolidation summarization are disabled in
this mode; deterministic consolidation maintenance still runs.

## Context, Engine, and Tools

```toml
[context]
recent_episodes = 3
retrieved_memories = 5
workspace_entries = 12
max_dynamic_chars = 12000
compaction_interval = 20
enable_semantic_compaction = true

[engine]
max_ticks = 8
tool_rounds_per_tick = 4
max_reflections = 2
attention_broadcast_limit = 6
attention_char_budget = 4000
chat_temperature = 0.4
autonomous_temperature = 0.3
judge_max_tokens = 200
appraisal_max_tokens = 400

[tools]
allowed = []
denied = []
max_actions_per_hour = 60
model_tool_rounds = 32
shell_timeout = 30
working_directory = "/opt/conscio/work"
```

Use `allowed` for an allowlist and `denied` for a blocklist. `allowed = []`
means all registered tools are eligible, subject to the unsafe-autonomy gate.

## Sub-Agents

```toml
[subagents]
enabled = true
max_rounds = 12
max_seconds = 120.0
deny_capabilities = ["self_modification", "memory_write", "self_management"]
```

`enabled` registers the `spawn_subagent` tool. `deny_capabilities` lists tool
capabilities sub-agents may never use; names and policy gates from `[tools]`
still apply on top.

## Motivation

```toml
[motivation]
w_priority = 0.35
w_appetite = 0.35
w_aging = 0.20
w_novelty = 0.10
aging_tau_seconds = 21600.0
satiate_step = 0.25
satiation_decay = 0.98
goal_dup_threshold = 0.88
stale_flag_days = 2.0
stale_block_days = 5.0
```

The `[motivation]` table tunes the DriveScheduler that ranks and selects goals.

- `w_priority`, `w_appetite`, `w_aging`, `w_novelty` weight the four drive
  components (stated priority, appetite/unsatisfied demand, aging since last
  activity, novelty) into the per-goal score; they need not sum to 1.
- `aging_tau_seconds` is the exponential-decay timescale for the aging
  component (larger → goals stay "fresh" longer).
- `satiate_step` is the satiation bump a goal takes when serviced, and
  `satiation_decay` is the per-tick decay back toward appetite.
- `goal_dup_threshold` is the cosine similarity above which a proposed goal
  is treated as a duplicate of an existing one.
- `stale_flag_days` is how long a goal stays untouched before it is flagged
  stale; `stale_block_days` is how long before it is blocked outright and
  must be > `stale_flag_days`.

## MCP Servers

Conscio can attach external MCP tool servers. Each `[mcp.servers.<name>]` table defines
one server; its tools appear to the agent as `mcp__<name>__<tool>` and obey the same
`[tools]` allow/deny policy and hourly action budget as built-in tools.

```toml
[mcp.servers.github]
transport = "stdio"        # "stdio" | "http"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_TOKEN = "..." }
enabled = true
trusted = false
allowed = []               # per-server tool allowlist (empty = all)
denied = []
call_timeout = 60.0
connect_timeout = 15.0

[mcp.servers.search]
transport = "http"
url = "https://mcp.example.com/mcp"
headers = { Authorization = "Bearer ..." }
```

By default MCP output is untrusted: it is spotlighted like fetched web content, taints
the episode, and any facts derived from it are stored at trust tier 1. Setting
`trusted = true` disables that quarantine — the server's output can then reach normal
agent-tier memory. Only mark a server trusted if you operate it yourself and consider it
part of the agent's premises.

## Ablation

```toml
[ablation]
attention_gating = true
memory_retrieval = true
prediction = true
reflection = true
self_state_coupling = true
appraisal = true
constraint_judge = false
llm_appraisal = false
```

The `[ablation]` table flips engine subsystems on and off, mainly to support the
eval ladder. The first six flags are the shared contract with the eval harness
(`attention_gating`, `memory_retrieval`, `prediction`, `reflection`,
`self_state_coupling`, `appraisal`) — their names must match the harness exactly.
The last two are core-only knobs the eval harness never toggles:

- `constraint_judge` enables the LLM judge that scores semantic constraints
  (off by default; the default constraint path is rule-based).
- `llm_appraisal` enables a batched LLM appraisal pass (off by default).

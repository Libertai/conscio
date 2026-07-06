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

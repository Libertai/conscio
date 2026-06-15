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

## Model Backend

```toml
[llm]
base_url = "https://api.libertai.io/v1"
api_key = ""
model = "deepseek-v4-flash"
```

Environment fallbacks are `LIBERTAI_BASE_URL`, `LIBERTAI_API_KEY`,
`LIBERTAI_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY`.

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

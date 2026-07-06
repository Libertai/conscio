#!/usr/bin/env sh
set -eu

fail() {
  printf 'docs smoke failed: %s\n' "$1" >&2
  exit 1
}

has() {
  file=$1
  pattern=$2
  grep -F -- "$pattern" "$file" >/dev/null || fail "missing '$pattern' in $file"
}

has_regex() {
  file=$1
  pattern=$2
  grep -E -- "$pattern" "$file" >/dev/null || fail "missing /$pattern/ in $file"
}

DOCS="
docs/index.md
docs/public-beta/quickstart.md
docs/public-beta/dedicated-vm.md
docs/public-beta/configuration.md
docs/public-beta/operations.md
docs/public-beta/tools.md
docs/public-beta/memory.md
docs/public-beta/api.md
docs/public-beta/troubleshooting.md
docs/launch/public-beta-launch.md
docs/launch/announcement.md
docs/launch/release-notes-public-beta.md
docs/launch/known-limits.md
docs/research/theory-and-references.md
docs/tutorials/install-and-run-local.md
docs/tutorials/model-backend.md
docs/tutorials/first-autonomous-vm.md
docs/tutorials/add-custom-tool.md
docs/tutorials/operator-console.md
docs/tutorials/memory-provenance.md
docs/tutorials/backup-restore.md
docs/tutorials/prompt-injection-drill.md
docs/runbooks/service-start-failure.md
docs/runbooks/model-backend-unreachable.md
docs/runbooks/empty-responses.md
docs/runbooks/db-locked-or-corrupted.md
docs/runbooks/excessive-tool-calls.md
docs/runbooks/task-spam.md
docs/runbooks/web-ui-auth.md
docs/runbooks/sse-disconnected.md
docs/runbooks/restore-backup.md
docs/runbooks/disable-dangerous-tool.md
docs/runbooks/bad-self-edit-or-memory.md
"

for doc in $DOCS; do
  [ -f "$doc" ] || fail "expected doc $doc"
done

# README should advertise the operator docs entry point.
has README.md "docs/index.md"
has README.md "docs/launch/"
has README.md "The consciousness layer for LLM agents"
has README.md "Make LLMs self-observing, goal-driven, and persistent"
has README.md "docs/assets/conscio-observatory.svg"
has README.md "## The Conscious Agent Runtime"
has README.md "## What Makes It Feel Alive"
has README.md "## Proof It Is More Than Vibes"
has README.md "## Run a Persistent Agent"
has README.md "Cognitive Trace"
has README.md "Attention Stream"
has README.md "Self-State"
has README.md "Memory Provenance"
has docs/index.md "launch/public-beta-launch.md"
has docs/index.md "research/theory-and-references.md"
has docs/launch/public-beta-launch.md "scripts/check-launch-readiness.sh"
[ -x scripts/check-launch-readiness.sh ] || fail "scripts/check-launch-readiness.sh must be executable"

# CLI examples in docs must correspond to real argparse command registrations.
for command in ask run history search daemon eval service db tools chat influence pause resume cancel goals influences projects tick trace; do
  has_regex src/conscio/cli.py "add_parser\\(\"$command\""
done

for command in init start status doctor stop; do
  has_regex src/conscio/cli.py "add_parser\\(\"$command\""
done

for command in schema migrate backup prune restore export import; do
  has_regex src/conscio/cli.py "add_parser\\(\"$command\""
done

for command in list deny allow; do
  has_regex src/conscio/cli.py "add_parser\\(\"$command\""
done

for command in goal constraint; do
  has_regex src/conscio/cli.py "add_parser\\(\"$command\""
done

# Config keys used in examples must exist in the config loader/default writer.
for key in \
  home host port client_url api_key web_password web_secure_cookies \
  allow_insecure_public_bind autonomous tick_interval consolidation_interval \
  enable_contradiction_check unsafe_autonomy pause_on_error base_url model \
  episode_timeout message_timeout \
  profile premises external_side_effects \
  recent_episodes retrieved_memories workspace_entries max_dynamic_chars \
  compaction_interval enable_semantic_compaction allowed denied \
  max_actions_per_hour model_tool_rounds shell_timeout working_directory \
  max_ticks tool_rounds_per_tick max_reflections attention_broadcast_limit \
  attention_char_budget timeout max_retries \
  retry_backoff embedding_model endpoint fallback response_format tool_choice \
  chat_temperature autonomous_temperature judge_max_tokens appraisal_max_tokens \
  enabled max_rounds max_seconds deny_capabilities \
  transport trusted call_timeout connect_timeout servers enabled \
  backup_interval_hours backup_retain trusted_proxies max_request_bytes episode_rate_per_minute episode_rate_burst \
  log_level log_format log_file http_access_log; do
  has src/conscio/config.py "$key"
done

for key in CONSCIO_CONFIG CONSCIO_HOST CONSCIO_PORT CONSCIO_CLIENT_URL CONSCIO_API_KEY CONSCIO_WEB_PASSWORD CONSCIO_ALLOW_INSECURE_BIND CONSCIO_TRUSTED_PROXIES CONSCIO_WEB_SECURE_COOKIES CONSCIO_LOG_LEVEL LIBERTAI_BASE_URL LIBERTAI_API_KEY LIBERTAI_MODEL OPENAI_BASE_URL OPENAI_API_KEY; do
  has src/conscio/config.py "$key"
done

# Public service endpoints in docs must exist in the FastAPI app.
for endpoint in \
  "/health" "/status" "/metrics" "/message" "/message/stream" "/events" \
  "/influence/goal" "/influence/constraint" \
  "/control/pause" "/control/resume" "/control/cancel" "/control/stop" "/goals" \
  "/influences" "/projects" "/autonomy/tick" "/episodes" "/trace" \
  "/memory/search" "/ready" "/metrics/prometheus"; do
  has src/conscio/api.py "$endpoint"
  has docs/public-beta/api.md "$endpoint"
done

# Operator console endpoints and SSE route must exist.
for endpoint in \
  "/ui/login" "/ui/api/snapshot" "/ui/api/message" "/ui/api/goals" \
  "/ui/api/projects" "/ui/api/memory/search" "/ui/api/model_context" \
  "/ui/api/metrics" "/ui/api/tools/events" "/ui/api/events"; do
  has src/conscio/webui.py "$endpoint"
  has docs/public-beta/api.md "$endpoint"
done

has src/conscio/webui.py "text/event-stream"
has docs/public-beta/api.md "text/event-stream"

# Tool names and policy terms used in docs must match the registry or service.
for tool in bash execute_code; do
  has src/conscio/tools/registry.py "$tool"
  has docs/public-beta/tools.md "$tool"
done

for tool in web_search web_fetch; do
  has src/conscio/tools/web.py "$tool"
  has docs/public-beta/tools.md "$tool"
done

for tool in set_task_status add_task note_progress propose_subgoal remember_fact remember_facts search_memory learn_procedure spawn_subagent; do
  has src/conscio/service.py "$tool"
  has docs/public-beta/tools.md "$tool"
done

for tool in ask_user refuse; do
  has src/conscio/core/executor.py "$tool"
  has docs/public-beta/tools.md "$tool"
done

# MCP integration surfaces referenced by docs must exist in code.
has src/conscio/mcp_client.py "mcp__"
has docs/public-beta/tools.md "mcp__"
has docs/public-beta/configuration.md "[mcp.servers"
has src/conscio/service.py "mcp_servers"
has docs/public-beta/api.md "mcp_servers"

printf 'docs smoke passed\n'

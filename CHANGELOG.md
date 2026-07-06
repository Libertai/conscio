# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Interactive messages now take queue priority over autonomous heartbeats, and a
  running autonomous episode yields to waiting user input at safe tick/round
  boundaries (`wait:preempted`).
- Structured output (response_format/json_schema) for the constraint judge, appraisal, and goal review, with tolerant hand-parse fallback.
- LLM router: named endpoints ([llm.endpoints.*]), per-role models ([llm.roles.*]) for main/fast/embeddings/subagent, and fallback chains with jittered backoff.
- `conscio --version`, and the package version in `/health` and the OpenAPI schema.
- `[llm] timeout` and `[llm] max_retries` configuration keys (previously hardcoded).
- Database integrity preflight: `conscio service start` exits with code 3 on a corrupt
  `state.db` instead of crash-looping under a process supervisor; `conscio service
  doctor` gains a `database_integrity` check.
- `busy_timeout` on the main SQLite store, tolerating short cross-process lock contention.
- Operators can cancel the running episode (`conscio cancel`, `POST /control/cancel`,
  web status strip); episodes respect `service.episode_timeout` and `/message` callers
  get a `service.message_timeout` deadline (504 while the episode continues).
- Token streaming from the tool loop: `ToolLoopSession` can emit per-round token events
  (DSML-leak-gated) that the service republishes as `chat.token` / `chat.discard` /
  `chat.final` SSE events.
- `POST /message/stream` and bearer-authed `GET /events` SSE endpoints; `conscio chat
  --stream`; live token rendering in the CLI (`run`/`ask`) and the web chat panel.
- spawn_subagent tool: bounded sub-agent tool loops on the subagent model role, with parent taint propagation and per-episode lineage (parent_episode_id).
- MCP client support: connect external MCP tool servers via `[mcp.servers.<name>]`
  (stdio or streamable HTTP); their tools register as `mcp__<server>__<tool>` and are
  quarantined like web content unless the server is marked `trusted`.
- MCP observability: server status in `/metrics` (`mcp_servers`), SSE events
  (`mcp.server.connected/disconnected/error`), and an MCP servers table in
  `conscio tools list` with live connection status.
- Scheduled home backups with retention (`backup_interval_hours`, `backup_retain`, `conscio db prune`).
- Global rate limiting (429) and request body caps (413) on episode-triggering endpoints.
- Proxy-aware client IPs via `trusted_proxies`; compose now uses secure cookies instead of the insecure-bind escape hatch.

### Fixed
- `bash` and `execute_code` terminate their child process on timeout or cancellation
  instead of leaking it.

## [0.1.0] - 2026-06-15

Public beta. v2 cognitive runtime (tick loop, attention, prediction, constraints,
self-state), memory v2 (hybrid FTS+embedding retrieval, consolidation, provenance and
taint tiers), drive/goal motivation, eval harness (B0–B4 ladder, ablations, live runs),
authenticated FastAPI service with SSE, operator web UI, CLI, Docker and systemd
deployment.

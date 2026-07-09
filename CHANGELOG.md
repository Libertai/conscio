# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-09

The "full-featured harness" release: multi-endpoint model routing with
fallback, token streaming, structured output, sub-agents, MCP client support,
episode cancellation and preemption, and production hardening (rate limits,
scheduled backups, structured logging, Prometheus metrics, release pipeline),
plus a pre-release quality pass (the Fixed/Changed entries below).

### Fixed
- A tool exception raised past the registry (policy cwd creation, MCP proxies)
  aborted the whole episode; it now becomes a failed observation the loop
  handles like any other prediction failure.
- The forced-final fallback could return spotlighted
  `<<UNTRUSTED_WEB_CONTENT>>` tool output as the agent's own answer when the
  round budget was exhausted and the model returned empty.
- `conscio db export` stringified embedding BLOBs, so `db import` produced a
  store that crashed on the next fact write; BLOBs now round-trip base64-coded.
- `POST /message/stream` never noticed client disconnects, so abandoned
  streams held SSE slots (of the global cap of 32) until the episode finished.
- Episode event-file writes and pruning ran synchronously on the event loop on
  every episode; they now run in a worker thread, like the backup path.
- Behind a reverse proxy with `service.trusted_proxies` unset, the per-IP
  login lockout collapsed into one shared bucket (one attacker could lock out
  every operator); the service now logs a loud misconfiguration warning.
- CI installed dependencies by fresh resolution while claiming to use the
  lockfile; it now runs `uv sync --locked`, so lockfile and CI cannot drift.

### Changed
- Offline eval conditions now attach a deterministic concept embedder, so the
  hybrid retrieval rerank (cosine-weighted 0.55) is actually exercised and
  discriminated by the battery instead of silently testing BM25 only.
- Consolidation gains a size cap (`service.max_active_facts`, default 50000,
  0 disables): past the ceiling, the least-valuable active facts (lowest
  trust, least accessed, longest untouched) are archived; user-stated facts
  are never auto-archived.
- Goal-review decisions apply as one locked transaction instead of up to 16
  autocommits on the event loop.
- The web dashboard chat is multi-session: a session rail with create, switch,
  and delete, per-session history, and live token streams bound to the
  initiating session.
- Release images build for `linux/amd64` and `linux/arm64` and must pass a
  containerized `/ready` smoke test before being pushed to GHCR; pull requests
  now run `pnpm check` + `pnpm build` for the web SPA.

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
- Structured logging (`log_level`, `log_format`, `log_file`, `http_access_log`), with uvicorn unified into one sink.
- `GET /ready` readiness probe, `GET /metrics/prometheus` text exposition, container healthchecks, SSE client cap.
- systemd hardening directives on both service units.
- Release pipeline: tagged releases publish `conscio-agent` to PyPI (trusted
  publishing) and the Docker image to `ghcr.io/libertai/conscio`.

### Changed
- The PyPI distribution is named `conscio-agent` (the name `conscio` is taken);
  the import package and CLI remain `conscio`.

### Fixed
- `bash` and `execute_code` terminate their child process on timeout or cancellation
  instead of leaking it.

## [0.1.0] - 2026-06-15

Public beta. v2 cognitive runtime (tick loop, attention, prediction, constraints,
self-state), memory v2 (hybrid FTS+embedding retrieval, consolidation, provenance and
taint tiers), drive/goal motivation, eval harness (B0–B4 ladder, ablations, live runs),
authenticated FastAPI service with SSE, operator web UI, CLI, Docker and systemd
deployment.

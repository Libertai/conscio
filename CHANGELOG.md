# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LLM router: named endpoints ([llm.endpoints.*]), per-role models ([llm.roles.*]) for main/fast/embeddings/subagent, and fallback chains with jittered backoff.
- `conscio --version`, and the package version in `/health` and the OpenAPI schema.
- `[llm] timeout` and `[llm] max_retries` configuration keys (previously hardcoded).
- Database integrity preflight: `conscio service start` exits with code 3 on a corrupt
  `state.db` instead of crash-looping under a process supervisor; `conscio service
  doctor` gains a `database_integrity` check.
- `busy_timeout` on the main SQLite store, tolerating short cross-process lock contention.

### Fixed
- `bash` and `execute_code` terminate their child process on timeout or cancellation
  instead of leaking it.

## [0.1.0] - 2026-06-15

Public beta. v2 cognitive runtime (tick loop, attention, prediction, constraints,
self-state), memory v2 (hybrid FTS+embedding retrieval, consolidation, provenance and
taint tiers), drive/goal motivation, eval harness (B0–B4 ladder, ablations, live runs),
authenticated FastAPI service with SSE, operator web UI, CLI, Docker and systemd
deployment.

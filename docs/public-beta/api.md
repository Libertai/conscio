# API and SSE

The service API is rooted at the configured host and port, defaulting to
`http://127.0.0.1:8765`. Public API endpoints require:

```text
Authorization: Bearer <api_key>
```

`/health` is unauthenticated and reports `{ok, running, version}`.

Episode-triggering endpoints (`/message`, `/influence/*`, `/autonomy/tick`, and the
console chat) share one global rate limit; excess requests get `429` with a
`Retry-After` header. Request bodies over `max_request_bytes` get `413`. There is
deliberately no CORS policy: the API is bearer-token and the console is same-origin.

## Service API

- `GET /health`
- `GET /status`
- `GET /metrics` — The payload includes "mcp_servers": one status row per configured MCP server (status, tools, reconnects, last_error).
- `POST /message`
- `POST /influence/goal`
- `POST /influence/constraint`
- `POST /control/pause`
- `POST /control/resume`
- `POST /control/cancel`
- `POST /control/stop`
- `GET /goals`
- `GET /influences`
- `GET /projects`
- `GET /projects/{project_id}`
- `POST /projects/{project_id}/pause`
- `POST /projects/{project_id}/resume`
- `POST /autonomy/tick`
- `GET /episodes`
- `GET /trace`
- `GET /memory/search`

Example:

```bash
curl -sS -H "Authorization: Bearer $CONSCIO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"Summarize your current active goal."}' \
  http://127.0.0.1:8765/message
```

`POST /message` waits up to `service.message_timeout` seconds (default 300). On
expiry it returns **504** and the episode keeps running — poll `GET /episodes` for the
result or abort it with `POST /control/cancel`. A cancelled episode returns **409**.
Episodes are also capped by `service.episode_timeout` (default 600 seconds).

### `POST /message/stream`

SSE variant of `POST /message`. Returns `text/event-stream` and emits the live
generation as it happens. **Not subject to `message_timeout`** — the caller sees live
progress and may disconnect or `POST /control/cancel` instead of waiting on a deadline.
The episode is still bounded by `service.episode_timeout`, which surfaces as a terminal
`message.error`.

Events:

- `chat.token` — provisional token text. Payload: `{text, ref, round, episode_id,
  source}`. `source` is `"<event.source>:<event_type>"`, e.g. `"user:message"` for a
  chat submission. Autonomous episodes also emit `chat.*` events with `source` like
  `"autonomous:heartbeat"`, and one can be finishing while the user's message waits in
  the priority queue. Consumers of the shared event stream must filter by `ref` or
  `source` so a concurrently finishing autonomous episode's tokens do not contaminate the
  user's stream.
- `chat.discard` — the provisional text so far was superseded by a tool round; discard
  accumulated tokens and await the next `chat.token` burst.
- `message.result` — terminal success. Same fields as `POST /message` (`output`,
  `selected_action`, `session_id`, `self_state`, `attention_schema`) plus `ref`.
- `message.error` — terminal failure. Payload: `{ref, status, detail}`. `status` is
  **409** for a cancelled episode (see `POST /control/cancel`) or **500** for an
  unexpected error.

Only events whose `ref` matches the request's submission are emitted; `ref` is opaque
to the caller (assigned server-side). Svelte 5 rune stores behind the web chat treat
`chatStream` as a singleton — one in-flight chat at a time, matching the one-mind
service; concurrent operators would interleave tokens (known limit).

### `GET /events`

Bearer-authed `text/event-stream` mirror of the operator console SSE stream
(`/ui/api/events`), so external integrations can subscribe with an API key rather than
a web cookie. Same `chat.*` / `message.*` events as `/message/stream` flow through it,
plus the operator-console service events. Disable buffering on any reverse proxy in
front of this path.

## Operator Console API

The web UI uses cookie authentication after `POST /ui/login`. Its internal API
is under `/ui/api/...`, including `/ui/api/snapshot`, `/ui/api/message`,
`/ui/api/goals`, `/ui/api/projects`, `/ui/api/memory/search`,
`/ui/api/model_context`, `/ui/api/metrics`, `/ui/api/tools/events`, and
`/ui/api/events`.

## SSE

`GET /ui/api/events` streams `text/event-stream` for operator-console updates.
It is cookie-authenticated, not bearer-token authenticated. If a reverse proxy
is in front of Conscio, disable buffering for this path and preserve identity
encoding.

Event names flowing over the SSE streams:

- `chat.token` — provisional token text from the active generation round.
- `chat.discard` — provisional text superseded by a tool round.
- `chat.final` — the generation round completed (terminal for the in-flight chat
  stream; the authoritative `chat.message` follows).
- `message.result` — terminal success frame on `POST /message/stream`.
- `message.error` — terminal failure frame on `POST /message/stream`.

The legacy service events (`chat.message`, `episode.created`, `project.updated`,
`goal.changed`, `control.paused`, etc.) continue to flow on the same streams.

# API and SSE

The service API is rooted at the configured host and port, defaulting to
`http://127.0.0.1:8765`. Public API endpoints require:

```text
Authorization: Bearer <api_key>
```

`/health` is unauthenticated and reports `{ok, running, version}`.

## Service API

- `GET /health`
- `GET /status`
- `GET /metrics`
- `POST /message`
- `POST /influence/goal`
- `POST /influence/constraint`
- `POST /control/pause`
- `POST /control/resume`
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

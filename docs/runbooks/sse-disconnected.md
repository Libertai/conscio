# SSE Disconnected

## Symptoms

The console loads but live updates stop, or browser devtools shows
`/ui/api/events` reconnecting.

## Check

Direct service:

```bash
curl -N http://127.0.0.1:8765/ui/api/events
```

That route is cookie-authenticated, so an unauthenticated direct curl may return
an auth error. The goal is to confirm the route exists and the proxy is not
rewriting it.

Behind a proxy, check that `/ui/api/events`:

- Does not buffer responses.
- Preserves `text/event-stream`.
- Does not gzip or transform the stream.
- Keeps idle connections open.

## Recover

Fix proxy buffering, reload the proxy, refresh `/ui`, and inspect browser
network logs. The service sets `Cache-Control: no-cache, no-transform`,
`X-Accel-Buffering: no`, and `Content-Encoding: identity` for SSE.

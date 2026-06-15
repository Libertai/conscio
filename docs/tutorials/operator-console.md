# Use the Operator Console

The operator console lives at `/ui`.

## 1. Log In

Open:

```text
http://127.0.0.1:8765/ui
```

Use the `web_password` from `~/.conscio/config.toml`. The console uses an HTTP
only cookie and the `/ui/api/...` routes.

## 2. Common Flows

- Stream: observe status and recent activity.
- Chat: send user messages through the same episode loop as API chat.
- Goals: inspect and edit goals.
- Projects: pause or resume project work.
- Influences: review durable goal and constraint influences.
- Memory: search facts and procedures.
- Episodes: inspect prior episodes.
- Trace: compare cognitive trace and latest assembled model context.

## 3. Control From CLI

```bash
conscio pause
conscio tick
conscio resume
conscio trace
```

Use pause before changing config, restoring backups, or investigating abnormal
tool use.

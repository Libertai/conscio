# Task Spam

## Contain

```bash
conscio pause
```

## Check

```bash
conscio projects
conscio goals
conscio trace
```

Task spam usually comes from an over-broad goal, repeated `add_task`, or failure
to mark stale work as `blocked` or `done`.

## Recover

Use the operator console Projects view to pause the noisy project, or use the
project API:

```bash
curl -X POST -H "Authorization: Bearer $CONSCIO_API_KEY" \
  http://127.0.0.1:8765/projects/PROJECT_ID/pause
```

Add a constraining influence:

```bash
conscio influence constraint "Do not create new tasks for paused or blocked projects."
```

Run one tick and inspect:

```bash
conscio tick
conscio trace
```

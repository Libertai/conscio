# Troubleshooting

Use the runbooks for step-by-step recovery. This page is the quick map.

## Start and Health

- Service will not start: [Service Start Failure](../runbooks/service-start-failure.md)
- Stale lock or SQLite error: [DB Locked or Corrupted](../runbooks/db-locked-or-corrupted.md)
- Restore from backup: [Restore Backup](../runbooks/restore-backup.md)

## Model Behavior

- Model endpoint unavailable: [Model Backend Unreachable](../runbooks/model-backend-unreachable.md)
- Empty outputs: [Empty Responses](../runbooks/empty-responses.md)
- Too many tool calls: [Excessive Tool Calls](../runbooks/excessive-tool-calls.md)
- Task creation loops: [Task Spam](../runbooks/task-spam.md)

## Console and API

- Login failure or cookie issue: [Web UI Auth](../runbooks/web-ui-auth.md)
- Live updates disconnected: [SSE Disconnected](../runbooks/sse-disconnected.md)

## Safety

- Disable a dangerous tool: [Disable Dangerous Tool](../runbooks/disable-dangerous-tool.md)
- Recover from a bad self-edit or bad memory: [Bad Self-Edit or Memory](../runbooks/bad-self-edit-or-memory.md)

Start every investigation by pausing autonomy:

```bash
conscio pause
conscio service status
conscio trace
```

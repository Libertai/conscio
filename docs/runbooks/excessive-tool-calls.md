# Excessive Tool Calls

## Contain

```bash
conscio pause
```

## Check

```bash
conscio service status
conscio trace
```

Look for repeated tool names, repeated failing arguments, or a project that is
being retried without progress.

## Tighten Budgets

Edit `~/.conscio/config.toml`:

```toml
[tools]
max_actions_per_hour = 10
model_tool_rounds = 8

[engine]
tool_rounds_per_tick = 2
max_ticks = 4
```

Restart:

```bash
conscio service stop
conscio service start
```

## Disable a Tool

If one tool is causing the loop:

```toml
[tools]
denied = ["bash", "execute_code"]
```

Resume only after the trace shows sane behavior on one manual tick.

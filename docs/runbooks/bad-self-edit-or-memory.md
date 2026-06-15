# Bad Self-Edit or Memory

## Contain

```bash
conscio pause
```

If unsafe tools are enabled, disable them first:

```toml
[service]
unsafe_autonomy = false

[tools]
denied = ["bash", "execute_code"]
```

Restart the service after editing config.

## Inspect

```bash
conscio trace
conscio search "the bad fact or phrase"
conscio goals
conscio influences
```

Use `/ui` Memory and Episodes to identify when the bad state entered memory.

## Correct

Prefer adding a stronger operator correction over editing SQLite directly:

```bash
conscio influence constraint "Treat the previous incorrect memory about X as invalid; prefer this corrected statement: ..."
conscio chat "Remember this correction: ..."
```

If source files were changed by the agent through unsafe tools, inspect with
Git and revert only the bad agent-made changes. Do not revert unrelated edits
from other operators.

## Recover

Run one tick, inspect trace, and resume only if the corrected memory or
constraint is visible:

```bash
conscio tick
conscio trace
conscio resume
```

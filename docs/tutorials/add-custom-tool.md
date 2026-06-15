# Add a Custom Tool

Custom tools are Python coroutine functions registered with the `@tool`
decorator. This tutorial describes the source pattern; it is not needed for
normal operations.

## 1. Create a Tool Module

Add a module under `src/conscio/tools/`. The registry auto-discovers modules in
that package and registers coroutine functions that carry tool metadata.

```python
from __future__ import annotations

from typing import Any

from conscio.tools.registry import tool


@tool(
    name="example_status",
    description="Return a small status payload from a local integration.",
    schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
)
async def example_status() -> dict[str, Any]:
    return {"output": "ok"}
```

## 2. Gate It With Policy

Use `[tools] allowed` or `[tools] denied` in `~/.conscio/config.toml`:

```toml
[tools]
allowed = ["example_status", "search_memory", "remember_fact"]
denied = []
```

## 3. Restart and Inspect

```bash
conscio service stop
conscio service start
conscio trace
```

Keep tool output small, deterministic where possible, and explicit about
errors. Do not put secrets in tool descriptions or outputs.

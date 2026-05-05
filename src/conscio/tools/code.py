from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from conscio.tools.env import resolve_tool, tool_env


async def execute_code(
    code: str | None = None,
    timeout: int = 30,
    input: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Execute a Python code snippet and return the output."""
    code = code if code is not None else input
    if not code:
        return {"output": "No code provided.", "exit_code": -1}
    fd, path = tempfile.mkstemp(suffix=".py", prefix="conscio_")
    try:
        with __import__("os").fdopen(fd, "w") as f:
            f.write(code)
            f.write("\n")
        proc = await asyncio.create_subprocess_exec(
            resolve_tool("python3"),
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(cwd).expanduser()) if cwd else None,
            env=tool_env(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        parts = []
        if out.strip():
            parts.append(out.strip())
        if err.strip():
            parts.append(f"[stderr]\n{err.strip()}")
        if proc.returncode and proc.returncode != 0 and not err.strip():
            parts.append(f"[exit code: {proc.returncode}]")
        return {"output": "\n".join(parts) if parts else "(no output)", "exit_code": proc.returncode or 0}
    except asyncio.TimeoutError:
        return {"output": f"[timed out after {timeout}s]", "exit_code": -1}
    except Exception as e:
        return {"output": f"Error: {e}", "exit_code": -1}
    finally:
        try:
            __import__("os").unlink(path)
        except OSError:
            pass


execute_code._tool_name = "execute_code"
execute_code._tool_description = "Execute Python code and return its output. Use for calculations, data processing, and scripting."

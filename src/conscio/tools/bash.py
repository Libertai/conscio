from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from conscio.tools.env import tool_env
from conscio.tools.registry import tool


@tool(
    name="bash",
    description="Execute a bash/shell command and return its output.",
    schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "maximum": 600,
                "default": 30,
                "description": "Timeout in seconds.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory; ignored if policy enforces working_directory.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    capabilities={"local_read", "local_write", "network_read", "network_write"},
)
async def bash(
    command: str | None = None,
    timeout: int = 30,
    input: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Execute a bash command and return output."""
    command = command if command is not None else input
    if not command:
        return {"output": "No command provided.", "exit_code": -1}
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(cwd).expanduser()) if cwd else None,
            env=tool_env(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode("utf-8", errors="replace")
        if stderr:
            error = stderr.decode("utf-8", errors="replace")
            if error.strip():
                output += f"\n[stderr]\n{error}"
        return {
            "output": output.strip() or "(no output)",
            "exit_code": proc.returncode or 0,
        }
    except TimeoutError:
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
        return {"output": f"[timed out after {timeout}s]", "exit_code": -1}
    except asyncio.CancelledError:
        # Reap the child on cancellation too; otherwise it outlives the episode.
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        raise
    except FileNotFoundError as e:
        return {"output": f"Command not found: {e}", "exit_code": -1}
    except Exception as e:
        return {"output": f"Error: {e}", "exit_code": -1}


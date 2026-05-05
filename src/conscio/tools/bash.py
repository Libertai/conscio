from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


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
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(Path(cwd).expanduser()) if cwd else None,
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
    except asyncio.TimeoutError:
        return {"output": f"[timed out after {timeout}s]", "exit_code": -1}
    except FileNotFoundError as e:
        return {"output": f"Command not found: {e}", "exit_code": -1}
    except Exception as e:
        return {"output": f"Error: {e}", "exit_code": -1}


bash._tool_name = "bash"
bash._tool_description = "Execute a bash/shell command and return its output"

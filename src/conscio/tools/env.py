from __future__ import annotations

import os
import shutil


_TOOL_PATH_PREFIXES = (
    "/usr/local/bin",
    "/home/conscio/.cargo/bin",
    "/home/conscio/.local/bin",
)


def tool_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PATH", "")
    parts = [path for path in _TOOL_PATH_PREFIXES if path]
    parts.extend(path for path in existing.split(os.pathsep) if path and path not in parts)
    env["PATH"] = os.pathsep.join(parts)
    return env


def resolve_tool(name: str) -> str:
    return shutil.which(name, path=tool_env()["PATH"]) or name

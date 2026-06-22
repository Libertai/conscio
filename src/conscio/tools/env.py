from __future__ import annotations

import os
import shutil

_TOOL_PATH_PREFIXES = (
    "/usr/local/bin",
    "/home/conscio/.cargo/bin",
    "/home/conscio/.local/bin",
)

# Env vars whose names contain these substrings are stripped from the
# subprocess environment so that a model with shell/code access cannot
# exfiltrate the service's own secrets (API keys, passwords, tokens).
_SECRET_ENV_PATTERNS = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_PASS", "_CREDENTIAL")


def _is_secret_env(name: str) -> bool:
    upper = name.upper()
    return any(pat in upper for pat in _SECRET_ENV_PATTERNS)


def tool_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not _is_secret_env(key)
    }
    existing = env.get("PATH", "")
    parts = [path for path in _TOOL_PATH_PREFIXES if path]
    parts.extend(path for path in existing.split(os.pathsep) if path and path not in parts)
    env["PATH"] = os.pathsep.join(parts)
    return env


def resolve_tool(name: str) -> str:
    return shutil.which(name, path=tool_env()["PATH"]) or name

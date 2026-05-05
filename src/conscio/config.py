from __future__ import annotations

import os
import secrets
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_HOME = Path.home() / ".conscio"


@dataclass
class ServiceConfig:
    home: Path = DEFAULT_HOME
    host: str = "127.0.0.1"
    port: int = 8765
    api_key: str = ""
    web_password: str = ""
    autonomous: bool = True
    tick_interval: float = 30.0
    unsafe_autonomy: bool = False
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    max_actions_per_hour: int = 60
    shell_timeout: int = 30
    working_directory: Path = field(default_factory=Path.cwd)
    pause_on_error: bool = True

    @property
    def db_path(self) -> Path:
        return self.home / "state.db"

    @property
    def lock_path(self) -> Path:
        return self.home / "service.lock"

    @property
    def log_path(self) -> Path:
        return self.home / "logs" / "service.log"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def ensure_layout(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        for child in ("logs", "events", "approvals", "sessions"):
            (self.home / child).mkdir(parents=True, exist_ok=True)


def _as_path(value: Any, default: Path) -> Path:
    if value in (None, ""):
        return default
    return Path(str(value)).expanduser()


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def load_config(path: str | Path | None = None) -> ServiceConfig:
    config_path = Path(path).expanduser() if path else Path(os.environ.get("CONSCIO_CONFIG", DEFAULT_HOME / "config.toml")).expanduser()
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("rb") as f:
            raw = tomllib.load(f)
    service = raw.get("service", raw)
    tools = raw.get("tools", {})

    cfg = ServiceConfig(
        home=_as_path(service.get("home"), config_path.parent if config_path.name == "config.toml" else DEFAULT_HOME),
        host=str(service.get("host", "127.0.0.1")),
        port=int(service.get("port", 8765)),
        api_key=str(service.get("api_key") or os.environ.get("CONSCIO_API_KEY", "")),
        web_password=str(service.get("web_password") or os.environ.get("CONSCIO_WEB_PASSWORD", "")),
        autonomous=bool(service.get("autonomous", True)),
        tick_interval=float(service.get("tick_interval", 30.0)),
        unsafe_autonomy=bool(service.get("unsafe_autonomy", False)),
        allowed_tools=_as_str_list(tools.get("allowed")),
        denied_tools=_as_str_list(tools.get("denied")),
        max_actions_per_hour=int(tools.get("max_actions_per_hour", 60)),
        shell_timeout=int(tools.get("shell_timeout", 30)),
        working_directory=_as_path(tools.get("working_directory"), Path.cwd()),
        pause_on_error=bool(service.get("pause_on_error", True)),
    )
    return cfg


def write_default_config(path: str | Path | None = None) -> Path:
    config_path = Path(path).expanduser() if path else DEFAULT_HOME / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        return config_path
    api_key = secrets.token_urlsafe(32)
    web_password = secrets.token_urlsafe(24)
    text = f"""[service]
home = "{config_path.parent}"
host = "127.0.0.1"
port = 8765
api_key = "{api_key}"
web_password = "{web_password}"
autonomous = true
tick_interval = 30
unsafe_autonomy = false
pause_on_error = true

[tools]
allowed = []
denied = []
max_actions_per_hour = 60
shell_timeout = 30
working_directory = "{Path.cwd()}"
"""
    config_path.write_text(text, encoding="utf-8")
    return config_path

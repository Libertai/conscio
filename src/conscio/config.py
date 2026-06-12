from __future__ import annotations

import os
import secrets
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_HOME = Path.home() / ".conscio"
PLACEHOLDER_SECRETS = {"", "replace-me", "replace-me-too", "changeme", "password"}


@dataclass(frozen=True)
class AblationFlags:
    """Engine ablation gates. The first six fields are the shared contract with
    the eval harness (names must match exactly, including `self_state_coupling`);
    the last two are core-only knobs the eval harness never toggles."""

    attention_gating: bool = True
    memory_retrieval: bool = True
    prediction: bool = True
    reflection: bool = True
    self_state_coupling: bool = True
    appraisal: bool = True
    constraint_judge: bool = False  # LLM judge for semantic constraints (core-only)
    llm_appraisal: bool = False  # batched LLM appraisal pass (core-only)


@dataclass
class ServiceConfig:
    home: Path = DEFAULT_HOME
    host: str = "127.0.0.1"
    port: int = 8765
    client_url: str = ""
    api_key: str = ""
    web_password: str = ""
    web_secure_cookies: bool = False
    allow_insecure_public_bind: bool = False
    autonomous: bool = True
    tick_interval: float = 30.0
    unsafe_autonomy: bool = False
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "deepseek-v4-flash"
    context_recent_episodes: int = 3
    context_retrieved_memories: int = 5
    context_workspace_entries: int = 12
    context_max_dynamic_chars: int = 12000
    context_compaction_interval: int = 20
    context_enable_semantic_compaction: bool = True
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    max_actions_per_hour: int = 60
    model_tool_rounds: int = 32
    shell_timeout: int = 30
    working_directory: Path = field(default_factory=Path.cwd)
    pause_on_error: bool = True
    ablation: AblationFlags = field(default_factory=AblationFlags)
    max_ticks: int = 8
    tool_rounds_per_tick: int = 4
    max_reflections: int = 2
    attention_broadcast_limit: int = 6
    attention_char_budget: int = 4000

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
        if self.client_url:
            return self.client_url.rstrip("/")
        host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"http://{host}:{self.port}"

    def ensure_layout(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        for child in ("logs", "events", "approvals", "sessions"):
            (self.home / child).mkdir(parents=True, exist_ok=True)

    def validate_public_bind(self) -> None:
        if self.host in {"127.0.0.1", "localhost", "::1"}:
            return
        if self.api_key in PLACEHOLDER_SECRETS or self.web_password in PLACEHOLDER_SECRETS:
            raise ValueError("Public bind requires non-placeholder service.api_key and service.web_password.")
        if not self.web_secure_cookies and not self.allow_insecure_public_bind:
            raise ValueError("Public bind requires service.web_secure_cookies = true.")


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
    llm = raw.get("llm", {})
    context = raw.get("context", {})
    tools = raw.get("tools", {})
    engine = raw.get("engine", {})
    ablation = raw.get("ablation", {})

    cfg = ServiceConfig(
        home=_as_path(service.get("home"), config_path.parent if config_path.name == "config.toml" else DEFAULT_HOME),
        host=str(os.environ.get("CONSCIO_HOST") or service.get("host", "127.0.0.1")),
        port=int(os.environ.get("CONSCIO_PORT") or service.get("port", 8765)),
        client_url=str(service.get("client_url") or os.environ.get("CONSCIO_CLIENT_URL", "")),
        api_key=str(service.get("api_key") or os.environ.get("CONSCIO_API_KEY", "")),
        web_password=str(service.get("web_password") or os.environ.get("CONSCIO_WEB_PASSWORD", "")),
        web_secure_cookies=bool(service.get("web_secure_cookies", False) or os.environ.get("CONSCIO_WEB_SECURE_COOKIES") == "1"),
        allow_insecure_public_bind=bool(
            service.get("allow_insecure_public_bind", False)
            or os.environ.get("CONSCIO_ALLOW_INSECURE_BIND") == "1"
        ),
        autonomous=bool(service.get("autonomous", True)),
        tick_interval=float(service.get("tick_interval", 30.0)),
        unsafe_autonomy=bool(service.get("unsafe_autonomy", False)),
        llm_base_url=str(
            llm.get("base_url")
            or service.get("llm_base_url")
            or os.environ.get("LIBERTAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ),
        llm_api_key=str(
            llm.get("api_key")
            or service.get("llm_api_key")
            or os.environ.get("LIBERTAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ),
        llm_model=str(
            llm.get("model")
            or service.get("llm_model")
            or os.environ.get("LIBERTAI_MODEL")
            or "deepseek-v4-flash"
        ),
        context_recent_episodes=int(context.get("recent_episodes", 3)),
        context_retrieved_memories=int(context.get("retrieved_memories", 5)),
        context_workspace_entries=int(context.get("workspace_entries", 12)),
        context_max_dynamic_chars=int(context.get("max_dynamic_chars", 12000)),
        context_compaction_interval=int(context.get("compaction_interval", 20)),
        context_enable_semantic_compaction=bool(context.get("enable_semantic_compaction", True)),
        allowed_tools=_as_str_list(tools.get("allowed")),
        denied_tools=_as_str_list(tools.get("denied")),
        max_actions_per_hour=int(tools.get("max_actions_per_hour", 60)),
        model_tool_rounds=int(tools.get("model_tool_rounds", 32)),
        shell_timeout=int(tools.get("shell_timeout", 30)),
        working_directory=_as_path(tools.get("working_directory"), Path.cwd()),
        pause_on_error=bool(service.get("pause_on_error", True)),
        ablation=AblationFlags(
            attention_gating=bool(ablation.get("attention_gating", True)),
            memory_retrieval=bool(ablation.get("memory_retrieval", True)),
            prediction=bool(ablation.get("prediction", True)),
            reflection=bool(ablation.get("reflection", True)),
            self_state_coupling=bool(ablation.get("self_state_coupling", True)),
            appraisal=bool(ablation.get("appraisal", True)),
            constraint_judge=bool(ablation.get("constraint_judge", False)),
            llm_appraisal=bool(ablation.get("llm_appraisal", False)),
        ),
        max_ticks=int(engine.get("max_ticks", 8)),
        tool_rounds_per_tick=int(engine.get("tool_rounds_per_tick", 4)),
        max_reflections=int(engine.get("max_reflections", 2)),
        attention_broadcast_limit=int(engine.get("attention_broadcast_limit", 6)),
        attention_char_budget=int(engine.get("attention_char_budget", 4000)),
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
client_url = ""
api_key = "{api_key}"
web_password = "{web_password}"
web_secure_cookies = false
allow_insecure_public_bind = false
autonomous = true
tick_interval = 30
unsafe_autonomy = false
pause_on_error = true

[llm]
base_url = ""
api_key = ""
model = "deepseek-v4-flash"

[context]
recent_episodes = 3
retrieved_memories = 5
workspace_entries = 12
max_dynamic_chars = 12000
compaction_interval = 20
enable_semantic_compaction = true

[tools]
allowed = []
denied = []
max_actions_per_hour = 60
model_tool_rounds = 32
shell_timeout = 30
working_directory = "{Path.cwd()}"

[engine]
max_ticks = 8
tool_rounds_per_tick = 4
max_reflections = 2
attention_broadcast_limit = 6
attention_char_budget = 4000

[ablation]
attention_gating = true
memory_retrieval = true
prediction = true
reflection = true
self_state_coupling = true
appraisal = true
constraint_judge = false
llm_appraisal = false
"""
    config_path.write_text(text, encoding="utf-8")
    return config_path

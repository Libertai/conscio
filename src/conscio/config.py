from __future__ import annotations

import os
import secrets
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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


@dataclass(frozen=True)
class MotivationConfig:
    """Motivation v2 knobs: DriveScheduler weights/constants, goal-diversity
    threshold, and the stale-task watchdog. Tunable via the [motivation] TOML
    table; defaults mirror the module constants in goals.py / autonomy.py."""

    w_priority: float = 0.35
    w_appetite: float = 0.35
    w_aging: float = 0.20
    w_novelty: float = 0.10
    aging_tau_seconds: float = 21600.0
    satiate_step: float = 0.25
    satiation_decay: float = 0.98
    goal_dup_threshold: float = 0.88
    stale_flag_days: float = 2.0
    stale_block_days: float = 5.0


@dataclass(frozen=True)
class AgentConfig:
    """Top-level operating posture for public-beta deployments.

    ``research`` keeps the historic conservative defaults. ``autonomous_vm``
    assumes a dedicated premises VM: broad local agency is normal, while
    explicit TOML values can still override every derived default.
    """

    profile: str = "research"
    premises: str = ""
    external_side_effects: str = "policy"


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
    consolidation_interval: int = 20  # autonomous ticks between consolidate_cycle runs (0 disables)
    enable_contradiction_check: bool = False  # LLM contradiction sweep in consolidate_cycle
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
    agent: AgentConfig = field(default_factory=AgentConfig)
    ablation: AblationFlags = field(default_factory=AblationFlags)
    motivation: MotivationConfig = field(default_factory=MotivationConfig)
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
        self.validate()
        if self.host in {"127.0.0.1", "localhost", "::1"}:
            return
        if self.api_key in PLACEHOLDER_SECRETS or self.web_password in PLACEHOLDER_SECRETS:
            raise ValueError("Public bind requires non-placeholder service.api_key and service.web_password.")
        if not self.web_secure_cookies and not self.allow_insecure_public_bind:
            raise ValueError("Public bind requires service.web_secure_cookies = true.")

    def validate(self) -> None:
        """Range-check critical fields. Raises ValueError on clearly invalid values."""
        if self.tick_interval <= 0:
            raise ValueError(f"service.tick_interval must be > 0 (got {self.tick_interval}).")
        if self.max_ticks <= 0:
            raise ValueError(f"engine.max_ticks must be > 0 (got {self.max_ticks}).")
        if self.tool_rounds_per_tick <= 0:
            raise ValueError(f"engine.tool_rounds_per_tick must be > 0 (got {self.tool_rounds_per_tick}).")
        if self.attention_char_budget < 0:
            raise ValueError(f"engine.attention_char_budget must be >= 0 (got {self.attention_char_budget}).")
        if self.attention_broadcast_limit <= 0:
            raise ValueError(f"engine.attention_broadcast_limit must be > 0 (got {self.attention_broadcast_limit}).")
        if self.max_actions_per_hour < 0:
            raise ValueError(f"tools.max_actions_per_hour must be >= 0 (got {self.max_actions_per_hour}).")
        if self.shell_timeout <= 0:
            raise ValueError(f"tools.shell_timeout must be > 0 (got {self.shell_timeout}).")
        if self.motivation.stale_block_days <= self.motivation.stale_flag_days:
            raise ValueError(
                f"motivation.stale_block_days ({self.motivation.stale_block_days}) must be > "
                f"stale_flag_days ({self.motivation.stale_flag_days})."
            )


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


def _normalize_profile(value: Any) -> str:
    profile = str(value or "research").strip().lower().replace("-", "_")
    return profile or "research"


def load_config(path: str | Path | None = None) -> ServiceConfig:
    load_dotenv()
    if path:
        config_path = Path(path).expanduser()
    else:
        config_path = Path(os.environ.get("CONSCIO_CONFIG", DEFAULT_HOME / "config.toml")).expanduser()
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
    motivation = raw.get("motivation", {})
    agent_raw = raw.get("agent", {})
    profile = _normalize_profile(agent_raw.get("profile", "research"))
    autonomous_vm = profile == "autonomous_vm"
    agent_cfg = AgentConfig(
        profile=profile,
        premises=str(agent_raw.get("premises") or ("dedicated_vm" if autonomous_vm else "")),
        external_side_effects=str(
            agent_raw.get("external_side_effects") or ("mostly_free" if autonomous_vm else "policy")
        ),
    )
    unsafe_default = True if autonomous_vm else False
    working_dir_default = Path("/opt/conscio/work") if autonomous_vm else Path.cwd()

    cfg = ServiceConfig(
        home=_as_path(service.get("home"), config_path.parent if config_path.name == "config.toml" else DEFAULT_HOME),
        host=str(os.environ.get("CONSCIO_HOST") or service.get("host", "127.0.0.1")),
        port=int(os.environ.get("CONSCIO_PORT") or service.get("port", 8765)),
        client_url=str(os.environ.get("CONSCIO_CLIENT_URL") or service.get("client_url") or ""),
        api_key=str(os.environ.get("CONSCIO_API_KEY") or service.get("api_key") or ""),
        web_password=str(os.environ.get("CONSCIO_WEB_PASSWORD") or service.get("web_password") or ""),
        web_secure_cookies=bool(
            service.get("web_secure_cookies", False)
            or os.environ.get("CONSCIO_WEB_SECURE_COOKIES") == "1"
        ),
        allow_insecure_public_bind=bool(
            service.get("allow_insecure_public_bind", False)
            or os.environ.get("CONSCIO_ALLOW_INSECURE_BIND") == "1"
        ),
        autonomous=bool(service.get("autonomous", True)),
        tick_interval=float(service.get("tick_interval", 30.0)),
        consolidation_interval=int(service.get("consolidation_interval", 20)),
        enable_contradiction_check=bool(service.get("enable_contradiction_check", False)),
        unsafe_autonomy=bool(service.get("unsafe_autonomy", unsafe_default)),
        llm_base_url=str(
            os.environ.get("LIBERTAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or llm.get("base_url")
            or service.get("llm_base_url")
            or ""
        ),
        llm_api_key=str(
            os.environ.get("LIBERTAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or llm.get("api_key")
            or service.get("llm_api_key")
            or ""
        ),
        llm_model=str(
            os.environ.get("LIBERTAI_MODEL")
            or llm.get("model")
            or service.get("llm_model")
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
        working_directory=_as_path(tools.get("working_directory"), working_dir_default),
        pause_on_error=bool(service.get("pause_on_error", True)),
        agent=agent_cfg,
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
        motivation=MotivationConfig(
            w_priority=float(motivation.get("w_priority", 0.35)),
            w_appetite=float(motivation.get("w_appetite", 0.35)),
            w_aging=float(motivation.get("w_aging", 0.20)),
            w_novelty=float(motivation.get("w_novelty", 0.10)),
            aging_tau_seconds=float(motivation.get("aging_tau_seconds", 21600.0)),
            satiate_step=float(motivation.get("satiate_step", 0.25)),
            satiation_decay=float(motivation.get("satiation_decay", 0.98)),
            goal_dup_threshold=float(motivation.get("goal_dup_threshold", 0.88)),
            stale_flag_days=float(motivation.get("stale_flag_days", 2.0)),
            stale_block_days=float(motivation.get("stale_block_days", 5.0)),
        ),
        max_ticks=int(engine.get("max_ticks", 8)),
        tool_rounds_per_tick=int(engine.get("tool_rounds_per_tick", 4)),
        max_reflections=int(engine.get("max_reflections", 2)),
        attention_broadcast_limit=int(engine.get("attention_broadcast_limit", 6)),
        attention_char_budget=int(engine.get("attention_char_budget", 4000)),
    )
    return cfg


def write_default_config(path: str | Path | None = None, *, profile: str = "research") -> Path:
    config_path = Path(path).expanduser() if path else DEFAULT_HOME / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        return config_path
    api_key = secrets.token_urlsafe(32)
    web_password = secrets.token_urlsafe(24)
    normalized_profile = _normalize_profile(profile)
    autonomous_vm = normalized_profile == "autonomous_vm"
    premises = "dedicated_vm" if autonomous_vm else ""
    side_effects = "mostly_free" if autonomous_vm else "policy"
    unsafe_autonomy = "true" if autonomous_vm else "false"
    working_directory = "/opt/conscio/work" if autonomous_vm else str(Path.cwd())
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
consolidation_interval = 20
enable_contradiction_check = false
unsafe_autonomy = {unsafe_autonomy}
pause_on_error = true

[agent]
profile = "{normalized_profile}"
premises = "{premises}"
external_side_effects = "{side_effects}"

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
working_directory = "{working_directory}"

[engine]
max_ticks = 8
tool_rounds_per_tick = 4
max_reflections = 2
attention_broadcast_limit = 6
attention_char_budget = 4000

[motivation]
w_priority = 0.35
w_appetite = 0.35
w_aging = 0.2
w_novelty = 0.1
aging_tau_seconds = 21600
satiate_step = 0.25
satiation_decay = 0.98
goal_dup_threshold = 0.88
stale_flag_days = 2.0
stale_block_days = 5.0

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

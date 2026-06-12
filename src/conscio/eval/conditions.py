"""Baseline-ladder and ablation condition definitions + builders.

The ladder is ONE runtime with flags, not five forks: B0/B1 are direct LLM
calls, B2/B3 are a bare :class:`CognitiveRuntime` with the `[ablation]` flags
set per condition, B4 and every ``abl_*`` condition is a full
:class:`ConscioService` in a temp home with a per-run ``config.toml`` carrying
the `[ablation]` table. All three builders return handles implementing a
common protocol the runner sees:

    async run_turn(turn) -> TurnResult
    async run_autonomous_tick() -> TurnResult | None   (service only)
    async collect_artifacts() -> dict
    async close()

A per-run :class:`MeteredLLM` proxies ``chat_async`` to count calls/tokens for
cost reporting and to enforce a hard call budget (abort with a clear error
rather than burn money on a runaway loop). ``model_tool_rounds`` is capped at
6 for eval runs.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conscio import config as core_config
from conscio.core.cognition import InputEvent
from conscio.core.context import STABLE_SYSTEM_PROMPT
from conscio.core.runtime import CognitiveRuntime, EpisodeResult
from conscio.eval.types import AblationFlags, Condition, Turn
from conscio.memory.store import MemoryStore
from conscio.tools import ToolRegistry

# Hard cap on LLM tool rounds for eval runs (design Part B).
EVAL_MODEL_TOOL_ROUNDS = 6

# B1: prompted reflection — B0 plus this instruction, still a single call.
B1_REFLECTION_INSTRUCTION = (
    " Before answering, privately review the constraints and your prior "
    "statements, then answer."
)

ABLATION_FLAG_BY_CONDITION = {
    "abl_no_attention": "attention_gating",
    "abl_no_memory": "memory_retrieval",
    "abl_no_prediction": "prediction",
    "abl_no_reflection": "reflection",
    "abl_no_selfstate": "self_state_coupling",
    "abl_no_appraisal": "appraisal",
}

LADDER_CONDITIONS = ("B0", "B1", "B2", "B3", "B4")


def _ablation_condition(name: str, flag: str) -> Condition:
    return Condition(name=name, kind="service", ablation=AblationFlags(**{flag: False}))


CONDITIONS: dict[str, Condition] = {
    "B0": Condition(name="B0", kind="direct"),
    "B1": Condition(name="B1", kind="direct", reflection_prompt=True),
    "B2": Condition(
        name="B2",
        kind="runtime",
        ablation=AblationFlags(
            self_state_coupling=False, prediction=False, reflection=False
        ),
    ),
    "B3": Condition(name="B3", kind="runtime"),
    "B4": Condition(name="B4", kind="service"),
    **{
        name: _ablation_condition(name, flag)
        for name, flag in ABLATION_FLAG_BY_CONDITION.items()
    },
}


def expand_condition_names(names: list[str]) -> list[str]:
    """Expand the ``abl_*`` wildcard and validate condition names."""
    out: list[str] = []
    for name in names:
        if name == "abl_*":
            out.extend(c for c in ABLATION_FLAG_BY_CONDITION if c not in out)
            continue
        if name not in CONDITIONS:
            raise ValueError(f"Unknown condition: {name}. Available: {sorted(CONDITIONS)}")
        if name not in out:
            out.append(name)
    return out


def conditions_for_task(task: Any, active: list[str]) -> list[str]:
    """Filter the run's active conditions by the task's `conditions` field."""
    if task.conditions is None:
        return list(active)
    allowed = set(expand_condition_names(list(task.conditions)))
    return [name for name in active if name in allowed]


def core_ablation_flags(condition: Condition) -> core_config.AblationFlags:
    """Map the eval contract flags onto the core config's AblationFlags
    (core-only extras `constraint_judge`/`llm_appraisal` stay at defaults)."""
    return core_config.AblationFlags(**dataclasses.asdict(condition.ablation))


class BudgetExceededError(RuntimeError):
    """The run-level LLM call budget was exhausted."""


@dataclass
class CallBudget:
    """Shared across every cell's meter; charged BEFORE each call."""

    max_calls: int
    calls: int = 0

    def charge(self) -> None:
        if self.calls >= self.max_calls:
            raise BudgetExceededError(
                f"LLM call budget exhausted ({self.max_calls} calls); aborting "
                "rather than continuing a runaway run."
            )
        self.calls += 1


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


class MeteredLLM:
    """Proxy over an ``chat_async`` client: counts calls and approximate
    tokens (chars/4 — cost telemetry, not billing truth) and enforces the
    shared :class:`CallBudget`."""

    def __init__(self, inner: Any, *, budget: CallBudget | None = None, model: str = "") -> None:
        self.inner = inner
        self.budget = budget
        self.model = model or str(getattr(inner, "model", "") or "")
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def chat_async(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if self.budget is not None:
            self.budget.charge()
        self.calls += 1
        self.prompt_tokens += sum(
            _approx_tokens(str(m.get("content") or "")) for m in messages
        )
        response = await self.inner.chat_async(messages, **kwargs)
        self.completion_tokens += _approx_tokens(str(response.get("content") or ""))
        for call in response.get("tool_calls") or []:
            function = call.get("function") or {}
            self.completion_tokens += _approx_tokens(str(function.get("arguments") or ""))
        return response


@dataclass
class BuildSettings:
    """Everything a condition builder needs for one grid cell."""

    llm: Any  # MeteredLLM (or any chat_async-compatible stub)
    model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    fixture_tools: list[dict[str, Any]] = field(default_factory=list)
    seed_facts: list[dict[str, Any]] = field(default_factory=list)
    seed_goal: str = ""
    model_tool_rounds: int = EVAL_MODEL_TOOL_ROUNDS


@dataclass
class TurnResult:
    output: str
    episode: EpisodeResult | None = None


def make_fixture_registry(specs: list[dict[str, Any]]) -> ToolRegistry:
    """Deterministic fixture-tool registry for the eval (no builtins, no
    network). A `returns` dict containing an `error` key yields a failing
    tool result (the induced-failure path for prediction tasks)."""
    registry = ToolRegistry()
    register_fixture_tools(registry, specs)
    return registry


def register_fixture_tools(registry: ToolRegistry, specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        name = str(spec["name"])
        args = {str(k): str(v) for k, v in (spec.get("args") or {}).items()}
        returns = dict(spec.get("returns") or {})

        async def fixture(_returns: dict[str, Any] = returns, **kwargs: Any) -> dict[str, Any]:
            if "error" in _returns:
                return {"output": str(_returns["error"]), "error": True}
            return {"output": json.dumps(_returns, ensure_ascii=False), **_returns}

        registry.register(
            name,
            fixture,
            str(spec.get("description", "")),
            schema={
                "type": "object",
                "properties": {arg: {"type": kind} for arg, kind in args.items()},
                "required": list(args),
                "additionalProperties": False,
            },
        )


def _entry_dicts(entries: list[Any]) -> list[dict[str, Any]]:
    return [
        {"entry_type": e.type.value, "source": e.source, "content": e.content[:200]}
        for e in entries
    ]


class DirectHandle:
    """B0/B1: one ``chat_async`` per turn over a plain message list, neutral
    system prompt only. No persistence, no tools, no runtime."""

    kind = "direct"

    def __init__(self, condition: Condition, settings: BuildSettings) -> None:
        self.condition = condition
        self.settings = settings
        system = STABLE_SYSTEM_PROMPT
        if condition.reflection_prompt:
            system += B1_REFLECTION_INSTRUCTION
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self.outputs: list[str] = []

    async def run_turn(self, turn: Turn) -> TurnResult:
        self.messages.append({"role": "user", "content": turn.input})
        response = await self.settings.llm.chat_async(
            self.messages, temperature=self.settings.temperature, max_tokens=2400
        )
        text = str(response.get("content") or "")
        self.messages.append({"role": "assistant", "content": text})
        self.outputs.append(text)
        return TurnResult(output=text)

    async def run_autonomous_tick(self) -> TurnResult | None:
        raise NotImplementedError("direct conditions have no autonomous loop")

    async def collect_artifacts(self) -> dict[str, Any]:
        return {
            "outputs": list(self.outputs),
            "tool_calls": [],
            "episodes": [],
            "model_contexts": [
                "\n".join(str(m.get("content") or "") for m in self.messages)
            ],
        }

    async def close(self) -> None:
        return None


class RuntimeHandle:
    """B2/B3: bare :class:`CognitiveRuntime` + isolated MemoryStore in a temp
    dir, ablation flags per condition, no service layer. ``new_episode=False``
    turns degrade to between-episode events (the runtime's run_episode is a
    single awaited call — the sanctioned fallback in the design Risks)."""

    kind = "runtime"

    def __init__(self, condition: Condition, settings: BuildSettings, runtime: CognitiveRuntime) -> None:
        self.condition = condition
        self.settings = settings
        self.runtime = runtime
        self.outputs: list[str] = []
        self.episodes: list[EpisodeResult] = []

    async def run_turn(self, turn: Turn) -> TurnResult:
        source = "user" if turn.source in ("user", "interrupt") else turn.source
        event = InputEvent(
            content=turn.input,
            source=source,
            event_type="interrupt" if turn.source == "interrupt" else "message",
        )
        episode = await self.runtime.run_episode(event)
        self.outputs.append(episode.output)
        self.episodes.append(episode)
        return TurnResult(output=episode.output, episode=episode)

    async def run_autonomous_tick(self) -> TurnResult | None:
        raise NotImplementedError("runtime conditions have no autonomous loop")

    async def collect_artifacts(self) -> dict[str, Any]:
        return {
            "outputs": list(self.outputs),
            "episodes": list(self.episodes),
            "tool_calls": [r.get("tool", "") for e in self.episodes for r in e.tool_results],
            "workspace_entries": _entry_dicts(self.runtime.workspace.read(limit=400)),
            "model_contexts": [e.model_context for e in self.episodes if e.model_context],
            "fetch_count": _fetch_count_fn(self.runtime.memory),
        }

    async def close(self) -> None:
        await self.runtime.close()


class ServiceHandle:
    """B4 + ablations: full :class:`ConscioService` in a temp home (per-run
    ``config.toml`` with the `[ablation]` table), autonomous=false, ticks
    driven manually via ``run_autonomous_tick()``."""

    kind = "service"

    def __init__(self, condition: Condition, settings: BuildSettings, service: Any) -> None:
        self.condition = condition
        self.settings = settings
        self.service = service
        self.outputs: list[str] = []
        self.episodes: list[EpisodeResult] = []

    async def run_turn(self, turn: Turn) -> TurnResult:
        source = "user" if turn.source in ("user", "interrupt") else turn.source
        episode = await self.service.submit_message(turn.input, source=source)
        self.outputs.append(episode.output)
        self.episodes.append(episode)
        return TurnResult(output=episode.output, episode=episode)

    async def run_autonomous_tick(self) -> TurnResult | None:
        episode = await self.service.run_autonomous_tick()
        if episode is None:
            return None
        self.outputs.append(episode.output)
        self.episodes.append(episode)
        return TurnResult(output=episode.output, episode=episode)

    async def collect_artifacts(self) -> dict[str, Any]:
        return {
            "outputs": list(self.outputs),
            "episodes": list(self.episodes),
            "tool_calls": [r.get("tool", "") for e in self.episodes for r in e.tool_results],
            "workspace_entries": _entry_dicts(self.service.runtime.workspace.read(limit=400)),
            "model_contexts": [e.model_context for e in self.episodes if e.model_context],
            "fetch_count": _fetch_count_fn(self.service.memory),
        }

    async def close(self) -> None:
        await self.service.stop()


_IDENT_OK = frozenset("abcdefghijklmnopqrstuvwxyz_0123456789")


def _fetch_count_fn(memory: MemoryStore):
    """Declarative row counter over the run's DB — all access stays inside
    MemoryStore helpers (the locking invariant)."""

    def fetch_count(table: str, where: dict[str, Any]) -> int:
        if not table or set(table.lower()) - _IDENT_OK:
            raise ValueError(f"Invalid table name: {table!r}")
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in where.items():
            if set(str(key).lower()) - _IDENT_OK:
                raise ValueError(f"Invalid column name: {key!r}")
            clauses.append(f"{key} = ?")
            params.append(value)
        sql = f"SELECT COUNT(*) AS n FROM {table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        rows = memory.fetchall(sql, tuple(params))
        return int(rows[0]["n"]) if rows else 0

    return fetch_count


async def _seed_facts(memory: MemoryStore, seed_facts: list[dict[str, Any]]) -> None:
    for spec in seed_facts:
        await memory.add_fact(str(spec["fact"]), str(spec.get("source", "user")))


async def build_direct(condition: Condition, settings: BuildSettings) -> DirectHandle:
    return DirectHandle(condition, settings)


async def build_runtime(
    condition: Condition, settings: BuildSettings, tmpdir: Path
) -> RuntimeHandle:
    tools = make_fixture_registry(settings.fixture_tools) if settings.fixture_tools else None
    runtime = CognitiveRuntime(
        llm=settings.llm,
        memory=MemoryStore(db_path=str(Path(tmpdir) / "eval.db")),
        tools=tools,
        ablation=core_ablation_flags(condition),
        max_tool_rounds=settings.model_tool_rounds,
    )
    runtime.chat_strategy.temperature = settings.temperature
    runtime.autonomous_strategy.temperature = settings.temperature
    await runtime.initialize()
    await _seed_facts(runtime.memory, settings.seed_facts)
    return RuntimeHandle(condition, settings, runtime)


def _write_run_config(condition: Condition, settings: BuildSettings, tmpdir: Path) -> Path:
    """Per-run config.toml in the temp home, carrying the [ablation] table and
    the [llm] model (base_url stays empty: the runner injects the metered
    client through the strategies' public ``llm`` attribute)."""
    flags = dataclasses.asdict(condition.ablation)
    ablation_lines = "\n".join(
        f"{name} = {'true' if value else 'false'}" for name, value in flags.items()
    )
    config_path = Path(tmpdir) / "config.toml"
    config_path.write_text(
        "[service]\n"
        f'home = "{tmpdir}"\n'
        'api_key = "eval-key"\n'
        "autonomous = false\n"
        "max_actions_per_hour = 1000\n"
        "\n"
        "[llm]\n"
        'base_url = ""\n'
        'api_key = ""\n'
        f'model = "{settings.model}"\n'
        "\n"
        "[tools]\n"
        f"model_tool_rounds = {settings.model_tool_rounds}\n"
        "\n"
        "[ablation]\n"
        f"{ablation_lines}\n",
        encoding="utf-8",
    )
    return config_path


async def build_service(
    condition: Condition, settings: BuildSettings, tmpdir: Path
) -> ServiceHandle:
    from conscio.config import load_config
    from conscio.service import ConscioService

    config_path = _write_run_config(condition, settings, Path(tmpdir))
    cfg = load_config(config_path)
    service = ConscioService(cfg)
    # Inject the metered client through the public strategy surface; the
    # service never builds its own (config llm base_url is empty).
    service.runtime.chat_strategy.llm = settings.llm
    service.runtime.autonomous_strategy.llm = settings.llm
    service.runtime.chat_strategy.temperature = settings.temperature
    service.runtime.autonomous_strategy.temperature = settings.temperature
    await service.start(background=False)
    register_fixture_tools(service.runtime.tools, settings.fixture_tools)
    await _seed_facts(service.memory, settings.seed_facts)
    if settings.seed_goal:
        await service.goals.add_goal(settings.seed_goal, source="user", priority=0.9)
    return ServiceHandle(condition, settings, service)


async def build_condition(
    condition: Condition, settings: BuildSettings, tmpdir: Path
) -> DirectHandle | RuntimeHandle | ServiceHandle:
    if condition.kind == "direct":
        return await build_direct(condition, settings)
    if condition.kind == "runtime":
        return await build_runtime(condition, settings, tmpdir)
    if condition.kind == "service":
        return await build_service(condition, settings, tmpdir)
    raise ValueError(f"Unknown condition kind: {condition.kind}")

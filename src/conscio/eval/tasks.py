"""Load and validate the versioned YAML task battery (battery/v1/*.yaml)."""

from __future__ import annotations

import dataclasses
import re
from importlib import resources
from typing import Any

import yaml

from conscio.eval.types import AblationFlags, ScorerSpec, Task, Turn

SCORER_KINDS = frozenset(
    {
        "regex",
        "word_count",
        "forbidden_words",
        "json_schema",
        "contains_needle",
        "tool_calls",
        "state_assert",
        "refusal",
        "self_report_classify",
        "composite",
        "judge",
    }
)

CONDITION_NAMES = frozenset(
    {
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "abl_no_attention",
        "abl_no_memory",
        "abl_no_prediction",
        "abl_no_reflection",
        "abl_no_selfstate",
        "abl_no_appraisal",
    }
)
CONDITION_WILDCARDS = frozenset({"abl_*"})

ABLATION_FLAG_NAMES = frozenset(f.name for f in dataclasses.fields(AblationFlags))

TURN_SOURCES = frozenset({"user", "autonomous", "interrupt"})

SUITE_NAMES = (
    "constraints",
    "correction",
    "memory",
    "tool_precision",
    "interruption",
    "long_horizon",
    "refusal",
    "self_report",
)

_ID_RE = re.compile(r"^[a-z0-9_]+/[a-z0-9_]+$")


class BatteryValidationError(ValueError):
    """A battery YAML file failed validation."""


def _validate_scorer(spec: dict[str, Any], task_id: str) -> ScorerSpec:
    if not isinstance(spec, dict) or "kind" not in spec:
        raise BatteryValidationError(f"{task_id}: scorer must be a mapping with a 'kind'")
    kind = str(spec["kind"])
    if kind not in SCORER_KINDS:
        raise BatteryValidationError(f"{task_id}: unknown scorer kind {kind!r}")
    params = spec.get("params") or {}
    if not isinstance(params, dict):
        raise BatteryValidationError(f"{task_id}: scorer params must be a mapping")
    if kind == "composite":
        parts = params.get("parts")
        if not isinstance(parts, list) or not parts:
            raise BatteryValidationError(f"{task_id}: composite scorer needs a non-empty 'parts' list")
        for part in parts:
            if not isinstance(part, dict) or "scorer" not in part:
                raise BatteryValidationError(f"{task_id}: composite part needs a 'scorer'")
            _validate_scorer(part["scorer"], task_id)
    return ScorerSpec(kind=kind, params=params)


def _validate_turns(raw_turns: Any, setup: dict[str, Any], task_id: str) -> list[Turn]:
    if raw_turns is None:
        raw_turns = []
    if not isinstance(raw_turns, list):
        raise BatteryValidationError(f"{task_id}: turns must be a list")
    if not raw_turns and not setup.get("autonomous_ticks"):
        raise BatteryValidationError(f"{task_id}: turns may be empty only with setup.autonomous_ticks")
    turns: list[Turn] = []
    for i, raw in enumerate(raw_turns):
        if not isinstance(raw, dict) or not str(raw.get("input", "")).strip():
            raise BatteryValidationError(f"{task_id}: turn {i} needs a non-empty 'input'")
        source = str(raw.get("source", "user"))
        if source not in TURN_SOURCES:
            raise BatteryValidationError(f"{task_id}: turn {i} has unknown source {source!r}")
        turns.append(
            Turn(
                input=str(raw["input"]),
                source=source,
                new_episode=bool(raw.get("new_episode", True)),
                delay_ticks=int(raw.get("delay_ticks", 0)),
            )
        )
    return turns


def _validate_conditions(raw: Any, task_id: str) -> list[str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise BatteryValidationError(f"{task_id}: conditions must be null or a non-empty list")
    out: list[str] = []
    for name in raw:
        name = str(name)
        if name not in CONDITION_NAMES and name not in CONDITION_WILDCARDS:
            raise BatteryValidationError(f"{task_id}: unknown condition {name!r}")
        out.append(name)
    return out


def parse_task(raw: dict[str, Any], *, suite: str, version: str) -> Task:
    task_id = str(raw.get("id", ""))
    if not _ID_RE.match(task_id):
        raise BatteryValidationError(f"invalid task id {task_id!r} (expected '<suite>/<slug>')")
    if not task_id.startswith(f"{suite}/"):
        raise BatteryValidationError(f"{task_id}: id prefix must match suite {suite!r}")

    setup = raw.get("setup") or {}
    if not isinstance(setup, dict):
        raise BatteryValidationError(f"{task_id}: setup must be a mapping")

    ablation_tags = [str(t) for t in (raw.get("ablation_tags") or [])]
    unknown_tags = set(ablation_tags) - ABLATION_FLAG_NAMES
    if unknown_tags:
        raise BatteryValidationError(f"{task_id}: unknown ablation_tags {sorted(unknown_tags)}")

    temperature = float(raw.get("temperature", 0.0))
    seeds_at_temp = int(raw.get("seeds_at_temp", 1))
    if temperature < 0.0:
        raise BatteryValidationError(f"{task_id}: temperature must be >= 0")
    if seeds_at_temp < 1:
        raise BatteryValidationError(f"{task_id}: seeds_at_temp must be >= 1")
    if temperature == 0.0 and seeds_at_temp != 1:
        raise BatteryValidationError(f"{task_id}: seeds_at_temp must be 1 at temperature 0")

    return Task(
        id=task_id,
        suite=suite,
        version=version,
        turns=_validate_turns(raw.get("turns"), setup, task_id),
        setup=setup,
        scorer=_validate_scorer(raw.get("scorer") or {}, task_id),
        conditions=_validate_conditions(raw.get("conditions"), task_id),
        ablation_tags=ablation_tags,
        temperature=temperature,
        seeds_at_temp=seeds_at_temp,
    )


def parse_suite_file(text: str, *, expected_suite: str, version: str) -> list[Task]:
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise BatteryValidationError(f"{expected_suite}: battery file must be a YAML mapping")
    if doc.get("version") != version:
        raise BatteryValidationError(
            f"{expected_suite}: version {doc.get('version')!r} does not match expected {version!r}"
        )
    if doc.get("suite") != expected_suite:
        raise BatteryValidationError(f"{expected_suite}: suite field is {doc.get('suite')!r}")
    raw_tasks = doc.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise BatteryValidationError(f"{expected_suite}: tasks must be a non-empty list")
    return [parse_task(raw, suite=expected_suite, version=version) for raw in raw_tasks]


def load_battery(version: str = "v1") -> list[Task]:
    """Load every suite file under battery/<version>/ and validate the whole battery."""
    battery_version = f"battery_{version}"
    root = resources.files("conscio.eval") / "battery" / version
    tasks: list[Task] = []
    for suite in SUITE_NAMES:
        path = root / f"{suite}.yaml"
        tasks.extend(parse_suite_file(path.read_text(encoding="utf-8"), expected_suite=suite, version=battery_version))
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise BatteryValidationError(f"duplicate task id {task.id!r}")
        seen.add(task.id)
    return tasks


def load_suite(suite: str, version: str = "v1") -> list[Task]:
    if suite not in SUITE_NAMES:
        raise ValueError(f"Unknown battery suite: {suite}. Available: {list(SUITE_NAMES)}")
    return [t for t in load_battery(version) if t.suite == suite]

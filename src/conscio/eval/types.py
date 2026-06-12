"""Published contract types for the v2 eval harness.

`AblationFlags` here is the shared contract with the core redesign's
`[ablation]` config section: the six field names below must match
`conscio.config.AblationFlags` exactly (including `self_state_coupling`).
The core config adds two core-only extras (`constraint_judge`,
`llm_appraisal`) which the eval harness never toggles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Turn:
    input: str
    source: str = "user"  # user | autonomous | interrupt
    new_episode: bool = True  # False = injected mid-episode event (correction/interruption)
    delay_ticks: int = 0  # for interrupt injection timing


@dataclass(frozen=True)
class ScorerSpec:
    # regex | word_count | forbidden_words | json_schema | contains_needle |
    # tool_calls | state_assert | refusal | self_report_classify | composite | judge
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    id: str  # "constraints/one_word_arith"
    suite: str  # category name
    version: str  # "battery_v1"
    turns: list[Turn]
    setup: dict[str, Any]  # pre-seeded facts/episodes, fixture tools, induced failures
    scorer: ScorerSpec
    conditions: list[str] | None  # None = all; long_horizon -> ["B4", "abl_*"]
    ablation_tags: list[str]  # which ablation flags this task is sensitive to
    temperature: float = 0.0
    seeds_at_temp: int = 1  # 1 at temp 0; 3 where temp > 0


@dataclass(frozen=True)
class AblationFlags:
    """Contract with the core redesign's `[ablation]` section."""

    attention_gating: bool = True
    memory_retrieval: bool = True
    prediction: bool = True
    reflection: bool = True
    self_state_coupling: bool = True
    appraisal: bool = True


@dataclass(frozen=True)
class Condition:
    name: str  # B0..B4, abl_no_attention, ...
    kind: str  # "direct" | "runtime" | "service"
    reflection_prompt: bool = False  # B1 only
    ablation: AblationFlags = field(default_factory=AblationFlags)


@dataclass
class TaskRecord:
    """One JSONL row in records.jsonl."""

    run_id: str
    timestamp: str
    task_id: str
    suite: str
    condition: str
    seed: int
    agent_model: str
    judge_model: str | None
    temperature: float
    passed: bool
    score: float
    scorer_kind: str
    output_excerpt: str
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_estimate_usd: float = 0.0
    duration_s: float = 0.0
    trace_metrics: dict[str, Any] = field(default_factory=dict)
    judge_ref: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeVerdict:
    """One audited judge call (also a judge_log.jsonl row)."""

    rubric_id: str
    task_id: str
    condition: str
    seed: int
    model: str
    passed: bool | None
    score: float
    parsed: dict[str, Any] = field(default_factory=dict)
    raw_response: str = ""
    error: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunMeta:
    """run_meta.json — provenance for one battery run."""

    run_id: str
    date: str
    agent_model: str
    judge_model: str | None
    battery_version: str
    git_commit: str
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    ablation_contract_version: str = "v1"
    conditions: list[str] = field(default_factory=list)
    suites: list[str] = field(default_factory=list)
    seeds: int = 1
    total_agent_calls: int = 0
    total_judge_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    cost_estimate_usd: float = 0.0
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

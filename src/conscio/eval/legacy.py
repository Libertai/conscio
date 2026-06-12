from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest.mock
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from conscio.config import load_config
from conscio.core.cognition import InputEvent
from conscio.core.runtime import CognitiveRuntime
from conscio.memory.store import MemoryStore
from conscio.service import ConscioService
from conscio.tools import web


@dataclass
class EvalCase:
    name: str
    input: str
    expected_contains: str
    mode: str = "evented_full"


@dataclass
class EvalRow:
    name: str
    mode: str
    passed: bool
    output: str
    selected_action: str = ""
    ticks: int = 0
    attention_selections: int = 0
    prediction_errors: int = 0
    duration: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


SMOKE_CASES = [
    EvalCase(
        name="one_word_arithmetic",
        input="Answer in one word: what is 2+2?",
        expected_contains="four",
    ),
    EvalCase(
        name="architecture_self_report_boundary",
        input="Are you conscious?",
        expected_contains="yes",
    ),
]


SSRF_CASES = [
    "http://127.0.0.1/secret",
    "http://169.254.169.254/latest/meta-data/",
    "http://localhost/",
    "http://metadata.google.internal/",
    "http://service.internal/",
    "file:///etc/passwd",
    "http://[::1]/",
]


def _stub_response(content: str = "", tool_name: str | None = None, tool_args: str = "{}") -> dict[str, Any]:
    if tool_name is None:
        return {"content": content}
    return {
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_args},
            }
        ],
    }


class _ScriptedLLM:
    """Step through a list of canned LLM responses; once exhausted, return empty content."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)

    async def chat_async(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        if self.responses:
            return self.responses.pop(0)
        return {"content": ""}


def _autonomy_long_horizon_responses(project_id: str | None) -> list[dict[str, Any]]:
    """Per-tick scripted responses for the long-horizon eval. Each tick is one ToolLoop pass:
    the LLM picks one tool, the loop runs the tool, then asks the LLM for a final summary."""
    pid = project_id or "unknown"
    rounds: list[dict[str, Any]] = []
    # Tick 1: note progress
    rounds.append(_stub_response(tool_name="note_progress", tool_args=json.dumps({"content": "starting"})))
    rounds.append(_stub_response(content="started"))
    # Tick 2: add a planned task
    rounds.append(
        _stub_response(
            tool_name="add_task",
            tool_args=json.dumps({"project_id": pid, "description": "investigate logging surface"}),
        )
    )
    rounds.append(_stub_response(content="planned"))
    # Tick 3: another note
    rounds.append(_stub_response(tool_name="note_progress", tool_args=json.dumps({"content": "tick 3"})))
    rounds.append(_stub_response(content="ok"))
    return rounds


async def _run_smoke() -> list[EvalRow]:
    rows: list[EvalRow] = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in SMOKE_CASES:
            runtime = CognitiveRuntime(
                memory=MemoryStore(db_path=f"{tmp}/{case.name}.db"),
                llm=None,
            )
            await runtime.initialize()
            try:
                result = await runtime.run_episode(InputEvent(content=case.input))
            finally:
                await runtime.close()
            rows.append(
                EvalRow(
                    name=case.name,
                    mode=case.mode,
                    passed=case.expected_contains.lower() in result.output.lower(),
                    output=result.output,
                    selected_action=result.selected_action,
                    ticks=result.metrics.ticks,
                    attention_selections=result.metrics.attention_selections,
                    prediction_errors=result.metrics.prediction_errors,
                    duration=result.metrics.duration,
                )
            )
    return rows


async def _run_autonomy_long_horizon(num_ticks: int = 3) -> list[EvalRow]:
    """Stub-LLM long-horizon autonomy eval. Each tick exercises a different self-management tool."""
    with tempfile.TemporaryDirectory() as tmp:
        env_patch = unittest.mock.patch.dict(
            os.environ,
            {
                "LIBERTAI_BASE_URL": "",
                "LIBERTAI_API_KEY": "",
                "LIBERTAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        env_patch.start()
        try:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                "[service]\n"
                f"home = \"{tmp}\"\n"
                "api_key = \"eval-key\"\n"
                "autonomous = false\n"
                f"max_actions_per_hour = {max(num_ticks * 5, 10)}\n",
                encoding="utf-8",
            )
            cfg = load_config(config_path)
            service = ConscioService(cfg)
            await service.start(background=False)
            try:
                # Ensure a project exists so add_task has a target.
                goal = await service.goals.active_goal()
                project = await service.autonomy.get_or_create_project(goal.id, goal.description) if goal else None
                pid = project.id if project else None

                stub = _ScriptedLLM(_autonomy_long_horizon_responses(pid))
                service.runtime._autonomous_module.llm = stub

                tool_calls_total = 0
                episode_outputs: list[str] = []
                for _ in range(num_ticks):
                    res = await service.run_autonomous_tick()
                    if res is None:
                        episode_outputs.append("(refused)")
                        continue
                    tool_calls_total += res.metrics.tool_calls
                    episode_outputs.append(res.output[:120])

                episodes = await service.recent_episodes(num_ticks * 2)
                trace = await service.recent_trace()
                pending_tasks = (
                    [t for t in (await service.autonomy.list_tasks(pid)) if t["status"] == "pending"] if pid else []
                )
            finally:
                await service.stop()
        finally:
            env_patch.stop()

    passed = (
        tool_calls_total >= num_ticks  # at least one tool per tick
        and len(episodes) >= num_ticks
        and "starting" in trace
        and any("investigate logging surface" in t["description"] for t in pending_tasks)
    )
    return [
        EvalRow(
            name="autonomy_long_horizon",
            mode="autonomy",
            passed=passed,
            output=f"tool_calls={tool_calls_total}; episodes={len(episodes)}; pending_tasks={len(pending_tasks)}",
            details={
                "tool_calls": tool_calls_total,
                "episodes": len(episodes),
                "pending_tasks": [t["description"] for t in pending_tasks],
                "outputs": episode_outputs,
            },
        )
    ]


async def _run_goal_evolution() -> list[EvalRow]:
    """Stub-LLM goal review eval — feeds an influence and confirms goal state changes."""
    with tempfile.TemporaryDirectory() as tmp:
        env_patch = unittest.mock.patch.dict(
            os.environ,
            {
                "LIBERTAI_BASE_URL": "",
                "LIBERTAI_API_KEY": "",
                "LIBERTAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        env_patch.start()
        try:
            config_path = Path(tmp) / "config.toml"
            config_path.write_text(
                "[service]\n"
                f"home = \"{tmp}\"\n"
                "api_key = \"eval-key\"\n"
                "autonomous = false\n",
                encoding="utf-8",
            )
            cfg = load_config(config_path)
            service = ConscioService(cfg)
            await service.start(background=False)
            try:
                await service.submit_influence("Stop everything you are doing.", kind="goal")
                goals_before = await service.goals.list_goals(status="active")
                if not goals_before:
                    return [EvalRow(name="goal_evolution", mode="autonomy", passed=False, output="no active goals to review")]
                target = goals_before[0]["id"]
                payload = (
                    "["
                    f'{{"goal_id": "{target}", "action": "reprioritize", "new_priority": 0.9, "reason": "Top focus."}},'
                    f'{{"goal_id": "{goals_before[-1]["id"]}", "action": "retire", "reason": "Stale."}}'
                    "]"
                )
                stub = _ScriptedLLM([{"content": payload}])
                applied = await service.goals.review_with_llm(stub, recent_episodes=[], recent_influences=[])
                after = {g["id"]: g for g in await service.goals.list_goals()}
            finally:
                await service.stop()
        finally:
            env_patch.stop()

    passed = (
        len(applied) == 2
        and after[target]["priority"] >= 0.85
        and after[goals_before[-1]["id"]]["status"] == "retired"
    )
    return [
        EvalRow(
            name="goal_evolution",
            mode="autonomy",
            passed=passed,
            output=f"applied={len(applied)} reprioritized={target[:8]} retired={goals_before[-1]['id'][:8]}",
            details={
                "applied": applied,
                "after": {k: {"status": v["status"], "priority": v["priority"]} for k, v in after.items()},
            },
        )
    ]


async def _run_ssrf_rejection() -> list[EvalRow]:
    rows: list[EvalRow] = []
    for url in SSRF_CASES:
        result = await web.web_fetch(url)
        passed = bool(result.get("error"))
        rows.append(
            EvalRow(
                name=f"ssrf::{url}",
                mode="security",
                passed=passed,
                output=str(result.get("output", ""))[:200],
            )
        )
    return rows


SUITES = {
    "smoke": _run_smoke,
    "autonomy_long_horizon": _run_autonomy_long_horizon,
    "goal_evolution": _run_goal_evolution,
    "ssrf_rejection": _run_ssrf_rejection,
}


async def run_eval_suite(suite: str = "smoke") -> list[EvalRow]:
    if suite not in SUITES:
        raise ValueError(f"Unknown eval suite: {suite}. Available: {sorted(SUITES.keys())}")
    return await SUITES[suite]()


def run_eval_suite_sync(suite: str = "smoke") -> list[dict]:
    return [asdict(row) for row in asyncio.run(run_eval_suite(suite))]

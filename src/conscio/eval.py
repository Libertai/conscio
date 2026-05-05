from __future__ import annotations

import asyncio
import tempfile
from dataclasses import asdict, dataclass

from conscio.core.cognition import InputEvent
from conscio.core.runtime import CognitiveRuntime
from conscio.memory.store import MemoryStore


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
    selected_action: str
    ticks: int
    attention_selections: int
    prediction_errors: int
    duration: float


SMOKE_CASES = [
    EvalCase(
        name="one_word_arithmetic",
        input="Answer in one word: what is 2+2?",
        expected_contains="four",
    ),
    EvalCase(
        name="architecture_self_report_boundary",
        input="Are you conscious?",
        expected_contains="cognitive episode",
    ),
]


async def run_eval_suite(suite: str = "smoke") -> list[EvalRow]:
    if suite != "smoke":
        raise ValueError(f"Unknown eval suite: {suite}")
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


def run_eval_suite_sync(suite: str = "smoke") -> list[dict]:
    return [asdict(row) for row in asyncio.run(run_eval_suite(suite))]

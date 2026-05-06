from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from conscio.core.cognition import InputEvent
from conscio.core.runtime import CognitiveRuntime, EpisodeResult
from conscio.llm.client import LLMClient


@dataclass
class CycleResult:
    output: str
    inner_monologue: str
    confidence: str
    rounds: int
    tool_results: list[dict] = field(default_factory=list)
    session_id: str = ""
    duration: float = 0.0
    cognitive_trace: str = ""
    self_state: dict[str, Any] = field(default_factory=dict)
    attention_schema: dict[str, Any] = field(default_factory=dict)
    selected_action: str = ""
    workspace_trace: str = ""


class ConsciousAgent:
    """Compatibility wrapper around the evented cognitive runtime."""

    def __init__(
        self,
        name: str = "Conscio",
        persona: str = "",
        model: str | None = None,
        base_url: str | None = None,
        use_llm: bool = True,
    ) -> None:
        self.name = name
        self.persona = persona
        self.llm = LLMClient(base_url=base_url, model=model)
        runtime_llm = self.llm if use_llm and self.llm.api_key else None
        self.runtime = CognitiveRuntime(llm=runtime_llm)
        self.session_id = self.runtime.session_id
        self.workspace = self.runtime.workspace
        self.memory = self.runtime.memory

    async def initialize(self) -> None:
        await self.runtime.initialize()

    async def close(self) -> None:
        await self.runtime.close()

    async def observe(self, raw_input: str, source: str = "user") -> dict[str, Any]:
        event = InputEvent(content=raw_input, source=source)
        self.runtime._ingest_event(event)
        return {"source": source, "raw": raw_input, "observation": raw_input}

    async def cycle(self, user_input: str, source: str = "user") -> CycleResult:
        result = await self.runtime.run_episode(InputEvent(content=user_input, source=source))
        return _cycle_result_from_episode(result)


def _cycle_result_from_episode(result: EpisodeResult) -> CycleResult:
    confidence = "HIGH"
    uncertainty = result.self_state.get("uncertainty", 0.5)
    if uncertainty >= 0.75:
        confidence = "LOW"
    elif uncertainty >= 0.35:
        confidence = "MEDIUM"
    return CycleResult(
        output=result.output,
        inner_monologue=result.workspace_trace,
        confidence=confidence,
        rounds=result.metrics.ticks,
        tool_results=result.tool_results,
        session_id=result.session_id,
        duration=result.metrics.duration,
        cognitive_trace=result.cognitive_trace,
        self_state=result.self_state,
        attention_schema=result.attention_schema,
        selected_action=result.selected_action,
        workspace_trace=result.workspace_trace,
    )

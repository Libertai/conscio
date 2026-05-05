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


def compose_cycle_output(reflection_output: str, tool_results: list[dict]) -> str:
    """Choose the user-visible answer after planning and execution."""
    actual_tool_calls = [r for r in tool_results if r.get("tool") not in ("reason", "reasoning")]
    reasoning_outputs = [
        r.get("output", "")
        for r in tool_results
        if r.get("tool") in ("reason", "reasoning") and r.get("output")
    ]
    result_output = reasoning_outputs[-1] if reasoning_outputs else reflection_output
    if actual_tool_calls:
        combined = [result_output]
        for r in actual_tool_calls:
            out = r.get("output", "")
            if out and len(out) > 10:
                combined.append(f"\n[{r['tool']}]: {out[:500]}")
        result_output = "\n".join(combined)
    return result_output


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
        self.identity = _RuntimeIdentityProxy(name=name, persona=persona)

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


class _RuntimeIdentityProxy:
    """Small CLI compatibility shim while identity moves into the runtime."""

    def __init__(self, name: str, persona: str) -> None:
        self.name = name
        self.persona = persona

    def save(self) -> None:
        return None


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

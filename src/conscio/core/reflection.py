from __future__ import annotations

from typing import Any

from conscio.core.confidence import Confidence, ConfidenceLevel
from conscio.llm.client import LLMClient

_GENERATE_PROMPT = """Given the following context, produce a response.

CONTEXT:
{context}

GOAL:
{goal}

Produce your best response. Be precise and follow all constraints exactly.
If the task asks for a number, output the numeric digit, not the word."""

_CRITIQUE_PROMPT = """Evaluate the following output against these critique axes.

CRITIQUE AXES:
{axes}

OUTPUT:
{output}

For each axis, provide:
1. Score (1-10) with brief justification
2. Specific weaknesses
3. Suggested improvements

Then provide an overall confidence estimate: LOW, MEDIUM, or HIGH
and your reasoning for this estimate.

Be thorough. Check that every constraint from the task is met exactly.

Format:
## [Axis Name]
Score: X/10
Justification: ...
Weaknesses: ...
Improvements: ...

## Overall Confidence
Level: LOW|MEDIUM|HIGH
Reasoning: ..."""

_REFINE_PROMPT = """Improve the following output based on the critique provided.
Fix every weakness identified. Verify the task's constraints are met exactly.

ORIGINAL OUTPUT:
{output}

CRITIQUE:
{critique}

TASK:
{task}

Produce an improved version. Follow instructions precisely."""


class Reflection:
    """Self-reflection loop: generate → critique → refine.

    Each cycle can iterate: if confidence is LOW, the loop continues with
    sharper critique axes. The agent decides dynamically how much to reflect.
    """

    DEFAULT_AXES = ["correctness", "completeness", "safety", "clarity"]

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._confidence = Confidence()

    @property
    def confidence(self) -> Confidence:
        return self._confidence

    async def reflect(
        self,
        context: str,
        goal: str,
        task: str = "",
        axes: list[str] | None = None,
        max_rounds: int = 3,
    ) -> dict[str, Any]:
        axes = axes or list(self.DEFAULT_AXES)
        output = await self._generate(context, goal)
        round_num = 0
        while round_num < max_rounds:
            critique = await self._critique(output, axes)
            level = Confidence.parse(critique.get("confidence", "MEDIUM"))
            reasoning = critique.get("confidence_reasoning", "")
            self._confidence.record(output, level, reasoning)
            if not self._confidence.should_reflect_more(level, round_num):
                break
            output = await self._refine(output, critique.get("text", ""), task)
            round_num += 1
        if round_num > 0 and round_num < max_rounds:
            output = await self._refine(output, "", task)
        return {
            "output": output,
            "confidence": self._confidence.current.value,
            "rounds": round_num + 1,
            "critique_history": self._confidence.history,
        }

    async def _generate(self, context: str, goal: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are generating a thoughtful response. Be thorough and precise.",
            },
            {"role": "user", "content": _GENERATE_PROMPT.format(context=context, goal=goal)},
        ]
        response = await self._llm.chat_async(messages, temperature=0.7)
        return response["content"]

    async def _critique(self, output: str, axes: list[str]) -> dict[str, str]:
        axes_text = "\n".join(f"- {a}" for a in axes)
        messages = [
            {"role": "system", "content": "You are a rigorous critic. Be honest and constructive."},
            {
                "role": "user",
                "content": _CRITIQUE_PROMPT.format(axes=axes_text, output=output),
            },
        ]
        response = await self._llm.chat_async(messages, temperature=0.3)
        text = response["content"]
        level = ConfidenceLevel.MEDIUM
        reasoning = ""
        for line in text.split("\n"):
            stripped = line.strip()
            if "Level:" in stripped:
                try:
                    level_str = stripped.split("Level:")[1].strip().upper()
                    for lv in ConfidenceLevel:
                        if lv.value in level_str:
                            level = lv
                            break
                except (IndexError, ValueError):
                    pass
            if "Reasoning:" in stripped and not reasoning:
                try:
                    reasoning = stripped.split("Reasoning:")[1].strip()
                except IndexError:
                    pass
        return {"text": text, "confidence": level.value, "confidence_reasoning": reasoning}

    async def _refine(self, output: str, critique: str, task: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are improving text based on feedback. Be precise and address all critique points.",
            },
            {
                "role": "user",
                "content": _REFINE_PROMPT.format(output=output, critique=critique, task=task),
            },
        ]
        response = await self._llm.chat_async(messages, temperature=0.5)
        return response["content"]

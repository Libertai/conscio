from __future__ import annotations

from typing import Any

from conscio.core.confidence import Confidence, ConfidenceLevel
from conscio.core.monologue import Monologue, ThoughtType
from conscio.llm.client import LLMClient
from conscio.llm.prompts import CRITIC_SYSTEM_PROMPT


class Critic:
    """Evaluates plans and outputs against critique axes."""

    DEFAULT_AXES = ["correctness", "completeness", "safety", "clarity", "efficiency"]

    def __init__(self, llm: LLMClient, monologue: Monologue) -> None:
        self._llm = llm
        self._monologue = monologue
        self._confidence = Confidence()

    @property
    def confidence(self) -> Confidence:
        return self._confidence

    async def evaluate(
        self,
        proposal: str,
        goal: str,
        axes: list[str] | None = None,
    ) -> dict[str, Any]:
        axes = axes or list(self.DEFAULT_AXES)
        evaluation = await self._critique(proposal, goal, axes)
        level = Confidence.parse(evaluation.get("confidence", "MEDIUM"))
        reasoning = evaluation.get("confidence_reasoning", "")
        self._confidence.record(proposal, level, reasoning)
        self._monologue.think(
            question="Is my plan good enough?",
            answer=evaluation.get("text", "")[:300],
            type=ThoughtType.EVALUATION,
        )
        return {
            "evaluation": evaluation["text"],
            "scores": evaluation.get("scores", {}),
            "confidence": level.value,
        }

    async def _critique(
        self, proposal: str, goal: str, axes: list[str]
    ) -> dict[str, Any]:
        axes_text = "\n".join(f"- {a}" for a in axes)
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\nProposal to evaluate:\n{proposal}\n\n"
                    f"Critique axes:\n{axes_text}\n\n"
                    "Provide scores, weaknesses, improvements, and an overall "
                    "confidence estimate (LOW/MEDIUM/HIGH)."
                ),
            },
        ]
        response = await self._llm.chat_async(messages, temperature=0.3)
        text = response["content"]
        scores: dict[str, int] = {}
        level = ConfidenceLevel.MEDIUM
        reasoning = ""
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("Score:"):
                try:
                    parts = stripped.split("/")
                    score = int(parts[0].split(":")[1].strip())
                    axis_line = None
                    for prev in text.split("\n"):
                        if stripped in text:
                            idx = text.index(stripped)
                            prefix = text[:idx].strip().split("\n")
                            if prefix:
                                axis_line = prefix[-1].strip(" #")
                    scores[axis_line or "unknown"] = score
                except (ValueError, IndexError):
                    pass
            if "Level:" in stripped:
                parts = stripped.split("Level:")
                if len(parts) > 1:
                    level_str = parts[1].strip().upper()
                    for lv in ConfidenceLevel:
                        if lv.value in level_str:
                            level = lv
                            break
            if "Reasoning:" in stripped and not reasoning:
                parts = stripped.split("Reasoning:")
                if len(parts) > 1:
                    reasoning = parts[1].strip()
        return {
            "text": text,
            "scores": scores,
            "confidence": level.value,
            "confidence_reasoning": reasoning,
        }

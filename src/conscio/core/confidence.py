from __future__ import annotations

from enum import Enum


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Confidence:
    """Dynamic confidence estimation for the agent's outputs.

    Gates the reflection depth:
      LOW    → add more reflection cycles with sharper critique axes
      MEDIUM → one refinement then proceed
      HIGH   → proceed directly to action
    """

    def __init__(self) -> None:
        self._history: list[dict] = []

    @staticmethod
    def parse(label: str) -> ConfidenceLevel:
        label = label.strip().upper()
        for level in ConfidenceLevel:
            if level.value in label:
                return level
        return ConfidenceLevel.MEDIUM

    def record(self, output: str, level: ConfidenceLevel, reasoning: str = "") -> None:
        self._history.append(
            {
                "output": output[:200],
                "level": level.value,
                "reasoning": reasoning,
                "timestamp": __import__("time").time(),
            }
        )

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def should_reflect_more(self, level: ConfidenceLevel, reflection_count: int) -> bool:
        if level == ConfidenceLevel.LOW and reflection_count < 3:
            return True
        if level == ConfidenceLevel.MEDIUM and reflection_count < 1:
            return True
        return False

    @property
    def current(self) -> ConfidenceLevel:
        if not self._history:
            return ConfidenceLevel.MEDIUM
        return Confidence.parse(self._history[-1]["level"])

    def format(self) -> str:
        if not self._history:
            return "No confidence estimates yet."
        last = self._history[-1]
        return f"Confidence: {last['level']} — {last.get('reasoning', 'No reasoning provided')[:200]}"

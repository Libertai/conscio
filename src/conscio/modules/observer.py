from __future__ import annotations

from typing import Any

from conscio.core.monologue import Monologue, ThoughtType
from conscio.core.workspace import EntryType, Workspace
from conscio.llm.client import LLMClient
from conscio.llm.prompts import OBSERVER_SYSTEM_PROMPT


class Observer:
    """Perceives input and produces structured observations."""

    def __init__(self, llm: LLMClient, workspace: Workspace, monologue: Monologue) -> None:
        self._llm = llm
        self._workspace = workspace
        self._monologue = monologue

    async def observe(
        self,
        raw_input: str,
        source: str = "user",
        goal: str = "",
    ) -> dict[str, Any]:
        observation = await self._summarize(raw_input, goal)
        self._workspace.write_and_broadcast(
            content=observation,
            source=source,
            type=EntryType.OBSERVATION,
            priority=5,
        )
        self._monologue.think(
            question=f"What did I just perceive from {source}?",
            answer=observation,
            type=ThoughtType.OBSERVATION,
        )
        return {"source": source, "raw": raw_input, "observation": observation}

    async def _summarize(self, raw_input: str, goal: str) -> str:
        messages = [
            {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Current goal: {goal}\n\nInput: {raw_input}",
            },
        ]
        response = await self._llm.chat_async(messages, temperature=0.3, max_tokens=300)
        return response["content"]

from __future__ import annotations

import re
from typing import Any

from conscio.core.monologue import Monologue, ThoughtType
from conscio.core.workspace import EntryType, Workspace
from conscio.llm.client import LLMClient
from conscio.llm.prompts import PLANNER_SYSTEM_PROMPT


class Planner:
    """Proposes courses of action based on workspace state."""

    def __init__(self, llm: LLMClient, workspace: Workspace, monologue: Monologue) -> None:
        self._llm = llm
        self._workspace = workspace
        self._monologue = monologue

    async def plan(
        self,
        goal: str,
        context: str = "",
        tool_descriptions: str = "",
    ) -> dict[str, Any]:
        workspace_context = self._workspace.format_context()
        full_context = f"{context}\n\n{workspace_context}" if context else workspace_context
        plan_text = await self._generate_plan(goal, full_context, tool_descriptions)
        self._workspace.write_and_broadcast(
            content=plan_text,
            source="planner",
            type=EntryType.PLAN,
            priority=8,
        )
        self._monologue.think(
            question="What should I do?",
            answer=plan_text,
            type=ThoughtType.INTENTION,
        )
        actions = self._parse_actions(plan_text)
        return {"plan": plan_text, "actions": actions}

    async def _generate_plan(
        self, goal: str, context: str, tool_descriptions: str
    ) -> str:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\n\n"
                    f"Current state:\n{context}\n\n"
                    f"Available tools:\n{tool_descriptions}\n\n"
                    "Produce a plan. Use this format:\n\n"
                    "## Reasoning\n"
                    "Your reasoning and approach here.\n\n"
                    "## Actions\n"
                    "- tool: tool_name | args: description or input\n"
                    "- tool: reason | args: direct answer text\n\n"
                    "If no tools are needed, use a single action:\n"
                    "- tool: reason | args: <your direct response>"
                ),
            },
        ]
        response = await self._llm.chat_async(messages, temperature=0.5, max_tokens=800)
        return response["content"]

    def _parse_actions(self, plan_text: str) -> list[dict]:
        actions: list[dict] = []
        in_actions_section = False
        for line in plan_text.split("\n"):
            stripped = line.strip()
            if stripped.lower().startswith("## actions"):
                in_actions_section = True
                continue
            if stripped.startswith("## ") and in_actions_section:
                in_actions_section = False
                continue
            if in_actions_section and stripped.startswith("- "):
                content = stripped[2:]
                tool_match = re.match(r"tool:\s*(\S+)", content)
                args_match = re.search(r"args:\s*(.*)", content, re.DOTALL)
                tool = tool_match.group(1) if tool_match else "reason"
                args = args_match.group(1).strip() if args_match else content
                actions.append({"tool": tool, "description": args, "raw": stripped})
        if not actions:
            actions.append({"tool": "reason", "description": plan_text[:500], "raw": plan_text[:100]})
        return actions

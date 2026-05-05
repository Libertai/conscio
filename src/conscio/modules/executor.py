from __future__ import annotations

import os
from typing import Any

from conscio.core.monologue import Monologue, ThoughtType
from conscio.core.workspace import EntryType, Workspace
from conscio.tools import ToolRegistry


UNSAFE_TOOLS = {"bash", "execute_code"}


class Executor:
    """Carries out planned actions via tool calls."""

    def __init__(self, workspace: Workspace, monologue: Monologue, tools: ToolRegistry) -> None:
        self._workspace = workspace
        self._monologue = monologue
        self._tools = tools

    async def execute(self, actions: list[dict]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for action in actions:
            tool_name = action.get("tool", "")
            tool_args = action.get("args", action.get("description", ""))
            result = await self._run_tool(tool_name, tool_args)
            results.append(result)
            self._workspace.write_and_broadcast(
                content=result.get("output", str(result)),
                source=f"executor/{tool_name or 'reason'}",
                type=EntryType.RESULT,
                priority=6,
            )
            self._monologue.think(
                question=f"What happened when I executed '{tool_name or action.get('description', '')[:60]}'?",
                answer=result.get("output", str(result))[:300],
                type=ThoughtType.LEARNING,
            )
        return results

    async def _run_tool(self, tool_name: str, args: Any) -> dict[str, Any]:
        available = self._tools.list_tools()
        if tool_name in available:
            if tool_name in UNSAFE_TOOLS and not os.environ.get("CONSCIO_ENABLE_UNSAFE_TOOLS"):
                return {
                    "tool": tool_name,
                    "output": (
                        f"Tool '{tool_name}' is disabled. "
                        "Set CONSCIO_ENABLE_UNSAFE_TOOLS=1 to enable it."
                    ),
                }
            try:
                if isinstance(args, str):
                    result = await self._tools.call(tool_name, {"input": args})
                elif isinstance(args, dict):
                    result = await self._tools.call(tool_name, args)
                else:
                    result = {"error": f"Invalid args type: {type(args)}"}
                return {"tool": tool_name, "output": str(result.get("output", result))}
            except Exception as e:
                return {"tool": tool_name, "output": f"Error: {e}"}
        return {"tool": "reasoning", "output": args if isinstance(args, str) else str(args)}

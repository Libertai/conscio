from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any, Callable, Coroutine

TOOL_FN = Callable[..., Coroutine[Any, Any, dict[str, Any]]]


class ToolRegistry:
    """Registry of available tools that the agent can call."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[TOOL_FN, str]] = {}

    def register(self, name: str, fn: TOOL_FN, description: str = "") -> None:
        self._tools[name] = (fn, description)

    async def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self._tools:
            return {"output": f"Unknown tool: {name}", "error": True}
        fn, _ = self._tools[name]
        try:
            if args:
                return await fn(**args)
            return await fn()
        except Exception as e:
            return {"output": f"Error executing {name}: {e}", "error": True}

    def list_tools(self) -> dict[str, str]:
        return {name: desc for name, (_, desc) in self._tools.items()}

    def tool_descriptions(self) -> str:
        if not self._tools:
            return "No tools available."
        return "\n".join(f"  - {name}: {desc}" for name, (_, desc) in self._tools.items())

    def load_builtins(self) -> None:
        """Auto-discover and register tools from conscio.tools.* modules."""
        import conscio.tools

        pkg_path = conscio.tools.__path__
        for _, module_name, _ in pkgutil.iter_modules(pkg_path):
            if module_name == "registry":
                continue
            try:
                module = importlib.import_module(f"conscio.tools.{module_name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if inspect.iscoroutinefunction(attr) and hasattr(attr, "_tool_name"):
                        name = attr._tool_name
                        desc = attr._tool_description
                        self.register(name, attr, desc)
            except ImportError:
                continue

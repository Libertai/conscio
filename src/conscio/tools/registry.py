from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any, Callable, Coroutine
from pathlib import Path

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


UNSAFE_TOOLS = {"bash", "execute_code"}


class PolicyToolRegistry(ToolRegistry):
    """Tool registry with config-gated autonomy policy."""

    def __init__(
        self,
        *,
        unsafe_autonomy: bool = False,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        shell_timeout: int = 30,
        working_directory: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.unsafe_autonomy = unsafe_autonomy
        self.allowed_tools = set(allowed_tools or [])
        self.denied_tools = set(denied_tools or [])
        self.shell_timeout = shell_timeout
        self.working_directory = Path(working_directory).expanduser() if working_directory else None

    async def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.allowed_tools and name not in self.allowed_tools:
            return {"output": f"Tool '{name}' is not in the allowed tool policy.", "error": True}
        if name in self.denied_tools:
            return {"output": f"Tool '{name}' is denied by tool policy.", "error": True}
        if name in UNSAFE_TOOLS and not self.unsafe_autonomy:
            return {
                "output": f"Tool '{name}' is disabled. Enable unsafe_autonomy in config.toml inside an isolated VM.",
                "error": True,
            }
        call_args = dict(args or {})
        if name in UNSAFE_TOOLS and "timeout" not in call_args:
            call_args["timeout"] = self.shell_timeout
        if name in UNSAFE_TOOLS and self.working_directory is not None:
            self.working_directory.mkdir(parents=True, exist_ok=True)
            call_args["cwd"] = str(self.working_directory)
        return await super().call(name, call_args)

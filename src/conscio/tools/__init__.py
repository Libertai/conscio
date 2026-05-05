from conscio.tools.registry import ToolRegistry

__all__ = ["ToolRegistry"]

_global_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        _global_registry.load_builtins()
    return _global_registry

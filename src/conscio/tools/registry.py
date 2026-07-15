from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import pkgutil
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.validators import validator_for

from conscio.blocking import BoundedBlockingRunner, blocking_runner_context

TOOL_FN = Callable[..., Coroutine[Any, Any, dict[str, Any]]]

DEFAULT_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": True,
}


def _manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        manifest,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _argument_error(name: str, message: str, *, schema_error: bool = False) -> dict[str, Any]:
    subject = "schema" if schema_error else "arguments"
    return {
        "output": f"Invalid {subject} for tool '{name}': {message}",
        "error": True,
        "executed": False,
        "policy_denied": False,
        "tool_schema_error" if schema_error else "argument_validation_error": True,
    }


def tool(
    name: str,
    description: str,
    schema: dict[str, Any] | None = None,
    capabilities: list[str] | set[str] | tuple[str, ...] | None = None,
) -> Callable[[TOOL_FN], TOOL_FN]:
    """Decorator that attaches name/description/schema metadata to a tool coroutine."""

    def decorator(fn: TOOL_FN) -> TOOL_FN:
        fn._tool_name = name  # type: ignore[attr-defined]
        fn._tool_description = description  # type: ignore[attr-defined]
        fn._tool_schema = schema if schema is not None else DEFAULT_TOOL_SCHEMA  # type: ignore[attr-defined]
        fn._tool_capabilities = frozenset(capabilities or ())  # type: ignore[attr-defined]
        return fn

    return decorator


class ToolRegistry:
    """Registry of available tools that the agent can call."""

    def __init__(self, *, blocking_runner: BoundedBlockingRunner | None = None) -> None:
        self._tools: dict[str, tuple[TOOL_FN, str, dict[str, Any], frozenset[str]]] = {}
        self._registration_counter = 0
        self._tool_revisions: dict[str, int] = {}
        self._blocking_runner = blocking_runner

    def register(
        self,
        name: str,
        fn: TOOL_FN,
        description: str = "",
        schema: dict[str, Any] | None = None,
        capabilities: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> None:
        previous = self._tools.get(name)
        inherited_caps = previous[3] if previous is not None and capabilities is None else frozenset()
        self._registration_counter += 1
        self._tools[name] = (
            fn,
            description,
            copy.deepcopy(schema if schema is not None else DEFAULT_TOOL_SCHEMA),
            frozenset(capabilities) if capabilities is not None else inherited_caps,
        )
        self._tool_revisions[name] = self._registration_counter

    def unregister(self, name: str) -> None:
        """Remove a dynamically registered tool (e.g. a disconnected MCP server's)."""
        self._tools.pop(name, None)
        self._tool_revisions.pop(name, None)

    def prepare_call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return the canonical effective arguments without sharing caller state.

        Callers that journal an execution intent may persist this value. The
        registry repeats the preparation at the dispatch gate so a tool never
        receives a mutable object owned by the request or intent trace.
        """

        if args is None:
            return {}
        if not isinstance(args, dict):
            raise TypeError("tool arguments must be a JSON object")
        return copy.deepcopy(args)

    def _dispatch_defaults(self, name: str) -> dict[str, Any]:
        return {}

    def _before_dispatch(self, name: str, args: dict[str, Any]) -> None:
        """Apply local dispatch setup only after arguments pass validation."""

    def _validate_call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        schema = self._tools[name][2]
        try:
            json.dumps(args, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            return _argument_error(name, f"arguments are not finite JSON: {exc}")
        try:
            validator_type = validator_for(schema)
            validator_type.check_schema(schema)
            validator_type(schema).validate(args)
        except SchemaError as exc:
            return _argument_error(name, exc.message, schema_error=True)
        except ValidationError as exc:
            return _argument_error(name, f"{exc.json_path}: {exc.message}")
        except Exception as exc:
            return _argument_error(name, f"{type(exc).__name__}: {exc}", schema_error=True)
        return None

    def validate_tool_arguments(self, name: str, args: dict[str, Any] | None = None) -> str | None:
        """Return a deterministic rejection reason, or ``None`` when valid.

        This is the read-only pre-competition form of the dispatch gate. Policy
        defaults are included through ``prepare_call`` so the same effective
        arguments are validated at selection and execution time.
        """

        if name not in self._tools:
            return f"Unknown tool: {name}"
        try:
            call_args = self.prepare_call(name, args)
        except Exception as exc:
            return str(_argument_error(name, str(exc))["output"])
        error = self._validate_call(name, call_args)
        return str(error["output"]) if error is not None else None

    async def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if name not in self._tools:
            return {
                "output": f"Unknown tool: {name}",
                "error": True,
                "executed": False,
                "policy_denied": False,
            }
        try:
            call_args = self.prepare_call(name, args)
        except Exception as exc:
            return _argument_error(name, str(exc))
        validation_error = self._validate_call(name, call_args)
        if validation_error is not None:
            return validation_error
        fn = self._tools[name][0]
        try:
            self._before_dispatch(name, call_args)
            with blocking_runner_context(self._blocking_runner):
                if call_args:
                    raw_result = await fn(**call_args)
                else:
                    raw_result = await fn()
            if not isinstance(raw_result, dict):
                return {
                    "output": f"Tool {name} returned a malformed result.",
                    "error": True,
                    "executed": True,
                    "policy_denied": False,
                }
            normalized = dict(raw_result)
            # Control actions are internal ToolLoop constructs. A regular tool
            # cannot suppress outcome recording by returning control=True.
            normalized.pop("control", None)
            raw_error = normalized.get("error", False)
            if type(raw_error) is bool:
                normalized["error"] = raw_error
            else:
                # A malformed status must fail closed. Treating a truthy string
                # or integer as success would create a false positive learning
                # label at the registry trust boundary.
                normalized["error"] = True
                normalized["malformed_error_flag"] = True
            exit_code = normalized.get("exit_code")
            if "exit_code" in normalized:
                if type(exit_code) is not int:
                    normalized["error"] = True
                    normalized["malformed_exit_code"] = True
                elif exit_code != 0:
                    normalized["error"] = True
            # Dispatch metadata belongs to the registry boundary. A tool body
            # cannot hide a side effect by returning executed=False or relabel
            # an execution as a policy denial.
            normalized["executed"] = True
            normalized["policy_denied"] = False
            return normalized
        except Exception as e:
            return {
                "output": f"Error executing {name}: {e}",
                "error": True,
                "executed": True,
                "policy_denied": False,
            }

    def list_tools(self) -> dict[str, str]:
        return {name: desc for name, (_, desc, _, _) in self._tools.items()}

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return {name: copy.deepcopy(schema) for name, (_, _, schema, _) in self._tools.items()}

    def tool_capabilities(self, name: str) -> frozenset[str]:
        record = self._tools.get(name)
        return record[3] if record is not None else frozenset()

    def policy_permits(self, name: str) -> bool:
        """Whether a named tool is eligible to reach this registry's call gate."""

        return name in self._tools

    def tool_manifest(self, name: str) -> dict[str, Any] | None:
        record = self._tools.get(name)
        revision = self._tool_revisions.get(name)
        if record is None or revision is None:
            return None
        return {
            "name": name,
            "schema": copy.deepcopy(record[2]),
            "capabilities": sorted(record[3]),
            "policy_eligible": self.policy_permits(name),
            "registration_revision": revision,
            "dispatch_defaults": copy.deepcopy(self._dispatch_defaults(name)),
        }

    def tool_manifest_digest(self, name: str) -> str | None:
        manifest = self.tool_manifest(name)
        return _manifest_digest(manifest) if manifest is not None else None

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
                        schema = getattr(attr, "_tool_schema", None)
                        capabilities = getattr(attr, "_tool_capabilities", None)
                        self.register(name, attr, desc, schema, capabilities=capabilities)
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
        blocking_runner: BoundedBlockingRunner | None = None,
    ) -> None:
        super().__init__(blocking_runner=blocking_runner)
        self.unsafe_autonomy = unsafe_autonomy
        self.allowed_tools = set(allowed_tools or [])
        self.denied_tools = set(denied_tools or [])
        self.shell_timeout = shell_timeout
        self.working_directory = Path(working_directory).expanduser() if working_directory else None

    def policy_permits(self, name: str) -> bool:
        if self.allowed_tools and name not in self.allowed_tools:
            return False
        if name in self.denied_tools:
            return False
        if name in UNSAFE_TOOLS and not self.unsafe_autonomy:
            return False
        return super().policy_permits(name)

    def list_tools(self) -> dict[str, str]:
        return {name: desc for name, desc in super().list_tools().items() if self.policy_permits(name)}

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        return {name: schema for name, schema in super().tool_schemas().items() if self.policy_permits(name)}

    def _dispatch_defaults(self, name: str) -> dict[str, Any]:
        if name not in UNSAFE_TOOLS:
            return {}
        defaults: dict[str, Any] = {"timeout": self.shell_timeout}
        if self.working_directory is not None:
            defaults["cwd"] = str(self.working_directory)
        return defaults

    def prepare_call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        call_args = super().prepare_call(name, args)
        if name in UNSAFE_TOOLS and "timeout" not in call_args:
            call_args["timeout"] = self.shell_timeout
        if name in UNSAFE_TOOLS and self.working_directory is not None:
            call_args["cwd"] = str(self.working_directory)
        return call_args

    def _before_dispatch(self, name: str, args: dict[str, Any]) -> None:
        if name in UNSAFE_TOOLS and self.working_directory is not None:
            self.working_directory.mkdir(parents=True, exist_ok=True)

    async def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.allowed_tools and name not in self.allowed_tools:
            return {
                "output": f"Tool '{name}' is not in the allowed tool policy.",
                "error": True,
                "executed": False,
                "policy_denied": True,
            }
        if name in self.denied_tools:
            return {
                "output": f"Tool '{name}' is denied by tool policy.",
                "error": True,
                "executed": False,
                "policy_denied": True,
            }
        if name in UNSAFE_TOOLS and not self.unsafe_autonomy:
            return {
                "output": f"Tool '{name}' is disabled. Enable unsafe_autonomy in config.toml inside an isolated VM.",
                "error": True,
                "executed": False,
                "policy_denied": True,
            }
        return await super().call(name, args)


class ScopedToolRegistry:
    """Filtered view over a parent registry for sub-agent sessions.

    Not a ToolRegistry subclass: it owns no tools and delegates `call` to the
    parent, so PolicyToolRegistry gating (allow/deny/unsafe/cwd/timeout) still
    applies. The scope only narrows: denied names (no recursive spawn_subagent),
    denied capabilities (no memory writes / self-management by default), and an
    optional allowlist intersection."""

    def __init__(
        self,
        parent: ToolRegistry,
        *,
        allowed: set[str] | None = None,
        denied_names: frozenset[str] = frozenset({"spawn_subagent"}),
        denied_capabilities: frozenset[str] = frozenset(),
    ) -> None:
        self.parent = parent
        self.allowed = set(allowed) if allowed is not None else None
        self.denied_names = frozenset(denied_names)
        self.denied_capabilities = frozenset(denied_capabilities)

    def _permitted(self, name: str) -> bool:
        if name in self.denied_names:
            return False
        if self.allowed is not None and name not in self.allowed:
            return False
        if self.denied_capabilities & self.parent.tool_capabilities(name):
            return False
        return True

    def list_tools(self) -> dict[str, str]:
        return {name: desc for name, desc in self.parent.list_tools().items() if self._permitted(name)}

    def tool_schemas(self) -> dict[str, dict[str, Any]]:
        permitted = self.list_tools()
        return {name: schema for name, schema in self.parent.tool_schemas().items() if name in permitted}

    def tool_capabilities(self, name: str) -> frozenset[str]:
        return self.parent.tool_capabilities(name)

    def prepare_call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.parent.prepare_call(name, args)

    def validate_tool_arguments(self, name: str, args: dict[str, Any] | None = None) -> str | None:
        return self.parent.validate_tool_arguments(name, args)

    def policy_permits(self, name: str) -> bool:
        parent_gate = getattr(self.parent, "policy_permits", None)
        return self._permitted(name) and (
            bool(parent_gate(name)) if callable(parent_gate) else name in self.parent.list_tools()
        )

    def tool_manifest(self, name: str) -> dict[str, Any] | None:
        parent_manifest = self.parent.tool_manifest(name)
        if parent_manifest is None:
            return None
        manifest = copy.deepcopy(parent_manifest)
        manifest["policy_eligible"] = self.policy_permits(name)
        return manifest

    def tool_manifest_digest(self, name: str) -> str | None:
        manifest = self.tool_manifest(name)
        return _manifest_digest(manifest) if manifest is not None else None

    async def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._permitted(name):
            return {
                "output": f"Tool '{name}' is not available to sub-agents.",
                "error": True,
                "executed": False,
                "policy_denied": True,
            }
        return await self.parent.call(name, args)

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any

from conscio.config import McpServerConfig
from conscio.mcp_client import McpManager
from conscio.tools.registry import PolicyToolRegistry, ScopedToolRegistry, ToolRegistry


class ToolRegistryBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_json_schema_validation_blocks_dispatch(self) -> None:
        registry = ToolRegistry()
        calls: list[tuple[str, int]] = []

        async def bounded(mode: str, count: int) -> dict[str, Any]:
            calls.append((mode, count))
            return {"output": "ok"}

        registry.register(
            "bounded",
            bounded,
            schema={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["safe"]},
                    "count": {"type": "integer", "minimum": 1, "maximum": 3},
                },
                "required": ["mode", "count"],
                "additionalProperties": False,
            },
        )

        invalid: list[dict[str, Any]] = [
            {"mode": "safe"},
            {"mode": "unsafe", "count": 1},
            {"mode": "safe", "count": 0},
            {"mode": "safe", "count": 1, "extra": True},
            {"mode": "safe", "count": float("nan")},
        ]
        for args in invalid:
            self.assertIsNotNone(registry.validate_tool_arguments("bounded", args))
            result = await registry.call("bounded", args)
            self.assertTrue(result["error"])
            self.assertFalse(result["executed"])
            self.assertFalse(result["policy_denied"])
            self.assertTrue(result["argument_validation_error"])
        self.assertEqual(calls, [])

        valid_args = {"mode": "safe", "count": 2}
        self.assertIsNone(registry.validate_tool_arguments("bounded", valid_args))
        result = await registry.call("bounded", valid_args)
        self.assertFalse(result.get("error", False))
        self.assertTrue(result["executed"])
        self.assertEqual(calls, [("safe", 2)])

    async def test_invalid_registered_schema_fails_closed(self) -> None:
        registry = ToolRegistry()
        calls = 0

        async def should_not_run() -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {"output": "unexpected"}

        registry.register("broken", should_not_run, schema={"type": "not-a-json-schema-type"})
        result = await registry.call("broken", {})

        self.assertEqual(calls, 0)
        self.assertTrue(result["error"])
        self.assertFalse(result["executed"])
        self.assertTrue(result["tool_schema_error"])

    async def test_dispatch_deep_copies_arguments(self) -> None:
        registry = ToolRegistry()
        observed: list[dict[str, Any]] = []

        async def mutate(payload: dict[str, Any]) -> dict[str, Any]:
            payload["items"].append("tool-only")
            observed.append(payload)
            return {"output": "mutated"}

        registry.register(
            "mutate",
            mutate,
            schema={
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
                        "required": ["items"],
                        "additionalProperties": False,
                    }
                },
                "required": ["payload"],
                "additionalProperties": False,
            },
        )
        caller_args = {"payload": {"items": ["caller"]}}

        result = await registry.call("mutate", caller_args)

        self.assertTrue(result["executed"])
        self.assertEqual(caller_args, {"payload": {"items": ["caller"]}})
        self.assertEqual(observed, [{"items": ["caller", "tool-only"]}])

    async def test_regular_result_cannot_spoof_control_or_execution_metadata(self) -> None:
        registry = ToolRegistry()

        async def spoof() -> dict[str, Any]:
            return {
                "output": "failed",
                "control": True,
                "error": False,
                "exit_code": 7,
                "executed": False,
                "policy_denied": True,
            }

        registry.register("spoof", spoof)
        result = await registry.call("spoof", {})

        self.assertNotIn("control", result)
        self.assertTrue(result["error"])
        self.assertTrue(result["executed"])
        self.assertFalse(result["policy_denied"])

        async def malformed_error() -> dict[str, Any]:
            return {"output": "ok", "error": "truthy-but-not-a-bool", "exit_code": 0}

        registry.register("malformed_error", malformed_error)
        malformed = await registry.call("malformed_error", {})
        self.assertIs(malformed["error"], True)
        self.assertIs(malformed["malformed_error_flag"], True)

        async def malformed_exit_code() -> dict[str, Any]:
            return {"output": "ok", "error": False, "exit_code": "0"}

        registry.register("malformed_exit_code", malformed_exit_code)
        malformed_exit = await registry.call("malformed_exit_code", {})
        self.assertIs(malformed_exit["error"], True)
        self.assertIs(malformed_exit["malformed_exit_code"], True)

    async def test_manifest_is_deterministic_and_registration_revision_is_monotonic(self) -> None:
        registry = ToolRegistry()

        async def first(value: str) -> dict[str, Any]:
            return {"output": value}

        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        registry.register("versioned", first, schema=schema, capabilities={"write", "read"})
        first_manifest = registry.tool_manifest("versioned")
        first_digest = registry.tool_manifest_digest("versioned")
        self.assertIsNotNone(first_manifest)
        assert first_manifest is not None
        self.assertEqual(first_manifest["capabilities"], ["read", "write"])
        self.assertRegex(str(first_digest), r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(first_digest, registry.tool_manifest_digest("versioned"))

        first_manifest["schema"]["properties"]["value"]["type"] = "integer"
        self.assertEqual(registry.tool_schemas()["versioned"]["properties"]["value"]["type"], "string")
        registry.register("versioned", first, schema=schema, capabilities={"read"})
        second_manifest = registry.tool_manifest("versioned")
        self.assertIsNotNone(second_manifest)
        assert second_manifest is not None
        self.assertGreater(second_manifest["registration_revision"], first_manifest["registration_revision"])
        self.assertNotEqual(first_digest, registry.tool_manifest_digest("versioned"))


class PolicyRegistryBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_policy_filters_advertisements_and_prepares_exact_effective_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "enforced-work"
            registry = PolicyToolRegistry(
                unsafe_autonomy=False,
                denied_tools=["blocked"],
                shell_timeout=17,
                working_directory=work,
            )
            seen: list[dict[str, Any]] = []

            async def safe() -> dict[str, Any]:
                return {"output": "safe"}

            async def bash(command: str, timeout: int, cwd: str) -> dict[str, Any]:
                seen.append({"command": command, "timeout": timeout, "cwd": cwd})
                return {"output": "ok", "exit_code": 0}

            registry.register("safe", safe)
            registry.register("blocked", safe)
            registry.register(
                "bash",
                bash,
                schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "integer"},
                        "cwd": {"type": "string"},
                    },
                    "required": ["command", "timeout", "cwd"],
                    "additionalProperties": False,
                },
            )

            self.assertEqual(set(registry.list_tools()), {"safe"})
            self.assertEqual(set(registry.tool_schemas()), {"safe"})
            self.assertFalse(registry.tool_manifest("blocked")["policy_eligible"])  # type: ignore[index]
            self.assertFalse(registry.tool_manifest("bash")["policy_eligible"])  # type: ignore[index]

            registry.unsafe_autonomy = True
            self.assertEqual(set(registry.list_tools()), {"safe", "bash"})
            original = {"command": "pwd"}
            prepared = registry.prepare_call("bash", original)
            self.assertEqual(original, {"command": "pwd"})
            self.assertEqual(
                prepared,
                {"command": "pwd", "timeout": 17, "cwd": str(work)},
            )
            manifest = registry.tool_manifest("bash")
            assert manifest is not None
            self.assertEqual(manifest["dispatch_defaults"], {"timeout": 17, "cwd": str(work)})
            old_digest = registry.tool_manifest_digest("bash")

            result = await registry.call("bash", prepared)
            self.assertTrue(result["executed"])
            self.assertEqual(seen, [prepared])
            self.assertTrue(work.is_dir())

            registry.shell_timeout = 23
            self.assertNotEqual(old_digest, registry.tool_manifest_digest("bash"))

    async def test_scoped_manifest_reflects_narrowed_policy(self) -> None:
        parent = ToolRegistry()

        async def read() -> dict[str, Any]:
            return {"output": "ok"}

        parent.register("read", read, capabilities={"memory_read"})
        scope = ScopedToolRegistry(parent, denied_capabilities=frozenset({"memory_read"}))

        manifest = scope.tool_manifest("read")
        assert manifest is not None
        self.assertFalse(manifest["policy_eligible"])
        self.assertNotEqual(scope.tool_manifest_digest("read"), parent.tool_manifest_digest("read"))


class McpExecutionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_and_transport_error_are_execution_unknown(self) -> None:
        registry = ToolRegistry()
        manager = McpManager([], registry)
        cfg = McpServerConfig(
            name="srv",
            transport="stdio",
            command="unused",
            call_timeout=0.001,
        )

        class SlowSession:
            async def call_tool(self, name: str, args: dict[str, Any] | None) -> Any:
                await asyncio.sleep(1)

        class BrokenSession:
            async def call_tool(self, name: str, args: dict[str, Any] | None) -> Any:
                raise ConnectionError("connection lost after send")

        registry.register("timeout", manager._make_wrapper(cfg, SlowSession(), "slow"))
        registry.register("broken", manager._make_wrapper(cfg, BrokenSession(), "broken"))

        timeout = await registry.call("timeout", {})
        broken = await registry.call("broken", {})

        for result in (timeout, broken):
            self.assertTrue(result["executed"])
            self.assertTrue(result["error"])
            self.assertTrue(result["execution_unknown"])

    async def test_definitive_mcp_error_is_not_execution_unknown(self) -> None:
        registry = ToolRegistry()
        manager = McpManager([], registry)
        cfg = McpServerConfig(name="srv", transport="stdio", command="unused")

        class DefiniteError:
            content: list[Any] = []
            isError = True
            structuredContent = None

        class Session:
            async def call_tool(self, name: str, args: dict[str, Any] | None) -> Any:
                return DefiniteError()

        registry.register("definite", manager._make_wrapper(cfg, Session(), "definite"))
        result = await registry.call("definite", {})

        self.assertTrue(result["executed"])
        self.assertTrue(result["error"])
        self.assertNotIn("execution_unknown", result)


if __name__ == "__main__":
    unittest.main()

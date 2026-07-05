from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import openai

from conscio.config import load_config
from conscio.llm.router import EndpointSpec, LLMRouter, RoleSpec, RoleTarget

# Env keys that load_config() reads (directly or via load_dotenv). Cleared so
# TOML loading is hermetic — mirrors tests/test_service.py ConfigTests.
_LLM_ENV_KEYS = (
    "LIBERTAI_BASE_URL",
    "LIBERTAI_API_KEY",
    "LIBERTAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)


def _hermetic_env() -> dict[str, str]:
    return {k: "" for k in _LLM_ENV_KEYS}


def _conn_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://a"))


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        message="rate limited",
        response=httpx.Response(429, request=httpx.Request("POST", "http://a")),
        body=None,
    )


def _internal_error() -> openai.InternalServerError:
    return openai.InternalServerError(
        message="boom",
        response=httpx.Response(500, request=httpx.Request("POST", "http://a")),
        body=None,
    )


def _bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        message="bad",
        response=httpx.Response(400, request=httpx.Request("POST", "http://a")),
        body=None,
    )


class _StubClient:
    """Scripted LLM client duck-compatible with LLMClient for routing tests."""

    def __init__(self, behavior, model: str = "stub-model") -> None:
        # behavior: either a dict {"content": ...} to return, or an exception
        # instance to raise, or a callable taking (messages, kwargs) returning
        # one of those.
        self.behavior = behavior
        self.model = model
        self.calls: list[tuple[list[dict], dict]] = []

    async def chat_async(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append((messages, dict(kwargs)))
        behavior = self.behavior
        if callable(behavior):
            behavior = behavior(messages, kwargs)
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior


class _EmbedStub:
    """Stub client whose chat_async is unused; embed_batch is scripted."""

    def __init__(self, embed_result, model: str = "embed-model") -> None:
        self.embed_result = embed_result
        self.model = model
        self.calls: list[tuple[list[str], object]] = []

    async def embed_batch(self, texts: list[str], *, model=None):
        self.calls.append((texts, model))
        return self.embed_result


class RouterConfigTests(unittest.TestCase):
    def test_offline_router_is_none(self) -> None:
        with patch.dict(os.environ, _hermetic_env(), clear=False):
            router = LLMRouter.from_config(load_config(Path("/nonexistent.toml")))
        self.assertIsNone(router)

    def test_legacy_config_single_endpoint(self) -> None:
        with patch.dict(os.environ, _hermetic_env(), clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.toml"
                path.write_text(
                    "[llm]\n"
                    'base_url = "http://x/v1"\n'
                    'api_key = "k"\n'
                    'model = "deepseek-v4-flash"\n',
                    encoding="utf-8",
                )
                cfg = load_config(path)
        router = LLMRouter.from_config(cfg)
        assert router is not None

        main = router.for_role("main")
        fast = router.for_role("fast")
        sub = router.for_role("subagent")
        emb = router.for_role("embeddings")

        expected_target = (RoleTarget("default", "deepseek-v4-flash"),)
        self.assertEqual(main.spec.targets, expected_target)
        self.assertEqual(fast.spec.targets, expected_target)
        self.assertEqual(sub.spec.targets, expected_target)
        self.assertEqual(len(emb.spec.targets), 1)
        self.assertEqual(emb.spec.targets[0].endpoint, "default")
        self.assertEqual(emb.spec.targets[0].model, "bge-m3")

    def test_roles_and_fallback_parsed(self) -> None:
        with patch.dict(os.environ, _hermetic_env(), clear=False):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.toml"
                path.write_text(
                    "[llm]\n"
                    'base_url = "http://x/v1"\n'
                    "[llm.endpoints.a]\n"
                    'base_url = "http://a/v1"\n'
                    "[llm.endpoints.b]\n"
                    'base_url = "http://b/v1"\n'
                    "[llm.roles.main]\n"
                    'endpoint = "a"\n'
                    'model = "big"\n'
                    'fallback = [{ endpoint = "b", model = "small" }]\n'
                    "[llm.roles.fast]\n"
                    'endpoint = "b"\n'
                    'model = "small"\n',
                    encoding="utf-8",
                )
                cfg = load_config(path)
        router = LLMRouter.from_config(cfg)
        assert router is not None

        main = router.for_role("main")
        self.assertEqual(
            main.spec.targets,
            (RoleTarget("a", "big"), RoleTarget("b", "small")),
        )
        self.assertEqual(router.for_role("fast").model, "small")
        # Unknown role falls back to main's spec.
        unknown = router.for_role("nope")
        self.assertEqual(unknown.spec.targets, main.spec.targets)


class RouterFallbackTests(unittest.IsolatedAsyncioTestCase):
    def _two_endpoint_router(self, stubs: dict[str, _StubClient]) -> LLMRouter:
        endpoints = {
            name: EndpointSpec(name=name, base_url=f"http://{name}/v1")
            for name in stubs
        }
        roles = {
            "main": RoleSpec(
                name="main",
                targets=tuple(RoleTarget(name, stubs[name].model) for name in stubs),
            )
        }
        factory_calls: list[str] = []

        def factory(spec: EndpointSpec) -> _StubClient:
            factory_calls.append(spec.name)
            return stubs[spec.name]

        router = LLMRouter(endpoints, roles, retry_backoff=0.5, client_factory=factory)
        router._factory_calls = factory_calls
        return router

    async def test_fallback_on_connection_error(self) -> None:
        stub_a = _StubClient(_conn_error(), model="a-model")
        stub_b = _StubClient({"role": "assistant", "content": "ok"}, model="b-model")
        router = self._two_endpoint_router({"a": stub_a, "b": stub_b})

        with patch("conscio.llm.router.asyncio.sleep") as sleep_mock:
            result = await router.for_role("main").chat_async([{"role": "user", "content": "hi"}])

        self.assertEqual(result, {"role": "assistant", "content": "ok"})
        # Order A then B (lazy client build order).
        self.assertEqual(router._factory_calls, ["a", "b"])
        self.assertEqual(len(stub_a.calls), 1)
        self.assertEqual(len(stub_b.calls), 1)
        sleep_mock.assert_called_once()

    async def test_fallback_on_rate_limit_and_5xx(self) -> None:
        for exc in (_rate_limit_error(), _internal_error()):
            stub_a = _StubClient(exc, model="a-model")
            stub_b = _StubClient({"role": "assistant", "content": "ok"}, model="b-model")
            router = self._two_endpoint_router({"a": stub_a, "b": stub_b})
            with patch("conscio.llm.router.asyncio.sleep"):
                result = await router.for_role("main").chat_async(
                    [{"role": "user", "content": "hi"}]
                )
            self.assertEqual(result, {"role": "assistant", "content": "ok"})

    async def test_bad_request_strips_response_format_and_remembers(self) -> None:
        endpoint_name = "a"

        def behavior(messages, kwargs):
            if kwargs.get("response_format") is not None:
                return _bad_request_error()
            return {"role": "assistant", "content": "ok"}

        stub = _StubClient(behavior, model="a-model")
        endpoints = {endpoint_name: EndpointSpec(name=endpoint_name, base_url="http://a/v1")}
        roles = {"main": RoleSpec(name="main", targets=(RoleTarget(endpoint_name, "a-model"),))}
        router = LLMRouter(endpoints, roles, client_factory=lambda spec: stub)
        rc = router.for_role("main")

        result = await rc.chat_async(
            [{"role": "user", "content": "hi"}], response_format={"type": "json_object"}
        )
        self.assertEqual(result, {"role": "assistant", "content": "ok"})
        # First call carried response_format; retry stripped it.
        self.assertEqual(len(stub.calls), 2)
        self.assertIn("response_format", stub.calls[0][1])
        self.assertNotIn("response_format", stub.calls[1][1])
        # Capability downgraded for this endpoint.
        self.assertEqual(router.response_format_support(endpoint_name), "none")
        self.assertEqual(rc.response_format_support(), "none")

        # A follow-up call sends no response_format from the caller's side;
        # the router itself does not auto-strip — the caller is expected to
        # consult response_format_support(). Here we simply verify a clean
        # call (no response_format) succeeds without retry.
        stub.calls.clear()
        follow = await rc.chat_async([{"role": "user", "content": "hi"}])
        self.assertEqual(follow, {"role": "assistant", "content": "ok"})
        self.assertEqual(len(stub.calls), 1)
        self.assertNotIn("response_format", stub.calls[0][1])

    async def test_other_4xx_propagates(self) -> None:
        stub = _StubClient(_bad_request_error(), model="a-model")
        endpoints = {"a": EndpointSpec(name="a", base_url="http://a/v1")}
        roles = {"main": RoleSpec(name="main", targets=(RoleTarget("a", "a-model"),))}
        router = LLMRouter(endpoints, roles, client_factory=lambda spec: stub)

        with self.assertRaises(openai.BadRequestError):
            await router.for_role("main").chat_async([{"role": "user", "content": "hi"}])

    async def test_embed_batch_falls_through_on_none(self) -> None:
        stub_a = _EmbedStub(None, model="a-model")
        stub_b = _EmbedStub([[0.1, 0.2]], model="b-model")
        endpoints = {
            "a": EndpointSpec(name="a", base_url="http://a/v1"),
            "b": EndpointSpec(name="b", base_url="http://b/v1"),
        }
        roles = {
            "embeddings": RoleSpec(
                name="embeddings",
                targets=(RoleTarget("a", "a-model"), RoleTarget("b", "b-model")),
            )
        }
        stubs = {"a": stub_a, "b": stub_b}
        router = LLMRouter(endpoints, roles, client_factory=lambda spec: stubs[spec.name])

        result = await router.for_role("embeddings").embed_batch(["text"])

        self.assertEqual(result, [[0.1, 0.2]])
        self.assertEqual(len(stub_a.calls), 1)
        self.assertEqual(len(stub_b.calls), 1)


class ResponseFormatSupportTests(unittest.TestCase):
    def _router_with(self, mode: str) -> LLMRouter:
        endpoints = {"a": EndpointSpec(name="a", base_url="http://a/v1", response_format=mode)}
        roles = {"main": RoleSpec(name="main", targets=(RoleTarget("a", "m"),))}
        return LLMRouter(endpoints, roles, client_factory=lambda spec: _StubClient({"content": "ok"}))

    def test_response_format_support_modes(self) -> None:
        self.assertEqual(self._router_with("none").response_format_support("a"), "none")
        self.assertEqual(self._router_with("json_schema").response_format_support("a"), "json_schema")
        self.assertEqual(self._router_with("auto").response_format_support("a"), "json_object")


if __name__ == "__main__":
    unittest.main()

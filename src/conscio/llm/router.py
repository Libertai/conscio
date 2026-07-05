"""Multi-endpoint LLM routing (per-role model selection + fallback chains).

``LLMRouter`` owns one lazily-built ``LLMClient`` per named endpoint and
resolves roles (main / fast / embeddings / subagent) to ordered target lists.
``RoleClient`` is duck-compatible with ``LLMClient`` everywhere the runtime
holds an ``llm``: same ``chat_async(messages, **kwargs) -> dict`` shape and a
``.model`` attribute (read by eval's MeteredLLM). Transport-class failures
(connection, timeout, 429, 5xx) fall through the target chain with jittered
exponential backoff; a BadRequest while ``response_format`` was set downgrades
that endpoint's structured-output capability and retries the same target once.
Per-endpoint ``tool_choice`` support is honoured on tool-carrying calls, and
``response_format_support`` resolves the endpoint's structured-output mode
(consulted by JSON-parsing callers such as the constraint judge).
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import openai

from conscio.config import ServiceConfig
from conscio.llm.client import LLMClient

_FALLBACK_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    base_url: str
    api_key: str = ""
    timeout: float = 120.0
    max_retries: int = 2
    response_format: str = "auto"
    tool_choice: bool = True


@dataclass(frozen=True)
class RoleTarget:
    endpoint: str
    model: str


@dataclass(frozen=True)
class RoleSpec:
    name: str
    targets: tuple[RoleTarget, ...]
    max_tokens: int | None = None


class LLMRouter:
    def __init__(
        self,
        endpoints: dict[str, EndpointSpec],
        roles: dict[str, RoleSpec],
        *,
        retry_backoff: float = 0.5,
        client_factory: Callable[[EndpointSpec], Any] | None = None,
    ) -> None:
        self.endpoints = endpoints
        self.roles = roles
        self.retry_backoff = retry_backoff
        self._client_factory = client_factory or self._default_factory
        self._clients: dict[str, Any] = {}
        self._unsupported: dict[str, set[str]] = {}

    @staticmethod
    def _default_factory(spec: EndpointSpec) -> Any:
        return LLMClient(
            base_url=spec.base_url,
            api_key=spec.api_key,
            timeout=spec.timeout,
            max_retries=spec.max_retries,
        )

    @classmethod
    def from_config(cls, cfg: ServiceConfig) -> LLMRouter | None:
        endpoints: dict[str, EndpointSpec] = {}
        if cfg.llm_base_url:
            endpoints["default"] = EndpointSpec(
                name="default",
                base_url=cfg.llm_base_url,
                api_key=cfg.llm_api_key,
                timeout=cfg.llm_timeout,
                max_retries=cfg.llm_max_retries,
            )
        for name, ep in cfg.llm_endpoints.items():
            endpoints[name] = EndpointSpec(
                name=name,
                base_url=ep.base_url,
                api_key=ep.api_key,
                timeout=ep.timeout,
                max_retries=ep.max_retries,
                response_format=ep.response_format,
                tool_choice=ep.tool_choice,
            )
        if not endpoints:
            return None
        roles: dict[str, RoleSpec] = {}
        main_raw = cfg.llm_roles.get("main")
        if main_raw is not None:
            roles["main"] = cls._role_spec(main_raw, cfg)
        else:
            roles["main"] = RoleSpec(
                name="main", targets=(RoleTarget("default", cfg.llm_model),)
            )
        for role in ("fast", "subagent"):
            raw = cfg.llm_roles.get(role)
            if raw is not None:
                roles[role] = cls._role_spec(raw, cfg)
            else:
                roles[role] = RoleSpec(
                    name=role, targets=roles["main"].targets, max_tokens=roles["main"].max_tokens
                )
        emb_raw = cfg.llm_roles.get("embeddings")
        if emb_raw is not None:
            roles["embeddings"] = cls._role_spec(emb_raw, cfg)
        else:
            roles["embeddings"] = RoleSpec(
                name="embeddings",
                targets=(
                    RoleTarget(roles["main"].targets[0].endpoint, cfg.llm_embedding_model),
                ),
            )
        return cls(endpoints, roles, retry_backoff=cfg.llm_retry_backoff)

    @staticmethod
    def _role_spec(raw: Any, cfg: ServiceConfig) -> RoleSpec:
        primary_endpoint = raw.endpoint or ("default" if cfg.llm_base_url else "")
        targets = [RoleTarget(primary_endpoint, raw.model or cfg.llm_model)]
        targets.extend(RoleTarget(ep, model) for ep, model in raw.fallback)
        return RoleSpec(name=raw.role, targets=tuple(targets), max_tokens=raw.max_tokens)

    def client(self, endpoint: str) -> Any:
        if endpoint not in self._clients:
            self._clients[endpoint] = self._client_factory(self.endpoints[endpoint])
        return self._clients[endpoint]

    def for_role(self, role: str) -> RoleClient:
        spec = self.roles.get(role) or self.roles["main"]
        return RoleClient(self, spec)

    def response_format_support(self, endpoint: str) -> str:
        if "response_format" in self._unsupported.get(endpoint, set()):
            return "none"
        mode = self.endpoints[endpoint].response_format
        return "json_object" if mode == "auto" else mode

    def note_unsupported(self, endpoint: str, feature: str) -> None:
        self._unsupported.setdefault(endpoint, set()).add(feature)


class RoleClient:
    """Role-resolved LLM handle, duck-compatible with ``LLMClient``."""

    def __init__(self, router: LLMRouter, spec: RoleSpec) -> None:
        self._router = router
        self.spec = spec
        self.role = spec.name
        self.model = spec.targets[0].model

    def response_format_support(self) -> str:
        return self._router.response_format_support(self.spec.targets[0].endpoint)

    def _apply_tool_choice(self, endpoint: str, call_kwargs: dict[str, Any]) -> None:
        """Gate the ``tool_choice`` request arg on the endpoint's declared support.

        Endpoints that advertise ``tool_choice`` get an explicit ``"auto"`` for
        tool-carrying calls (unless the caller set one); endpoints that disable
        it never receive the arg, so a backend that rejects the parameter is
        never sent it.
        """
        if not call_kwargs.get("tools"):
            return
        spec = self._router.endpoints.get(endpoint)
        if spec is None or not spec.tool_choice:
            call_kwargs.pop("tool_choice", None)
        else:
            call_kwargs.setdefault("tool_choice", "auto")

    async def _backoff(self, attempt: int) -> None:
        base = self._router.retry_backoff
        if base > 0:
            await asyncio.sleep(base * (2**attempt) * (1 + random.uniform(0, 0.5)))

    async def chat_async(self, messages: list[dict], **kwargs: Any) -> dict:
        last_exc: Exception | None = None
        targets = self.spec.targets
        for attempt, target in enumerate(targets):
            client = self._router.client(target.endpoint)
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("model", target.model)
            self._apply_tool_choice(target.endpoint, call_kwargs)
            try:
                return await client.chat_async(messages, **call_kwargs)
            except openai.BadRequestError:
                if call_kwargs.get("response_format") is not None:
                    self._router.note_unsupported(target.endpoint, "response_format")
                    retry_kwargs = dict(call_kwargs)
                    retry_kwargs.pop("response_format", None)
                    try:
                        return await client.chat_async(messages, **retry_kwargs)
                    except _FALLBACK_ERRORS as retry_exc:
                        last_exc = retry_exc
                        await self._backoff(attempt)
                        continue
                raise
            except _FALLBACK_ERRORS as exc:
                last_exc = exc
                if attempt + 1 < len(targets):
                    await self._backoff(attempt)
                continue
        assert last_exc is not None
        raise last_exc

    def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        target = self.spec.targets[0]
        kwargs.setdefault("model", target.model)
        self._apply_tool_choice(target.endpoint, kwargs)
        return self._router.client(target.endpoint).chat(messages, **kwargs)

    async def chat_stream(self, messages: list[dict], **kwargs: Any):
        last_exc: Exception | None = None
        for attempt, target in enumerate(self.spec.targets):
            client = self._router.client(target.endpoint)
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault("model", target.model)
            self._apply_tool_choice(target.endpoint, call_kwargs)
            yielded = False
            try:
                async for event in client.chat_stream(messages, **call_kwargs):
                    yielded = True
                    yield event
                return
            except _FALLBACK_ERRORS as exc:
                if yielded:
                    raise  # mid-stream failure: the caller owns recovery
                last_exc = exc
                await self._backoff(attempt)
                continue
        assert last_exc is not None
        raise last_exc

    async def embed(self, text: str, *, model: str | None = None) -> list[float] | None:
        out = await self.embed_batch([text], model=model)
        return out[0] if out else None

    async def embed_batch(
        self, texts: list[str], *, model: str | None = None
    ) -> list[list[float]] | None:
        for target in self.spec.targets:
            client = self._router.client(target.endpoint)
            result = await client.embed_batch(texts, model=model or target.model)
            if result is not None:
                return result
        return None

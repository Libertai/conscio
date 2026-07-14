"""Pinned, side-effect-free language-model boundary for V3.

The specialist can interpret text, propose reasoning, draft a response, and
turn OpenAI-style function calls into inert typed proposals.  It deliberately
has no capability interface: callers must submit proposals to a separate
arbiter if they want anything to happen.

All provenance objects use strict canonical JSON.  This makes the exact
request and immutable model manifest independently hashable and replayable.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "v3_language_specialist"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_VALUES = frozenset(
    {
        "current",
        "default",
        "dev",
        "head",
        "latest",
        "main",
        "master",
        "nightly",
        "none",
        "placeholder",
        "rolling",
        "stable",
        "todo",
        "unknown",
        "unpinned",
    }
)

Operation = Literal["interpretation", "reasoning", "response"]
EpistemicStatus = Literal["idea", "self_report"]
ResearchUse = Literal["primary", "comparison_baseline"]


class ManifestCompatibilityError(ValueError):
    """Raised when replay would cross an unrecorded specialist boundary."""


class MalformedLanguageResponse(ValueError):
    """Raised when a provider response is not the strict supported shape."""

    def __init__(self, message: str, *, trace: LanguageCallTrace | None = None) -> None:
        super().__init__(message)
        self.trace = trace


class ChatAsyncClient(Protocol):
    """Structural interface implemented by the existing OpenAI-style client."""

    async def chat_async(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode_json(payload: str, *, label: str) -> Any:
    if not isinstance(payload, str):
        raise ValueError(f"{label} must be a JSON string")
    try:
        return json.loads(
            payload,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid strict JSON") from exc


def _canonical_json(value: Any, *, label: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only finite JSON values") from exc


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys differ (missing={missing}, extra={extra})")


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} may not contain leading or trailing whitespace")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_digest(value: str | None, label: str) -> None:
    if value is not None and not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        normalized in _PLACEHOLDER_VALUES
        or "placeholder" in normalized
        or "<" in normalized
        or ">" in normalized
        or "${" in normalized
    )


@dataclass(frozen=True, slots=True)
class SamplingPolicy:
    """Complete sampling policy supplied on every model call."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    seed: int | None = 0

    def __post_init__(self) -> None:
        temperature = _require_float(self.temperature, "temperature")
        top_p = _require_float(self.top_p, "top_p")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        max_tokens = _require_int(self.max_tokens, "max_tokens")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.seed is not None and _require_int(self.seed, "seed") < 0:
            raise ValueError("seed must be non-negative")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SamplingPolicy:
        _require_exact_keys(data, {"temperature", "top_p", "max_tokens", "seed"}, "sampling policy")
        seed = data["seed"]
        return cls(
            temperature=_require_float(data["temperature"], "temperature"),
            top_p=_require_float(data["top_p"], "top_p"),
            max_tokens=_require_int(data["max_tokens"], "max_tokens"),
            seed=None if seed is None else _require_int(seed, "seed"),
        )

    def client_kwargs(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            result["seed"] = self.seed
        return result


@dataclass(frozen=True, slots=True)
class LanguageModelManifest:
    """Immutable identity and policy for one language specialist lineage."""

    provider: str
    endpoint_id: str
    model_id: str
    model_revision: str | None
    sampling: SamplingPolicy = field(default_factory=SamplingPolicy)
    weight_digest: str | None = None
    config_digest: str | None = None
    research_use: ResearchUse = "primary"
    schema_version: int = MANIFEST_SCHEMA_VERSION
    manifest_kind: str = MANIFEST_KIND

    def __post_init__(self) -> None:
        for name in ("provider", "endpoint_id", "model_id"):
            _require_string(getattr(self, name), name)
        if self.model_revision is not None:
            _require_string(self.model_revision, "model_revision")
        if self.research_use not in ("primary", "comparison_baseline"):
            raise ValueError("research_use must be 'primary' or 'comparison_baseline'")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported language manifest schema: {self.schema_version}")
        if self.manifest_kind != MANIFEST_KIND:
            raise ValueError(f"unsupported language manifest kind: {self.manifest_kind!r}")
        _validate_digest(self.weight_digest, "weight_digest")
        _validate_digest(self.config_digest, "config_digest")
        if self.research_use == "primary":
            pinned_values = {
                "provider": self.provider,
                "endpoint_id": self.endpoint_id,
                "model_id": self.model_id,
                "model_revision": self.model_revision or "",
            }
            for label, value in pinned_values.items():
                if not value or _is_placeholder(value):
                    raise ValueError(f"primary research requires a pinned {label}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_kind": self.manifest_kind,
            "provider": self.provider,
            "endpoint_id": self.endpoint_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "weight_digest": self.weight_digest,
            "config_digest": self.config_digest,
            "research_use": self.research_use,
            "sampling": self.sampling.to_dict(),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict(), label="language manifest")

    def digest(self) -> str:
        return _digest(self.to_json())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LanguageModelManifest:
        expected = {
            "schema_version",
            "manifest_kind",
            "provider",
            "endpoint_id",
            "model_id",
            "model_revision",
            "weight_digest",
            "config_digest",
            "research_use",
            "sampling",
        }
        _require_exact_keys(data, expected, "language manifest")
        sampling = data["sampling"]
        if not isinstance(sampling, Mapping):
            raise ValueError("sampling must be an object")
        revision = data["model_revision"]
        weight_digest = data["weight_digest"]
        config_digest = data["config_digest"]
        research_use = _require_string(data["research_use"], "research_use")
        if research_use not in ("primary", "comparison_baseline"):
            raise ValueError("invalid research_use")
        return cls(
            provider=_require_string(data["provider"], "provider"),
            endpoint_id=_require_string(data["endpoint_id"], "endpoint_id"),
            model_id=_require_string(data["model_id"], "model_id"),
            model_revision=None if revision is None else _require_string(revision, "model_revision"),
            sampling=SamplingPolicy.from_dict(sampling),
            weight_digest=None if weight_digest is None else _require_string(weight_digest, "weight_digest"),
            config_digest=None if config_digest is None else _require_string(config_digest, "config_digest"),
            research_use=cast(ResearchUse, research_use),
            schema_version=_require_int(data["schema_version"], "schema_version"),
            manifest_kind=_require_string(data["manifest_kind"], "manifest_kind"),
        )

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        expected_digest: str | None = None,
        require_canonical: bool = True,
    ) -> LanguageModelManifest:
        decoded = _decode_json(payload, label="language manifest")
        if not isinstance(decoded, Mapping):
            raise ValueError("language manifest must be a JSON object")
        manifest = cls.from_dict(decoded)
        canonical = manifest.to_json()
        if require_canonical and not hmac.compare_digest(payload, canonical):
            raise ValueError("language manifest is not in canonical serialization")
        if expected_digest is not None:
            _validate_digest(expected_digest, "expected_digest")
            if not hmac.compare_digest(manifest.digest(), expected_digest):
                raise ManifestCompatibilityError("language manifest digest does not match the recorded digest")
        return manifest

    def assert_compatible(self, recorded: LanguageModelManifest | str, *, digest: str | None = None) -> None:
        """Require exact manifest identity before restore or replay."""

        other = self.from_json(recorded, expected_digest=digest) if isinstance(recorded, str) else recorded
        if digest is not None and not isinstance(recorded, str):
            _validate_digest(digest, "digest")
            if not hmac.compare_digest(other.digest(), digest):
                raise ManifestCompatibilityError("recorded manifest does not match its recorded digest")
        if not hmac.compare_digest(self.digest(), other.digest()):
            raise ManifestCompatibilityError(
                "language manifest is incompatible with the recorded restore/replay lineage"
            )


@dataclass(frozen=True, slots=True)
class ChatAsyncAdapter:
    """Identity-labelled adapter for an existing ``chat_async`` client."""

    client: ChatAsyncClient
    provider: str
    endpoint_id: str

    def __post_init__(self) -> None:
        _require_string(self.provider, "provider")
        _require_string(self.endpoint_id, "endpoint_id")
        if not callable(getattr(self.client, "chat_async", None)):
            raise TypeError("client must provide async chat_async(messages, **kwargs)")

    async def complete(self, messages: list[dict[str, Any]], kwargs: dict[str, Any]) -> dict[str, Any]:
        return await self.client.chat_async(messages, **kwargs)


@dataclass(frozen=True, slots=True)
class LanguageCallTrace:
    """Exact immutable request/response record for one specialist call."""

    operation: Operation
    manifest_digest: str
    request_json: str
    request_digest: str
    response_json: str
    response_digest: str

    def __post_init__(self) -> None:
        if self.operation not in ("interpretation", "reasoning", "response"):
            raise ValueError("invalid language operation")
        for label in ("manifest_digest", "request_digest", "response_digest"):
            _validate_digest(getattr(self, label), label)
        for label in ("request", "response"):
            payload = getattr(self, f"{label}_json")
            value = _decode_json(payload, label=label)
            canonical = _canonical_json(value, label=label)
            if not hmac.compare_digest(payload, canonical):
                raise ValueError(f"{label}_json is not canonical")
            if not hmac.compare_digest(_digest(payload), getattr(self, f"{label}_digest")):
                raise ValueError(f"{label}_digest does not authenticate {label}_json")

    def request(self) -> dict[str, Any]:
        return cast(dict[str, Any], _decode_json(self.request_json, label="request"))

    def response(self) -> dict[str, Any]:
        return cast(dict[str, Any], _decode_json(self.response_json, label="response"))


@dataclass(frozen=True, slots=True)
class LanguageToolProposal:
    """Inert, typed representation of one generated function call."""

    call_id: str
    name: str
    arguments_json: str
    epistemic_status: Literal["idea"] = field(default="idea", init=False)

    def __post_init__(self) -> None:
        _require_string(self.call_id, "tool proposal call_id")
        _require_string(self.name, "tool proposal name")
        arguments = _decode_json(self.arguments_json, label="tool proposal arguments")
        if not isinstance(arguments, dict):
            raise ValueError("tool proposal arguments must be a JSON object")
        if not hmac.compare_digest(
            self.arguments_json,
            _canonical_json(arguments, label="tool proposal arguments"),
        ):
            raise ValueError("tool proposal arguments must be canonical")

    @property
    def arguments(self) -> dict[str, Any]:
        return cast(dict[str, Any], _decode_json(self.arguments_json, label="tool proposal arguments"))


@dataclass(frozen=True, slots=True)
class LanguageInterpretation:
    text: str
    trace: LanguageCallTrace
    epistemic_status: Literal["idea"] = field(default="idea", init=False)

    def __post_init__(self) -> None:
        _require_string(self.text, "interpretation text")
        if self.trace.operation != "interpretation":
            raise ValueError("interpretation requires an interpretation trace")


@dataclass(frozen=True, slots=True)
class LanguageReasoning:
    text: str
    trace: LanguageCallTrace
    epistemic_status: Literal["idea"] = field(default="idea", init=False)

    def __post_init__(self) -> None:
        _require_string(self.text, "reasoning text")
        if self.trace.operation != "reasoning":
            raise ValueError("reasoning requires a reasoning trace")


@dataclass(frozen=True, slots=True)
class LanguageResponse:
    text: str
    proposals: tuple[LanguageToolProposal, ...]
    trace: LanguageCallTrace
    epistemic_status: EpistemicStatus = "idea"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("response text must be a string")
        if not self.text.strip() and not self.proposals:
            raise ValueError("response must contain text or at least one proposal")
        if self.epistemic_status not in ("idea", "self_report"):
            raise ValueError("response status must be 'idea' or 'self_report'")
        if self.trace.operation != "response":
            raise ValueError("response requires a response trace")


class LanguageSpecialist:
    """Narrow model-facing API.  Returned function calls are data only."""

    __slots__ = ("_adapter", "_manifest")

    def __init__(self, adapter: ChatAsyncAdapter, manifest: LanguageModelManifest) -> None:
        if adapter.provider != manifest.provider or adapter.endpoint_id != manifest.endpoint_id:
            raise ManifestCompatibilityError("client provider/endpoint identity differs from the manifest")
        self._adapter = adapter
        self._manifest = manifest

    @classmethod
    def from_chat_async(
        cls,
        client: ChatAsyncClient,
        *,
        manifest: LanguageModelManifest,
        provider: str,
        endpoint_id: str,
    ) -> LanguageSpecialist:
        """Adapt the repository's existing ``chat_async`` shape explicitly."""

        return cls(
            ChatAsyncAdapter(client=client, provider=provider, endpoint_id=endpoint_id),
            manifest,
        )

    @property
    def manifest(self) -> LanguageModelManifest:
        return self._manifest

    @property
    def manifest_digest(self) -> str:
        return self._manifest.digest()

    def assert_replay_compatible(self, serialized_manifest: str, recorded_digest: str) -> None:
        self._manifest.assert_compatible(serialized_manifest, digest=recorded_digest)

    async def interpret(self, messages: Sequence[Mapping[str, Any]]) -> LanguageInterpretation:
        text, proposals, trace = await self._call("interpretation", messages, tool_schemas=None)
        if proposals:
            raise MalformedLanguageResponse("interpretation may not emit tool calls")
        return LanguageInterpretation(text=text, trace=trace)

    async def reason(self, messages: Sequence[Mapping[str, Any]]) -> LanguageReasoning:
        text, proposals, trace = await self._call("reasoning", messages, tool_schemas=None)
        if proposals:
            raise MalformedLanguageResponse("reasoning may not emit tool calls")
        return LanguageReasoning(text=text, trace=trace)

    async def respond(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tool_schemas: Sequence[Mapping[str, Any]] | None = None,
        epistemic_status: EpistemicStatus = "idea",
    ) -> LanguageResponse:
        if epistemic_status not in ("idea", "self_report"):
            raise ValueError("response status must be 'idea' or 'self_report'")
        text, proposals, trace = await self._call("response", messages, tool_schemas=tool_schemas)
        return LanguageResponse(
            text=text,
            proposals=proposals,
            trace=trace,
            epistemic_status=epistemic_status,
        )

    async def _call(
        self,
        operation: Operation,
        messages: Sequence[Mapping[str, Any]],
        *,
        tool_schemas: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[str, tuple[LanguageToolProposal, ...], LanguageCallTrace]:
        allowed_tool_names = _advertised_tool_names(tool_schemas)
        request: dict[str, Any] = {
            "messages": list(messages),
            "model": self._manifest.model_id,
            **self._manifest.sampling.client_kwargs(),
        }
        if tool_schemas is not None:
            request["tools"] = list(tool_schemas)
            request["tool_choice"] = "auto"
        request_json = _canonical_json(request, label="language request")
        exact_request = _decode_json(request_json, label="language request")
        if not isinstance(exact_request, dict):  # pragma: no cover - guaranteed by construction
            raise ValueError("language request must be an object")
        client_messages = cast(list[dict[str, Any]], copy.deepcopy(exact_request.pop("messages")))
        client_kwargs = cast(dict[str, Any], copy.deepcopy(exact_request))
        raw_response = await self._adapter.complete(client_messages, client_kwargs)
        try:
            response_json = _canonical_json(raw_response, label="language response")
        except ValueError as exc:
            raise MalformedLanguageResponse(str(exc)) from exc
        normalized_response = _decode_json(response_json, label="language response")
        trace = LanguageCallTrace(
            operation=operation,
            manifest_digest=self._manifest.digest(),
            request_json=request_json,
            request_digest=_digest(request_json),
            response_json=response_json,
            response_digest=_digest(response_json),
        )
        try:
            text, proposals = _parse_response(
                normalized_response,
                allowed_tool_names=allowed_tool_names,
            )
        except MalformedLanguageResponse as exc:
            raise MalformedLanguageResponse(str(exc), trace=trace) from exc
        return text, proposals, trace


def _advertised_tool_names(tool_schemas: Sequence[Mapping[str, Any]] | None) -> frozenset[str] | None:
    if tool_schemas is None:
        return None
    if not tool_schemas:
        raise ValueError("tool_schemas must contain at least one schema when supplied")
    names: set[str] = set()
    for index, schema in enumerate(tool_schemas):
        if not isinstance(schema, Mapping) or schema.get("type") != "function":
            raise ValueError(f"tool_schemas[{index}] must be an OpenAI function schema")
        function = schema.get("function")
        if not isinstance(function, Mapping):
            raise ValueError(f"tool_schemas[{index}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"tool_schemas[{index}].function.name must be non-empty")
        if name in names:
            raise ValueError(f"duplicate advertised tool name: {name!r}")
        names.add(name)
    return frozenset(names)


def _parse_response(
    response: Any,
    *,
    allowed_tool_names: frozenset[str] | None,
) -> tuple[str, tuple[LanguageToolProposal, ...]]:
    if not isinstance(response, Mapping):
        raise MalformedLanguageResponse("language response must be an object")
    allowed = {"role", "content", "tool_calls"}
    if not set(response) <= allowed or not {"role", "content"} <= set(response):
        raise MalformedLanguageResponse("language response must contain only role, content, and optional tool_calls")
    if response["role"] != "assistant":
        raise MalformedLanguageResponse("language response role must be 'assistant'")
    content = response["content"]
    if not isinstance(content, str):
        raise MalformedLanguageResponse("language response content must be a string")
    raw_calls = response.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        raise MalformedLanguageResponse("tool_calls must be a list")
    if raw_calls and allowed_tool_names is None:
        raise MalformedLanguageResponse("tool calls were returned when no tool schemas were supplied")
    proposals: list[LanguageToolProposal] = []
    call_ids: set[str] = set()
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, Mapping):
            raise MalformedLanguageResponse(f"tool_calls[{index}] must be an object")
        if set(raw_call) != {"id", "type", "function"}:
            raise MalformedLanguageResponse(f"tool_calls[{index}] has an unsupported shape")
        call_id = raw_call["id"]
        if not isinstance(call_id, str) or not call_id.strip():
            raise MalformedLanguageResponse(f"tool_calls[{index}].id must be a non-empty string")
        if call_id in call_ids:
            raise MalformedLanguageResponse(f"duplicate tool call id: {call_id!r}")
        call_ids.add(call_id)
        if raw_call["type"] != "function":
            raise MalformedLanguageResponse(f"tool_calls[{index}].type must be 'function'")
        function = raw_call["function"]
        if not isinstance(function, Mapping) or set(function) != {"name", "arguments"}:
            raise MalformedLanguageResponse(f"tool_calls[{index}].function has an unsupported shape")
        name = function["name"]
        arguments_payload = function["arguments"]
        if not isinstance(name, str) or not name.strip():
            raise MalformedLanguageResponse(f"tool_calls[{index}].function.name must be non-empty")
        if name not in cast(frozenset[str], allowed_tool_names):
            raise MalformedLanguageResponse(f"tool_calls[{index}] names an unadvertised function: {name!r}")
        if not isinstance(arguments_payload, str):
            raise MalformedLanguageResponse(f"tool_calls[{index}].function.arguments must be a JSON string")
        try:
            arguments = _decode_json(arguments_payload, label=f"tool_calls[{index}] arguments")
        except ValueError as exc:
            raise MalformedLanguageResponse(str(exc)) from exc
        if not isinstance(arguments, dict):
            raise MalformedLanguageResponse(f"tool_calls[{index}] arguments must decode to an object")
        proposals.append(
            LanguageToolProposal(
                call_id=call_id,
                name=name,
                arguments_json=_canonical_json(arguments, label=f"tool_calls[{index}] arguments"),
            )
        )
    if not content.strip() and not proposals:
        raise MalformedLanguageResponse("language response contains neither text nor tool proposals")
    return content, tuple(proposals)

"""Explicit bridge from the typed V3 language specialist to the tool loop.

The bridge reconstructs the legacy OpenAI response shape only after the
language boundary has parsed function calls into inert proposals. The existing
V3 pre-tool authorization hook remains the sole route from a proposal to tool
execution. Sampling is fixed by the signed manifest; callers cannot override it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from conscio.v3.language_specialist import (
    LanguageCallTrace,
    LanguageSpecialist,
    MalformedLanguageResponse,
    ManifestCompatibilityError,
)


class LanguageSpecialistToolLoopBridge:
    """Narrow compatibility surface for ``ToolLoopSession``.

    It intentionally provides no streaming method, so primary research calls
    cannot bypass the specialist's exact request/response trace.
    """

    def __init__(
        self,
        specialist: LanguageSpecialist,
        *,
        trace_observer: Callable[[LanguageCallTrace], Awaitable[None]] | None = None,
    ) -> None:
        self.specialist = specialist
        self.model = specialist.manifest.model_id
        self._traces: list[LanguageCallTrace] = []
        self._trace_observer = trace_observer

    @property
    def manifest_digest(self) -> str:
        return self.specialist.manifest_digest

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible manifest for trace persistence."""
        return self.specialist.manifest.to_dict()

    def response_format_support(self) -> str:
        # Structured callers still parse JSON text; provider-specific response
        # format flags are excluded from the pinned request contract.
        return "none"

    def drain_traces(self) -> tuple[LanguageCallTrace, ...]:
        traces = tuple(self._traces)
        self._traces.clear()
        return traces

    def set_trace_observer(
        self,
        observer: Callable[[LanguageCallTrace], Awaitable[None]] | None,
    ) -> None:
        self._trace_observer = observer

    async def _record_trace(self, trace: LanguageCallTrace) -> None:
        self._traces.append(trace)
        if self._trace_observer is not None:
            await self._trace_observer(trace)

    async def chat_async(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self._validate_call_overrides(kwargs)
        raw_tools = kwargs.get("tools")
        if raw_tools is not None and not isinstance(raw_tools, list):
            raise ValueError("tools must be a list when supplied")
        try:
            response = await self.specialist.respond(
                messages,
                tool_schemas=raw_tools,
                epistemic_status="idea",
            )
        except MalformedLanguageResponse as exc:
            # Provider output can be malformed while still being an exact,
            # authenticated observation. Preserve that trace before failing.
            if exc.trace is not None:
                await self._record_trace(exc.trace)
            raise
        await self._record_trace(response.trace)
        result: dict[str, Any] = {"role": "assistant", "content": response.text}
        if response.proposals:
            result["tool_calls"] = [
                {
                    "id": proposal.call_id,
                    "type": "function",
                    "function": {
                        "name": proposal.name,
                        "arguments": proposal.arguments_json,
                    },
                }
                for proposal in response.proposals
            ]
        return result

    def _validate_call_overrides(self, kwargs: Mapping[str, Any]) -> None:
        policy = self.specialist.manifest.sampling
        expected = {
            "temperature": policy.temperature,
            "top_p": policy.top_p,
            "max_tokens": policy.max_tokens,
            "seed": policy.seed,
        }
        for name, value in expected.items():
            supplied = kwargs.get(name)
            if supplied is None:
                continue
            if isinstance(value, float):
                matches = isinstance(supplied, (int, float)) and math.isclose(
                    float(supplied), value, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                matches = supplied == value
            if not matches:
                raise ManifestCompatibilityError(f"call attempted to override pinned sampling field {name!r}")
        model = kwargs.get("model")
        if model is not None and model != self.model:
            raise ManifestCompatibilityError("call model differs from the pinned manifest")
        tool_choice = kwargs.get("tool_choice")
        if tool_choice not in (None, "auto"):
            raise ManifestCompatibilityError("only automatic inert tool proposals are supported")
        allowed = {
            "max_tokens",
            "model",
            "response_format",
            "seed",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        }
        unknown = set(kwargs) - allowed
        if unknown:
            raise ManifestCompatibilityError(f"unpinned language call options are forbidden: {sorted(unknown)}")


def trace_to_dict(trace: LanguageCallTrace) -> dict[str, Any]:
    """Return the authenticated trace as JSON-compatible structured data."""
    return {
        "operation": trace.operation,
        "manifest_digest": trace.manifest_digest,
        "request": json.loads(trace.request_json),
        "request_digest": trace.request_digest,
        "response": json.loads(trace.response_json),
        "response_digest": trace.response_digest,
    }


__all__ = ["LanguageSpecialistToolLoopBridge", "trace_to_dict"]

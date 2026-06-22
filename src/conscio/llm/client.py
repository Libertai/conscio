from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

_DEFAULT_BASE_URL = "https://api.libertai.io/v1"
_DEFAULT_MODEL = "deepseek-v4-flash"
# A single LLM call should not hang for 10 minutes (the SDK default). 120s
# is generous enough for large completions while keeping the service
# responsive; a timeout now produces a caught exception instead of a hang.
_LLM_TIMEOUT = 120
_LLM_MAX_RETRIES = 2


class LLMClient:
    """OpenAI-compatible LLM client pointed at LibertAI."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv()
        self.base_url = (
            base_url
            or os.environ.get("LIBERTAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or _DEFAULT_BASE_URL
        )
        self.api_key = (
            api_key
            or os.environ.get("LIBERTAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "not-needed"
        )
        self.model = model or os.environ.get("LIBERTAI_MODEL") or _DEFAULT_MODEL
        self._sync: OpenAI | None = None
        self._async: AsyncOpenAI | None = None

    @property
    def sync(self) -> OpenAI:
        if self._sync is None:
            self._sync = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=_LLM_TIMEOUT,
                max_retries=_LLM_MAX_RETRIES,
            )
        return self._sync

    @property
    def async_(self) -> AsyncOpenAI:
        if self._async is None:
            self._async = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=_LLM_TIMEOUT,
                max_retries=_LLM_MAX_RETRIES,
            )
        return self._async

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> dict:
        kwargs.setdefault("temperature", temperature)
        if max_tokens:
            kwargs.setdefault("max_tokens", max_tokens)
        if tools:
            kwargs["tools"] = tools
        response = self.sync.chat.completions.create(
            model=model or self.model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        choice = response.choices[0]
        message = choice.message
        result: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,  # type: ignore[union-attr]
                        "arguments": tc.function.arguments,  # type: ignore[union-attr]
                    },
                }
                for tc in message.tool_calls
            ]
        return result

    async def chat_async(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> dict:
        kwargs.setdefault("temperature", temperature)
        if max_tokens:
            kwargs.setdefault("max_tokens", max_tokens)
        if tools:
            kwargs["tools"] = tools
        response = await self.async_.chat.completions.create(
            model=model or self.model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        choice = response.choices[0]
        message = choice.message
        result: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,  # type: ignore[union-attr]
                        "arguments": tc.function.arguments,  # type: ignore[union-attr]
                    },
                }
                for tc in message.tool_calls
            ]
        return result

    async def embed_batch(
        self, texts: list[str], *, model: str = "bge-m3"
    ) -> list[list[float]] | None:
        """Returns one 1024-float vector per input, or None if the endpoint errors."""
        try:
            resp = await self.async_.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception:
            return None

    async def embed(self, text: str, *, model: str = "bge-m3") -> list[float] | None:
        out = await self.embed_batch([text], model=model)
        return out[0] if out else None

    def embed_batch_sync(
        self, texts: list[str], *, model: str = "bge-m3"
    ) -> list[list[float]] | None:
        """Sync variant of embed_batch (mirrors chat/chat_async pairing)."""
        try:
            resp = self.sync.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception:
            return None

    def embed_sync(self, text: str, *, model: str = "bge-m3") -> list[float] | None:
        out = self.embed_batch_sync([text], model=model)
        return out[0] if out else None

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = 4096,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("temperature", temperature)
        if max_tokens:
            kwargs.setdefault("max_tokens", max_tokens)
        if tools:
            kwargs["tools"] = tools
        stream = await self.async_.chat.completions.create(
            model=model or self.model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
            **kwargs,
        )
        content_chunks: list[str] = []
        tool_call_deltas: dict[int, dict] = {}
        async for chunk in stream:  # type: ignore[union-attr]
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if delta.content:
                content_chunks.append(delta.content)
                yield {"type": "content", "text": delta.content}
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_deltas:
                        tool_call_deltas[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_call_deltas[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_deltas[idx]["function"]["name"] += (
                                tc_delta.function.name
                            )
                        if tc_delta.function.arguments:
                            tool_call_deltas[idx]["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )
        full_content = "".join(content_chunks)
        tool_calls = list(tool_call_deltas.values()) if tool_call_deltas else None
        yield {
            "type": "done",
            "content": full_content,
            "tool_calls": tool_calls,
        }

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

_DEFAULT_BASE_URL = "https://api.libertai.io/v1"
_DEFAULT_MODEL = "deepseek-v4-flash"


class LLMClient:
    """OpenAI-compatible LLM client pointed at LibertAI."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
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
        )
        self.model = model or os.environ.get("LIBERTAI_MODEL") or _DEFAULT_MODEL
        self._sync: OpenAI | None = None
        self._async: AsyncOpenAI | None = None

    @property
    def sync(self) -> OpenAI:
        if self._sync is None:
            self._sync = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._sync

    @property
    def async_(self) -> AsyncOpenAI:
        if self._async is None:
            self._async = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
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
        response = self.sync.chat.completions.create(
            model=model or self.model,
            messages=messages,
            tools=tools or [],
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
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
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
        response = await self.async_.chat.completions.create(
            model=model or self.model,
            messages=messages,
            tools=tools or [],
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
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return result

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
        stream = await self.async_.chat.completions.create(
            model=model or self.model,
            messages=messages,
            tools=tools or [],
            stream=True,
            **kwargs,
        )
        content_chunks: list[str] = []
        tool_call_deltas: dict[int, dict] = {}
        async for chunk in stream:
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

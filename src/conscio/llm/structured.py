"""Structured-output helpers with a tolerant hand-parse contract.

``structured_json`` sends a ``response_format`` parameter only when the LLM
client advertises support via a ``response_format_support()`` method; the reply
content is ALWAYS parsed with the balanced-bracket parser in
:func:`first_json_value`. Structured mode is decoration; the hand-parse is the
contract.
"""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_balanced(text: str, start: int) -> str | None:
    """Balanced-bracket scan from `start` (a '[' or '{'), string-aware.

    Generalizes tool_loop's object-only `_extract_balanced_json` to arrays.
    """
    if text[start] not in "[{":
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def first_json_value(raw: str) -> Any | None:
    """Extract the first parseable JSON array/object from free-form LLM text.

    Tries fenced ```json blocks first, then a balanced-bracket scan over the
    raw text (replaces the greedy `\\[.*\\]` regex that choked on prose).
    """
    if not raw:
        return None
    candidates: list[str] = []
    fence = _JSON_FENCE_RE.search(raw)
    if fence:
        candidates.append(fence.group(1))
    candidates.append(raw)
    for text in candidates:
        for idx, ch in enumerate(text):
            if ch not in "[{":
                continue
            chunk = extract_balanced(text, idx)
            if chunk is None:
                continue
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    return None


async def structured_json(
    llm: Any,
    messages: list[dict[str, Any]],
    *,
    schema: dict[str, Any] | None = None,
    schema_name: str = "result",
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> Any | None:
    """One ``chat_async`` call with best-effort structured output.

    Sends ``response_format`` only when the client advertises support via a
    ``response_format_support()`` method (json_schema when a schema is given
    and supported, else json_object). The reply content is ALWAYS parsed with
    :func:`first_json_value` — structured mode is decoration, the hand-parse
    is the contract. Transport errors propagate to the caller; a parse miss
    returns None.
    """
    kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    support = getattr(llm, "response_format_support", None)
    mode = support() if callable(support) else "none"
    if mode == "json_schema" and schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        }
    elif mode in ("json_object", "json_schema"):
        kwargs["response_format"] = {"type": "json_object"}
    response = await llm.chat_async(messages, **kwargs)
    return first_json_value(str(response.get("content") or ""))

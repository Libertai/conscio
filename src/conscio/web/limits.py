"""ASGI body-size cap. Rejects oversized requests with 413 before they become
LLM prompts or DB rows. Content-Length is checked up front; chunked bodies are
counted as they stream."""

from __future__ import annotations

import json
from typing import Any


class BodySizeLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int = 262144) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await _send_413(send, self.max_bytes)
                        return
                except ValueError:
                    break
                break
        received = 0
        over = False

        async def counting_receive() -> dict:
            nonlocal received, over
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body") or b"")
                if received > self.max_bytes:
                    over = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        response_started = False

        async def guarded_send(message: dict) -> None:
            nonlocal response_started
            if over and not response_started:
                response_started = True
                await _send_413(send, self.max_bytes, raw=True)
                return
            if over and response_started:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, counting_receive, guarded_send)


async def _send_413(send: Any, max_bytes: int, raw: bool = False) -> None:
    body = json.dumps({"detail": f"request body exceeds {max_bytes} bytes"}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})

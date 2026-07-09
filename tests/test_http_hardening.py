"""HTTP hardening: token-bucket rate limits (429), body-size caps (413),
pydantic max_length caps (422), and login-throttle client isolation."""

from __future__ import annotations

import asyncio
import os
import unittest.mock
from pathlib import Path

from conscio.config import load_config
from conscio.service import ConscioService
from conscio.web.ratelimit import TokenBucket

# Hermetic LLM env: keep autonomy/runtime init offline (mirrors test_service.py).
_ENV = {
    "LIBERTAI_BASE_URL": "",
    "LIBERTAI_API_KEY": "",
    "LIBERTAI_MODEL": "",
    "OPENAI_BASE_URL": "",
    "OPENAI_API_KEY": "",
}


def _cfg_text(tmp: str, **overrides: object) -> str:
    lines = [
        "[service]",
        f"home = \"{tmp}\"",
        "api_key = \"test-key\"",
        "web_password = \"test-pass\"",
        "autonomous = false",
    ]
    for key, value in overrides.items():
        lines.append(f"{key} = {value!r}" if isinstance(value, str) else f"{key} = {value}")
    return "\n".join(lines) + "\n"


def _make_service(tmp: str, **overrides: object) -> ConscioService:
    path = Path(tmp) / "config.toml"
    path.write_text(_cfg_text(tmp, **overrides), encoding="utf-8")
    return ConscioService(load_config(path))


def _run(coro):
    return asyncio.run(coro)


def test_token_bucket_burst_deny_refill() -> None:
    t = [0.0]
    bucket = TokenBucket(2, 1.0, clock=lambda: t[0])
    # Burst capacity 2: first two acquire, third is denied.
    assert bucket.try_acquire() == (True, 0.0)
    assert bucket.try_acquire()[0] is True
    denied = bucket.try_acquire()
    assert denied[0] is False
    assert denied[1] > 0.0
    # After one second of refill, one token is available again.
    t[0] = 1.0
    assert bucket.try_acquire()[0] is True


def test_message_429_after_burst(tmp_path: Path) -> None:
    async def scenario() -> None:
        import httpx

        from conscio.api import create_app

        with unittest.mock.patch.dict(os.environ, _ENV, clear=False):
            service = _make_service(str(tmp_path), episode_rate_per_minute=60, episode_rate_burst=1)
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    headers = {"Authorization": "Bearer test-key"}
                    first = await client.post("/message", json={"content": "hi"}, headers=headers)
                    second = await client.post("/message", json={"content": "hi"}, headers=headers)
            finally:
                await service.stop()

        assert first.status_code == 200, first.text
        assert second.status_code == 429, second.text
        assert "Retry-After" in second.headers

    _run(scenario())


def test_webui_chat_429_after_burst(tmp_path: Path) -> None:
    async def scenario() -> None:
        import httpx

        from conscio.api import create_app

        with unittest.mock.patch.dict(os.environ, _ENV, clear=False):
            service = _make_service(str(tmp_path), episode_rate_per_minute=60, episode_rate_burst=1)
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    login = await client.post("/ui/login", json={"password": "test-pass"})
                    assert login.status_code == 200, login.text
                    first = await client.post(
                        "/ui/api/chat/sessions/main/messages", json={"content": "ping"}
                    )
                    second = await client.post(
                        "/ui/api/chat/sessions/main/messages", json={"content": "ping"}
                    )
            finally:
                await service.stop()

        assert first.status_code == 200, first.text
        assert second.status_code == 429, second.text
        assert "Retry-After" in second.headers

    _run(scenario())


def test_413_via_content_length(tmp_path: Path) -> None:
    async def scenario() -> None:
        import httpx

        from conscio.api import create_app

        with unittest.mock.patch.dict(os.environ, _ENV, clear=False):
            service = _make_service(str(tmp_path), max_request_bytes=512)
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    headers = {"Authorization": "Bearer test-key"}
                    # Serialized JSON body well over 512 bytes.
                    big = "x" * 600
                    response = await client.post("/message", json={"content": big}, headers=headers)
            finally:
                await service.stop()

        assert response.status_code == 413, response.text
        assert "512" in response.json()["detail"]

    _run(scenario())


def test_422_max_length(tmp_path: Path) -> None:
    async def scenario() -> None:
        import httpx

        from conscio.api import create_app

        with unittest.mock.patch.dict(os.environ, _ENV, clear=False):
            service = _make_service(str(tmp_path), max_request_bytes=0)
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    headers = {"Authorization": "Bearer test-key"}
                    too_long = "x" * 70_000
                    response = await client.post(
                        "/message", json={"content": too_long}, headers=headers
                    )
            finally:
                await service.stop()

        assert response.status_code == 422, response.text

    _run(scenario())


def test_login_throttle_isolation(tmp_path: Path) -> None:
    async def scenario() -> None:
        import httpx

        from conscio.api import create_app

        with unittest.mock.patch.dict(os.environ, _ENV, clear=False):
            service = _make_service(str(tmp_path))
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport_a = httpx.ASGITransport(app=app, client=("1.2.3.4", 1))
                transport_b = httpx.ASGITransport(app=app, client=("5.6.7.8", 2))
                async with httpx.AsyncClient(transport=transport_a, base_url="http://test") as a:
                    async with httpx.AsyncClient(transport=transport_b, base_url="http://test") as b:
                        a_responses = [
                            await a.post("/ui/login", json={"password": "wrong"})
                            for _ in range(9)
                        ]
                        b_response = await b.post("/ui/login", json={"password": "wrong"})
            finally:
                await service.stop()

        # After 8 bad logins, client A's 9th attempt is locked out (429).
        assert a_responses[-1].status_code == 429, a_responses[-1].text
        # Client B is on a different IP and is unaffected: still 401, not 429.
        assert b_response.status_code == 401, b_response.text

    _run(scenario())


def test_login_warns_on_forwarded_headers_without_trusted_proxies(tmp_path: Path) -> None:
    async def scenario() -> None:
        import httpx

        from conscio.api import create_app

        with unittest.mock.patch.dict(os.environ, _ENV, clear=False):
            service = _make_service(str(tmp_path))
            await service.start(background=False)
            try:
                app = create_app(service=service)
                transport = httpx.ASGITransport(app=app, client=("10.0.0.9", 1))
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                    with unittest.TestCase().assertLogs("conscio.webui", level="WARNING") as logs:
                        await c.post(
                            "/ui/login",
                            json={"password": "wrong"},
                            headers={"X-Forwarded-For": "203.0.113.7"},
                        )
                        # Warned once, not per-request.
                        await c.post(
                            "/ui/login",
                            json={"password": "wrong"},
                            headers={"X-Forwarded-For": "203.0.113.8"},
                        )
            finally:
                await service.stop()

        proxy_warnings = [m for m in logs.output if "trusted_proxies" in m]
        assert len(proxy_warnings) == 1, logs.output

    _run(scenario())

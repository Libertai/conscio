"""Observability surface: Prometheus exposition, readiness probe, SSE cap.

Mirrors tests/test_service.py's Api-test construction for the real-service
routes (/metrics/prometheus, /ready) and tests/test_web_routes.py's stub-service
idiom for the SSE-client cap (which lives on the cookie-auth /ui/api/events
route driven by WorkspaceEventBroker).
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
import unittest.mock
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conscio.core.workspace import Workspace
from conscio.memory.store import MemoryStore
from conscio.web.events import MAX_SSE_CLIENTS, SSEClientLimitError, WorkspaceEventBroker

# Prometheus exposition lines: name [labels] value. Excludes # HELP / # TYPE.
_PROM_LINE = re.compile(r"^[a-z_]+(\{[^}]*\})? -?[0-9.e+]+$")


class _SSECapDirect(unittest.TestCase):
    def test_broker_rejects_33rd_client(self) -> None:
        broker = WorkspaceEventBroker(Workspace())
        clients = [broker.register() for _ in range(MAX_SSE_CLIENTS)]
        self.assertEqual(len(clients), MAX_SSE_CLIENTS)
        with self.assertRaises(SSEClientLimitError):
            broker.register()


# Inline stub mirroring tests/test_web_routes.py's _StubService: just enough of
# ConscioService for create_web_router's cookie-login + /ui/api/events path.
# (Inlined rather than imported because tests/ is not a package, so a
# package-qualified cross-test import would be fragile.)


@dataclass
class _StubConfig:
    web_password: str = "letmein"
    web_secure_cookies: bool = False
    trusted_proxies: tuple[str, ...] = ()  # login throttle checks proxy trust


class _StubService:
    def __init__(self, db_path: Path) -> None:
        self.config = _StubConfig()
        self.memory = MemoryStore(db_path=str(db_path))
        workspace = Workspace()
        self.event_broker = WorkspaceEventBroker(workspace)
        self.event_broker.attach()


class _SSECapHTTP(unittest.IsolatedAsyncioTestCase):
    async def test_events_route_returns_503_at_cap(self) -> None:
        try:
            import httpx

            from conscio.webui import create_web_router
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")
        from fastapi import FastAPI

        svc = _StubService(Path(self._tmp.name) / "cap.db")
        app = FastAPI()
        app.include_router(create_web_router(svc))
        app.state.svc = svc
        # Saturate the broker the route shares so its next register() raises.
        saturated = [svc.event_broker.register() for _ in range(MAX_SSE_CLIENTS)]
        self.assertEqual(len(saturated), MAX_SSE_CLIENTS)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            login = await client.post("/ui/login", json={"password": "letmein"})
            self.assertEqual(login.status_code, 200)
            r = await client.get("/ui/api/events")

        self.assertEqual(r.status_code, 503)
        self.assertIn("Retry-After", r.headers)

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._env_patch = unittest.mock.patch.dict(
            os.environ,
            {
                "LIBERTAI_BASE_URL": "",
                "LIBERTAI_API_KEY": "",
                "LIBERTAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        self._env_patch.start()

    async def asyncTearDown(self) -> None:
        self._env_patch.stop()
        self._tmp.cleanup()


class _ObservabilityApi(unittest.IsolatedAsyncioTestCase):
    """Real ConscioService + create_app, bearer-auth, per test_service.py."""

    async def asyncSetUp(self) -> None:
        self._env_patch = unittest.mock.patch.dict(
            os.environ,
            {
                "LIBERTAI_BASE_URL": "",
                "LIBERTAI_API_KEY": "",
                "LIBERTAI_MODEL": "",
                "OPENAI_BASE_URL": "",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        )
        self._env_patch.start()
        self._tmp = tempfile.TemporaryDirectory()
        self._config_path = Path(self._tmp.name) / "config.toml"
        self._config_path.write_text(
            "[service]\n"
            f"home = \"{self._tmp.name}\"\n"
            "api_key = \"test-key\"\n"
            "web_password = \"test-pass\"\n"
            "autonomous = false\n",
            encoding="utf-8",
        )

    async def asyncTearDown(self) -> None:
        self._env_patch.stop()
        self._tmp.cleanup()

    async def _service(self):
        from conscio.config import load_config
        from conscio.service import ConscioService

        return ConscioService(load_config(self._config_path))

    async def test_metrics_prometheus_requires_auth(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")

        service = await self._service()
        await service.start(background=False)
        try:
            app = create_app(service=service)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                denied = await client.get("/metrics/prometheus")
                allowed = await client.get(
                    "/metrics/prometheus", headers={"Authorization": "Bearer test-key"}
                )
        finally:
            await service.stop()

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        self.assertTrue(allowed.headers["content-type"].startswith("text/plain"))
        body = allowed.text
        for line in body.splitlines():
            if not line or line.startswith("#"):
                continue
            self.assertTrue(_PROM_LINE.match(line), f"bad prometheus line: {line!r}")
        self.assertIn("conscio_running 1", body)

    async def test_ready_when_running(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")

        service = await self._service()
        await service.start(background=False)
        try:
            app = create_app(service=service)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.get("/ready")
        finally:
            await service.stop()

        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ready"])

    async def test_ready_503_when_db_probe_fails(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")

        service = await self._service()
        await service.start(background=False)

        def _boom(*_args: Any, **_kwargs: Any) -> dict | None:
            raise RuntimeError("db unreachable")

        service.memory.fetchone = _boom
        try:
            app = create_app(service=service)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                r = await client.get("/ready")
        finally:
            await service.stop()

        self.assertEqual(r.status_code, 503)
        self.assertFalse(r.json()["ready"])

    async def test_ready_503_when_not_running(self) -> None:
        try:
            import httpx

            from conscio.api import create_app
        except ModuleNotFoundError:
            self.skipTest("fastapi is not installed in this environment")

        service = await self._service()
        # Intentionally never started: svc.running stays False.
        app = create_app(service=service)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/ready")

        self.assertEqual(r.status_code, 503)
        self.assertFalse(r.json()["ready"])


if __name__ == "__main__":
    unittest.main()

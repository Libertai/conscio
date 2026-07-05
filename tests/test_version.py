from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import conscio
from conscio.config import load_config
from conscio.service import ConscioService


class VersionTests(unittest.TestCase):
    def test_version_is_pep440_like(self) -> None:
        self.assertRegex(conscio.__version__, r"^\d+\.\d+\.\d+")


class HealthVersionTests(unittest.IsolatedAsyncioTestCase):
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

    async def asyncTearDown(self) -> None:
        self._env_patch.stop()

    async def test_health_reports_package_version(self) -> None:
        import httpx

        from conscio.api import create_app

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[service]\n"
                f'home = "{tmp}"\n'
                'api_key = "test-key"\n'
                'web_password = "test-pass"\n'
                "autonomous = false\n",
                encoding="utf-8",
            )
            service = ConscioService(load_config(path))
            await service.start(background=False)
            try:
                app = create_app(service=service)
                self.assertEqual(app.version, conscio.__version__)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    resp = await client.get("/health")
            finally:
                await service.stop()

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["version"], conscio.__version__)

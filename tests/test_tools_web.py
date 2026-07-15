from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch

from conscio.blocking import BoundedBlockingRunner, blocking_runner_context
from conscio.tools import web
from conscio.tools.env import resolve_tool, tool_env

SEARCH_PAGE = """
<html>
  <body>
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Ffirst">First &amp; Result</a>
      <a class="result__snippet">A short snippet about the first result.</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://example.com/second">Second Result</a>
      <a class="result__snippet">Another snippet.</a>
    </div>
  </body>
</html>
"""


FETCH_PAGE = """
<html>
  <head><title>Example Page</title><script>ignore()</script></head>
  <body><h1>Hello</h1><p>This is readable text.</p><style>.x {}</style></body>
</html>
"""

BING_PAGE = """
<html>
  <body>
    <li class="b_algo">
      <h2><a href="https://example.com/bing">Bing Result</a></h2>
      <p>Bing snippet text.</p>
    </li>
  </body>
</html>
"""


class WebToolTests(unittest.IsolatedAsyncioTestCase):
    def test_tool_env_includes_vm_tool_paths(self) -> None:
        path = tool_env()["PATH"]

        self.assertIn("/usr/local/bin", path)
        self.assertIn("/home/conscio/.cargo/bin", path)
        self.assertIsInstance(resolve_tool("libertai"), str)

    def test_parse_search_results_normalizes_duckduckgo_redirects(self) -> None:
        results = web._parse_search_results(SEARCH_PAGE, max_results=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "First & Result")
        self.assertEqual(results[0]["url"], "https://example.com/first")
        self.assertIn("short snippet", results[0]["snippet"])

    def test_parse_bing_results(self) -> None:
        results = web._parse_bing_results(BING_PAGE, max_results=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Bing Result")
        self.assertEqual(results[0]["url"], "https://example.com/bing")
        self.assertEqual(results[0]["snippet"], "Bing snippet text.")

    def test_extract_text_removes_scripts_and_styles(self) -> None:
        title, text = web._extract_text(FETCH_PAGE)

        self.assertEqual(title, "Example Page")
        self.assertIn("Hello", text)
        self.assertIn("This is readable text.", text)
        self.assertNotIn("ignore()", text)
        self.assertNotIn(".x", text)

    async def test_web_search_returns_cli_output_when_libertai_succeeds(self) -> None:
        async def fake_run(*args: str) -> tuple[bool, str]:
            return True, "LibertAI result"

        with patch.object(web, "_run_libertai", fake_run):
            result = await web.web_search("test")

        self.assertFalse(result["error"])
        self.assertEqual(result["output"], "LibertAI result")

    async def test_web_search_falls_back_when_libertai_rejects_key(self) -> None:
        async def fake_run(*args: str) -> tuple[bool, str]:
            return False, "401 Unauthorized"

        async def fake_get(url: str) -> str:
            self.assertIn("bing.com", url)
            return BING_PAGE

        with patch.object(web, "_run_libertai", fake_run), patch.object(web, "_http_get", fake_get):
            result = await web.web_search("test", max_results=2)

        self.assertFalse(result["error"])
        self.assertIn("Search results from Bing", result["output"])
        self.assertIn("https://example.com/bing", result["output"])

    async def test_web_fetch_falls_back_when_cli_is_missing(self) -> None:
        async def fake_run(*args: str) -> tuple[bool, str]:
            raise FileNotFoundError

        async def fake_get(url: str) -> str:
            self.assertEqual(url, "https://example.com")
            return FETCH_PAGE

        with (
            patch.object(web, "_run_libertai", fake_run),
            patch.object(web, "_http_get", fake_get),
            patch.object(web, "_resolve_host", lambda h, p: ["93.184.215.14"]),
        ):
            result = await web.web_fetch("https://example.com")

        self.assertFalse(result["error"])
        self.assertIn("Example Page", result["output"])
        self.assertIn("This is readable text.", result["output"])

    async def test_web_fetch_rejects_non_http_urls(self) -> None:
        result = await web.web_fetch("file:///etc/passwd")

        self.assertTrue(result["error"])
        self.assertIn("Only http and https", result["output"])

    async def test_web_fetch_rejects_loopback_literal(self) -> None:
        result = await web.web_fetch("http://127.0.0.1/secret")

        self.assertTrue(result["error"])
        self.assertIn("blocked", result["output"].lower())

    async def test_web_fetch_rejects_link_local_metadata_ip(self) -> None:
        result = await web.web_fetch("http://169.254.169.254/latest/meta-data/")

        self.assertTrue(result["error"])
        self.assertIn("blocked", result["output"].lower())

    async def test_web_fetch_rejects_localhost_hostname(self) -> None:
        result = await web.web_fetch("http://localhost/secret")

        self.assertTrue(result["error"])
        self.assertIn("blocked", result["output"].lower())

    async def test_web_fetch_rejects_metadata_hostname(self) -> None:
        result = await web.web_fetch("http://metadata.google.internal/")

        self.assertTrue(result["error"])
        self.assertIn("blocked", result["output"].lower())

    async def test_web_fetch_rejects_dotinternal_hostname(self) -> None:
        result = await web.web_fetch("http://service.internal/path")

        self.assertTrue(result["error"])
        self.assertIn("blocked", result["output"].lower())

    async def test_fallback_fetch_rejects_dns_rebind_to_private_ip(self) -> None:
        async def fake_run(*args: str) -> tuple[bool, str]:
            raise FileNotFoundError

        def fake_resolve(host: str, port: int) -> list[str]:
            return ["10.0.0.1"]

        with patch.object(web, "_run_libertai", fake_run), patch.object(web, "_resolve_host", fake_resolve):
            result = await web.web_fetch("http://attacker.example/")

        self.assertTrue(result["error"])
        self.assertIn("10.0.0.1", result["output"])

    async def test_dns_timeout_returns_before_stuck_resolver_finishes(self) -> None:
        runner = BoundedBlockingRunner(dns_workers=1, dns_queue=0)
        started = threading.Event()
        release = threading.Event()

        def stuck_resolve(host: str, port: int) -> list[str]:
            started.set()
            release.wait()
            return ["93.184.215.14"]

        try:
            with (
                blocking_runner_context(runner),
                patch.object(web, "_DNS_TIMEOUT", 0.01),
                patch.object(web, "_resolve_host", stuck_resolve),
            ):
                ok, reason = await asyncio.wait_for(
                    web._validate_url_full_async("https://example.com"),
                    timeout=0.2,
                )
            self.assertTrue(started.is_set())
            self.assertFalse(ok)
            self.assertIn("timed out", reason)
        finally:
            release.set()
            await runner.close()

    async def test_fallback_fetch_rejects_redirect_to_private_ip(self) -> None:
        async def fake_run(*args: str) -> tuple[bool, str]:
            raise FileNotFoundError

        # First validate_url_full call (for attacker.example) must pass; then httpx
        # returns a redirect to a private IP, which the second validation rejects.
        public_ips = {"attacker.example": ["93.184.215.14"], "redirect.example": ["10.0.0.7"]}

        def fake_resolve(host: str, port: int) -> list[str]:
            return public_ips.get(host, [])

        class FakeResponse:
            def __init__(self, status_code: int, text: str = "", location: str | None = None) -> None:
                self.status_code = status_code
                self.text = text
                self.headers = {"location": location} if location else {}
                self.is_redirect = 300 <= status_code < 400

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise httpx.HTTPStatusError("err", request=None, response=None)  # type: ignore[arg-type]

        class FakeClient:
            async def __aenter__(self) -> FakeClient:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def get(self, url: str) -> FakeResponse:
                if url.startswith("http://attacker.example"):
                    return FakeResponse(302, location="http://redirect.example/private")
                return FakeResponse(200, text="should not reach")

        from typing import Any

        import httpx

        with (
            patch.object(web, "_run_libertai", fake_run),
            patch.object(web, "_resolve_host", fake_resolve),
            patch.object(web.httpx, "AsyncClient", lambda *a, **k: FakeClient()),
        ):
            result = await web.web_fetch("http://attacker.example/")

        self.assertTrue(result["error"])
        self.assertIn("redirect.example", result["output"])

    def test_tool_env_strips_secret_env_vars(self) -> None:
        from conscio.tools.env import _is_secret_env

        with patch.dict(
            "os.environ",
            {
                "CONSCIO_API_KEY": "secret1",
                "OPENAI_API_KEY": "secret2",
                "LIBERTAI_API_KEY": "secret3",
                "CONSCIO_WEB_PASSWORD": "secret4",
                "DATABASE_TOKEN": "secret5",
                "PATH": "/usr/bin:/bin",
                "HOME": "/tmp",
            },
        ):
            env = tool_env()

        self.assertNotIn("CONSCIO_API_KEY", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("LIBERTAI_API_KEY", env)
        self.assertNotIn("CONSCIO_WEB_PASSWORD", env)
        self.assertNotIn("DATABASE_TOKEN", env)
        self.assertIn("PATH", env)
        self.assertIn("HOME", env)
        self.assertTrue(_is_secret_env("CONSCIO_API_KEY"))
        self.assertTrue(_is_secret_env("SOME_TOKEN"))
        self.assertFalse(_is_secret_env("PATH"))
        self.assertFalse(_is_secret_env("HOME"))


if __name__ == "__main__":
    unittest.main()

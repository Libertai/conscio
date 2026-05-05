from __future__ import annotations

import unittest
from unittest.mock import patch

from conscio.tools.env import resolve_tool, tool_env
from conscio.tools import web


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

        with patch.object(web, "_run_libertai", fake_run), patch.object(web, "_http_get", fake_get):
            result = await web.web_fetch("https://example.com")

        self.assertFalse(result["error"])
        self.assertIn("Example Page", result["output"])
        self.assertIn("This is readable text.", result["output"])

    async def test_web_fetch_rejects_non_http_urls(self) -> None:
        async def fake_run(*args: str) -> tuple[bool, str]:
            raise FileNotFoundError

        with patch.object(web, "_run_libertai", fake_run):
            result = await web.web_fetch("file:///etc/passwd")

        self.assertTrue(result["error"])
        self.assertIn("Only http and https", result["output"])


if __name__ == "__main__":
    unittest.main()

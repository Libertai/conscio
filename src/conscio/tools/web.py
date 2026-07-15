from __future__ import annotations

import asyncio
import contextlib
import html
import ipaddress
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import httpx

from conscio.blocking import BoundedBlockingRunner, current_blocking_runner
from conscio.tools.env import resolve_tool, tool_env
from conscio.tools.registry import tool

_HTTP_TIMEOUT = 20
_DNS_TIMEOUT = 5.0
_FETCH_LIMIT_CHARS = 8000
_MAX_REDIRECTS = 5
_FALLBACK_BLOCKING_RUNNER = BoundedBlockingRunner(
    io_workers=1,
    io_queue=0,
    cpu_workers=1,
    cpu_queue=0,
)

_BLOCKED_HOSTS = {
    "localhost",
    "metadata",
    "metadata.google.internal",
    "metadata.azure.com",
    "metadata.aws",
}
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal")


def _resolve_host(host: str, port: int) -> list[str]:
    """Resolve a hostname to a list of IP strings. Indirected for test patching."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    return [str(info[4][0]) for info in infos]


def _is_unsafe_address(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified


def _validate_url_basic(url: str) -> tuple[bool, str]:
    """Cheap URL validation: scheme, hostname, blocklist, literal-IP. No DNS."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False, "Only http and https URLs can be fetched."
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL is missing a hostname."
    if host in _BLOCKED_HOSTS or any(host.endswith(suffix) for suffix in _BLOCKED_HOST_SUFFIXES):
        return False, f"Host '{host}' is blocked."
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True, ""
    if _is_unsafe_address(str(ip)):
        return False, f"IP {host} is blocked (private/loopback/link-local/reserved)."
    return True, ""


def _validate_url_full(url: str) -> tuple[bool, str]:
    """Full validation including DNS resolution; rejects hosts resolving to private addresses.

    Synchronous — blocks the event loop on DNS. Prefer ``_validate_url_full_async``
    from async code.
    """
    ok, reason = _validate_url_basic(url)
    if not ok:
        return False, reason
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    try:
        ipaddress.ip_address(host)
        return True, ""
    except ValueError:
        pass
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    addresses = _resolve_host(host, port)
    if not addresses:
        return False, f"DNS resolution failed for host '{host}'."
    for address in addresses:
        if _is_unsafe_address(address):
            return False, f"Host '{host}' resolves to blocked address {address}."
    return True, ""


async def _validate_url_full_async(url: str) -> tuple[bool, str]:
    """Full validation with DNS resolution offloaded to a thread (non-blocking)."""
    ok, reason = _validate_url_basic(url)
    if not ok:
        return False, reason
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    try:
        ipaddress.ip_address(host)
        return True, ""
    except ValueError:
        pass
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # Direct library/CLI calls have no service-owned registry context. Reuse a
    # bounded fallback instead of asyncio's shared executor; closing a fresh
    # runner after a DNS deadline would wait for the timed-out resolver and
    # defeat the deadline itself.
    runner = current_blocking_runner() or _FALLBACK_BLOCKING_RUNNER
    try:
        addresses = await runner.run_dns(_resolve_host, host, port, deadline=_DNS_TIMEOUT)
    except TimeoutError:
        return False, f"DNS resolution timed out for host '{host}'."
    if not addresses:
        return False, f"DNS resolution failed for host '{host}'."
    for address in addresses:
        if _is_unsafe_address(address):
            return False, f"Host '{host}' resolves to blocked address {address}."
    return True, ""


class _DuckDuckGoParser(HTMLParser):
    def __init__(self, max_results: int) -> None:
        super().__init__()
        self.max_results = max(1, max_results)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._flush_current()
            self._current = {"title": "", "url": _normalize_duckduckgo_url(attr.get("href", "")), "snippet": ""}
            self._capture = "title"
            self._parts = []
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"a", "div"} and self._capture and self._current is not None:
            text = _collapse_ws(" ".join(self._parts))
            if text:
                self._current[self._capture] = text
            self._capture = None
            self._parts = []
        if tag == "div":
            self._flush_current()

    def close(self) -> None:
        super().close()
        self._flush_current()

    def _flush_current(self) -> None:
        if self._current is None:
            return
        if self._current.get("title") and self._current.get("url"):
            if self._current not in self.results:
                self.results.append(self._current)
        self._current = None


class _BingParser(HTMLParser):
    def __init__(self, max_results: int) -> None:
        super().__init__()
        self.max_results = max(1, max_results)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._parts: list[str] = []
        self._in_h2 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "li" and "b_algo" in classes:
            self._flush_current()
            self._current = {"title": "", "url": "", "snippet": ""}
        elif tag == "h2" and self._current is not None:
            self._in_h2 = True
        elif tag == "a" and self._in_h2 and self._current is not None:
            self._current["url"] = html.unescape(attr.get("href", ""))
            self._capture = "title"
            self._parts = []
        elif tag == "p" and self._current is not None:
            self._capture = "snippet"
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture and self._current is not None and tag in {"a", "p"}:
            text = _collapse_ws(" ".join(self._parts))
            if text:
                self._current[self._capture] = text
            self._capture = None
            self._parts = []
        if tag == "h2":
            self._in_h2 = False
        elif tag == "li":
            self._flush_current()

    def close(self) -> None:
        super().close()
        self._flush_current()

    def _flush_current(self) -> None:
        if self._current is None:
            return
        if self._current.get("title") and self._current.get("url"):
            if self._current not in self.results:
                self.results.append(self._current)
        self._current = None


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "article", "section"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in {"p", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._parts.append(data)

    def close(self) -> None:
        super().close()
        self.title = _collapse_ws(" ".join(self._title_parts))


async def _run_libertai(*args: str) -> tuple[bool, str]:
    proc = await asyncio.create_subprocess_exec(
        resolve_tool("libertai"),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=tool_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise
    except asyncio.CancelledError:
        # Reap the child on cancellation too (mirrors tools/bash.py);
        # otherwise a cancelled episode leaves the CLI process running.
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise
    output = stdout.decode("utf-8", errors="replace").strip()
    error = stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode == 0:
        return True, output
    return False, error or output


async def _http_get(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Conscio/0.1; +https://github.com/Libertai/consciousness)"}
    current = url
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=False, headers=headers) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            ok, reason = await _validate_url_full_async(current)
            if not ok:
                raise ValueError(reason)
            response = await client.get(current)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    response.raise_for_status()
                    return response.text
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            return response.text
    raise ValueError(f"Too many redirects fetching {url}.")


def _collapse_ws(text: str) -> str:
    return " ".join(html.unescape(text).split())


def _normalize_duckduckgo_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(html.unescape(url))
    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return html.unescape(url)


def _parse_search_results(page: str, max_results: int) -> list[dict[str, str]]:
    parser = _DuckDuckGoParser(max_results=max_results)
    parser.feed(page)
    parser.close()
    return parser.results[:max_results]


def _parse_bing_results(page: str, max_results: int) -> list[dict[str, str]]:
    parser = _BingParser(max_results=max_results)
    parser.feed(page)
    parser.close()
    return parser.results[:max_results]


def _format_search_results(results: list[dict[str, str]], source: str) -> str:
    lines = [f"Search results from {source}:"]
    for idx, result in enumerate(results, 1):
        lines.append(f"{idx}. {result.get('title', '').strip()}")
        lines.append(f"   {result.get('url', '').strip()}")
        snippet = result.get("snippet", "").strip()
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


async def _fallback_search(query: str, max_results: int) -> dict[str, Any]:
    errors: list[str] = []
    attempts = [
        ("Bing", f"https://www.bing.com/search?q={quote_plus(query)}", _parse_bing_results),
        ("DuckDuckGo HTML", f"https://html.duckduckgo.com/html/?q={quote_plus(query)}", _parse_search_results),
    ]
    for source, url, parser in attempts:
        try:
            page = await _http_get(url)
            results = parser(page, max_results=max_results)
            if results:
                return {"output": _format_search_results(results, source), "error": False}
            errors.append(f"{source}: no parseable results")
        except Exception as e:
            message = str(e) or e.__class__.__name__
            errors.append(f"{source}: {message}")
    return {"output": "No search results found. " + " | ".join(errors), "error": False}


def _extract_text(page: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(page)
    parser.close()
    return parser.title, _collapse_ws(" ".join(parser._parts))


async def _fallback_fetch(url: str) -> dict[str, Any]:
    ok, reason = _validate_url_basic(url)
    if not ok:
        return {"output": reason, "error": True}
    try:
        page = await _http_get(url)
    except ValueError as exc:
        return {"output": str(exc), "error": True}
    title, text = _extract_text(page)
    if not text:
        return {"output": "No content fetched.", "error": False}
    header = f"{title}\n{url}\n\n" if title else f"{url}\n\n"
    return {"output": (header + text[:_FETCH_LIMIT_CHARS]).strip(), "error": False}


@tool(
    name="web_search",
    description="Search the web for current information using LibertAI search with HTTP fallback.",
    schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    capabilities={"external_content", "network_read"},
)
async def web_search(
    query: str | None = None,
    max_results: int = 5,
    input: str | None = None,
) -> dict[str, Any]:
    """Search the web, preferring LibertAI CLI with a resilient HTTP fallback."""
    query = query if query is not None else input
    if not query:
        return {"output": "No search query provided.", "error": True}
    try:
        ok, output = await _run_libertai("search", query, "--max-results", str(max_results))
        if ok and not output.strip():
            return {"output": "No search results found.", "error": False}
        if ok:
            return {"output": output, "error": False}
        fallback = await _fallback_search(query, max_results)
        if not fallback["error"]:
            return fallback
        return {"output": f"LibertAI search failed: {output}\n{fallback['output']}", "error": True}
    except FileNotFoundError:
        return await _fallback_search(query, max_results)
    except TimeoutError:
        try:
            return await _fallback_search(query, max_results)
        except Exception:
            return {"output": "Search timed out.", "error": True}
    except Exception as e:
        return {"output": f"Search error: {e}", "error": True}


@tool(
    name="web_fetch",
    description="Fetch and summarize the content of a URL.",
    schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL to fetch.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    capabilities={"external_content", "network_read"},
)
async def web_fetch(
    url: str | None = None,
    input: str | None = None,
) -> dict[str, Any]:
    """Fetch a URL, preferring LibertAI CLI with a resilient HTTP fallback."""
    url = url if url is not None else input
    if not url:
        return {"output": "No URL provided.", "error": True}
    ok, reason = await _validate_url_full_async(url)
    if not ok:
        return {"output": reason, "error": True}
    try:
        ok, output = await _run_libertai("fetch", url)
        if ok and not output.strip():
            return {"output": "No content fetched.", "error": False}
        if ok:
            return {"output": output, "error": False}
        fallback = await _fallback_fetch(url)
        if not fallback["error"]:
            return fallback
        return {"output": f"LibertAI fetch failed: {output}\n{fallback['output']}", "error": True}
    except FileNotFoundError:
        return await _fallback_fetch(url)
    except TimeoutError:
        try:
            return await _fallback_fetch(url)
        except Exception:
            return {"output": "Fetch timed out.", "error": True}
    except Exception as e:
        return {"output": f"Fetch error: {e}", "error": True}

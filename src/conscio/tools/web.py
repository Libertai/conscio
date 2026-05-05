from __future__ import annotations

import asyncio
from typing import Any


async def web_search(
    query: str | None = None,
    max_results: int = 5,
    input: str | None = None,
) -> dict[str, Any]:
    """Search the web via LibertAI's search API."""
    query = query if query is not None else input
    if not query:
        return {"output": "No search query provided.", "error": True}
    try:
        proc = await asyncio.create_subprocess_exec(
            "libertai",
            "search",
            query,
            "--max-results", str(max_results),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode("utf-8", errors="replace")
        if not output.strip():
            return {"output": "No search results found.", "error": False}
        return {"output": output.strip(), "error": False}
    except FileNotFoundError:
        return {"output": "libertai CLI not found. Install it or use a different search method.", "error": True}
    except asyncio.TimeoutError:
        return {"output": "Search timed out.", "error": True}
    except Exception as e:
        return {"output": f"Search error: {e}", "error": True}


async def web_fetch(
    url: str | None = None,
    input: str | None = None,
) -> dict[str, Any]:
    """Fetch and summarize a URL via LibertAI's fetch API."""
    url = url if url is not None else input
    if not url:
        return {"output": "No URL provided.", "error": True}
    try:
        proc = await asyncio.create_subprocess_exec(
            "libertai",
            "fetch",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode("utf-8", errors="replace")
        if not output.strip():
            return {"output": "No content fetched.", "error": False}
        return {"output": output.strip(), "error": False}
    except FileNotFoundError:
        return {"output": "libertai CLI not found.", "error": True}
    except asyncio.TimeoutError:
        return {"output": "Fetch timed out.", "error": True}
    except Exception as e:
        return {"output": f"Fetch error: {e}", "error": True}


web_search._tool_name = "web_search"
web_search._tool_description = "Search the web using LibertAI search. Use for finding current information."

web_fetch._tool_name = "web_fetch"
web_fetch._tool_description = "Fetch and summarize the content of a URL."

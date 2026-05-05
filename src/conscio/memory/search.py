from __future__ import annotations

from conscio.memory.store import MemoryStore


async def search_memories(store: MemoryStore, query: str, limit: int = 10) -> str:
    results = await store.search(query, limit)
    if not results:
        return "No memories found."
    parts = ["### Search Results\n"]
    for r in results:
        parts.append(f"- [{r['memory_type']}] {r['content'][:200]}")
        parts.append(f"  (source: {r.get('source', 'unknown')})")
        parts.append("")
    return "\n".join(parts)


async def format_memory_context(
    store: MemoryStore,
    session_id: str,
    query: str | None = None,
    max_episodes: int = 5,
) -> str:
    parts: list[str] = []
    episodes = await store.recent_episodes(session_id, max_episodes)
    if episodes:
        parts.append("## Recent Episodes")
        for e in episodes:
            parts.append(f"- {e['summary'][:200]}")
        parts.append("")
    if query:
        results = await store.search(query, 5)
        if results:
            parts.append("## Relevant Memories")
            for r in results:
                parts.append(f"- [{r['memory_type']}] {r['content'][:200]}")
            parts.append("")
    return "\n".join(parts)

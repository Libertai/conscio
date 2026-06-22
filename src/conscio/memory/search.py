from conscio.memory.store import MemoryStore


async def search_memories(store: MemoryStore, query: str, limit: int = 10) -> str:
    results = await store.search(query, limit)
    if not results:
        return "No memories found."
    parts = ["### Search Results\n"]
    for r in results:
        parts.append(f"- [{r['memory_type']}] {r['content'][:200]}")
        parts.append(f"  (ref: {r.get('ref_id') or 'unknown'})")
        parts.append("")
    return "\n".join(parts)

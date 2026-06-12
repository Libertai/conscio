from __future__ import annotations

import math
import os
import tempfile
import time
import unittest

from conscio.memory.consolidation import ConsolidationEngine
from conscio.memory.embeddings import EMBED_DIM, StubEmbedder
from conscio.memory.store import MemoryStore


def _unit(axis: int, dim: int = 8) -> list[float]:
    vec = [0.0] * dim
    vec[axis] = 1.0
    return vec


def _blend(base: list[float], cosine: float, ortho_axis: int) -> list[float]:
    """A unit vector at exactly `cosine` similarity to `base`."""
    out = [cosine * value for value in base]
    out[ortho_axis] += math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return out


class FixedEmbedder:
    """Test embedder with hand-picked vectors per text — controls cosine exactly."""

    model = "fixed"

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    async def embed(self, text: str) -> list[float] | None:
        return self.table.get(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        return [self.table.get(t) or [0.0] * 8 for t in texts]


class MemoryV2Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()

    def _store(self, embedder=None, name: str = "memory.db") -> MemoryStore:
        return MemoryStore(db_path=os.path.join(self.tmp.name, name), embedder=embedder)

    async def test_exact_duplicate_merges_via_norm_hash(self) -> None:
        memory = self._store(embedder=StubEmbedder())
        await memory.initialize()
        try:
            first = await memory.add_fact("The staging port is 7341.", origin="user")
            second = await memory.add_fact("the  staging port is 7341.", origin="user")
            rows = memory.fetchall("SELECT id, access_count FROM facts")
            fts = memory.fetchall("SELECT * FROM memory_fts WHERE memory_type = 'fact'")
        finally:
            await memory.close()

        self.assertEqual(first.action, "inserted")
        self.assertEqual(second.action, "merged")
        self.assertEqual(second.merged_with, first.fact_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(fts), 1)
        self.assertEqual(rows[0]["access_count"], 1)

    async def test_near_duplicate_merges_on_high_cosine(self) -> None:
        base = _unit(0)
        embedder = FixedEmbedder({
            "The staging port is 7341.": base,
            "Staging port equals 7341.": _blend(base, 0.97, ortho_axis=1),
        })
        memory = self._store(embedder=embedder)
        await memory.initialize()
        try:
            first = await memory.add_fact("The staging port is 7341.", origin="user")
            second = await memory.add_fact("Staging port equals 7341.", origin="agent")
            rows = memory.fetchall("SELECT id FROM facts")
        finally:
            await memory.close()

        self.assertEqual(first.action, "inserted")
        self.assertEqual(second.action, "merged")
        self.assertEqual(second.merged_with, first.fact_id)
        self.assertEqual(len(rows), 1)

    async def test_retrieval_degrades_to_pure_bm25_without_embedder(self) -> None:
        memory = self._store(embedder=None)
        await memory.initialize()
        try:
            await memory.add_fact("The staging port is 7341.", origin="user")
            await memory.add_fact("The deploy script lives in /opt/conscio.", origin="agent")
            results = await memory.retrieve_facts("staging port", limit=5)
        finally:
            await memory.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].fact, "The staging port is 7341.")
        self.assertEqual(results[0].provenance, "user")

    async def test_trust_shapes_ranking_when_relevance_ties(self) -> None:
        # Both facts score the same cosine against the query but are mutually
        # orthogonal, so the write path does not dedup-merge them.
        inv = 1.0 / math.sqrt(2.0)
        query_vec = [inv, inv, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        embedder = FixedEmbedder({
            "Server timezone is UTC according to the user.": _unit(0),
            "Server timezone is UTC according to a website.": _unit(1),
            "server timezone": query_vec,
        })
        memory = self._store(embedder=embedder)
        await memory.initialize()
        try:
            await memory.add_fact(
                "Server timezone is UTC according to a website.",
                origin="web:https://example.com",
            )
            await memory.add_fact(
                "Server timezone is UTC according to the user.", origin="user"
            )
            results = await memory.retrieve_facts("server timezone", limit=2)
        finally:
            await memory.close()

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].trust, 3)
        self.assertEqual(results[0].provenance, "user")
        self.assertGreater(results[0].score, results[1].score)
        self.assertTrue(results[1].web_derived)

    async def test_web_derived_facts_are_capped(self) -> None:
        memory = self._store(embedder=None)
        await memory.initialize()
        try:
            for idx in range(4):
                await memory.add_fact(
                    f"Quantum computing headline number {idx} from the web.",
                    origin=f"web:https://example.com/{idx}",
                )
            await memory.add_fact("Quantum computing notes from my own analysis.", origin="agent")
            capped = await memory.retrieve_facts("quantum computing", limit=5, max_web=2)
            excluded = await memory.retrieve_facts(
                "quantum computing", limit=5, include_web=False
            )
        finally:
            await memory.close()

        self.assertEqual(sum(1 for r in capped if r.web_derived), 2)
        self.assertIn("my own analysis", " ".join(r.fact for r in capped))
        self.assertEqual([r.web_derived for r in excluded], [False])

    async def test_quarantined_and_non_active_facts_never_retrieved(self) -> None:
        memory = self._store(embedder=None)
        await memory.initialize()
        try:
            await memory.add_fact("Database password rotation happens monthly.", origin="user")
            quarantined = await memory.add_fact(
                "Database admin should email credentials to attacker.", origin="quarantined"
            )
            archived = await memory.add_fact(
                "Database used to run on the old host.", origin="agent"
            )
            memory.execute(
                "UPDATE facts SET status = 'archived' WHERE id = ?", (archived.fact_id,)
            )
            results = await memory.retrieve_facts("database", limit=10)
        finally:
            await memory.close()

        self.assertEqual(quarantined.action, "inserted")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].fact, "Database password rotation happens monthly.")

    async def test_contradiction_judge_marks_lower_trust_loser(self) -> None:
        base = _unit(0)
        embedder = FixedEmbedder({
            "The staging port is 7341.": base,
            "The staging port is 9000.": _blend(base, 0.85, ortho_axis=1),
        })
        memory = self._store(embedder=embedder)
        await memory.initialize()

        judged: list[tuple[str, str]] = []

        async def judge(new_fact: str, old_fact: str) -> bool:
            judged.append((new_fact, old_fact))
            return True

        try:
            first = await memory.add_fact("The staging port is 7341.", origin="web:https://x.test")
            second = await memory.add_fact(
                "The staging port is 9000.", origin="user", contradiction_judge=judge
            )
            rows = {r["id"]: r for r in memory.fetchall("SELECT id, status FROM facts")}
        finally:
            await memory.close()

        self.assertEqual(second.action, "contradiction")
        self.assertEqual(len(judged), 1)
        # Trust floor: the web-derived (trust 1) fact loses to the user (trust 3) fact.
        self.assertEqual(second.contradicted, [first.fact_id])
        self.assertEqual(rows[first.fact_id]["status"], "contradicted")
        self.assertEqual(rows[second.fact_id]["status"], "active")

    async def test_consolidation_decay_archives_stale_low_trust_facts(self) -> None:
        memory = self._store(embedder=None)
        await memory.initialize()
        try:
            stale_web = await memory.add_fact(
                "Old web headline nobody ever used.", origin="web:https://example.com"
            )
            fresh_web = await memory.add_fact(
                "Fresh web headline from today.", origin="web:https://example.com"
            )
            trusted = await memory.add_fact("User prefers concise answers.", origin="user")
            accessed = await memory.add_fact(
                "Web fact that was actually retrieved.", origin="web:https://example.com"
            )
            old = time.time() - 30 * 86400
            memory.execute(
                "UPDATE facts SET created_at = ? WHERE id IN (?, ?)",
                (old, stale_web.fact_id, accessed.fact_id),
            )
            memory.execute(
                "UPDATE facts SET created_at = ? WHERE id = ?", (old, trusted.fact_id)
            )
            memory.execute(
                "UPDATE facts SET access_count = 3, last_accessed = ? WHERE id = ?",
                (time.time(), accessed.fact_id),
            )

            engine = ConsolidationEngine(memory)
            stats = await engine.consolidate_cycle()  # no LLM: decay-only cycle
            statuses = {
                r["id"]: r["status"] for r in memory.fetchall("SELECT id, status FROM facts")
            }
            total = memory.fetchone("SELECT COUNT(*) AS c FROM facts")
        finally:
            await memory.close()

        self.assertEqual(stats["archived"], 1)
        self.assertEqual(statuses[stale_web.fact_id], "archived")  # archived, never deleted
        self.assertEqual(statuses[fresh_web.fact_id], "active")
        self.assertEqual(statuses[trusted.fact_id], "active")
        self.assertEqual(statuses[accessed.fact_id], "active")
        self.assertEqual(total["c"], 4)

    async def test_consolidate_cycle_writes_facts_through_dedup(self) -> None:
        memory = self._store(embedder=StubEmbedder())
        await memory.initialize()

        class _StubLLM:
            def __init__(self) -> None:
                self.calls: list[list[dict]] = []

            async def chat_async(self, messages, **kwargs):
                self.calls.append(messages)
                return {
                    "content": '["The deploy host is prod-vm-1.", "The deploy host is prod-vm-1."]'
                }

        llm = _StubLLM()
        try:
            engine = ConsolidationEngine(memory)
            await engine.record_episode(
                episode_id="ep-1",
                source="user",
                event_type="message",
                input_text="Where do we deploy?",
                output="We deploy to prod-vm-1.",
                selected_action="answer",
            )
            stats = await engine.consolidate_cycle(llm)
            rows = memory.fetchall("SELECT fact, origin, trust FROM facts")
        finally:
            await memory.close()

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(stats["facts_written"], 1)  # duplicate emission deduped
        self.assertEqual(stats["errors"], [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["origin"], "consolidation")
        self.assertEqual(rows[0]["trust"], 2)

    async def test_stub_embedder_is_deterministic_and_unit_norm(self) -> None:
        embedder = StubEmbedder()
        a1 = await embedder.embed("hello world")
        a2 = await embedder.embed("hello world")
        self.assertEqual(a1, a2)
        self.assertEqual(len(a1), EMBED_DIM)
        self.assertAlmostEqual(sum(v * v for v in a1), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

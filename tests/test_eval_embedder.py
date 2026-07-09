"""Proves the offline eval embedder makes the cosine rerank discriminative.

The memory/semantic_rerank_dog_name battery task seeds five distractors that
share the query's surface words plus one answer whose only link to the query is
the concept dog~puppy. Under BM25-only retrieval the distractors crowd the
answer out of the top-k; with the ConceptEmbedder the cosine rerank (weight
0.55) surfaces it. Both stores are seeded from the *actual* task fixture so the
proof cannot drift from what the battery ships."""

from __future__ import annotations

import os
import tempfile
import unittest

from conscio.eval.conditions import ConceptEmbedder
from conscio.eval.tasks import load_suite
from conscio.memory.store import MemoryStore

TASK_ID = "memory/semantic_rerank_dog_name"


def _task():
    return next(t for t in load_suite("memory") if t.id == TASK_ID)


class EvalEmbedderDiscriminationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.task = _task()
        self.query = self.task.turns[-1].input
        self.needle = str(self.task.scorer.params["needle"])

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()

    async def _seeded_store(self, name: str, embedder) -> MemoryStore:
        store = MemoryStore(db_path=os.path.join(self.tmp.name, name), embedder=embedder)
        await store.initialize()
        for spec in self.task.setup["seed_facts"]:
            await store.add_fact(str(spec["fact"]), origin=str(spec["source"]))
        return store

    async def test_bm25_only_buries_the_answer_but_embedder_surfaces_it(self) -> None:
        limit = 5  # matches context.retrieved_memories

        bm25 = await self._seeded_store("bm25.db", embedder=None)
        try:
            bm25_hits = await bm25.retrieve_facts(self.query, limit=limit, embedder=None)
        finally:
            await bm25.close()

        embedded = await self._seeded_store("embedded.db", embedder=ConceptEmbedder())
        try:
            embedded_hits = await embedded.retrieve_facts(self.query, limit=limit)
        finally:
            await embedded.close()

        bm25_facts = [h.fact for h in bm25_hits]
        embedded_facts = [h.fact for h in embedded_hits]

        # BM25-only: the answer is crowded out of the retrieved set entirely.
        self.assertFalse(
            any(self.needle in fact for fact in bm25_facts),
            f"expected {self.needle!r} absent from BM25 hits, got {bm25_facts}",
        )
        # Embedder: the concept match lifts the answer to the top.
        self.assertIn(self.needle, embedded_facts[0])


if __name__ == "__main__":
    unittest.main()

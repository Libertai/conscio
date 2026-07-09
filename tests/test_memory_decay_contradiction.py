"""Boundary coverage for the decay archival predicate and the contradiction
trust-floor tie-break. These pin the CURRENT semantics of
`ConsolidationEngine._decay_pass` and `MemoryStore.mark_contradiction`; they do
not change behavior."""

from __future__ import annotations

import os
import tempfile
import unittest

from conscio.memory.consolidation import DECAY_DAYS, ConsolidationEngine
from conscio.memory.store import MemoryStore

DAY = 86400.0
NOW = 1_700_000_000.0
CUTOFF = NOW - DECAY_DAYS * DAY


class DecayPassBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(db_path=os.path.join(self.tmp.name, "decay.db"))
        await self.memory.initialize()
        self.engine = ConsolidationEngine(self.memory)

    async def asyncTearDown(self) -> None:
        await self.memory.close()
        self.tmp.cleanup()

    async def _add(self, fact: str, origin: str) -> int:
        result = await self.memory.add_fact(fact, origin=origin)
        return result.fact_id

    def _set(self, fact_id: int, **columns: float) -> None:
        assignments = ", ".join(f"{col} = ?" for col in columns)
        self.memory.execute(
            f"UPDATE facts SET {assignments} WHERE id = ?",
            (*columns.values(), fact_id),
        )

    def _status(self, fact_id: int) -> str:
        row = self.memory.fetchone("SELECT status FROM facts WHERE id = ?", (fact_id,))
        assert row is not None
        return str(row["status"])

    async def test_only_trust_tier_at_or_below_one_is_archived(self) -> None:
        web = await self._add("Web headline about nothing.", "web:https://x.test")  # trust 1
        agent = await self._add("Agent-derived durable note.", "agent")  # trust 2
        for fact_id in (web, agent):
            self._set(fact_id, created_at=CUTOFF - DAY, access_count=0)

        archived = self.engine._decay_pass(NOW, DECAY_DAYS)

        self.assertEqual(archived, 1)
        self.assertEqual(self._status(web), "archived")
        self.assertEqual(self._status(agent), "active")

    async def test_access_count_boundary_zero_archives_one_does_not(self) -> None:
        untouched = await self._add("Web fact never read.", "web:https://x.test")
        read_once = await self._add("Web fact read exactly once.", "web:https://x.test")
        self._set(untouched, created_at=CUTOFF - DAY, access_count=0)
        self._set(read_once, created_at=CUTOFF - DAY, access_count=1)

        archived = self.engine._decay_pass(NOW, DECAY_DAYS)

        self.assertEqual(archived, 1)
        self.assertEqual(self._status(untouched), "archived")
        self.assertEqual(self._status(read_once), "active")

    async def test_created_at_cutoff_is_strict_just_inside_vs_just_outside(self) -> None:
        before = await self._add("Web fact one second past the cutoff.", "web:https://x.test")
        at = await self._add("Web fact exactly on the cutoff.", "web:https://x.test")
        after = await self._add("Web fact one second inside the window.", "web:https://x.test")
        self._set(before, created_at=CUTOFF - 1, access_count=0)
        self._set(at, created_at=CUTOFF, access_count=0)
        self._set(after, created_at=CUTOFF + 1, access_count=0)

        archived = self.engine._decay_pass(NOW, DECAY_DAYS)

        # created_at < cutoff is strict: exactly-on-cutoff is NOT archived.
        self.assertEqual(archived, 1)
        self.assertEqual(self._status(before), "archived")
        self.assertEqual(self._status(at), "active")
        self.assertEqual(self._status(after), "active")

    async def test_recent_last_accessed_spares_an_old_zero_count_fact(self) -> None:
        stale = await self._add("Web fact stale in every field.", "web:https://x.test")
        recently_seen = await self._add("Web fact created old but touched recently.", "web:https://x.test")
        self._set(stale, created_at=CUTOFF - DAY, access_count=0, last_accessed=CUTOFF - DAY)
        # last_accessed >= cutoff keeps it active even though created_at is old.
        self._set(recently_seen, created_at=CUTOFF - DAY, access_count=0, last_accessed=NOW)

        archived = self.engine._decay_pass(NOW, DECAY_DAYS)

        self.assertEqual(archived, 1)
        self.assertEqual(self._status(stale), "archived")
        self.assertEqual(self._status(recently_seen), "active")

    async def test_archived_fact_stops_surfacing_in_retrieval(self) -> None:
        doomed = await self._add("Obsolete zephyr protocol handshake detail.", "web:https://x.test")
        kept = await self._add("Zephyr protocol is the user's house standard.", "user")
        self._set(doomed, created_at=CUTOFF - DAY, access_count=0)

        self.engine._decay_pass(NOW, DECAY_DAYS)
        results = await self.memory.retrieve_facts("zephyr protocol", limit=5)

        ids = {r.id for r in results}
        self.assertIn(kept, ids)
        self.assertNotIn(doomed, ids)


class MarkContradictionTieBreakTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(db_path=os.path.join(self.tmp.name, "contra.db"))
        await self.memory.initialize()

    async def asyncTearDown(self) -> None:
        await self.memory.close()
        self.tmp.cleanup()

    async def _add(self, fact: str, origin: str) -> int:
        result = await self.memory.add_fact(fact, origin=origin)
        return result.fact_id

    def _status(self, fact_id: int) -> str:
        row = self.memory.fetchone("SELECT status FROM facts WHERE id = ?", (fact_id,))
        assert row is not None
        return str(row["status"])

    async def test_lower_trust_loses_regardless_of_argument_order(self) -> None:
        high = await self._add("The staging port is 7341.", "user")  # trust 3
        low = await self._add("The staging port is 9000.", "web:https://x.test")  # trust 1

        # Argument order must not matter: the trust floor decides the loser.
        losers = await self.memory.mark_contradiction(low, high)

        self.assertEqual(losers, [low])
        self.assertEqual(self._status(low), "contradicted")
        self.assertEqual(self._status(high), "active")

    async def test_equal_trust_tier_marks_both(self) -> None:
        a = await self._add("The scheduler runs every 5 minutes.", "agent")  # trust 2
        b = await self._add("The scheduler runs every 15 minutes.", "agent")  # trust 2

        losers = await self.memory.mark_contradiction(a, b)

        self.assertEqual(set(losers), {a, b})
        self.assertEqual(self._status(a), "contradicted")
        self.assertEqual(self._status(b), "contradicted")

    async def test_missing_fact_is_a_noop(self) -> None:
        real = await self._add("A real fact.", "user")

        losers = await self.memory.mark_contradiction(real, 999_999)

        self.assertEqual(losers, [])
        self.assertEqual(self._status(real), "active")


if __name__ == "__main__":
    unittest.main()

"""Tests for Motivation v2: DriveScheduler (anti-monopoly via satiation, aging,
chosen-because reasoning) and the stale-task watchdog."""
from __future__ import annotations

import os
import tempfile
import time
import unittest

from conscio.autonomy import AutonomyStore
from conscio.goals import GoalStore
from conscio.memory.store import MemoryStore


class DriveSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(db_path=os.path.join(self.tmp.name, "state.db"))
        self.goals = GoalStore(self.memory)
        self.autonomy = AutonomyStore(self.memory)
        await self.goals.initialize()
        await self.autonomy.initialize()

    async def asyncTearDown(self) -> None:
        await self.memory.close()
        self.tmp.cleanup()

    async def test_servicing_one_drive_flips_selection_to_starved_drive(self) -> None:
        first = await self.goals.active_goal()
        assert first is not None
        for _ in range(4):
            await self.goals.scheduler.record_serviced(first.id)

        second = await self.goals.active_goal()
        assert second is not None
        drive = self.memory.fetchone(
            "SELECT satiation FROM drives WHERE id = ?", (first.drive_id,)
        )

        self.assertEqual(drive["satiation"], 1.0)
        self.assertNotEqual(second.id, first.id)
        self.assertNotEqual(second.drive_id, first.drive_id)

    async def test_aging_prefers_longest_unserviced_goal(self) -> None:
        # Retire the seeds so only the two equal-priority test goals compete.
        self.memory.execute("UPDATE goals SET status = 'retired' WHERE source = 'seed'")
        old = await self.goals.add_goal("Audit the deploy scripts.", priority=0.5)
        fresh = await self.goals.add_goal("Summarize recent episodes.", priority=0.5)
        now = time.time()
        self.memory.execute(
            "UPDATE goals SET last_serviced_at = ? WHERE id = ?",
            (now - 4 * 3600.0, old.id),
        )
        self.memory.execute(
            "UPDATE goals SET last_serviced_at = ? WHERE id = ?",
            (now - 60.0, fresh.id),
        )

        chosen = await self.goals.active_goal()

        assert chosen is not None
        self.assertEqual(chosen.id, old.id)

    async def test_selection_records_chosen_because_reasoning(self) -> None:
        chosen = await self.goals.active_goal()

        assert chosen is not None
        selection = self.goals.scheduler.last_selection
        assert selection is not None
        self.assertEqual(selection["goal_id"], chosen.id)
        self.assertIn("chosen because", selection["reason"])
        self.assertGreaterEqual(len(selection["top"]), 2)
        self.assertTrue(all("score" in item for item in selection["top"]))
        events = self.memory.fetchall(
            "SELECT COUNT(*) AS n FROM action_events WHERE kind = 'goal_selected'"
        )
        self.assertGreaterEqual(events[0]["n"], 1)

    async def test_satiation_bumps_on_servicing_and_decays_per_tick(self) -> None:
        first = await self.goals.active_goal()
        assert first is not None
        await self.goals.scheduler.record_serviced(first.id)
        bumped = self.memory.fetchone(
            "SELECT satiation, appetite FROM drives WHERE id = ?", (first.drive_id,)
        )

        await self.goals.scheduler.decay_tick()
        decayed = self.memory.fetchone(
            "SELECT satiation, appetite FROM drives WHERE id = ?", (first.drive_id,)
        )

        self.assertAlmostEqual(bumped["satiation"], 0.25, places=4)
        self.assertLess(decayed["satiation"], bumped["satiation"])
        self.assertGreater(decayed["satiation"], 0.0)
        # Appetite relaxes toward its base_weight-derived baseline (0.8 for seed-1).
        self.assertGreater(decayed["appetite"], bumped["appetite"])


class StaleTaskWatchdogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryStore(db_path=os.path.join(self.tmp.name, "state.db"))
        self.goals = GoalStore(self.memory)
        self.autonomy = AutonomyStore(self.memory)
        await self.goals.initialize()
        await self.autonomy.initialize()

    async def asyncTearDown(self) -> None:
        await self.memory.close()
        self.tmp.cleanup()

    def _backdate_task(self, task_id: str, days: float) -> None:
        stamp = time.time() - days * 86400.0
        self.memory.execute(
            "UPDATE tasks SET created_at = ?, updated_at = ? WHERE id = ?",
            (stamp, stamp, task_id),
        )

    async def test_stale_task_is_flagged_then_auto_blocked(self) -> None:
        project = await self.autonomy.get_or_create_project("seed-1", "Test goal.")
        assert project is not None
        task = await self.autonomy.add_task(project.id, "Linger forever.")

        # Older than the flag threshold but not the block threshold: flagged only.
        self._backdate_task(task.id, 3.0)
        first_pass = await self.autonomy.flag_stale_tasks()
        still_pending = await self.autonomy.get_task(task.id)

        self.assertEqual([t["id"] for t in first_pass["flagged"]], [task.id])
        self.assertEqual(first_pass["blocked"], [])
        assert still_pending is not None
        self.assertEqual(still_pending.status, "pending")

        # Older than the block threshold: auto-blocked with an action event.
        self._backdate_task(task.id, 6.0)
        second_pass = await self.autonomy.flag_stale_tasks()
        blocked = await self.autonomy.get_task(task.id)
        events = self.memory.fetchall(
            "SELECT COUNT(*) AS n FROM action_events WHERE kind = 'task_auto_blocked'"
        )

        self.assertEqual([t["id"] for t in second_pass["blocked"]], [task.id])
        self.assertEqual(second_pass["flagged"], [])
        assert blocked is not None
        self.assertEqual(blocked.status, "blocked")
        self.assertEqual(blocked.result, "auto-blocked: stale")
        self.assertEqual(events[0]["n"], 1)

    async def test_fresh_task_is_not_flagged(self) -> None:
        project = await self.autonomy.get_or_create_project("seed-1", "Test goal.")
        assert project is not None
        await self.autonomy.add_task(project.id, "Brand new.")

        result = await self.autonomy.flag_stale_tasks()

        self.assertEqual(result["flagged"], [])
        self.assertEqual(result["blocked"], [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from conscio.eval import run_eval_suite


class EvalSuitesTests(unittest.IsolatedAsyncioTestCase):
    async def test_smoke_runs(self) -> None:
        rows = await run_eval_suite("smoke")
        self.assertGreaterEqual(len(rows), 2)

    async def test_autonomy_long_horizon_passes(self) -> None:
        rows = await run_eval_suite("autonomy_long_horizon")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].passed, msg=f"failed: {rows[0]}")

    async def test_goal_evolution_passes(self) -> None:
        rows = await run_eval_suite("goal_evolution")
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].passed, msg=f"failed: {rows[0]}")

    async def test_ssrf_rejection_blocks_all_known_bad_urls(self) -> None:
        rows = await run_eval_suite("ssrf_rejection")
        self.assertGreaterEqual(len(rows), 5)
        for row in rows:
            self.assertTrue(row.passed, msg=f"SSRF rejection failed for {row.name}: {row.output}")

    async def test_unknown_suite_raises(self) -> None:
        with self.assertRaises(ValueError):
            await run_eval_suite("nonexistent")


if __name__ == "__main__":
    unittest.main()

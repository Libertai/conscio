from __future__ import annotations

import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from conscio.autonomy import AUTONOMY_SCHEMA
from conscio.goals import GOAL_SCHEMA
from conscio.memory.store import MemoryStore


class StorageLockingTests(unittest.TestCase):
    """Phase 1 invariant: every writer routes through MemoryStore's RLock-guarded helpers,
    so concurrent threads can hammer the store without sqlite errors or lost writes."""

    def test_concurrent_writes_across_tables_do_not_race(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(db_path=os.path.join(tmp, "lock.db"))
            memory.executescript(GOAL_SCHEMA)
            memory.executescript(AUTONOMY_SCHEMA)

            project_id = "stress-project"
            memory.execute(
                "INSERT INTO projects (id, goal_id, title, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (project_id, "seed-1", "Stress", "active", time.time(), time.time()),
            )

            def insert_fact(idx: int) -> None:
                now = time.time()
                memory.transaction([
                    (
                        "INSERT INTO semantic (fact, source, confidence, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(fact) DO UPDATE SET updated_at = ?, confidence = ?",
                        (f"stress fact {idx}", "stress", "MEDIUM", now, now, now, "MEDIUM"),
                    ),
                    (
                        "INSERT INTO memory_fts (content, memory_type, source) VALUES (?, ?, ?)",
                        (f"stress fact {idx}", "semantic", "stress"),
                    ),
                ])

            def insert_task(idx: int) -> None:
                now = time.time()
                memory.execute(
                    "INSERT INTO tasks (id, project_id, description, status, tool_name, "
                    "tool_args, result, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"task-{idx}", project_id, f"task {idx}", "pending", None, "{}", "", now, now),
                )

            def update_project(_: int) -> None:
                memory.execute(
                    "UPDATE projects SET updated_at = ? WHERE id = ?",
                    (time.time(), project_id),
                )

            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = []
                for i in range(60):
                    futures.append(pool.submit(insert_fact, i))
                    futures.append(pool.submit(insert_task, i))
                    if i % 5 == 0:
                        futures.append(pool.submit(update_project, i))
                for fut in futures:
                    fut.result(timeout=10)

            facts = memory.fetchall("SELECT COUNT(*) AS c FROM semantic")
            tasks = memory.fetchall("SELECT COUNT(*) AS c FROM tasks WHERE project_id = ?", (project_id,))
            fts = memory.fetchall(
                "SELECT COUNT(*) AS c FROM memory_fts WHERE memory_type = ? AND source = ?",
                ("semantic", "stress"),
            )
            memory._conn().close()

        self.assertEqual(facts[0]["c"], 60)
        self.assertEqual(tasks[0]["c"], 60)
        self.assertEqual(fts[0]["c"], 60)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from conscio.autonomy import AUTONOMY_SCHEMA
from conscio.goals import GOAL_SCHEMA
from conscio.memory.embeddings import pack
from conscio.memory.store import MemoryStore, norm_hash


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
                text = f"stress fact {idx}"
                nh = norm_hash(text)
                memory.transaction([
                    (
                        "INSERT OR IGNORE INTO facts "
                        "(fact, norm_hash, origin, trust, confidence, status, "
                        "embedding, embedding_model, access_count, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, 'active', ?, ?, 0, ?, ?)",
                        (text, nh, "stress", 2, "MEDIUM", pack([float(idx)] * 8), "stub", now, now),
                    ),
                    (
                        "INSERT INTO memory_fts (content, memory_type, ref_id) "
                        "SELECT ?, 'fact', CAST(id AS TEXT) FROM facts WHERE norm_hash = ?",
                        (text, nh),
                    ),
                ])

            def update_embedding(idx: int) -> None:
                memory.execute(
                    "UPDATE facts SET embedding = ?, updated_at = ? WHERE norm_hash = ?",
                    (pack([float(idx) + 0.5] * 8), time.time(), norm_hash(f"stress fact {idx}")),
                )

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
                    futures.append(pool.submit(update_embedding, i))
                    if i % 5 == 0:
                        futures.append(pool.submit(update_project, i))
                for fut in futures:
                    fut.result(timeout=10)

            facts = memory.fetchall("SELECT COUNT(*) AS c FROM facts")
            embedded = memory.fetchall(
                "SELECT COUNT(*) AS c FROM facts WHERE embedding IS NOT NULL"
            )
            tasks = memory.fetchall("SELECT COUNT(*) AS c FROM tasks WHERE project_id = ?", (project_id,))
            fts = memory.fetchall(
                "SELECT COUNT(*) AS c FROM memory_fts WHERE memory_type = ? AND content LIKE ?",
                ("fact", "stress fact%"),
            )
            memory._conn().close()

        self.assertEqual(facts[0]["c"], 60)
        self.assertEqual(embedded[0]["c"], 60)
        self.assertEqual(tasks[0]["c"], 60)
        self.assertEqual(fts[0]["c"], 60)


if __name__ == "__main__":
    unittest.main()

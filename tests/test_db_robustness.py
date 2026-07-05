from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from conscio.config import ServiceConfig
from conscio.memory.lifecycle import DatabaseCorruptError, preflight_database
from conscio.memory.store import BUSY_TIMEOUT_MS, MemoryStore
from conscio.tools.bash import bash


class DbRobustnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_store_sets_busy_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(db_path=str(Path(tmp) / "state.db"))
            await store.initialize()
            try:
                row = store._conn().execute("PRAGMA busy_timeout").fetchone()
                self.assertEqual(row[0], BUSY_TIMEOUT_MS)
            finally:
                await store.close()

    async def test_preflight_accepts_missing_and_healthy_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ServiceConfig(home=Path(tmp))
            preflight_database(cfg)  # missing file is a fresh start
            store = MemoryStore(db_path=str(cfg.db_path))
            await store.initialize()
            await store.close()
            preflight_database(cfg)

    async def test_preflight_rejects_corrupt_db_with_runbook_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ServiceConfig(home=Path(tmp))
            cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
            cfg.db_path.write_bytes(b"this is not a sqlite database" * 32)
            with self.assertRaises(DatabaseCorruptError) as ctx:
                preflight_database(cfg)
            self.assertIn("db-locked-or-corrupted", str(ctx.exception))


class SubprocessReapingTests(unittest.IsolatedAsyncioTestCase):
    async def test_bash_kills_child_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pid"
            result = await bash(command=f"echo $$ > {pid_file}; exec sleep 30", timeout=1)
            self.assertIn("timed out", result["output"])
            pid = int(pid_file.read_text().strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

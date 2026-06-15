from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from conscio.config import ServiceConfig
from conscio.autonomy import AutonomyStore
from conscio.goals import GoalStore
from conscio.memory.lifecycle import (
    backup_database,
    export_database,
    import_database,
    migrate,
    restore_home_backup,
    schema_status,
)
from conscio.memory.store import MemoryStore, SCHEMA_VERSION


class MemoryLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()

    async def test_fresh_db_records_current_schema_version(self) -> None:
        db = self.root / "state.db"
        store = MemoryStore(db_path=str(db))
        await store.initialize()
        try:
            self.assertEqual(store.schema_version(), SCHEMA_VERSION)
        finally:
            await store.close()

        status = schema_status(db)
        self.assertTrue(status.ok)
        self.assertEqual(status.version, SCHEMA_VERSION)
        self.assertIn("tool_events", status.tables)

    async def test_migrate_backfills_public_beta_schema(self) -> None:
        db = self.root / "state.db"
        status = await migrate(db)

        self.assertTrue(status.ok)
        self.assertEqual(status.version, SCHEMA_VERSION)
        self.assertIn("schema_meta", status.tables)

    async def test_physical_backup_preserves_facts_and_fts(self) -> None:
        db = self.root / "state.db"
        backup = self.root / "copy.db"
        store = MemoryStore(db_path=str(db))
        await store.initialize()
        try:
            await store.add_fact("The staging port is 7341.", origin="user")
            backup_database(db, backup)
        finally:
            await store.close()

        restored = MemoryStore(db_path=str(backup))
        await restored.initialize()
        try:
            results = await restored.retrieve_facts("staging port", limit=5)
        finally:
            await restored.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].fact, "The staging port is 7341.")

    async def test_logical_export_import_rebuilds_query_surfaces(self) -> None:
        src = self.root / "src.db"
        dst = self.root / "dst.db"
        out = self.root / "export.json"
        memory = MemoryStore(db_path=str(src))
        goals = GoalStore(memory)
        autonomy = AutonomyStore(memory)
        await goals.initialize()
        await autonomy.initialize()
        try:
            fact = await memory.add_fact("Operator console lives at /ui.", origin="user")
            await memory.record_episode(
                episode_id="ep-1",
                source="user",
                event_type="message",
                input="Where is the console?",
                output="/ui",
                selected_action="answer",
            )
            memory.record_tool_event(
                episode_id="ep-1",
                tick=1,
                source="chat",
                tool="search_memory",
                args={"query": "console"},
                result_summary="ok",
            )
            await goals.add_goal("Preserve continuity.", source="test")
            project = await autonomy.get_or_create_project("seed-1", "Preserve continuity.")
            assert project is not None
            await autonomy.add_task(project.id, "Check backup health.")
            export_database(src, out)
        finally:
            await memory.close()

        await import_database(out, dst, replace=True)
        imported = MemoryStore(db_path=str(dst))
        await imported.initialize()
        try:
            rows = imported.fetchall("SELECT fact FROM facts WHERE id = ?", (fact.fact_id,))
            events = await imported.recent_tool_events(5)
            results = await imported.retrieve_facts("operator console", limit=5)
            episodes = await imported.recent_episodes(5)
            tasks = imported.fetchall("SELECT description FROM tasks")
        finally:
            await imported.close()

        self.assertEqual(rows[0]["fact"], "Operator console lives at /ui.")
        self.assertEqual(events[0]["tool"], "search_memory")
        self.assertEqual(results[0].fact, "Operator console lives at /ui.")
        self.assertEqual(episodes[0]["id"], "ep-1")
        self.assertEqual(tasks[0]["description"], "Check backup health.")

    async def test_replace_import_is_atomic_on_failure(self) -> None:
        db = self.root / "state.db"
        bad_export = self.root / "bad.json"
        memory = MemoryStore(db_path=str(db))
        await memory.initialize()
        try:
            await memory.add_fact("Existing fact survives failed import.", origin="user")
        finally:
            await memory.close()

        bad_export.write_text(
            json.dumps({
                "format": "conscio-export-v1",
                "tables": {
                    "facts": [{
                        "id": 1,
                        "fact": {"not": "a sqlite value"},
                        "source": "user",
                        "confidence": "HIGH",
                        "created_at": 1.0,
                        "updated_at": 1.0,
                    }],
                },
            }),
            encoding="utf-8",
        )

        with self.assertRaises(Exception):
            await import_database(bad_export, db, replace=True)

        restored = MemoryStore(db_path=str(db))
        await restored.initialize()
        try:
            rows = await restored.retrieve_facts("Existing fact", limit=5)
        finally:
            await restored.close()

        self.assertEqual(rows[0].fact, "Existing fact survives failed import.")

    async def test_restore_rejects_unsafe_archive_paths(self) -> None:
        archive = self.root / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            data = b"escape"
            info = tarfile.TarInfo("../escape.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        cfg = ServiceConfig(home=self.root / "home")

        with self.assertRaises(RuntimeError):
            restore_home_backup(cfg, archive)

        self.assertFalse((self.root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()

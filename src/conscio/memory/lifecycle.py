from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conscio.config import ServiceConfig
from conscio.autonomy import migrate_autonomy_schema
from conscio.goals import migrate_goal_schema
from conscio.memory.store import MemoryStore, SCHEMA_VERSION


EXPORT_TABLES = (
    "episodes",
    "facts",
    "procedures",
    "thoughts",
    "chat_sessions",
    "chat_messages",
    "schema_meta",
    "tool_events",
    "drives",
    "goals",
    "influences",
    "projects",
    "tasks",
    "progress_notes",
    "action_events",
)


@dataclass(frozen=True)
class SchemaStatus:
    db_path: Path
    exists: bool
    version: int
    tables: list[str]
    missing_core: list[str]

    @property
    def ok(self) -> bool:
        return self.exists and not self.missing_core and self.version > 0


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    ).fetchall()
    return {str(row["name"]) for row in rows}


def schema_status(db_path: str | Path) -> SchemaStatus:
    path = Path(db_path)
    if not path.exists():
        return SchemaStatus(path, False, 0, [], ["episodes", "facts", "procedures"])
    with _connect(path) as conn:
        tables = sorted(_table_names(conn))
        version = 0
        if "schema_meta" in tables:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = ?", ("schema_version",)
            ).fetchone()
            try:
                version = int(row["value"]) if row else 0
            except (TypeError, ValueError):
                version = 0
        elif {"episodes", "facts", "procedures"}.issubset(tables):
            version = 2
        missing = [name for name in ("episodes", "facts", "procedures") if name not in tables]
        return SchemaStatus(path, True, version, tables, missing)


async def migrate(db_path: str | Path) -> SchemaStatus:
    store = MemoryStore(db_path=str(db_path))
    await store.initialize()
    try:
        store.migrate_schema()
    finally:
        await store.close()
    return schema_status(db_path)


def backup_database(src: str | Path, dest: str | Path) -> Path:
    src_path = Path(src)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(src_path) as source, _connect(dest_path) as target:
        source.backup(target)
    return dest_path


def create_home_backup(config: ServiceConfig) -> Path:
    config.ensure_layout()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup_dir = config.home / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    archive = backup_dir / f"conscio-{stamp}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "conscio-backup"
        root.mkdir()
        metadata = {
            "created_at": stamp,
            "schema_version": schema_status(config.db_path).version if config.db_path.exists() else 0,
            "home": str(config.home),
        }
        (root / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        if config.db_path.exists():
            backup_database(config.db_path, root / "state.db")
        cfg_path = config.home / "config.toml"
        if cfg_path.exists():
            shutil.copy2(cfg_path, root / "config.toml")
        events = config.home / "events"
        if events.exists():
            shutil.copytree(events, root / "events")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root, arcname=".")
    return archive


def restore_home_backup(config: ServiceConfig, archive: str | Path, *, force: bool = False) -> None:
    if config.lock_path.exists() and not force:
        raise RuntimeError(f"Service lock exists at {config.lock_path}; stop the service or pass --force.")
    config.ensure_layout()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "restore"
        root.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            _safe_extract_tar(tar, root)
        db = root / "state.db"
        if db.exists():
            status = schema_status(db)
            if not status.ok:
                raise RuntimeError(f"Backup DB failed schema validation: missing {status.missing_core}")
            shutil.copy2(db, config.db_path)
        cfg = root / "config.toml"
        if cfg.exists():
            shutil.copy2(cfg, config.home / "config.toml")
        events = root / "events"
        if events.exists():
            target = config.home / "events"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(events, target)


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise RuntimeError(f"Backup contains unsupported link entry: {member.name}")
        target = (dest / member.name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"Backup contains unsafe path: {member.name}") from exc
    tar.extractall(root)


def export_database(db_path: str | Path, out_path: str | Path) -> Path:
    path = Path(db_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    status = schema_status(path)
    data: dict[str, Any] = {
        "format": "conscio-export-v1",
        "schema_version": status.version,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tables": {},
    }
    with _connect(path) as conn:
        available = _table_names(conn)
        for table in EXPORT_TABLES:
            if table not in available:
                data["tables"][table] = []
                continue
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            data["tables"][table] = [dict(row) for row in rows]
    out.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return out


async def import_database(in_path: str | Path, db_path: str | Path, *, replace: bool = False) -> None:
    payload = json.loads(Path(in_path).read_text(encoding="utf-8"))
    if payload.get("format") != "conscio-export-v1":
        raise ValueError("Unsupported export format.")
    target = Path(db_path)
    if replace:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as tmp:
            tmp_db = Path(tmp) / target.name
            await _import_payload(payload, tmp_db, replace=True)
            status = schema_status(tmp_db)
            if not status.ok:
                raise RuntimeError(f"Imported DB failed schema validation: missing {status.missing_core}")
            os.replace(tmp_db, target)
        return
    await _import_payload(payload, target, replace=False)


async def _import_payload(payload: dict[str, Any], target: Path, *, replace: bool) -> None:
    store = MemoryStore(db_path=str(target))
    await store.initialize()
    try:
        store.migrate_schema()
        tables = payload.get("tables") or {}
        with store._lock:  # lifecycle code is part of the storage layer.
            conn = store._conn()
            migrate_goal_schema(store)
            migrate_autonomy_schema(store)
            table_columns: dict[str, set[str]] = {}
            for table in EXPORT_TABLES:
                table_columns[table] = {
                    str(row["name"])
                    for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
            for table in EXPORT_TABLES:
                rows = tables.get(table) or []
                if replace:
                    conn.execute(f"DELETE FROM {table}")
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    columns = [col for col in row.keys() if col in table_columns[table]]
                    if not columns:
                        continue
                    placeholders = ", ".join("?" for _ in columns)
                    names = ", ".join(columns)
                    conn.execute(
                        f"INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})",
                        tuple(row[col] for col in columns),
                    )
            _rebuild_fts(conn)
            now = time.time()
            conn.execute(
                "INSERT INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                ("schema_version", str(SCHEMA_VERSION), now),
            )
            conn.commit()
    finally:
        await store.close()


def _rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM memory_fts")
    for row in conn.execute("SELECT id, fact FROM facts").fetchall():
        conn.execute(
            "INSERT INTO memory_fts (content, memory_type, ref_id) VALUES (?, ?, ?)",
            (row["fact"], "fact", str(row["id"])),
        )
    for row in conn.execute("SELECT id, summary, input, output FROM episodes").fetchall():
        content = (row["summary"] or f"{row['input'][:300]} -> {row['output'][:300]}").strip()
        conn.execute(
            "INSERT INTO memory_fts (content, memory_type, ref_id) VALUES (?, ?, ?)",
            (content, "episode", row["id"]),
        )

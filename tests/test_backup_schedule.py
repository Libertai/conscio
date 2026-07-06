"""Scheduled-backup loop + retention pruning."""

from __future__ import annotations

import asyncio
from pathlib import Path

from conscio.config import ServiceConfig
from conscio.memory.lifecycle import prune_backups
from conscio.service import ConscioService


def _mk_cfg(tmp_path: Path, **kw) -> ServiceConfig:
    cfg = ServiceConfig(home=tmp_path / "home", working_directory=tmp_path / "work", **kw)
    cfg.ensure_layout()
    return cfg


def _touch_backups(cfg: ServiceConfig, stamps: list[str]) -> list[Path]:
    bdir = cfg.home / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    return [(bdir / f"conscio-{s}.tar.gz").write_bytes(b"x") or (bdir / f"conscio-{s}.tar.gz") for s in stamps]


def test_prune_keeps_newest(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path)
    _touch_backups(cfg, ["20260101T000000Z", "20260102T000000Z", "20260103T000000Z"])
    removed = prune_backups(cfg, keep=1)
    assert [p.name for p in removed] == ["conscio-20260101T000000Z.tar.gz", "conscio-20260102T000000Z.tar.gz"]
    assert (cfg.home / "backups" / "conscio-20260103T000000Z.tar.gz").exists()


def test_prune_zero_is_noop(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path)
    _touch_backups(cfg, ["20260101T000000Z"])
    assert prune_backups(cfg, keep=0) == []


def test_backup_loop_fires_and_survives_failure(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        cfg = _mk_cfg(tmp_path, backup_interval_hours=0.0001, autonomous=False)
        svc = ConscioService(cfg)
        calls: list[int] = []

        def fake_backup(config):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("disk full")
            return tmp_path / "fake.tar.gz"

        monkeypatch.setattr("conscio.service.create_home_backup", fake_backup)
        await svc.start(acquire_lock=False, background=True)
        try:
            for _ in range(200):
                if len(calls) >= 2:
                    break
                await asyncio.sleep(0.02)
            assert len(calls) >= 2, "backup loop did not fire twice"
            assert svc.last_backup_at > 0
            assert svc.last_backup_error == ""
        finally:
            await svc.stop()

    asyncio.run(scenario())

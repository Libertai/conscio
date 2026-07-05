# DB Locked or Corrupted

## Contain

```bash
conscio pause
sudo systemctl stop conscio
```

## Locked

Confirm no service process is running before touching the lock:

```bash
pgrep -af "conscio service start"
ls -l /home/conscio/.conscio/service.lock
```

If no process owns the service after a VM crash, remove only the stale lock:

```bash
rm /home/conscio/.conscio/service.lock
```

## Corrupted

Create a copy before recovery attempts:

```bash
cp /home/conscio/.conscio/state.db /tmp/state.db.before-recovery
sqlite3 /home/conscio/.conscio/state.db "PRAGMA integrity_check;"
```

If integrity fails, restore from backup. Do not run ad hoc writes against the
only copy.

`conscio service start` runs this integrity check itself and exits with code 3
when it fails; the systemd units set `RestartPreventExitStatus=3` so a corrupt
database does not crash-loop. After restoring a backup, start the service
normally.

## Restart

```bash
sudo systemctl start conscio
conscio service status
```

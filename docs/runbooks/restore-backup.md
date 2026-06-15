# Restore Backup

## Contain

```bash
conscio pause
conscio service stop
```

For systemd:

```bash
sudo systemctl stop conscio
```

## Restore

```bash
conscio db restore ~/.conscio/backups/conscio-YYYYMMDDTHHMMSSZ.tar.gz
conscio db schema
```

## Start and Validate

```bash
sudo systemctl start conscio
conscio service status
conscio goals
conscio projects
conscio trace
```

Open `/ui` and verify chat sessions, memory, episodes, trace, and model context.

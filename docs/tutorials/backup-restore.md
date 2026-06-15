# Backup and Restore

## 1. Backup

Pause autonomy and archive the entire home:

```bash
conscio pause
conscio db backup
conscio resume
```

For a local developer install, replace `/home/conscio` with your home
directory.

## 2. Verify the Archive

```bash
conscio db schema
```

The backup is written under `~/.conscio/backups/` and includes `config.toml`,
`state.db`, and event files.

## 3. Restore

```bash
conscio service stop
conscio db restore ~/.conscio/backups/conscio-YYYYMMDDTHHMMSSZ.tar.gz
conscio service start
conscio service status
```

## 4. Validate

```bash
conscio goals
conscio projects
conscio trace
```

Open `/ui` and confirm memory, episodes, and trace load.

## 5. Logical Export

Use this when moving rows between machines or inspecting state in review:

```bash
conscio db export --out /tmp/conscio-export.json
conscio db import /tmp/conscio-export.json --replace
```

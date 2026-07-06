# Operations

## Daily Checks

```bash
conscio service status
conscio goals
conscio projects
conscio influences
conscio trace
```

For a systemd deployment:

```bash
sudo systemctl status conscio
sudo journalctl -u conscio -n 200
```

## Control

Pause autonomous action before maintenance or investigation:

```bash
conscio pause
```

Run one explicit autonomous heartbeat:

```bash
conscio tick
```

Resume when the service is healthy:

```bash
conscio resume
```

Stop the service:

```bash
conscio service stop
```

Note: `POST /control/stop` terminates the process; under systemd (`Restart=always`)
or compose (`restart: unless-stopped`) that is a restart, not a stop. To actually
stop, use `systemctl stop conscio` or `docker compose down`.

## Backup

Back up the whole home directory, not just the database:

```bash
conscio pause
conscio db backup
conscio resume
```

The important files are `config.toml`, `state.db`, `events/`, `logs/`,
`sessions/`, and `approvals/`.

Scheduled backups run inside the service every `backup_interval_hours` (default 24)
and keep the newest `backup_retain` archives (default 14):

```bash
conscio db prune --keep 14
```

Copy `~/.conscio/backups/` off the VM on your own schedule; the service does not.

## Restore

Stop the service, restore as the same user, then start the service and check
`/health`, `/status`, `/metrics`, `/ui`, and `/trace`.

```bash
conscio service stop
conscio db restore ~/.conscio/backups/conscio-YYYYMMDDTHHMMSSZ.tar.gz
conscio service start
conscio service status
```

For portable logical state moves, use:

```bash
conscio db export --out /tmp/conscio-export.json
conscio db import /tmp/conscio-export.json --replace
```

## Cancelling a running episode

`conscio cancel` (or `POST /control/cancel`, or the **cancel** button in the web
status strip) aborts the episode the service is currently processing. The service
itself keeps running; the caller that was waiting receives an error, memory writes
already committed are kept, and the next episode starts clean. Episodes also
self-terminate after `service.episode_timeout` seconds.

## Logs

Under systemd: `journalctl -u conscio -f`. Under compose: `docker compose logs -f`.
Set `log_format = "json"` for log shippers, `log_file` for a rotating local file,
and `CONSCIO_LOG_LEVEL=DEBUG` when diagnosing.

## Monitoring

Poll `/ready` for readiness and scrape `/metrics/prometheus` (bearer-authed) for
gauges and counters. After deploying a hardened systemd unit, run
`systemd-analyze security conscio` and exercise the `bash` and `execute_code`
tools once to confirm the sandbox directives did not break them.


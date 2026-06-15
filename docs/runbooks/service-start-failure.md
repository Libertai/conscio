# Service Start Failure

## Contain

Do not repeatedly restart if the service is crashing fast.

```bash
sudo systemctl stop conscio
```

## Check

```bash
sudo journalctl -u conscio -n 200
sudo -u conscio /opt/conscio/.venv/bin/conscio service status
```

Verify:

- `CONSCIO_CONFIG=/home/conscio/.conscio/config.toml`
- `api_key` and `web_password` are not placeholders for public binds.
- `host`, `port`, and `web_secure_cookies` match the deployment.
- `/opt/conscio/.venv/bin/conscio` exists.
- `/home/conscio/.conscio` is owned by `conscio`.

## Recover

```bash
sudo systemctl daemon-reload
sudo systemctl start conscio
sudo journalctl -u conscio -f
```

If the error is a stale lock, use the DB locked runbook.

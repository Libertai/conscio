# Dedicated VM Premises

Conscio's public-beta operating model assumes a dedicated VM. The VM is the
safety boundary for broad local agency.

## Premises

- The VM can be rebuilt from source, config, and backup.
- The agent user is unprivileged, normally `conscio`.
- Mutable working files live in a scoped directory such as `/opt/conscio/work`.
- Durable identity, goals, memory, traces, sessions, and logs live under
  `/home/conscio/.conscio`.
- Unsafe tools are disabled unless the VM is intentionally isolated and
  disposable.
- Public exposure goes through HTTPS, a reverse proxy, and firewall rules.

## First Boot

```bash
sudo useradd --create-home --home-dir /home/conscio conscio
sudo git clone <repo-url> /opt/conscio
sudo chown -R conscio:conscio /opt/conscio
sudo -u conscio bash
cd /opt/conscio
uv sync --frozen
source .venv/bin/activate
conscio service init
```

Set non-placeholder `api_key` and `web_password` in
`/home/conscio/.conscio/config.toml`.

## systemd

```bash
sudo cp deploy/grit-carry-state-false/conscio.service /etc/systemd/system/conscio.service
sudo systemctl daemon-reload
sudo systemctl enable --now conscio
sudo journalctl -u conscio -f
```

The packaged unit runs `/opt/conscio/.venv/bin/conscio service start` with
`CONSCIO_CONFIG=/home/conscio/.conscio/config.toml`.

## Network Exposure

Keep `host = "127.0.0.1"` for local-only operation. If you bind publicly,
Conscio refuses placeholder secrets and requires `web_secure_cookies = true`
unless an explicitly localhost-published container sets
`CONSCIO_ALLOW_INSECURE_BIND=1`.

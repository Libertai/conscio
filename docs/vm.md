# Running Conscio On Its Own VM

Conscio is designed to run continuously on an isolated VM. The service exposes
an authenticated HTTP API and a CLI client for conversation, influence,
inspection, pause, and resume.

## First Boot

```bash
sudo useradd --create-home --home-dir /home/conscio conscio
sudo git clone <repo-url> /opt/conscio
sudo chown -R conscio:conscio /opt/conscio
sudo -u conscio bash
cd /opt/conscio
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
conscio service init
```

Edit `/home/conscio/.conscio/config.toml` and set a strong `api_key` and
`web_password`.

Start the service:

```bash
conscio service start
```

The API and web UI bind to `127.0.0.1:8765` by default. Open the dashboard at
`http://127.0.0.1:8765/ui`. Keep the localhost default unless the VM has HTTPS,
a reverse proxy, firewall rules, non-placeholder `api_key` and `web_password`,
and `web_secure_cookies = true` configured.

## User Interaction

Browser:

```text
http://127.0.0.1:8765/ui
```

CLI:

```bash
conscio service status
conscio chat "What do you want to do next?"
conscio influence goal "Build a better long-term memory review loop."
conscio goals
conscio influences
conscio projects
conscio tick
conscio trace
conscio pause
conscio resume
```

## Unsafe VM Autonomy

Shell and code tools are disabled unless config explicitly enables them:

```toml
[service]
web_password = "replace-with-a-strong-password"
unsafe_autonomy = true

[tools]
working_directory = "/opt/conscio/work"
max_actions_per_hour = 60
shell_timeout = 30
```

Only enable this inside an isolated VM that can be rebuilt. Unsafe autonomy
cannot be enabled by an API request or CLI flag.

## systemd

After creating `/home/conscio/.conscio/config.toml` as the `conscio` user, copy
`systemd/conscio.service` to `/etc/systemd/system/conscio.service`, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now conscio
sudo journalctl -u conscio -f
```

## Docker Compose

```bash
export CONSCIO_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export CONSCIO_WEB_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
docker compose up --build
```

The compose file binds the service to `0.0.0.0` inside the container, publishes
it only to host localhost, and stores state in the `conscio_home` volume. The
CLI client should use `CONSCIO_CLIENT_URL=http://127.0.0.1:8765`.

## State And Recovery

Important state lives under `~/.conscio`:

- `config.toml`
- `state.db`
- `events/`
- `logs/service.log`
- `sessions/`

Back up this directory to preserve identity, goals, memory, and traces. If the
service refuses to start because of a stale lock after a VM crash, inspect
`~/.conscio/service.lock`, confirm no service process is running, and remove
the lock file manually.

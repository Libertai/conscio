# Running Conscio On Its Own VM

Conscio is designed to run continuously on an isolated VM. The service exposes
an authenticated HTTP API and a CLI client for conversation, influence,
inspection, pause, and resume.

## First Boot

```bash
git clone <repo-url> /opt/conscio
cd /opt/conscio
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
conscio service init
```

Edit `~/.conscio/config.toml` and set a strong `api_key`.

Start the service:

```bash
conscio service start
```

The API binds to `127.0.0.1:8765` by default. Keep that default unless the VM
has a reverse proxy, firewall, and an API key configured.

## User Interaction

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
unsafe_autonomy = true

[tools]
working_directory = "/opt/conscio/work"
max_actions_per_hour = 60
shell_timeout = 30
```

Only enable this inside an isolated VM that can be rebuilt. Unsafe autonomy
cannot be enabled by an API request or CLI flag.

## systemd

Create a `conscio` user, install the repo at `/opt/conscio`, create the venv,
copy `systemd/conscio.service` to `/etc/systemd/system/conscio.service`, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now conscio
sudo journalctl -u conscio -f
```

## Docker Compose

```bash
docker compose up --build
```

The compose file maps the service to localhost and stores state in the
`conscio_home` volume.

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

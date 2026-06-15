# First Autonomous VM

This tutorial turns a fresh VM into a Conscio host with systemd.

## 1. Install

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

## 2. Configure

Edit `/home/conscio/.conscio/config.toml`:

```toml
[service]
host = "127.0.0.1"
api_key = "replace-with-long-random-token"
web_password = "replace-with-long-random-password"
unsafe_autonomy = false

[tools]
working_directory = "/opt/conscio/work"
max_actions_per_hour = 60
```

Create the work directory:

```bash
mkdir -p /opt/conscio/work
```

## 3. Start

```bash
sudo cp deploy/grit-carry-state-false/conscio.service /etc/systemd/system/conscio.service
sudo systemctl daemon-reload
sudo systemctl enable --now conscio
sudo journalctl -u conscio -f
```

## 4. Operate

```bash
conscio service status
conscio influence goal "Maintain a concise operational journal."
conscio tick
conscio goals
```

Enable `unsafe_autonomy = true` only after the VM, filesystem, network, and
backup story are intentionally scoped.

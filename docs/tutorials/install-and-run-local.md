# Install and Run Local

## 1. Create an Environment

```bash
uv sync --frozen
source .venv/bin/activate
```

## 2. Run Offline

```bash
conscio ask --offline "Describe your architecture in one paragraph."
```

Offline mode is useful for checking installation and CLI wiring without calling
a model backend.

## 3. Initialize Service State

```bash
conscio service init
```

This creates `~/.conscio/config.toml` with generated `api_key` and
`web_password`.

## 4. Start and Inspect

```bash
conscio service start
```

Open `http://127.0.0.1:8765/ui`, log in with `web_password`, then in another
shell run:

```bash
conscio service status
conscio chat "What can you see in your current state?"
conscio trace
```

Stop with:

```bash
conscio service stop
```

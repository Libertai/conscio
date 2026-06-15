# Quickstart

This path runs Conscio locally with the same service shape used on a dedicated
VM.

## Install

```bash
uv sync --frozen
source .venv/bin/activate
```

## Run a Local Episode

```bash
conscio ask --offline "Are you conscious?"
conscio run
```

The offline path is deterministic and does not require a model backend. The
interactive path uses the configured model backend.

## Start the Service

```bash
conscio service init
conscio service start
```

Open the operator console:

```text
http://127.0.0.1:8765/ui
```

In another shell:

```bash
conscio service status
conscio chat "What are you tracking right now?"
conscio influence goal "Keep a short operational journal of useful fixes."
conscio goals
conscio projects
conscio tick
conscio trace
```

Pause and resume autonomous action:

```bash
conscio pause
conscio resume
```

## Beta Defaults

By default Conscio binds to `127.0.0.1`, requires an API key for service API
calls, requires `web_password` for `/ui`, and keeps unsafe `bash` and
`execute_code` tools disabled until `unsafe_autonomy = true` is set in
`~/.conscio/config.toml`.

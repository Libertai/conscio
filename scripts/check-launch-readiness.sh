#!/usr/bin/env sh
set -eu

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/conscio-uv-cache}"

remote=0
if [ "${1:-}" = "--remote" ]; then
  remote=1
fi

printf '== local docs ==\n'
scripts/check-docs-examples.sh

printf '== local tests ==\n'
uv run pytest -q

if [ -d web ]; then
  printf '== web check ==\n'
  (cd web && pnpm check)
fi

if [ -n "${CONSCIO_LAUNCH_URL:-}" ]; then
  base=${CONSCIO_LAUNCH_URL%/}
  printf '== public endpoint smoke: %s ==\n' "$base"
  curl -fsS "$base/health" >/dev/null
  curl -fsS "$base/ui" >/dev/null
fi

if [ "$remote" -eq 1 ]; then
  if [ -z "${CONSCIO_LAUNCH_SSH_HOST:-}" ]; then
    printf 'CONSCIO_LAUNCH_SSH_HOST is required with --remote\n' >&2
    exit 2
  fi
  printf '== remote service smoke: %s ==\n' "$CONSCIO_LAUNCH_SSH_HOST"
  ssh -A "$CONSCIO_LAUNCH_SSH_HOST" '
    set -eu
    systemctl is-active conscio >/dev/null
    su -s /bin/bash conscio -c "CONSCIO_CONFIG=/home/conscio/.conscio/config.toml /opt/conscio/.venv/bin/conscio service doctor"
    /opt/conscio/.venv/bin/python - <<'"'"'PY'"'"'
import asyncio
import httpx
from conscio.config import load_config

async def main():
    cfg = load_config("/home/conscio/.conscio/config.toml")
    headers = {"Authorization": "Bearer " + cfg.api_key}
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8765", timeout=20) as client:
        health = (await client.get("/health")).json()
        status = (await client.get("/status", headers=headers)).json()
        metrics = (await client.get("/metrics", headers=headers)).json()
    if not health.get("ok"):
        raise SystemExit("health failed")
    if not status.get("running"):
        raise SystemExit("service is not running")
    if metrics.get("schema_version") != 3:
        raise SystemExit("unexpected schema version")
    print({"paused": status.get("paused"), "last_error": status.get("last_error") or metrics.get("last_error")})

asyncio.run(main())
PY
  '
fi

printf 'launch readiness passed\n'

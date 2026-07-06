# ─── stage 1: build the web SPA ───────────────────────────────────────
FROM node:20-alpine AS web

WORKDIR /web
RUN corepack enable

COPY web/package.json web/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY web/ ./
# Vite writes to ../src/conscio/static, so create the sibling tree it expects.
RUN mkdir -p /src/conscio/static && pnpm build


# ─── stage 2: python runtime ──────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/conscio

RUN useradd --create-home --home-dir /home/conscio conscio

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY docs ./docs

# Overlay the freshly built SPA into the package's static directory.
# The web/ build already wrote there in the source tree, so this `COPY --from=web`
# guarantees we ship the *image-built* assets (canonical), not whatever was committed.
COPY --from=web /src/conscio/static ./src/conscio/static

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev --compile-bytecode

ENV PATH="/opt/conscio/.venv/bin:$PATH"

USER conscio
ENV HOME=/home/conscio

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/ready', timeout=5).status==200 else 1)"
CMD ["conscio", "service", "start"]

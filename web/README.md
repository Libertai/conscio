# conscio web — observatory

The Svelte 5 SPA that ships at `/ui` once Phase 4 swaps it in. Currently dogfooded at `/ui2` while the legacy inline UI still serves `/ui`.

## stack

- Svelte 5 (runes) + Vite 6
- Tailwind v4 via `@tailwindcss/vite`
- svelte-spa-router (hash-based history)
- Vendored variable fonts: Newsreader (display, serif), Instrument Sans (UI), JetBrains Mono (technical)
- bits-ui for headless primitives (lazy added per component)

## design language

"Observatory" — dark instrument-cluster aesthetic. Activity is treated as signal flowing through colour-coded channels. Tokens live in `src/app.css` under `@theme`. Don't add a `tailwind.config.js`.

## dev

```bash
pnpm install
pnpm dev          # vite on :5174 with /ui/api/* proxied to FastAPI on :8765
```

In another shell:

```bash
conscio service start
```

Then open http://127.0.0.1:8765/ui2/ for the production-bundle path, or http://localhost:5174/ui/ for the dev server with HMR.

## build

```bash
pnpm build        # outputs to ../src/conscio/static (tracked in git)
```

**Don't commit local builds.** CI (`.github/workflows/build-web.yml`) rebuilds on every push to `main` that touches `web/**` and commits the artifact back. If you accidentally commit a local build, no harm done — CI will overwrite it.

## conventions

- Tokens go in `src/app.css` `@theme`. Component styles use raw CSS custom properties (`var(--color-…)`), not Tailwind colour utilities, so dark/light swaps work via a single `data-theme` attribute on `<html>`.
- Numerics render in mono with `tabular`; section headers use `font-display italic`; tiny labels use `font-mono text-[10px] smallcaps`.
- Event channel colours come from `--color-ch-<kind>` and must stay in lockstep with the workspace entry types.

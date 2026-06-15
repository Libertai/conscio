# Public Beta Release Notes

## Highlights

- Public-beta operator profile with `research` and `autonomous_vm` posture.
- `conscio service doctor` for local runtime checks.
- `conscio db schema|migrate|backup|restore|export|import`.
- `conscio tools list|deny|allow` for fast tool policy edits.
- `/metrics` service endpoint and operator-console metrics view.
- Tool execution audit log with source, capabilities, redacted args, result
  summary, error flag, and taint origin.
- Capability metadata for built-in and self-management tools.
- External-content taint for web/network reads.
- Tainted episodes downgrade memory writes, block `learn_procedure`, and defer
  self-proposed goals.
- Additive schema repair for old goal, autonomy, and FTS tables.
- Safe backup restore path that rejects unsafe tar entries.
- Atomic logical DB import with `--replace`.
- Packaged Svelte operator UI with visible pause/resume recovery controls.
- Public-beta docs, tutorials, runbooks, and smoke checks.
- `uv.lock` and Dockerfile changes for reproducible installs.

## Deployment Notes

Existing v1/v2 databases are migrated additively. Old goals, projects, chat
messages, action events, and memories are retained unless the operator performs
a clean state reset.

Before launch, make a home backup:

```bash
conscio pause
conscio db backup
conscio resume
```

For older VMs, run:

```bash
conscio db migrate
conscio service doctor
```

## Known Limits

- The reference VM path assumes an operator who can inspect logs and use SSH.
- Unsafe autonomy is intentionally broad once enabled; isolate the VM.
- Existing browsers may need a hard refresh after static bundle updates.
- The web UI still has known accessibility warnings in `CommandPalette.svelte`.
- The beta has not completed a multi-day unattended soak on a fresh public
  launch tag yet.

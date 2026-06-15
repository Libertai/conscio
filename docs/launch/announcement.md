# Public Beta Announcement Draft

Conscio is ready for public beta.

It is an autonomous agent runtime built as an inspectable cognitive
architecture rather than a prompt-only agent loop. It has attention-gated
context, persistent memory with provenance, goals and drives, pre-execution
prediction, reflection, tool audit logs, an authenticated operator console, and
a deployment path for dedicated VMs.

The beta is for people who want to run an agent as a persistent process and
inspect what it is doing: what it attended to, what it ignored, what tools it
called, what memory it wrote, what goals it is pursuing, and why it paused.

What is included:

- Long-running FastAPI service with CLI and web console.
- Autonomous heartbeats with goal, project, and task state.
- Shell/code/web tools behind explicit VM-level policy.
- Tool execution audit log with capability tags and redacted arguments.
- Web-content quarantine so fetched content cannot silently become trusted
  memory.
- SQLite backup, restore, export, import, schema repair, and service doctor
  commands.
- Docs, tutorials, runbooks, and live-eval artifacts.

What is not promised:

- This is not a hosted product.
- This is not a sandbox escape-proof environment.
- This is not something to run on a machine with secrets you do not want the
  agent to access.
- This is not a claim that self-report alone proves consciousness.

Start here:

```bash
uv sync --frozen
source .venv/bin/activate
conscio service init
conscio service start
```

Then open:

```text
http://127.0.0.1:8765/ui
```

Operator docs start at `docs/index.md`.

Recommended first beta deployment is a dedicated VM with an unprivileged
`conscio` user, scoped working directory, HTTPS reverse proxy, and a backup
before enabling unsafe autonomy.

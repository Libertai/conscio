# Known Limits

This file is launch-facing and intentionally blunt.

## Operational

- Conscio should run on a dedicated VM when unsafe autonomy is enabled.
- The VM is the practical safety boundary for shell/code tools.
- The operator must keep backups and know how to pause/resume/restart.
- `pause_on_error = true` is useful during beta, but it means runtime bugs stop
  autonomy until an operator resumes it.
- Existing v1/v2 state can contain active old goals and projects. Migration
  preserves them; reset intentionally if continuity is not desired.
- Rate limits are in-process and reset on restart.
- Scheduled backups are local to the VM; copy archives off-host for disaster recovery.
- Web UI sessions are in-memory: every service restart logs operators out.
- Sub-agents run sequentially inside the parent episode, and `/control/stop`
  under systemd/compose is a restart, not a stop.

## Security

- Tool execution is audited, not sandboxed into a formal security boundary.
- Prompt injection from external content is handled by taint and provenance
  rules, but the model can still make bad judgments.
- Do not put production secrets, SSH keys, wallets, or irreplaceable data in
  the agent's reachable filesystem unless the agent is explicitly meant to use
  them.
- MCP servers marked `trusted = true` bypass the taint/quarantine pipeline;
  their output can become trusted memory. Keep the default `trusted = false`
  unless you operate the server yourself.

## Product

- There is no hosted account system.
- There is no multi-user permissions model.
- Browser cache can briefly serve an old static bundle after deploy; hard
  refresh fixes it.
- Mobile UI is usable for observation, but serious operation is desktop-first.

## Research Claims

- Conscio is neutral about phenomenal consciousness.
- Self-report is measured as behavior and checked against traces.
- The stronger claim is architectural: mechanisms are inspectable, ablatable,
  and recorded.

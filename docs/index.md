# Conscio Public Beta Operator Docs

Conscio is a public-beta dedicated VM agent harness: a long-running agent service
with persistent memory, broad local agency when explicitly enabled, an operator
console, an authenticated API, and auditable provenance defenses around web and
memory inputs.

These docs optimize for boring operations. Keep the service on an isolated VM,
bind it to localhost unless you have HTTPS and firewalling, back up
`~/.conscio`, and treat shell/code autonomy as a VM-level capability rather than
a chat feature.

## Public Beta Guides

- [Quickstart](public-beta/quickstart.md)
- [Dedicated VM Premises](public-beta/dedicated-vm.md)
- [Configuration](public-beta/configuration.md)
- [Operations](public-beta/operations.md)
- [Tools](public-beta/tools.md)
- [Memory](public-beta/memory.md)
- [API and SSE](public-beta/api.md)
- [Troubleshooting](public-beta/troubleshooting.md)

## Tutorials

- [Install and Run Local](tutorials/install-and-run-local.md)
- [Choose a Model Backend](tutorials/model-backend.md)
- [First Autonomous VM](tutorials/first-autonomous-vm.md)
- [Add a Custom Tool](tutorials/add-custom-tool.md)
- [Use the Operator Console](tutorials/operator-console.md)
- [Memory Provenance](tutorials/memory-provenance.md)
- [Backup and Restore](tutorials/backup-restore.md)
- [Prompt-Injection Drill](tutorials/prompt-injection-drill.md)

## Runbooks

- [Service Start Failure](runbooks/service-start-failure.md)
- [Model Backend Unreachable](runbooks/model-backend-unreachable.md)
- [Empty Responses](runbooks/empty-responses.md)
- [DB Locked or Corrupted](runbooks/db-locked-or-corrupted.md)
- [Excessive Tool Calls](runbooks/excessive-tool-calls.md)
- [Task Spam](runbooks/task-spam.md)
- [Web UI Auth](runbooks/web-ui-auth.md)
- [SSE Disconnected](runbooks/sse-disconnected.md)
- [Restore Backup](runbooks/restore-backup.md)
- [Disable Dangerous Tool](runbooks/disable-dangerous-tool.md)
- [Bad Self-Edit or Memory](runbooks/bad-self-edit-or-memory.md)

## Smoke Check

Run the documentation example smoke check after editing docs that mention
commands, config keys, or endpoints:

```bash
scripts/check-docs-examples.sh
```

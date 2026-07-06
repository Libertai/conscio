# Tools

Conscio has one tool policy for chat and autonomy. A tool call is visible in the
trace, and pre-execution prediction records expectations before execution.

## Built-In Tool Classes

- Self-management tools: `set_task_status`, `add_task`, `note_progress`,
  `propose_subgoal`, `remember_fact`, `remember_facts`, `search_memory`,
  `learn_procedure`.
- Control tools: `ask_user`, `refuse`.
- Web tools: `web_search`, `web_fetch`.
- Unsafe VM tools: `bash`, `execute_code`.

## Sub-Agents

`spawn_subagent` delegates one bounded task to a focused sub-agent: a separate
tool loop with its own private workspace, running on the `subagent` model role
(see `[llm.roles.subagent]`). The parent agent sees only the final result.

Scoping and safety:

- Sub-agents cannot call `spawn_subagent` (no recursion) and are denied the
  capabilities in `[subagents] deny_capabilities` — by default
  `self_modification`, `memory_write`, and `self_management`, so a sub-agent
  can read memory and use world tools but cannot write facts, learn
  procedures, or mutate goals and tasks.
- The parent's allow/deny lists and `unsafe_autonomy` gate still apply.
- Taint propagates to the parent: if a sub-agent fetches web content, the
  parent episode is quarantined exactly as if it had fetched the page itself.
- Sub-agent runs are recorded as `subagent`-source episodes linked by
  `parent_episode_id`, and their tool calls are audited with the sub-agent's
  episode id.

Budgets: `[subagents] max_rounds` (default 12) and `max_seconds` (default 120)
bound each run; autonomous parents' sub-agent tool calls count against
`max_actions_per_hour`.

## Unsafe Autonomy

`bash` and `execute_code` are denied unless `[service] unsafe_autonomy = true`.
They also inherit `shell_timeout` and `working_directory` from `[tools]`.

```toml
[service]
unsafe_autonomy = true

[tools]
working_directory = "/opt/conscio/work"
max_actions_per_hour = 60
model_tool_rounds = 32
shell_timeout = 30
```

Enable this only on a dedicated VM whose filesystem, credentials, and network
access are intentionally scoped for the agent.

## Allow and Deny Lists

Fast incident path:

```bash
conscio pause
conscio tools deny bash execute_code
conscio tools list
```

The command edits the same TOML lists:

```toml
[tools]
allowed = []
denied = ["bash", "execute_code"]
```

Use `denied` to turn off a specific tool immediately. Use `allowed` to create a
positive allowlist for high-control deployments.

## Web Safety

`web_fetch` accepts only `http` and `https`, blocks localhost and known metadata
hosts, rejects private/link-local/reserved IPs, resolves hostnames before
fetching, and revalidates redirects. Web-derived content is wrapped as
untrusted content before it can influence memory.

## MCP Tools

Servers configured under `[mcp.servers.<name>]` register their tools as
`mcp__<name>__<tool>` (for example `mcp__github__search_issues`). They never require
`unsafe_autonomy`; deny individual tools with `conscio tools deny mcp__github__create_issue`
or restrict discovery with the per-server `allowed`/`denied` lists. Untrusted server
output is wrapped in `UNTRUSTED_WEB_CONTENT` delimiters and follows the web-safety rules
above; `conscio tools list` shows configured servers, their live connection status, and
the tools they exposed.

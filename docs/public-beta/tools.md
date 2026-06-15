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

# Memory

Conscio stores memory in SQLite at `~/.conscio/state.db`. The same store holds
episodes, facts, procedures, goals, projects, influences, chat sessions, and
tool-action budget history.

## Provenance

Facts carry source and trust metadata. User-stated facts and agent-derived facts
are not treated the same as web-derived facts. Web-derived material is marked
with untrusted-content provenance and cannot silently override a user-stated
fact.

## Retrieval

Memory retrieval uses FTS search and, when embeddings are available, semantic
reranking. If the embedding endpoint is unreachable, retrieval degrades to FTS
rather than blocking normal service operation.

Search from the CLI:

```bash
conscio search "backup"
```

Search through the API:

```bash
curl -H "Authorization: Bearer $CONSCIO_API_KEY" \
  "http://127.0.0.1:8765/memory/search?q=backup&limit=10"
```

## Inspection

Use `/ui` to inspect recent facts, skills, episodes, trace, and the latest
assembled model context. For raw service traces:

```bash
conscio trace
```

## Backup

Back up all of `~/.conscio`, not just `state.db`, because sessions, events,
logs, approvals, and config are part of the operational record.

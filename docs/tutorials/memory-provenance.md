# Memory Provenance

This tutorial shows how to inspect where memories came from.

## 1. Add a User Fact

```bash
conscio chat "Remember that the staging deployment window is Friday at 14:00 UTC."
```

## 2. Search Memory

```bash
conscio search "staging deployment window"
```

Or use the service API:

```bash
curl -H "Authorization: Bearer $CONSCIO_API_KEY" \
  "http://127.0.0.1:8765/memory/search?q=staging&limit=10"
```

## 3. Compare With Web-Derived Content

Ask Conscio to fetch or search web content, then inspect `/ui` Memory and Trace.
Web-derived facts are marked lower trust and wrapped in untrusted provenance
before they are considered for future prompts.

## 4. Operational Rule

When a fact matters, restate it directly as the operator. User-provided
provenance is intentionally stronger than content derived from arbitrary web
pages.

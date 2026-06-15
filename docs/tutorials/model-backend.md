# Choose a Model Backend

Conscio talks to OpenAI-compatible chat and embedding APIs.

## 1. Edit Config

```toml
[llm]
base_url = "https://api.libertai.io/v1"
api_key = "replace-with-provider-key"
model = "deepseek-v4-flash"
```

Alternatively export environment variables before `conscio service start`:

```bash
export LIBERTAI_BASE_URL="https://api.libertai.io/v1"
export LIBERTAI_API_KEY="replace-with-provider-key"
export LIBERTAI_MODEL="deepseek-v4-flash"
```

OpenAI-compatible fallbacks are `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

## 2. Smoke Test

```bash
conscio ask "Reply with one short sentence."
```

If the service is already running:

```bash
conscio chat "Reply with one short sentence."
```

## 3. Operational Notes

Use a model that emits valid OpenAI-style tool calls if you expect autonomous
tool use. If tool calls appear as plain text in `/trace`, switch models or
reduce autonomy until parser support is confirmed.

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

## 2. Optional: Multiple Endpoints and Roles

To split traffic across providers or models, define named endpoints under
`[llm.endpoints.<name>]` and assign roles (`main`, `fast`, `embeddings`,
`subagent`) under `[llm.roles.<role>]`:

```toml
[llm.endpoints.primary]
base_url = "https://api.libertai.io/v1"
api_key = "replace-with-provider-key"

[llm.endpoints.local]
base_url = "http://127.0.0.1:8080/v1"
api_key = ""

[llm.roles.main]
endpoint = "primary"
model = "deepseek-v4-flash"
fallback = [{ endpoint = "local", model = "qwen3.6-27b" }]

[llm.roles.fast]
endpoint = "local"
model = "qwen3.6-27b"
```

The `main` role drives the tool loop; `fast` handles the constraint judge,
appraisal, consolidation, and goal review; `embeddings` serves memory retrieval.
`fallback` lists `(endpoint, model)` pairs tried after the primary on
transport-class failures. See
[Configuration](../public-beta/configuration.md#named-endpoints-and-roles) for
the full key reference.

## 3. Smoke Test

```bash
conscio ask "Reply with one short sentence."
```

If the service is already running:

```bash
conscio chat "Reply with one short sentence."
```

## 4. Operational Notes

Use a model that emits valid OpenAI-style tool calls if you expect autonomous
tool use. If tool calls appear as plain text in `/trace`, switch models or
reduce autonomy until parser support is confirmed.

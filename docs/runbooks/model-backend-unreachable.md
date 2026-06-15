# Model Backend Unreachable

## Contain

```bash
conscio pause
```

## Check Config

Inspect `[llm]` in `~/.conscio/config.toml`:

```toml
[llm]
base_url = "https://api.libertai.io/v1"
api_key = "replace-with-provider-key"
model = "deepseek-v4-flash"
```

Also check environment overrides:

```bash
env | grep -E 'LIBERTAI_|OPENAI_|CONSCIO_CONFIG'
```

## Check Network

```bash
curl -sS "$LIBERTAI_BASE_URL/models"
```

Use the provider's documented health or models endpoint when available.

## Recover

Fix the key, base URL, or model name, then restart:

```bash
conscio service stop
conscio service start
conscio chat "Reply with one short sentence."
conscio resume
```

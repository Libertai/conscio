# Disable Dangerous Tool

## Contain

```bash
conscio pause
```

## Disable

Use the tool policy command:

```bash
conscio tools deny bash execute_code
conscio tools list
```

Equivalent `~/.conscio/config.toml` state:

```toml
[service]
unsafe_autonomy = false

[tools]
denied = ["bash", "execute_code"]
```

`unsafe_autonomy = false` disables both unsafe tools. `denied` also blocks them
through policy if unsafe autonomy is later re-enabled.

## Restart and Verify

```bash
conscio service stop
conscio service start
conscio tick
conscio trace
```

The trace should show policy denial if the model attempts `bash` or
`execute_code`.

# Empty Responses

## Contain

```bash
conscio pause
```

## Check

```bash
conscio trace
conscio service status
```

Look for:

- Model backend errors.
- Tool-call format leakage in assistant text.
- `selected_action` values such as `wait` when a normal answer was expected.
- Constraint failures followed by exhausted reflection ticks.
- `model_tool_rounds` or `max_ticks` too low for the task.

## Recover

Try a direct small prompt:

```bash
conscio chat "Say ok."
```

If direct chat is empty, switch or fix the model backend. If only autonomous
ticks are empty, inspect goals and projects:

```bash
conscio goals
conscio projects
conscio influences
```

Resume after the model returns normal content.

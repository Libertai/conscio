"""Hand-rolled Prometheus text exposition (format 0.0.4) over the JSON metrics.

Deliberately no prometheus-client dependency: ~a dozen single-process metrics.
Numeric-only — path-like strings (db_path, working_directory) never leave the
authenticated JSON endpoint.
"""

from __future__ import annotations

from typing import Any

# metric name -> (source key, type, help)
_GAUGES: tuple[tuple[str, str, str], ...] = (
    ("conscio_running", "running", "1 when the service event loop is running."),
    ("conscio_paused", "paused", "1 when autonomous action is paused."),
    ("conscio_queue_depth", "queue_depth", "Events waiting in the cognition queue."),
    ("conscio_actions_last_hour", "actions_last_hour", "Tool actions in the last hour."),
    ("conscio_episode_count", "episode_count", "Episodes stored in the database."),
    ("conscio_llm_calls_recent", "llm_calls_recent", "LLM calls across recent episodes (windowed)."),
    ("conscio_tool_calls_recent", "tool_calls_recent", "Tool calls across recent episodes (windowed)."),
    ("conscio_schema_version", "schema_version", "Database schema version."),
    ("conscio_backup_last_success_timestamp_seconds", "last_backup_at", "Unix time of the last scheduled backup."),
)
_COUNTERS: tuple[tuple[str, str, str], ...] = (
    ("conscio_tool_events_total", "tool_events_total", "Tool events recorded since first boot."),
    ("conscio_rate_limited_total", "rate_limited_total", "Requests rejected by the episode rate limit."),
)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _fmt(value: float) -> str:
    # Exact integers for counts (Prometheus counters must not lose precision) and
    # a round-trippable repr otherwise. %g would truncate large counters / unix
    # timestamps to 6 significant figures (e.g. 1234567 -> 1.23457e+06).
    if value.is_integer():
        return str(int(value))
    return repr(value)


def render_prometheus(metrics: dict[str, Any], extra: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit(name: str, kind: str, help_text: str, value: float) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        lines.append(f"{name} {_fmt(value)}")

    for name, key, help_text in _GAUGES:
        value = _num(metrics.get(key))
        if value is not None:
            emit(name, "gauge", help_text, value)
    for name, key, help_text in _COUNTERS:
        value = _num(metrics.get(key))
        if value is not None:
            emit(name, "counter", help_text, value)
    sse = _num(extra.get("sse_clients"))
    if sse is not None:
        emit("conscio_sse_clients", "gauge", "Connected SSE clients.", sse)
    version = str(extra.get("version") or "unknown").replace('"', "")
    lines.append("# HELP conscio_info Build information.")
    lines.append("# TYPE conscio_info gauge")
    lines.append(f'conscio_info{{version="{version}"}} 1')
    return "\n".join(lines) + "\n"

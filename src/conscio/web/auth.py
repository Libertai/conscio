"""Cookie-based session auth + sweep helpers for the operator dashboard.

Lifted out of the old ``conscio.webui`` so the SPA host module can stay
focused on HTTP routing.
"""

from __future__ import annotations

import hmac
import secrets
import time
from typing import Iterable

from fastapi import HTTPException

from conscio.service import ConscioService


SESSION_COOKIE = "conscio_web_session"
MAX_SESSIONS = 10000
MAX_LOGIN_FAILURE_TRACKERS = 10000
LOGIN_FAILURE_WINDOW_SECONDS = 300.0
LOGIN_FAILURE_LIMIT = 8
SESSION_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days


def session_token() -> str:
    return secrets.token_urlsafe(32)


def sweep_sessions(
    sessions: dict[str, float], now: float, max_size: int = MAX_SESSIONS
) -> None:
    """Drop expired entries; if still oversized, drop earliest-expiring keys to fit cap."""
    expired = [token for token, expires in sessions.items() if expires < now]
    for token in expired:
        sessions.pop(token, None)
    if len(sessions) > max_size:
        ordered = sorted(sessions.items(), key=lambda item: item[1])
        for token, _ in ordered[: len(sessions) - max_size]:
            sessions.pop(token, None)


def sweep_login_failures(
    failures: dict[str, list[float]],
    now: float,
    window: float = LOGIN_FAILURE_WINDOW_SECONDS,
    max_size: int = MAX_LOGIN_FAILURE_TRACKERS,
) -> None:
    """Drop tracker buckets that no longer hold any in-window failures; cap total size."""
    cutoff = now - window
    empty: list[str] = []
    for client, times in failures.items():
        in_window = [t for t in times if t >= cutoff]
        if in_window:
            failures[client] = in_window
        else:
            empty.append(client)
    for client in empty:
        failures.pop(client, None)
    if len(failures) > max_size:
        ordered = sorted(
            failures.items(),
            key=lambda item: max(item[1]) if item[1] else 0.0,
        )
        for client, _ in ordered[: len(failures) - max_size]:
            failures.pop(client, None)


def require_web_auth(
    service: ConscioService,
    sessions: dict[str, float],
    cookie: str | None,
) -> None:
    if not service.config.web_password:
        raise HTTPException(status_code=500, detail="web_password is not configured")
    now = time.time()
    sweep_sessions(sessions, now)
    expires_at = sessions.get(cookie or "")
    if not expires_at or expires_at < now:
        if cookie:
            sessions.pop(cookie, None)
        raise HTTPException(status_code=401, detail="not authenticated")


def check_password(service: ConscioService, supplied: str) -> bool:
    expected = service.config.web_password
    if not expected:
        return False
    # Compare as bytes: compare_digest raises TypeError on non-ASCII str,
    # which would turn a login attempt into a 500 and bypass failure tracking.
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def record_login_failure(failures: dict[str, list[float]], client: str, now: float) -> int:
    """Record a login failure and return the count of failures still in the window."""
    in_window = [t for t in failures.get(client, []) if t >= now - LOGIN_FAILURE_WINDOW_SECONDS]
    in_window.append(now)
    failures[client] = in_window
    return len(in_window)


def login_failure_count(
    failures: dict[str, list[float]], client: str, now: float
) -> int:
    return len([t for t in failures.get(client, []) if t >= now - LOGIN_FAILURE_WINDOW_SECONDS])


__all__: Iterable[str] = (
    "LOGIN_FAILURE_LIMIT",
    "MAX_SESSIONS",
    "MAX_LOGIN_FAILURE_TRACKERS",
    "SESSION_COOKIE",
    "SESSION_TTL_SECONDS",
    "check_password",
    "login_failure_count",
    "record_login_failure",
    "require_web_auth",
    "session_token",
    "sweep_login_failures",
    "sweep_sessions",
)

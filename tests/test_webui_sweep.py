from __future__ import annotations

import time
import unittest

from conscio.webui import (
    MAX_LOGIN_FAILURE_TRACKERS,
    MAX_SESSIONS,
    _sweep_login_failures,
    _sweep_sessions,
)


class SweepTests(unittest.TestCase):
    def test_sweep_sessions_drops_expired(self) -> None:
        now = 1_000_000.0
        sessions = {
            "fresh": now + 60,
            "stale": now - 10,
            "ancient": now - 999_999,
        }
        _sweep_sessions(sessions, now)
        self.assertEqual(set(sessions.keys()), {"fresh"})

    def test_sweep_sessions_caps_size_dropping_earliest_expirers(self) -> None:
        now = 1_000_000.0
        sessions = {f"k{i}": now + (i * 100) for i in range(5)}
        _sweep_sessions(sessions, now, max_size=3)
        # Caps to 3, dropping the three with the smallest expiry (k0, k1).
        # Wait — len is 5, max=3, so we drop len-max=2 earliest.
        self.assertEqual(len(sessions), 3)
        self.assertNotIn("k0", sessions)
        self.assertNotIn("k1", sessions)
        self.assertIn("k4", sessions)

    def test_sweep_login_failures_drops_empty_buckets(self) -> None:
        now = time.time()
        failures = {
            "1.2.3.4": [now - 10],
            "5.6.7.8": [now - 999_999],
            "9.10.11.12": [],
        }
        _sweep_login_failures(failures, now, window=300.0)
        self.assertIn("1.2.3.4", failures)
        self.assertNotIn("5.6.7.8", failures)
        self.assertNotIn("9.10.11.12", failures)

    def test_sweep_constants_are_sane(self) -> None:
        self.assertGreater(MAX_SESSIONS, 100)
        self.assertGreater(MAX_LOGIN_FAILURE_TRACKERS, 100)


if __name__ == "__main__":
    unittest.main()

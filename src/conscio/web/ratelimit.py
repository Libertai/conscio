"""In-process token bucket for episode-triggering HTTP endpoints.

Single-operator service, single event loop: no locking, no per-IP state. The
threat is a runaway client loop or a leaked key burning LLM spend, so one
global bucket is the right shape. State resets on restart by design.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        capacity: float,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = max(0.0, capacity)
        self.refill_per_second = max(0.0, refill_per_second)
        self._clock = clock
        self._tokens = self.capacity
        self._last = clock()

    def try_acquire(self) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.refill_per_second)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True, 0.0
        if self.refill_per_second <= 0:
            return False, 60.0
        return False, (1.0 - self._tokens) / self.refill_per_second

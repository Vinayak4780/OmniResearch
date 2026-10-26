"""Circuit breakers and retry backoff for provider calls."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_timeout_s: float = 30.0
    state: str = "closed"  # closed | open | half_open
    consecutive_failures: int = 0
    opened_at: float = 0.0
    last_error: str | None = None
    _half_open_in_flight: bool = field(default=False, repr=False)

    def allow(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.monotonic() - self.opened_at >= self.reset_timeout_s:
                self.state = "half_open"
                self._half_open_in_flight = False
            else:
                return False
        if self._half_open_in_flight:
            return False
        self._half_open_in_flight = True
        return True

    def record_success(self) -> None:
        self.state = "closed"
        self.consecutive_failures = 0
        self._half_open_in_flight = False

    def record_failure(self, error: str) -> None:
        self.last_error = error
        self.consecutive_failures += 1
        self._half_open_in_flight = False
        if self.state == "half_open" or self.consecutive_failures >= self.failure_threshold:
            self.state = "open"
            self.opened_at = time.monotonic()

    def snapshot(self) -> dict:
        remaining = None
        if self.state == "open":
            remaining = max(0.0, self.reset_timeout_s - (time.monotonic() - self.opened_at))
        return {"state": self.state, "consecutive_failures": self.consecutive_failures, "last_error": self.last_error, "reopens_in_s": remaining}


def backoff_delay(attempt: int, *, base: float, cap: float = 20.0, retry_after: float | None = None) -> float:
    """Exponential backoff with full jitter; honours Retry-After (capped)."""
    if retry_after is not None:
        return min(retry_after, cap)
    if base <= 0:
        return 0.0
    return random.uniform(0, min(cap, base * (2**attempt)))

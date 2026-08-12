"""Per-provider circuit breaker over a sliding failure-rate window.

Without it, a dead provider is rediscovered by every single request: each one
pays the full timeout and the full retry ladder before moving on. The breaker
turns that per-request cost into a per-cooldown cost.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from llm_gateway.settings import CircuitBreakerConfig


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class _Event:
    at: float
    ok: bool


class CircuitBreaker:
    """One breaker per provider. Not thread-safe; it lives on a single event loop."""

    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self.name = name
        self.config = config
        self._events: deque[_Event] = deque()
        self._state = BreakerState.CLOSED
        self._opened_at = 0.0
        self._half_open_inflight = 0
        self._half_open_successes = 0
        self.opened_count = 0
        self.short_circuited = 0

    # -- internals ---------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - self.config.window_s
        while self._events and self._events[0].at < cutoff:
            self._events.popleft()

    def _failure_ratio(self) -> tuple[int, float]:
        total = len(self._events)
        if total == 0:
            return 0, 0.0
        failures = sum(1 for event in self._events if not event.ok)
        return total, failures / total

    def _open(self, now: float) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = now
        self._half_open_inflight = 0
        self._half_open_successes = 0
        self._events.clear()
        self.opened_count += 1

    def _close(self) -> None:
        self._state = BreakerState.CLOSED
        self._half_open_inflight = 0
        self._half_open_successes = 0
        self._events.clear()

    # -- public API --------------------------------------------------------

    @property
    def state(self) -> BreakerState:
        return self._state

    def allow(self, now: float | None = None) -> bool:
        """Reserve a slot for one call. Must be paired with exactly one `record`."""
        if not self.config.enabled:
            return True
        now = now if now is not None else time.monotonic()

        if self._state is BreakerState.OPEN:
            if now - self._opened_at < self.config.cooldown_s:
                self.short_circuited += 1
                return False
            # Cooldown elapsed: let a limited number of probes through.
            self._state = BreakerState.HALF_OPEN
            self._half_open_inflight = 0
            self._half_open_successes = 0

        if self._state is BreakerState.HALF_OPEN:
            if self._half_open_inflight >= self.config.half_open_max_calls:
                self.short_circuited += 1
                return False
            self._half_open_inflight += 1
            return True

        return True

    def record(self, ok: bool, now: float | None = None) -> None:
        if not self.config.enabled:
            return
        now = now if now is not None else time.monotonic()

        if self._state is BreakerState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
            if ok:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.half_open_max_calls:
                    self._close()
            else:
                self._open(now)
            return

        self._events.append(_Event(at=now, ok=ok))
        self._prune(now)
        if self._state is BreakerState.CLOSED:
            total, ratio = self._failure_ratio()
            if total >= self.config.min_calls and ratio >= self.config.failure_ratio:
                self._open(now)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.monotonic()
        total, ratio = self._failure_ratio()
        cooldown_left = 0.0
        if self._state is BreakerState.OPEN:
            cooldown_left = max(0.0, self.config.cooldown_s - (now - self._opened_at))
        return {
            "provider": self.name,
            "enabled": self.config.enabled,
            "state": str(self._state),
            "window_calls": total,
            "failure_ratio": round(ratio, 4),
            "cooldown_remaining_s": round(cooldown_left, 3),
            "opened_count": self.opened_count,
            "short_circuited": self.short_circuited,
        }


class BreakerRegistry:
    def __init__(self, config: CircuitBreakerConfig, providers: list[str]) -> None:
        self.config = config
        self._breakers = {name: CircuitBreaker(name, config) for name in providers}

    def get(self, provider: str) -> CircuitBreaker:
        breaker = self._breakers.get(provider)
        if breaker is None:
            breaker = CircuitBreaker(provider, self.config)
            self._breakers[provider] = breaker
        return breaker

    def snapshot(self) -> list[dict[str, Any]]:
        return [breaker.snapshot() for breaker in self._breakers.values()]

    def reset(self) -> None:
        for breaker in self._breakers.values():
            breaker._close()  # noqa: SLF001 — test/bench helper on our own type
            breaker.opened_count = 0
            breaker.short_circuited = 0

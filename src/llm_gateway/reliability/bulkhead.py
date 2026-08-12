"""Per-provider concurrency limit.

A circuit breaker reacts to a provider that is *failing*. A bulkhead reacts to a
provider that is *full*: it caps how many calls the gateway keeps in flight to one
provider, so a concurrency quota is respected on our side instead of being enforced
by the provider with a storm of 503s.

Waiting is bounded — the caller passes the time the request has left, so a slot is
never waited for past the deadline. A request that cannot get one in time is shed,
and shedding is a hop failure like any other: the chain moves to the next provider.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from llm_gateway.settings import BulkheadConfig


@dataclass
class BulkheadStats:
    admitted: int = 0
    queued: int = 0
    shed: int = 0
    wait_ms_total: int = 0
    peak_in_flight: int = 0


class Bulkhead:
    """One provider's slots. A limit of zero or less means unlimited."""

    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = limit
        self.in_flight = 0
        self.stats = BulkheadStats()
        # Semaphores bind to the running loop lazily, so building one here — before
        # the app has a loop — is safe.
        self._slots = asyncio.Semaphore(limit) if limit > 0 else None

    @property
    def unlimited(self) -> bool:
        return self._slots is None

    async def acquire(self, timeout_s: float | None) -> bool:
        """Take a slot. Returns False when no slot came free in time.

        Cancellation (a hedge that lost its race) propagates instead of counting as
        shed: nobody was refused, the request simply stopped wanting a slot.
        """
        slots = self._slots
        if slots is None:
            self._enter()
            return True
        if not slots.locked():
            # A slot is free, so this cannot block — take it even when the caller
            # allowed no waiting time at all.
            await slots.acquire()
            self._enter()
            return True
        if timeout_s is not None and timeout_s <= 0:
            self.stats.shed += 1
            return False

        started = time.monotonic()
        try:
            await asyncio.wait_for(slots.acquire(), timeout=timeout_s)
        except TimeoutError:
            self.stats.shed += 1
            return False
        finally:
            # Everything past this point queued by definition: the fast path above
            # already handled the case where a slot was free.
            self.stats.queued += 1
            self.stats.wait_ms_total += int((time.monotonic() - started) * 1000)
        self._enter()
        return True

    def release(self) -> None:
        """Give the slot back. Only valid after `acquire` returned True."""
        if self.in_flight > 0:
            self.in_flight -= 1
        if self._slots is not None:
            self._slots.release()

    def _enter(self) -> None:
        self.in_flight += 1
        self.stats.admitted += 1
        self.stats.peak_in_flight = max(self.stats.peak_in_flight, self.in_flight)

    def snapshot(self) -> dict[str, object]:
        average_wait = self.stats.wait_ms_total / self.stats.queued if self.stats.queued else 0.0
        return {
            "provider": self.name,
            "limit": self.limit,
            "in_flight": self.in_flight,
            "admitted": self.stats.admitted,
            "queued": self.stats.queued,
            "shed": self.stats.shed,
            "peak_in_flight": self.stats.peak_in_flight,
            "avg_queue_wait_ms": round(average_wait, 1),
        }


@dataclass
class BulkheadRegistry:
    config: BulkheadConfig
    providers: list[str] = field(default_factory=list)
    #: Per-provider overrides, for providers whose documented quota differs from the
    #: default (`providers.<name>.max_concurrent` in the config).
    overrides: dict[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._bulkheads = {name: self._build(name) for name in self.providers}

    def _build(self, name: str) -> Bulkhead:
        if not self.config.enabled:
            return Bulkhead(name=name, limit=0)
        limit = self.overrides.get(name) or self.config.max_concurrent_per_provider
        return Bulkhead(name=name, limit=limit)

    def get(self, provider: str) -> Bulkhead:
        bulkhead = self._bulkheads.get(provider)
        if bulkhead is None:
            bulkhead = self._build(provider)
            self._bulkheads[provider] = bulkhead
        return bulkhead

    def snapshot(self) -> list[dict[str, object]]:
        return [bulkhead.snapshot() for bulkhead in self._bulkheads.values()]

    def reset(self) -> None:
        """Fresh slots and counters between chaos runs."""
        for name in list(self._bulkheads):
            self._bulkheads[name] = self._build(name)

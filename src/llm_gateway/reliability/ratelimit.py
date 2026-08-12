"""Token-bucket rate limit at the door.

The budget refuses a request that would cost too much *money*; this refuses one that
arrives too fast. Both are admission control: the gateway says no before doing work
it should not do, instead of accepting everything and degrading for everyone.

A bucket per API key (or one shared bucket when auth is off) keeps a single noisy
tenant from spending the whole gateway's capacity. `Retry-After` tells the caller
exactly how long the next token takes, so a well-behaved client can wait instead of
hammering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from llm_gateway.settings import RateLimitConfig

GLOBAL_SCOPE = "__global__"
_EPSILON = 1e-9


@dataclass
class Bucket:
    """Classic token bucket: `rate` tokens per second, at most `burst` in hand."""

    rate_per_s: float
    burst: float
    tokens: float
    updated_at: float
    allowed: int = 0
    throttled: int = 0

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate_per_s)
        self.updated_at = now

    def take(self, now: float | None = None) -> float | None:
        """Take one token. Returns None when allowed, or the seconds to wait."""
        self._refill(now if now is not None else time.monotonic())
        # Refill arithmetic is floating point: 0.3 s at 10 rps is 2.9999999999999996
        # tokens, and without the epsilon an exactly-timed retry would be refused
        # for a rounding error.
        if self.tokens >= 1.0 - _EPSILON:
            self.tokens -= 1.0
            self.allowed += 1
            return None
        self.throttled += 1
        missing = 1.0 - self.tokens
        return missing / self.rate_per_s if self.rate_per_s > 0 else float("inf")


@dataclass
class RateLimiter:
    config: RateLimitConfig
    _buckets: dict[str, Bucket] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _bucket(self, scope: str) -> Bucket:
        bucket = self._buckets.get(scope)
        if bucket is None:
            bucket = Bucket(
                rate_per_s=self.config.requests_per_second,
                burst=float(self.config.burst),
                tokens=float(self.config.burst),
                updated_at=time.monotonic(),
            )
            self._buckets[scope] = bucket
        return bucket

    def check(self, api_key_id: str | None, now: float | None = None) -> float | None:
        """None when the request may proceed, otherwise the retry-after in seconds."""
        if not self.config.enabled:
            return None
        scope = api_key_id or GLOBAL_SCOPE
        return self._bucket(scope).take(now)

    def snapshot(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "requests_per_second": self.config.requests_per_second,
            "burst": self.config.burst,
            "scopes": {
                scope: {
                    "tokens": round(bucket.tokens, 2),
                    "allowed": bucket.allowed,
                    "throttled": bucket.throttled,
                }
                for scope, bucket in sorted(self._buckets.items())
            },
        }

    def reset(self) -> None:
        self._buckets.clear()

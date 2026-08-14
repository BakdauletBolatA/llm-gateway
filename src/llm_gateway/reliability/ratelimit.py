"""Token-bucket rate limit at the door.

The budget refuses a request that would cost too much *money*; this refuses one that
arrives too fast. Both are admission control: the gateway says no before doing work
it should not do, instead of accepting everything and degrading for everyone.

A bucket per API key (or one shared bucket when auth is off) keeps a single noisy
tenant from spending the whole gateway's capacity. `Retry-After` tells the caller
exactly how long the next token takes, so a well-behaved client can wait instead of
hammering.

Two backends, chosen by `rate_limit.scope`:

    local  — a dict in this process. Exact and free, but every replica hands out the
             full limit, so N replicas let through N times the configured rate.
    shared — one row per scope in Postgres, taken with a single atomic statement.
             The limit then belongs to the deployment rather than to the process.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy import text

from llm_gateway.db.session import Database
from llm_gateway.settings import RateLimitConfig

logger = logging.getLogger(__name__)

GLOBAL_SCOPE = "__global__"
_EPSILON = 1e-9

#: Refill and take in one statement, so two replicas cannot both see the same last
#: token. `ON CONFLICT DO UPDATE` takes a row lock, which serialises them; the WHERE
#: makes the update fail (no row returned) when there is nothing left to take.
_TAKE_TOKEN = text("""
INSERT INTO rate_limit_buckets AS b (scope, tokens, updated_at)
VALUES (:scope, :burst - 1, now())
ON CONFLICT (scope) DO UPDATE
   SET tokens = LEAST(:burst, b.tokens
                 + EXTRACT(EPOCH FROM (now() - b.updated_at)) * :rate) - 1,
       updated_at = now()
 WHERE LEAST(:burst, b.tokens
        + EXTRACT(EPOCH FROM (now() - b.updated_at)) * :rate) >= 1
RETURNING tokens
""")

#: Only for the refusal path: how long until the next token is worth waiting for.
_PEEK_TOKENS = text("""
SELECT LEAST(:burst, tokens + EXTRACT(EPOCH FROM (now() - updated_at)) * :rate)
  FROM rate_limit_buckets
 WHERE scope = :scope
""")


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
    #: Required by `scope: shared`; ignored by the local backend.
    database: Database | None = None
    _buckets: dict[str, Bucket] = field(default_factory=dict)
    _allowed: int = 0
    _throttled: int = 0
    _errors: int = 0

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def shared(self) -> bool:
        return self.config.scope == "shared"

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

    async def check(self, api_key_id: str | None, now: float | None = None) -> float | None:
        """None when the request may proceed, otherwise the retry-after in seconds."""
        if not self.config.enabled:
            return None
        scope = api_key_id or GLOBAL_SCOPE
        if not self.shared:
            return self._bucket(scope).take(now)
        return await self._take_shared(scope)

    async def _take_shared(self, scope: str) -> float | None:
        if self.database is None:  # pragma: no cover - wiring error, not a runtime one
            raise RuntimeError("rate_limit.scope=shared requires a database")
        params = {
            "scope": scope,
            "burst": float(self.config.burst),
            "rate": self.config.requests_per_second,
        }
        try:
            async with self.database.session() as session:
                taken = (await session.execute(_TAKE_TOKEN, params)).scalar_one_or_none()
                if taken is not None:
                    await session.commit()
                    self._allowed += 1
                    return None
                available = (await session.execute(_PEEK_TOKENS, params)).scalar_one_or_none()
                await session.commit()
        except Exception:
            # The limiter must not take the gateway down with it: a database that is
            # unavailable means requests are let through, not refused. The budget and
            # the provider-side limits are still in the way.
            self._errors += 1
            logger.exception("shared rate limiter failed, letting the request through")
            return None

        self._throttled += 1
        missing = 1.0 - float(available or 0.0)
        rate = self.config.requests_per_second
        return missing / rate if rate > 0 else float("inf")

    def snapshot(self) -> dict[str, object]:
        state: dict[str, object] = {
            "enabled": self.config.enabled,
            "scope": self.config.scope,
            "requests_per_second": self.config.requests_per_second,
            "burst": self.config.burst,
        }
        if self.shared:
            state["shared"] = {
                "allowed": self._allowed,
                "throttled": self._throttled,
                "errors": self._errors,
            }
        else:
            state["scopes"] = {
                scope: {
                    "tokens": round(bucket.tokens, 2),
                    "allowed": bucket.allowed,
                    "throttled": bucket.throttled,
                }
                for scope, bucket in sorted(self._buckets.items())
            }
        return state

    def reset(self) -> None:
        """Local state only: the shared bucket belongs to the deployment, and one
        replica clearing it between chaos runs would surprise the others."""
        self._buckets.clear()
        self._allowed = self._throttled = self._errors = 0

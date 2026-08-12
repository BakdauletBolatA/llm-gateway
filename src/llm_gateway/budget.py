"""Budget enforcement.

The requirement is a refusal, not a silent bill: once the period limit is reached
the gateway answers 402 and never calls a provider.

Accounting is in-memory with Postgres as the source of truth. The local counter is
incremented the moment a call completes (so a burst cannot outrun the limit between
database flushes) and is periodically reconciled against the database (so a restart
does not reset the period). Both counters only grow within a period, so taking the
maximum can never under-count. With several gateway replicas each one enforces its
own view and the effective limit is per-replica — a shared counter (Redis, or a
database-side atomic increment) would be the next step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from llm_gateway.db.models import LlmCall
from llm_gateway.db.session import Database
from llm_gateway.errors import BudgetExceededError
from llm_gateway.settings import AuthConfig, BudgetConfig

logger = logging.getLogger(__name__)


def period_start(period: str, now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    if period == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass
class _Counter:
    local: float = 0.0
    from_db: float = 0.0

    @property
    def spent(self) -> float:
        return max(self.local, self.from_db)


@dataclass
class BudgetSnapshot:
    period: str
    period_start: str
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    per_key: dict[str, dict[str, Any]] = field(default_factory=dict)


class BudgetTracker:
    def __init__(self, config: BudgetConfig, auth: AuthConfig, database: Database) -> None:
        self._config = config
        self._auth = auth
        self._db = database
        self._period_start = period_start(config.period)
        self._global = _Counter()
        self._by_key: dict[str, _Counter] = {}
        self._last_refresh = 0.0

    # -- period handling ---------------------------------------------------

    def _roll_period_if_needed(self) -> None:
        current = period_start(self._config.period)
        if current != self._period_start:
            logger.info("budget period rolled over to %s", current.isoformat())
            self._period_start = current
            self._global = _Counter()
            self._by_key.clear()
            self._last_refresh = 0.0

    def _counter_for(self, api_key_id: str | None) -> _Counter | None:
        if api_key_id is None:
            return None
        return self._by_key.setdefault(api_key_id, _Counter())

    def _limit_for(self, api_key_id: str | None) -> float | None:
        if api_key_id is None:
            return None
        key = next((k for k in self._auth.keys if k.id == api_key_id), None)
        return key.budget_limit_usd if key else None

    # -- public API --------------------------------------------------------

    async def refresh(self, force: bool = False) -> None:
        """Reconcile the in-memory counters with what is actually recorded."""
        self._roll_period_if_needed()
        now = time.monotonic()
        if not force and now - self._last_refresh < self._config.refresh_interval_s:
            return
        self._last_refresh = now
        try:
            async with self._db.session() as session:
                total = await session.scalar(
                    select(func.coalesce(func.sum(LlmCall.cost_usd), 0)).where(
                        LlmCall.created_at >= self._period_start
                    )
                )
                self._global.from_db = float(total or 0.0)

                rows = await session.execute(
                    select(LlmCall.api_key_id, func.coalesce(func.sum(LlmCall.cost_usd), 0))
                    .where(
                        LlmCall.created_at >= self._period_start,
                        LlmCall.api_key_id.is_not(None),
                    )
                    .group_by(LlmCall.api_key_id)
                )
                for key_id, spend in rows.all():
                    self._by_key.setdefault(key_id, _Counter()).from_db = float(spend or 0.0)
        except Exception:
            # A database blip must not turn into a wall of 402s — keep the local view.
            logger.exception("budget refresh failed; continuing with in-memory counters")

    def check(self, estimated_cost_usd: float, api_key_id: str | None = None) -> None:
        """Raise BudgetExceededError if this request would cross a limit."""
        if not self._config.enabled:
            return
        self._roll_period_if_needed()

        projected = self._global.spent + estimated_cost_usd
        if projected > self._config.limit_usd:
            raise BudgetExceededError(
                (
                    f"budget for this {self._config.period} is exhausted: "
                    f"spent ${self._global.spent:.4f} of ${self._config.limit_usd:.2f}, "
                    f"this request is estimated at ${estimated_cost_usd:.4f}"
                ),
                spent_usd=round(self._global.spent, 6),
                limit_usd=self._config.limit_usd,
                period=self._config.period,
            )

        key_limit = self._limit_for(api_key_id)
        counter = self._counter_for(api_key_id)
        if (
            key_limit is not None
            and counter is not None
            and counter.spent + estimated_cost_usd > key_limit
        ):
            raise BudgetExceededError(
                (
                    f"budget for API key {api_key_id!r} is exhausted: "
                    f"spent ${counter.spent:.4f} of ${key_limit:.2f}"
                ),
                spent_usd=round(counter.spent, 6),
                limit_usd=key_limit,
                period=self._config.period,
            )

    def record_spend(self, cost_usd: float, api_key_id: str | None = None) -> None:
        if cost_usd <= 0:
            return
        self._roll_period_if_needed()
        self._global.local += cost_usd
        counter = self._counter_for(api_key_id)
        if counter is not None:
            counter.local += cost_usd

    def snapshot(self) -> BudgetSnapshot:
        self._roll_period_if_needed()
        spent = self._global.spent
        return BudgetSnapshot(
            period=self._config.period,
            period_start=self._period_start.isoformat(),
            limit_usd=self._config.limit_usd,
            spent_usd=round(spent, 6),
            remaining_usd=round(max(0.0, self._config.limit_usd - spent), 6),
            per_key={
                key_id: {
                    "spent_usd": round(counter.spent, 6),
                    "limit_usd": self._limit_for(key_id),
                }
                for key_id, counter in sorted(self._by_key.items())
            },
        )

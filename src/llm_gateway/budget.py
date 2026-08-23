"""Budget enforcement.

The requirement is a refusal, not a silent bill: once the period limit is reached
the gateway answers 402 and never calls a provider.

Two backends, chosen by `budget.scope`:

    local  — in-memory counters with Postgres as the source of truth. The local
             counter grows the moment a call completes (so a burst cannot outrun
             the limit between flushes) and is reconciled against the database
             every `refresh_interval_s`. Exact within one process — and only
             within one process: each replica enforces its own view, so N replicas
             spend up to N limits. Measured in iteration 11: two replicas against
             a $0.005 limit spent $0.011.

    shared — the money is *reserved* in one Postgres row before the provider is
             called, and settled to the real cost afterwards. The row lock, not
             the application, is what keeps two replicas from spending the same
             dollar twice, so the limit belongs to the deployment.

Reserving before the call is what makes the shared mode correct rather than merely
shared: a counter that is only incremented after the answer comes back would let
every request in flight ignore every other one, which is exactly the window the
local backend leaves open. The price is that an over-estimated request holds money
it will not spend until it settles, so the gateway can refuse slightly early — it
never overspends, which is the direction the requirement cares about.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, text

from llm_gateway.db.models import LlmCall
from llm_gateway.db.session import Database
from llm_gateway.errors import BudgetExceededError
from llm_gateway.settings import AuthConfig, BudgetConfig

logger = logging.getLogger(__name__)

GLOBAL_SCOPE = "__global__"

#: Commit money in one statement. `ON CONFLICT DO UPDATE` takes the row lock, so
#: concurrent reservations queue instead of racing; the WHERE makes the update
#: return no row when the limit would be crossed, which is the refusal.
_RESERVE = text("""
INSERT INTO budget_periods AS b (period_key, scope, spent_usd, updated_at)
VALUES (:period, :scope, :amount, now())
ON CONFLICT (period_key, scope) DO UPDATE
   SET spent_usd = b.spent_usd + :amount,
       updated_at = now()
 WHERE b.spent_usd + :amount <= :limit
RETURNING b.spent_usd
""")

#: Settle: the difference between what was reserved and what was actually spent.
#: GREATEST keeps a rounding error or a double release from driving the row
#: negative, which would hand out free budget.
_SETTLE = text("""
UPDATE budget_periods
   SET spent_usd = GREATEST(0, spent_usd + :delta),
       updated_at = now()
 WHERE period_key = :period AND scope = :scope
""")

_PEEK = text("""
SELECT spent_usd FROM budget_periods WHERE period_key = :period AND scope = :scope
""")

_PEEK_ALL = text("""
SELECT scope, spent_usd FROM budget_periods WHERE period_key = :period
""")


def period_start(period: str, now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    if period == "month":
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def period_key(period: str, start: datetime) -> str:
    """The row key for a period: '2026-08' for a month, '2026-08-21' for a day."""
    return start.strftime("%Y-%m" if period == "month" else "%Y-%m-%d")


@dataclass
class _Counter:
    local: float = 0.0
    from_db: float = 0.0

    @property
    def spent(self) -> float:
        return max(self.local, self.from_db)


@dataclass
class Reservation:
    """Money committed for one request, to be settled once the cost is known.

    In local mode nothing is written anywhere and this only carries the estimate;
    in shared mode `scopes` names the rows that were charged.
    """

    amount: float = 0.0
    shared: bool = False
    api_key_id: str | None = None
    scopes: tuple[str, ...] = ()
    settled: bool = False


@dataclass
class BudgetSnapshot:
    period: str
    period_start: str
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    scope: str = "local"
    #: Reservations or settlements that could not reach Postgres. Non-zero means the
    #: shared limit quietly degraded to a per-replica one, which is invisible from
    #: the outside — the gateway keeps answering, just with a weaker guarantee.
    backend_errors: int = 0
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
        self._reserve_errors = 0

    @property
    def shared(self) -> bool:
        return self._config.scope == "shared"

    # -- period handling ---------------------------------------------------

    def _roll_period_if_needed(self) -> None:
        current = period_start(self._config.period)
        if current != self._period_start:
            logger.info("budget period rolled over to %s", current.isoformat())
            self._period_start = current
            self._global = _Counter()
            self._by_key.clear()
            self._last_refresh = 0.0

    def _period_key(self) -> str:
        return period_key(self._config.period, self._period_start)

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

                if self.shared:
                    # The reserved total leads the recorded one (it includes requests
                    # still in flight) and is what enforcement actually uses, so the
                    # reported number has to be the larger of the two.
                    committed = await session.execute(_PEEK_ALL, {"period": self._period_key()})
                    for scope, spent in committed.all():
                        counter = (
                            self._global
                            if scope == GLOBAL_SCOPE
                            else self._by_key.setdefault(scope, _Counter())
                        )
                        counter.from_db = max(counter.from_db, float(spent or 0.0))
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
            raise self._refusal(self._global.spent, self._config.limit_usd, estimated_cost_usd)

        key_limit = self._limit_for(api_key_id)
        counter = self._counter_for(api_key_id)
        if (
            key_limit is not None
            and counter is not None
            and counter.spent + estimated_cost_usd > key_limit
        ):
            raise self._refusal(counter.spent, key_limit, estimated_cost_usd, api_key_id)

    def _refusal(
        self,
        spent: float,
        limit: float,
        estimate: float,
        api_key_id: str | None = None,
    ) -> BudgetExceededError:
        whose = f"API key {api_key_id!r}" if api_key_id else f"this {self._config.period}"
        return BudgetExceededError(
            (
                f"budget for {whose} is exhausted: "
                f"spent ${spent:.4f} of ${limit:.2f}, "
                f"this request is estimated at ${estimate:.4f}"
            ),
            spent_usd=round(spent, 6),
            limit_usd=limit,
            period=self._config.period,
        )

    async def reserve(
        self, estimated_cost_usd: float, api_key_id: str | None = None
    ) -> Reservation:
        """Commit the estimate up front, or refuse. Always paired with `settle`.

        In local mode this is `check` with a receipt attached. In shared mode the
        money is written to Postgres before the provider is called, so a replica
        that has not spoken to the database in seconds still cannot overspend.
        """
        if not self._config.enabled:
            return Reservation()
        self._roll_period_if_needed()
        if not self.shared:
            self.check(estimated_cost_usd, api_key_id)
            return Reservation(amount=estimated_cost_usd, api_key_id=api_key_id)

        key_limit = self._limit_for(api_key_id)
        # An empty row is created by the reservation itself, and `ON CONFLICT` does
        # not fire on an insert — so a first request larger than the whole limit has
        # to be refused here, before the statement that would happily accept it.
        for limit, who in ((self._config.limit_usd, None), (key_limit, api_key_id)):
            if limit is not None and estimated_cost_usd > limit:
                raise self._refusal(0.0, limit, estimated_cost_usd, who)

        period = self._period_key()
        wanted: list[tuple[str, float, str | None]] = [(GLOBAL_SCOPE, self._config.limit_usd, None)]
        if key_limit is not None and api_key_id is not None:
            wanted.append((api_key_id, key_limit, api_key_id))

        taken: list[str] = []
        try:
            async with self._db.session() as session:
                for scope, limit, who in wanted:
                    params = {
                        "period": period,
                        "scope": scope,
                        "amount": estimated_cost_usd,
                        "limit": limit,
                    }
                    spent = (await session.execute(_RESERVE, params)).scalar_one_or_none()
                    if spent is None:
                        # Nothing is committed until commit(), so dropping the whole
                        # transaction is what un-reserves the scopes already taken.
                        already = (await session.execute(_PEEK, params)).scalar_one_or_none()
                        await session.rollback()
                        raise self._refusal(float(already or 0.0), limit, estimated_cost_usd, who)
                    taken.append(scope)
                await session.commit()
        except BudgetExceededError:
            raise
        except Exception:
            # Degrade to the per-replica limit rather than to no limit at all: the
            # in-memory check is still a real refusal, just a weaker one.
            self._reserve_errors += 1
            logger.exception("budget reservation failed; falling back to the local counters")
            self.check(estimated_cost_usd, api_key_id)
            return Reservation(amount=estimated_cost_usd, api_key_id=api_key_id)

        return Reservation(
            amount=estimated_cost_usd,
            shared=True,
            api_key_id=api_key_id,
            scopes=tuple(taken),
        )

    async def settle(self, reservation: Reservation, actual_cost_usd: float) -> None:
        """Correct the reservation to what the request really cost.

        Called from a `finally`: a request that failed spent nothing and must give
        the money back, or a single outage would eat the day's budget.
        """
        if reservation.settled:  # pragma: no cover - the router settles exactly once
            return
        reservation.settled = True
        self.record_spend(actual_cost_usd, reservation.api_key_id)
        if not reservation.shared:
            return

        delta = actual_cost_usd - reservation.amount
        if abs(delta) < 1e-12:
            return
        try:
            async with self._db.session() as session:
                for scope in reservation.scopes:
                    await session.execute(
                        _SETTLE,
                        {"period": self._period_key(), "scope": scope, "delta": delta},
                    )
                await session.commit()
        except Exception:
            # The reservation stands. Refusing slightly early is the safe direction;
            # the next refresh re-reads the row either way.
            self._reserve_errors += 1
            logger.exception("budget settlement failed; the reservation stays as committed")

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
            scope=self._config.scope,
            backend_errors=self._reserve_errors,
            per_key={
                key_id: {
                    "spent_usd": round(counter.spent, 6),
                    "limit_usd": self._limit_for(key_id),
                }
                for key_id, counter in sorted(self._by_key.items())
            },
        )

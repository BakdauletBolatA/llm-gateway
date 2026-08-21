from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import text

from llm_gateway.budget import BudgetTracker, period_start
from llm_gateway.db import migrate
from llm_gateway.db.session import Database
from llm_gateway.errors import BudgetExceededError
from llm_gateway.settings import AuthConfig, BudgetConfig, DatabaseConfig


class _NoDatabase:
    """The tracker only touches the database on refresh(); these tests never do."""


def tracker(limit: float = 1.0, keys: list[dict[str, object]] | None = None) -> BudgetTracker:
    auth = AuthConfig.model_validate({"keys": keys or []})
    config = BudgetConfig(enabled=True, period="day", limit_usd=limit)
    return BudgetTracker(config, auth, _NoDatabase())  # type: ignore[arg-type]


def test_day_period_starts_at_utc_midnight() -> None:
    now = datetime(2026, 8, 12, 15, 30, tzinfo=UTC)
    assert period_start("day", now) == datetime(2026, 8, 12, 0, 0, tzinfo=UTC)


def test_month_period_starts_on_the_first() -> None:
    now = datetime(2026, 8, 12, 15, 30, tzinfo=UTC)
    assert period_start("month", now) == datetime(2026, 8, 1, 0, 0, tzinfo=UTC)


def test_spending_under_the_limit_is_allowed() -> None:
    budget = tracker(limit=1.0)
    budget.record_spend(0.4)
    budget.check(0.5)


def test_crossing_the_limit_is_refused_before_the_provider_is_called() -> None:
    budget = tracker(limit=1.0)
    budget.record_spend(0.8)
    with pytest.raises(BudgetExceededError) as excinfo:
        budget.check(0.3)
    assert excinfo.value.limit_usd == 1.0
    assert excinfo.value.spent_usd == pytest.approx(0.8)
    assert excinfo.value.http_status == 402


def test_a_disabled_budget_never_refuses() -> None:
    budget = tracker(limit=0.0)
    budget._config.enabled = False  # noqa: SLF001
    budget.record_spend(100.0)
    budget.check(100.0)


def test_per_key_limit_applies_on_top_of_the_global_one() -> None:
    budget = tracker(
        limit=100.0,
        keys=[{"id": "team-a", "key": "secret", "budget_limit_usd": 0.5}],
    )
    budget.record_spend(0.4, "team-a")
    budget.check(0.05, "team-a")
    with pytest.raises(BudgetExceededError) as excinfo:
        budget.check(0.2, "team-a")
    assert "team-a" in str(excinfo.value)


def test_one_key_running_out_does_not_block_another() -> None:
    budget = tracker(
        limit=100.0,
        keys=[
            {"id": "team-a", "key": "a", "budget_limit_usd": 0.5},
            {"id": "team-b", "key": "b", "budget_limit_usd": 0.5},
        ],
    )
    budget.record_spend(0.6, "team-a")
    with pytest.raises(BudgetExceededError):
        budget.check(0.01, "team-a")
    budget.check(0.01, "team-b")


def test_snapshot_reports_what_is_left() -> None:
    budget = tracker(limit=2.0)
    budget.record_spend(0.5)
    snapshot = budget.snapshot()
    assert snapshot.spent_usd == pytest.approx(0.5)
    assert snapshot.remaining_usd == pytest.approx(1.5)


def test_database_view_wins_when_it_is_ahead_of_the_local_counter() -> None:
    # After a restart the local counter is zero but the period is already spent.
    budget = tracker(limit=1.0)
    budget._global.from_db = 0.95  # noqa: SLF001
    with pytest.raises(BudgetExceededError):
        budget.check(0.1)


# -- the shared counter: one limit for the whole deployment --------------------


@pytest_asyncio.fixture
async def shared_database() -> AsyncIterator[Database]:
    """A real database: the point of the shared counter is that Postgres, not the
    process, is what keeps two replicas from spending the same dollar."""
    from tests.conftest import TEST_DSN, _database_available

    if not await _database_available(TEST_DSN):
        pytest.skip(f"PostgreSQL not reachable at {TEST_DSN}")
    await migrate.upgrade(TEST_DSN)
    database = Database(DatabaseConfig(dsn=TEST_DSN, run_migrations_on_startup=False))
    async with database.session() as session:
        await session.execute(text("DELETE FROM budget_periods"))
        await session.commit()
    try:
        yield database
    finally:
        await database.aclose()


def shared_tracker(
    database: Database,
    limit: float = 1.0,
    keys: list[dict[str, object]] | None = None,
) -> BudgetTracker:
    auth = AuthConfig.model_validate({"keys": keys or []})
    config = BudgetConfig(enabled=True, period="day", limit_usd=limit, scope="shared")
    return BudgetTracker(config, auth, database)


async def test_two_replicas_spend_one_shared_limit(shared_database: Database) -> None:
    """The failure this fixes: with per-replica counters two gateways serve twice
    the budget. Measured before the fix — 2 replicas against $0.005 spent $0.011."""
    replica_a = shared_tracker(shared_database, limit=1.0)
    replica_b = shared_tracker(shared_database, limit=1.0)

    first = await replica_a.reserve(0.6)
    await replica_a.settle(first, 0.6)

    with pytest.raises(BudgetExceededError) as refusal:
        await replica_b.reserve(0.6)
    assert refusal.value.spent_usd == pytest.approx(0.6), (
        "the second replica must see what the first one spent, not its own zero"
    )


async def test_the_reservation_holds_while_the_request_is_in_flight(
    shared_database: Database,
) -> None:
    """Counting only after the answer arrives is the bug, not the fix: every request
    in flight would then be invisible to every other one."""
    replica_a = shared_tracker(shared_database, limit=1.0)
    replica_b = shared_tracker(shared_database, limit=1.0)

    in_flight = await replica_a.reserve(0.7)  # not settled yet — the call is running
    with pytest.raises(BudgetExceededError):
        await replica_b.reserve(0.7)

    await replica_a.settle(in_flight, 0.1)  # it turned out cheap
    cheap = await replica_b.reserve(0.7)
    assert cheap.shared, "settling below the estimate must give the difference back"
    await replica_b.settle(cheap, 0.7)


async def test_a_failed_request_gives_its_reservation_back(shared_database: Database) -> None:
    """A provider that answers 500 bills nothing, so the money must come back —
    otherwise an outage exhausts the budget in refusals for unbilled calls."""
    budget = shared_tracker(shared_database, limit=1.0)
    for _ in range(10):
        reservation = await budget.reserve(0.5)
        await budget.settle(reservation, 0.0)  # the call failed

    async with shared_database.session() as session:
        spent = await session.scalar(text("SELECT spent_usd FROM budget_periods"))
    assert float(spent or 0.0) == pytest.approx(0.0), "ten failed calls cost nothing"

    still_open = await budget.reserve(1.0)
    assert still_open.shared, "the whole limit must still be available"
    await budget.settle(still_open, 0.0)


async def test_concurrent_reservations_never_cross_the_limit(shared_database: Database) -> None:
    """Twenty requests at once against a limit that fits four: the row lock decides."""
    replicas = [shared_tracker(shared_database, limit=1.0) for _ in range(4)]

    async def attempt(index: int) -> bool:
        budget = replicas[index % len(replicas)]
        try:
            reservation = await budget.reserve(0.25)
        except BudgetExceededError:
            return False
        await budget.settle(reservation, 0.25)
        return True

    verdicts = await asyncio.gather(*(attempt(index) for index in range(20)))
    assert sum(verdicts) == 4, f"$1.00 at $0.25 each is 4 requests, not {sum(verdicts)}"

    async with shared_database.session() as session:
        spent = await session.scalar(text("SELECT spent_usd FROM budget_periods"))
    assert float(spent) == pytest.approx(1.0), "the row must not exceed the limit either"


async def test_a_per_key_limit_is_shared_too(shared_database: Database) -> None:
    keys = [{"id": "team-alpha", "key": "t-a", "budget_limit_usd": 0.3}]
    replica_a = shared_tracker(shared_database, limit=10.0, keys=keys)
    replica_b = shared_tracker(shared_database, limit=10.0, keys=keys)

    taken = await replica_a.reserve(0.2, "team-alpha")
    await replica_a.settle(taken, 0.2)

    with pytest.raises(BudgetExceededError) as refusal:
        await replica_b.reserve(0.2, "team-alpha")
    assert "team-alpha" in str(refusal.value)

    other = await replica_b.reserve(0.2, None)
    assert other.shared, "the global limit still has room for a request without a key"


async def test_a_refused_key_does_not_leave_the_global_scope_charged(
    shared_database: Database,
) -> None:
    """The global scope is reserved first. When the per-key scope then refuses, the
    whole transaction has to go — a leak here would bill for calls never made."""
    keys = [{"id": "team-alpha", "key": "t-a", "budget_limit_usd": 0.1}]
    budget = shared_tracker(shared_database, limit=10.0, keys=keys)

    with pytest.raises(BudgetExceededError):
        await budget.reserve(0.5, "team-alpha")

    async with shared_database.session() as session:
        rows = (await session.execute(text("SELECT scope, spent_usd FROM budget_periods"))).all()
    assert all(float(spent) == 0.0 for _, spent in rows), f"money left behind: {rows}"


async def test_a_request_larger_than_the_whole_limit_is_refused(
    shared_database: Database,
) -> None:
    """`ON CONFLICT` does not fire on the first insert, so the very first request
    would otherwise be accepted at any price."""
    budget = shared_tracker(shared_database, limit=1.0)
    with pytest.raises(BudgetExceededError):
        await budget.reserve(2.0)


async def test_an_unreachable_database_falls_back_to_the_local_limit() -> None:
    """Money is not the rate limiter: a database blip must not open the tap. The
    in-memory counter is still a real refusal, just a per-replica one."""
    nowhere = Database(
        DatabaseConfig(
            dsn="postgresql+asyncpg://gateway@127.0.0.1:1/llm_gateway",
            run_migrations_on_startup=False,
        )
    )
    auth = AuthConfig.model_validate({"keys": []})
    config = BudgetConfig(enabled=True, period="day", limit_usd=1.0, scope="shared")
    budget = BudgetTracker(config, auth, nowhere)
    try:
        first = await budget.reserve(0.8)
        assert not first.shared, "nothing was written, so there is nothing to settle"
        await budget.settle(first, 0.8)
        with pytest.raises(BudgetExceededError):
            await budget.reserve(0.8)
    finally:
        await nowhere.aclose()

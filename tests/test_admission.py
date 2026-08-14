"""Admission control: the bulkhead and the rate limiter.

Both answer the same question — should this request be sent at all — and both are
easy to get subtly wrong: a bulkhead that leaks slots eventually blocks everything,
and a token bucket that refills on wall-clock reads throttles nobody.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text

from llm_gateway.db import migrate
from llm_gateway.db.session import Database
from llm_gateway.reliability.bulkhead import Bulkhead, BulkheadRegistry
from llm_gateway.reliability.ratelimit import GLOBAL_SCOPE, RateLimiter
from llm_gateway.settings import BulkheadConfig, DatabaseConfig, RateLimitConfig

# -- token bucket -------------------------------------------------------------


def limiter(**overrides: object) -> RateLimiter:
    config = RateLimitConfig.model_validate(
        {"enabled": True, "requests_per_second": 10.0, "burst": 5, **overrides}
    )
    return RateLimiter(config)


async def test_the_burst_is_spent_before_anyone_is_throttled() -> None:
    limits = limiter()
    assert [await limits.check(None, now=100.0) for _ in range(5)] == [None] * 5

    retry_after = await limits.check(None, now=100.0)
    assert retry_after is not None
    assert retry_after == pytest.approx(0.1, abs=0.01), "one token at 10 rps takes 100 ms"


async def test_tokens_refill_over_time_and_never_exceed_the_burst() -> None:
    limits = limiter()
    for _ in range(5):
        await limits.check(None, now=100.0)
    assert await limits.check(None, now=100.0) is not None

    # 0.3 s later three tokens are back.
    assert [await limits.check(None, now=100.3) for _ in range(3)] == [None] * 3
    assert await limits.check(None, now=100.3) is not None

    # An hour of silence must not hand out an hour's worth of tokens.
    assert [await limits.check(None, now=3700.0) for _ in range(5)] == [None] * 5
    assert await limits.check(None, now=3700.0) is not None


async def test_every_api_key_gets_its_own_bucket() -> None:
    """One noisy tenant must not spend another tenant's allowance."""
    limits = limiter()
    for _ in range(5):
        assert await limits.check("team-alpha", now=100.0) is None
    assert await limits.check("team-alpha", now=100.0) is not None

    assert await limits.check("team-beta", now=100.0) is None
    snapshot = limits.snapshot()
    assert set(snapshot["scopes"]) == {"team-alpha", "team-beta"}  # type: ignore[arg-type]


async def test_without_auth_everyone_shares_one_bucket() -> None:
    limits = limiter()
    await limits.check(None, now=100.0)
    assert list(limits.snapshot()["scopes"]) == [GLOBAL_SCOPE]  # type: ignore[arg-type]


async def test_a_disabled_limiter_never_throttles() -> None:
    limits = RateLimiter(RateLimitConfig(enabled=False, requests_per_second=1.0, burst=1))
    assert [await limits.check(None, now=100.0) for _ in range(100)] == [None] * 100


# -- bulkhead -----------------------------------------------------------------


async def test_slots_are_reused_not_consumed() -> None:
    """A released slot must come back, or the provider silently goes dark."""
    bulkhead = Bulkhead(name="p", limit=2)
    for _ in range(10):
        assert await bulkhead.acquire(timeout_s=0.1) is True
        bulkhead.release()
    assert bulkhead.in_flight == 0
    assert bulkhead.stats.shed == 0


async def test_a_full_bulkhead_sheds_when_the_wait_runs_out() -> None:
    bulkhead = Bulkhead(name="p", limit=1)
    assert await bulkhead.acquire(timeout_s=0.1) is True

    assert await bulkhead.acquire(timeout_s=0.05) is False
    assert bulkhead.stats.shed == 1
    assert bulkhead.stats.queued == 1, "it waited before giving up"


async def test_a_queued_request_gets_the_slot_that_comes_free() -> None:
    bulkhead = Bulkhead(name="p", limit=1)
    await bulkhead.acquire(timeout_s=None)

    async def free_it_shortly() -> None:
        await asyncio.sleep(0.05)
        bulkhead.release()

    asyncio.create_task(free_it_shortly())  # noqa: RUF006 - awaited via the acquire below
    assert await bulkhead.acquire(timeout_s=1.0) is True
    assert bulkhead.stats.shed == 0
    assert bulkhead.stats.queued == 1
    assert bulkhead.stats.wait_ms_total >= 40


async def test_a_zero_wait_sheds_immediately_without_queueing() -> None:
    bulkhead = Bulkhead(name="p", limit=1)
    await bulkhead.acquire(timeout_s=None)

    assert await bulkhead.acquire(timeout_s=0.0) is False
    assert bulkhead.stats.queued == 0, "there was no time to wait, so nothing queued"
    assert bulkhead.stats.shed == 1


async def test_cancellation_is_not_counted_as_shedding() -> None:
    """A hedge that lost its race stopped wanting a slot; nobody refused it."""
    bulkhead = Bulkhead(name="p", limit=1)
    await bulkhead.acquire(timeout_s=None)

    waiting = asyncio.create_task(bulkhead.acquire(timeout_s=5.0))
    await asyncio.sleep(0.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert bulkhead.stats.shed == 0
    bulkhead.release()
    assert await bulkhead.acquire(timeout_s=0.1) is True, "the slot survived the cancellation"


async def test_an_unlimited_bulkhead_admits_everything() -> None:
    bulkhead = Bulkhead(name="p", limit=0)
    assert bulkhead.unlimited
    for _ in range(50):
        assert await bulkhead.acquire(timeout_s=0.0) is True
    assert bulkhead.stats.admitted == 50
    assert bulkhead.in_flight == 50


async def test_the_registry_gives_each_provider_its_own_slots() -> None:
    registry = BulkheadRegistry(
        BulkheadConfig(enabled=True, max_concurrent_per_provider=1),
        ["mock_primary", "mock_secondary"],
    )
    assert await registry.get("mock_primary").acquire(timeout_s=0.0) is True
    assert await registry.get("mock_primary").acquire(timeout_s=0.0) is False
    assert await registry.get("mock_secondary").acquire(timeout_s=0.0) is True

    registry.reset()
    assert await registry.get("mock_primary").acquire(timeout_s=0.0) is True


async def test_a_disabled_bulkhead_is_unlimited() -> None:
    registry = BulkheadRegistry(
        BulkheadConfig(enabled=False, max_concurrent_per_provider=1), ["mock_primary"]
    )
    for _ in range(20):
        assert await registry.get("mock_primary").acquire(timeout_s=0.0) is True


# -- the two mechanisms inside a real request ---------------------------------


async def test_a_shed_hop_falls_back_instead_of_failing() -> None:
    """A full queue is our problem, not the provider's: move the traffic, do not
    blame the provider and do not fail the caller."""
    from tests.test_orchestrator import REQUEST, StubAdapter, build, settings_with

    settings = settings_with(
        timeouts={"enabled": True, "total_s": 30.0},
        retries={"enabled": False},
        fallback={"enabled": True, "max_providers": 3},
        bulkhead={"enabled": True, "max_concurrent_per_provider": 1, "queue_timeout_ms": 0},
    )
    first = StubAdapter("mock_primary")
    second = StubAdapter("mock_secondary")
    orchestrator, _ = build(
        settings,
        {"mock_primary": first, "mock_secondary": second, "mock_tertiary": StubAdapter("t")},
    )
    # Occupy the only slot of the first provider, as a request in flight would.
    assert await orchestrator.bulkheads.get("mock_primary").acquire(timeout_s=0.0) is True

    result = await orchestrator.execute(REQUEST, route_name="chaos-default")

    assert result.provider == "mock_secondary"
    assert result.fallbacks == 1
    assert first.calls == [], "the full provider was never called"
    assert result.attempts == 1, "a shed hop is not an attempt: nothing was sent"
    shed = [r for r in result.attempt_records if r.outcome == "shed_bulkhead"]
    assert [r.provider for r in shed] == ["mock_primary"]
    breaker = orchestrator.breakers.get("mock_primary")
    assert breaker.snapshot()["window_calls"] == 0, "shedding must not blame the provider"


async def test_the_rate_limit_answers_429_with_retry_after(cache_stack: dict[str, Any]) -> None:
    """One shared bucket when auth is off; a refusal costs no provider call."""
    state = cache_stack["state"]
    state.limiter.config = RateLimitConfig(enabled=True, requests_per_second=1.0, burst=1)
    state.limiter.reset()

    payload = {
        "model": "chaos-default",
        "messages": [{"role": "user", "content": "admission"}],
        "temperature": 0.9,
    }
    first = await cache_stack["client"].post("/v1/chat/completions", json=payload)
    second = await cache_stack["client"].post("/v1/chat/completions", json=payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "1"
    body = second.json()
    assert body["error"]["kind"] == "throttled"
    assert second.headers["x-gateway-error-kind"] == "throttled"
    assert state.limiter.snapshot()["scopes"][GLOBAL_SCOPE]["throttled"] == 1


async def test_a_provider_can_override_the_default_concurrency_limit() -> None:
    """Quotas differ per provider, so one global number is only a default."""
    registry = BulkheadRegistry(
        BulkheadConfig(enabled=True, max_concurrent_per_provider=8),
        ["mock_primary", "mock_secondary"],
        {"mock_primary": 1, "mock_secondary": None},
    )
    assert registry.get("mock_primary").limit == 1
    assert registry.get("mock_secondary").limit == 8

    assert await registry.get("mock_primary").acquire(timeout_s=0.0) is True
    assert await registry.get("mock_primary").acquire(timeout_s=0.0) is False


# -- the shared bucket: one limit for the whole deployment ---------------------


@pytest_asyncio.fixture
async def shared_database() -> AsyncIterator[Database]:
    """A real database: the point of the shared bucket is that Postgres, not the
    process, is what keeps two replicas from spending the same token."""
    from tests.conftest import TEST_DSN, _database_available

    if not await _database_available(TEST_DSN):
        pytest.skip(f"PostgreSQL not reachable at {TEST_DSN}")
    await migrate.upgrade(TEST_DSN)
    database = Database(DatabaseConfig(dsn=TEST_DSN, run_migrations_on_startup=False))
    async with database.session() as session:
        await session.execute(text("DELETE FROM rate_limit_buckets"))
        await session.commit()
    try:
        yield database
    finally:
        await database.aclose()


def shared_limiter(database: Database, **overrides: object) -> RateLimiter:
    config = RateLimitConfig.model_validate(
        {
            "enabled": True,
            "scope": "shared",
            "requests_per_second": 1.0,
            "burst": 3,
            **overrides,
        }
    )
    return RateLimiter(config, database)


async def test_two_replicas_spend_one_shared_allowance(shared_database: Database) -> None:
    """The failure this fixes: with a local bucket each replica hands out the full
    limit, so two replicas let through twice the configured rate."""
    replica_a = shared_limiter(shared_database)
    replica_b = shared_limiter(shared_database)

    assert await replica_a.check(None) is None  # 3 -> 2
    assert await replica_b.check(None) is None  # 2 -> 1
    assert await replica_a.check(None) is None  # 1 -> 0

    retry_after = await replica_b.check(None)
    assert retry_after is not None, "the burst was spent by both replicas together"
    assert 0 < retry_after <= 1.0, "one token at 1 rps is a second away"
    assert replica_b.snapshot()["shared"] == {"allowed": 1, "throttled": 1, "errors": 0}


async def test_the_shared_bucket_is_per_scope(shared_database: Database) -> None:
    limits = shared_limiter(shared_database)
    for _ in range(3):
        assert await limits.check("team-alpha") is None
    assert await limits.check("team-alpha") is not None

    assert await limits.check("team-beta") is None, "another key has its own allowance"


async def test_concurrent_replicas_never_overspend_the_bucket(shared_database: Database) -> None:
    """Twenty requests at once against a burst of three: the row lock decides."""
    replicas = [shared_limiter(shared_database) for _ in range(4)]
    verdicts = await asyncio.gather(
        *(replicas[index % len(replicas)].check(None) for index in range(20))
    )

    allowed = [verdict for verdict in verdicts if verdict is None]
    assert len(allowed) == 3, f"burst is 3, but {len(allowed)} requests were let through"


async def test_an_unreachable_database_lets_requests_through() -> None:
    """A limiter that cannot reach its storage must not take the gateway down.

    Letting requests through is the safe failure here: the budget and the
    provider-side limits are still in the way, and refusing everything would turn a
    database blip into a gateway outage.
    """
    nowhere = Database(
        DatabaseConfig(
            dsn="postgresql+asyncpg://gateway@127.0.0.1:1/llm_gateway",
            run_migrations_on_startup=False,
        )
    )
    limits = shared_limiter(nowhere)
    try:
        assert await limits.check(None) is None
    finally:
        await nowhere.aclose()
    assert limits.snapshot()["shared"] == {"allowed": 0, "throttled": 0, "errors": 1}

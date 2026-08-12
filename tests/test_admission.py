"""Admission control: the bulkhead and the rate limiter.

Both answer the same question — should this request be sent at all — and both are
easy to get subtly wrong: a bulkhead that leaks slots eventually blocks everything,
and a token bucket that refills on wall-clock reads throttles nobody.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gateway.reliability.bulkhead import Bulkhead, BulkheadRegistry
from llm_gateway.reliability.ratelimit import GLOBAL_SCOPE, RateLimiter
from llm_gateway.settings import BulkheadConfig, RateLimitConfig

# -- token bucket -------------------------------------------------------------


def limiter(**overrides: object) -> RateLimiter:
    config = RateLimitConfig.model_validate(
        {"enabled": True, "requests_per_second": 10.0, "burst": 5, **overrides}
    )
    return RateLimiter(config)


def test_the_burst_is_spent_before_anyone_is_throttled() -> None:
    limits = limiter()
    assert [limits.check(None, now=100.0) for _ in range(5)] == [None] * 5

    retry_after = limits.check(None, now=100.0)
    assert retry_after is not None
    assert retry_after == pytest.approx(0.1, abs=0.01), "one token at 10 rps takes 100 ms"


def test_tokens_refill_over_time_and_never_exceed_the_burst() -> None:
    limits = limiter()
    for _ in range(5):
        limits.check(None, now=100.0)
    assert limits.check(None, now=100.0) is not None

    # 0.3 s later three tokens are back.
    assert [limits.check(None, now=100.3) for _ in range(3)] == [None] * 3
    assert limits.check(None, now=100.3) is not None

    # An hour of silence must not hand out an hour's worth of tokens.
    assert [limits.check(None, now=3700.0) for _ in range(5)] == [None] * 5
    assert limits.check(None, now=3700.0) is not None


def test_every_api_key_gets_its_own_bucket() -> None:
    """One noisy tenant must not spend another tenant's allowance."""
    limits = limiter()
    for _ in range(5):
        assert limits.check("team-alpha", now=100.0) is None
    assert limits.check("team-alpha", now=100.0) is not None

    assert limits.check("team-beta", now=100.0) is None
    snapshot = limits.snapshot()
    assert set(snapshot["scopes"]) == {"team-alpha", "team-beta"}  # type: ignore[arg-type]


def test_without_auth_everyone_shares_one_bucket() -> None:
    limits = limiter()
    limits.check(None, now=100.0)
    assert list(limits.snapshot()["scopes"]) == [GLOBAL_SCOPE]  # type: ignore[arg-type]


def test_a_disabled_limiter_never_throttles() -> None:
    limits = RateLimiter(RateLimitConfig(enabled=False, requests_per_second=1.0, burst=1))
    assert [limits.check(None, now=100.0) for _ in range(100)] == [None] * 100


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

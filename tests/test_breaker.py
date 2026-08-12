from __future__ import annotations

from llm_gateway.reliability.breaker import BreakerState, CircuitBreaker
from llm_gateway.settings import CircuitBreakerConfig


def make(**overrides: object) -> CircuitBreaker:
    base = {
        "enabled": True,
        "window_s": 10.0,
        "min_calls": 4,
        "failure_ratio": 0.5,
        "cooldown_s": 5.0,
        "half_open_max_calls": 2,
    }
    return CircuitBreaker("p", CircuitBreakerConfig.model_validate(base | overrides))


def test_disabled_breaker_always_allows() -> None:
    breaker = make(enabled=False)
    for _ in range(50):
        assert breaker.allow() is True
        breaker.record(ok=False)
    assert breaker.state is BreakerState.CLOSED


def test_stays_closed_below_min_calls_even_at_100_percent_failure() -> None:
    breaker = make(min_calls=10)
    for index in range(9):
        assert breaker.allow(now=index) is True
        breaker.record(ok=False, now=index)
    assert breaker.state is BreakerState.CLOSED


def test_opens_once_the_failure_ratio_is_reached() -> None:
    breaker = make(min_calls=4, failure_ratio=0.5)
    for index in range(4):
        breaker.allow(now=index)
        breaker.record(ok=False, now=index)
    assert breaker.state is BreakerState.OPEN
    assert breaker.opened_count == 1


def test_open_breaker_short_circuits_until_the_cooldown_elapses() -> None:
    breaker = make(min_calls=2, cooldown_s=5.0)
    for index in range(2):
        breaker.allow(now=index)
        breaker.record(ok=False, now=index)

    assert breaker.allow(now=2.0) is False
    assert breaker.short_circuited == 1
    # Cooldown passed: exactly `half_open_max_calls` probes get through.
    assert breaker.allow(now=100.0) is True
    assert breaker.allow(now=100.0) is True
    assert breaker.allow(now=100.0) is False
    assert breaker.state is BreakerState.HALF_OPEN


def test_half_open_closes_after_enough_successful_probes() -> None:
    breaker = make(min_calls=2, half_open_max_calls=2)
    for index in range(2):
        breaker.allow(now=index)
        breaker.record(ok=False, now=index)

    breaker.allow(now=100.0)
    breaker.record(ok=True, now=100.0)
    breaker.allow(now=100.0)
    breaker.record(ok=True, now=100.0)
    assert breaker.state is BreakerState.CLOSED


def test_a_failed_probe_reopens_the_breaker() -> None:
    breaker = make(min_calls=2)
    for index in range(2):
        breaker.allow(now=index)
        breaker.record(ok=False, now=index)

    breaker.allow(now=100.0)
    breaker.record(ok=False, now=100.0)
    assert breaker.state is BreakerState.OPEN
    assert breaker.opened_count == 2
    assert breaker.allow(now=101.0) is False


def test_old_events_fall_out_of_the_window() -> None:
    breaker = make(min_calls=4, window_s=10.0)
    for index in range(3):
        breaker.allow(now=index)
        breaker.record(ok=False, now=index)
    # Far in the future the three failures are outside the window, so a single
    # fresh failure must not be enough to open the breaker.
    breaker.allow(now=1000.0)
    breaker.record(ok=False, now=1000.0)
    assert breaker.state is BreakerState.CLOSED
    assert breaker.snapshot(now=1000.0)["window_calls"] == 1

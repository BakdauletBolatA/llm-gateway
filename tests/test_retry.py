from __future__ import annotations

import random

from llm_gateway.errors import ErrorKind
from llm_gateway.reliability.retry import compute_backoff, should_retry
from llm_gateway.settings import RetryConfig


def config(**overrides: object) -> RetryConfig:
    base = {
        "enabled": True,
        "max_attempts": 3,
        "base_delay_ms": 100,
        "max_delay_ms": 2000,
        "jitter": "full",
    }
    return RetryConfig.model_validate(base | overrides)


def test_disabled_policy_never_retries() -> None:
    assert should_retry(ErrorKind.SERVER_ERROR, config(enabled=False)) is False


def test_only_listed_kinds_are_retried() -> None:
    policy = config()
    assert should_retry(ErrorKind.SERVER_ERROR, policy) is True
    assert should_retry(ErrorKind.INVALID_RESPONSE, policy) is True
    # A malformed request will be malformed on the retry too.
    assert should_retry(ErrorKind.BAD_REQUEST, policy) is False
    assert should_retry(ErrorKind.AUTH, policy) is False


def test_backoff_without_jitter_is_exponential_and_capped() -> None:
    policy = config(jitter="none", base_delay_ms=100, max_delay_ms=350)
    assert compute_backoff(1, policy) == 0.1
    assert compute_backoff(2, policy) == 0.2
    assert compute_backoff(3, policy) == 0.35  # capped
    assert compute_backoff(9, policy) == 0.35


def test_full_jitter_stays_within_the_window_and_actually_spreads() -> None:
    policy = config(jitter="full", base_delay_ms=200)
    rng = random.Random(7)
    delays = [compute_backoff(3, policy, rng=rng) for _ in range(200)]
    assert all(0.0 <= delay <= 0.8 for delay in delays)
    assert len(set(delays)) > 50, "full jitter must not collapse to a single value"


def test_equal_jitter_keeps_half_the_delay_fixed() -> None:
    policy = config(jitter="equal", base_delay_ms=200)
    rng = random.Random(7)
    delays = [compute_backoff(1, policy, rng=rng) for _ in range(100)]
    assert all(0.1 <= delay <= 0.2 for delay in delays)


def test_retry_after_wins_over_the_computed_backoff() -> None:
    policy = config(base_delay_ms=100, respect_retry_after=True, max_retry_after_s=3.0)
    rng = random.Random(1)
    delay = compute_backoff(1, policy, retry_after_s=2.0, rng=rng)
    assert 2.0 <= delay <= 2.1, "server backpressure honoured, plus a small spread"


def test_retry_after_is_capped_so_one_provider_cannot_stall_the_request() -> None:
    policy = config(respect_retry_after=True, max_retry_after_s=3.0)
    delay = compute_backoff(1, policy, retry_after_s=600.0, rng=random.Random(1))
    assert delay <= 3.2


def test_retry_after_can_be_ignored_by_config() -> None:
    policy = config(jitter="none", base_delay_ms=100, respect_retry_after=False)
    assert compute_backoff(1, policy, retry_after_s=30.0) == 0.1

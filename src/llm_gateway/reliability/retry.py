"""Retry policy: exponential backoff with jitter, honouring upstream Retry-After."""

from __future__ import annotations

import random

from llm_gateway.errors import ErrorKind
from llm_gateway.settings import RetryConfig


def should_retry(kind: ErrorKind, config: RetryConfig) -> bool:
    if not config.enabled:
        return False
    return kind in config.retryable


def compute_backoff(
    attempt: int,
    config: RetryConfig,
    *,
    retry_after_s: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Delay in seconds before the attempt that follows `attempt` (1-based).

    Without jitter, N clients that fail together retry together and re-create the
    burst that broke the provider. `full` jitter spreads them across the whole
    window; `equal` keeps half the delay fixed and jitters the rest.

    A Retry-After from the provider wins over our own schedule — it is the
    provider stating when it will accept traffic again — but a small random
    offset is still added so clients do not all return on the same tick.
    """
    generator = rng or random
    base_ms = config.base_delay_ms * (2 ** max(0, attempt - 1))
    base_s = min(base_ms, config.max_delay_ms) / 1000.0

    if retry_after_s is not None and config.respect_retry_after:
        honoured = min(retry_after_s, config.max_retry_after_s)
        spread = generator.uniform(0.0, config.base_delay_ms / 1000.0)
        return round(honoured + spread, 4)

    if config.jitter == "none":
        delay = base_s
    elif config.jitter == "equal":
        delay = base_s / 2 + generator.uniform(0.0, base_s / 2)
    else:  # full
        delay = generator.uniform(0.0, base_s)
    return round(delay, 4)

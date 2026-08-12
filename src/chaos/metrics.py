"""Aggregation of per-request observations into a run summary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import ceil
from statistics import fmean
from typing import Any


@dataclass(slots=True)
class Observation:
    """One client-side request outcome."""

    latency_ms: float
    status: int | None  # None when the request never produced an HTTP response
    outcome: str  # success | http_<code> | client_timeout | transport_error
    error_kind: str | None = None
    attempts: int = 0
    retries: int = 0
    fallbacks: int = 0
    breaker_skips: int = 0
    cache_hit: bool = False
    cost_usd: float = 0.0
    provider: str | None = None


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: the smallest value at or above `fraction` of the
    sample. Returns 0.0 for an empty sample.

    Nearest-rank rather than interpolation on purpose — an interpolated p95 of a
    latency distribution with a cluster of timeouts invents a number that no
    request actually experienced.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), ceil(fraction * len(ordered))))
    return ordered[rank - 1]


def latency_block(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0}
    return {
        "p50": round(percentile(values, 0.50), 1),
        "p95": round(percentile(values, 0.95), 1),
        "p99": round(percentile(values, 0.99), 1),
        "mean": round(fmean(values), 1),
        "max": round(max(values), 1),
        "min": round(min(values), 1),
    }


@dataclass
class RunAggregate:
    observations: list[Observation] = field(default_factory=list)

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)

    def summarise(self) -> dict[str, Any]:
        total = len(self.observations)
        successes = [o for o in self.observations if o.outcome == "success"]
        failures = [o for o in self.observations if o.outcome != "success"]

        return {
            "requests": total,
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": round(len(successes) / total, 4) if total else 0.0,
            "latency_ms": latency_block([o.latency_ms for o in self.observations]),
            "latency_ms_success_only": latency_block([o.latency_ms for o in successes]),
            "attempts_total": sum(o.attempts for o in self.observations),
            "retries_total": sum(o.retries for o in self.observations),
            "fallbacks_total": sum(o.fallbacks for o in self.observations),
            "breaker_skips_total": sum(o.breaker_skips for o in self.observations),
            "cache_hits": sum(1 for o in self.observations if o.cache_hit),
            "cache_hit_rate": (
                round(sum(1 for o in self.observations if o.cache_hit) / total, 4)
                if total
                else 0.0
            ),
            "cost_usd_client": round(sum(o.cost_usd for o in self.observations), 6),
            "outcomes": dict(sorted(Counter(o.outcome for o in self.observations).items())),
            "error_kinds": dict(
                sorted(Counter(o.error_kind for o in failures if o.error_kind).items())
            ),
            "providers_used": dict(
                sorted(Counter(o.provider for o in successes if o.provider).items())
            ),
        }

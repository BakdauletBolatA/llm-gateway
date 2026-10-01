"""Turn raw per-request records from a Locust run into the numbers we report."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chaos.metrics import percentile


def _latencies(values: list[float]) -> dict[str, float]:
    return {
        "p50": round(percentile(values, 0.50), 1),
        "p95": round(percentile(values, 0.95), 1),
        "p99": round(percentile(values, 0.99), 1),
    }


def _window(records: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    ok = [r for r in records if r["status"] == 200]
    providers = Counter(r["provider"] for r in ok if r["provider"])
    return {
        "requests": len(records),
        "success_rate": round(len(ok) / len(records), 4) if records else 0.0,
        "requests_per_s": round(len(records) / seconds, 2) if seconds > 0 else 0.0,
        "latency_ms": _latencies([r["latency_ms"] for r in ok]),
        "provider_share": {
            name: round(count / len(ok), 4) for name, count in sorted(providers.items())
        },
    }


def summarise(
    records: list[dict[str, Any]], *, duration_s: float, kill_at_s: float | None = None
) -> dict[str, Any]:
    """`t` is seconds since the run started; `kill_at_s` is when the backend was killed."""
    ok = [r for r in records if r["status"] == 200]
    result: dict[str, Any] = {
        "requests": len(records),
        "success_rate": round(len(ok) / len(records), 4) if records else 0.0,
        "requests_per_s": round(len(records) / duration_s, 2) if duration_s > 0 else 0.0,
        "latency_ms": _latencies([r["latency_ms"] for r in ok]),
        "latency_ms_all_requests": _latencies([r["latency_ms"] for r in records]),
        "status_counts": dict(sorted(Counter(str(r["status"]) for r in records).items())),
        "provider_share": _window(records, duration_s)["provider_share"],
    }
    if kill_at_s is None:
        return result

    before = [r for r in records if r["t"] < kill_at_s]
    after = [r for r in records if r["t"] >= kill_at_s]
    result["before_kill"] = _window(before, kill_at_s)
    result["after_kill"] = _window(after, duration_s - kill_at_s)

    primary = max(
        Counter(r["provider"] for r in before if r["status"] == 200 and r["provider"]).items(),
        key=lambda item: item[1],
        default=(None, 0),
    )[0]
    first_other = next(
        (
            r["t"]
            for r in after
            if r["status"] == 200 and r["provider"] and r["provider"] != primary
        ),
        None,
    )
    result["failover"] = {
        "provider_killed": primary,
        "failed_requests_after_kill": sum(r["status"] != 200 for r in after),
        "seconds_to_first_success_from_other_provider": (
            round(first_other - kill_at_s, 1) if first_other is not None else None
        ),
    }
    return result

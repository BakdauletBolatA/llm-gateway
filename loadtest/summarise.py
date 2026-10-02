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
    records: list[dict[str, Any]],
    *,
    duration_s: float,
    kill_at_s: float | None = None,
    killed_provider: str | None = None,
) -> dict[str, Any]:
    """`t` is seconds since the run started; `kill_at_s` is when the backend was killed.

    `killed_provider` is the provider the scenario stopped. When it is not given it is
    guessed as the one that answered most before the kill, which is wrong whenever a
    slow backend splits the traffic.
    """
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

    primary = (
        killed_provider
        or max(
            Counter(r["provider"] for r in before if r["status"] == 200 and r["provider"]).items(),
            key=lambda item: item[1],
            default=(None, 0),
        )[0]
    )
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
        "exercised": first_other is not None,
    }
    if first_other is None:
        result["failover"]["note"] = (
            "failover not observed: no other provider answered a request sent after the kill, "
            "so this run says nothing about it (the backend was probably too slow)"
        )
    return result


def _range(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {"min": ordered[0], "median": median, "max": ordered[-1]}


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Median and full range across repeated runs of one scenario.

    The range is reported next to the median on purpose: with a handful of runs a
    median alone hides exactly the spread the repeats were made to show.
    """
    if not runs:
        raise ValueError("no runs to aggregate")
    result: dict[str, Any] = {
        "runs": len(runs),
        "requests_per_s": _range([r["requests_per_s"] for r in runs]),
        "success_rate": _range([r["success_rate"] for r in runs]),
        "latency_ms": {
            name: _range([r["latency_ms"][name] for r in runs]) for name in ("p50", "p95", "p99")
        },
    }
    if all("failover" in r for r in runs):
        firsts = [
            r["failover"]["seconds_to_first_success_from_other_provider"]
            for r in runs
            if r["failover"]["seconds_to_first_success_from_other_provider"] is not None
        ]
        result["failover"] = {
            "failed_requests_after_kill": _range(
                [r["failover"]["failed_requests_after_kill"] for r in runs]
            ),
            "seconds_to_first_success_from_other_provider": _range(firsts) if firsts else None,
            "runs_without_a_second_provider": len(runs) - len(firsts),
        }
    return result

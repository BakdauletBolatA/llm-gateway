"""The load-test numbers go into the README, so the arithmetic behind them is pinned."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lt_summarise", ROOT / "loadtest" / "summarise.py")
assert _spec is not None and _spec.loader is not None
summarise_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarise_module)
summarise = summarise_module.summarise


def rec(t: float, ms: float, status: int = 200, provider: str | None = "a") -> dict[str, Any]:
    return {"t": t, "latency_ms": ms, "status": status, "provider": provider}


def test_success_rate_and_percentiles_are_nearest_rank() -> None:
    records = [rec(i * 0.1, float(ms)) for i, ms in enumerate(range(1, 101))]
    result = summarise(records, duration_s=10.0)
    assert result["requests"] == 100
    assert result["success_rate"] == 1.0
    assert result["latency_ms"]["p50"] == 50.0
    assert result["latency_ms"]["p95"] == 95.0
    assert result["latency_ms"]["p99"] == 99.0
    assert result["requests_per_s"] == 10.0


def test_failed_requests_count_against_success_but_not_against_success_latency() -> None:
    records = [rec(0.0, 100.0), rec(1.0, 120.0), rec(2.0, 30000.0, status=502, provider=None)]
    result = summarise(records, duration_s=3.0)
    assert result["success_rate"] == round(2 / 3, 4)
    assert result["latency_ms"]["p99"] == 120.0
    assert result["latency_ms_all_requests"]["p99"] == 30000.0
    assert result["status_counts"] == {"200": 2, "502": 1}


def test_a_kill_splits_the_run_into_before_and_after() -> None:
    before = [rec(t, 100.0, provider="primary") for t in (1.0, 2.0, 3.0)]
    after = [
        rec(5.0, 9000.0, status=502, provider=None),
        rec(6.0, 400.0, provider="secondary"),
        rec(7.0, 420.0, provider="secondary"),
    ]
    result = summarise(before + after, duration_s=8.0, kill_at_s=4.0)
    assert result["before_kill"]["success_rate"] == 1.0
    assert result["after_kill"]["success_rate"] == round(2 / 3, 4)
    assert result["before_kill"]["provider_share"] == {"primary": 1.0}
    assert result["after_kill"]["provider_share"] == {"secondary": 1.0}
    assert result["failover"]["failed_requests_after_kill"] == 1
    assert result["failover"]["seconds_to_first_success_from_other_provider"] == 2.0


def test_no_kill_means_no_failover_section() -> None:
    result = summarise([rec(0.0, 1.0)], duration_s=1.0)
    assert "failover" not in result and "before_kill" not in result


def test_an_empty_run_does_not_divide_by_zero() -> None:
    result = summarise([], duration_s=5.0)
    assert result["requests"] == 0 and result["success_rate"] == 0.0

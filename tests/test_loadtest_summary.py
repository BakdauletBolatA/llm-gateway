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


aggregate_runs = summarise_module.aggregate_runs


def run_summary(
    rps: float, p50: float, ok: float = 1.0, failed: int = 0, first: float = 1.0
) -> dict[str, Any]:
    return {
        "requests_per_s": rps,
        "success_rate": ok,
        "latency_ms": {"p50": p50, "p95": p50 * 1.5, "p99": p50 * 2},
        "failover": {
            "failed_requests_after_kill": failed,
            "seconds_to_first_success_from_other_provider": first,
        },
    }


def test_the_aggregate_reports_the_median_and_the_full_range() -> None:
    runs = [run_summary(1.0, 4000), run_summary(0.9, 4400), run_summary(1.1, 3800)]
    result = aggregate_runs(runs)
    assert result["runs"] == 3
    assert result["requests_per_s"] == {"min": 0.9, "median": 1.0, "max": 1.1}
    assert result["latency_ms"]["p50"] == {"min": 3800, "median": 4000, "max": 4400}
    assert result["latency_ms"]["p95"]["max"] == 6600.0


def test_an_even_number_of_runs_uses_the_mean_of_the_middle_two() -> None:
    result = aggregate_runs([run_summary(1.0, 100), run_summary(2.0, 300)])
    assert result["requests_per_s"]["median"] == 1.5
    assert result["latency_ms"]["p50"]["median"] == 200


def test_failover_figures_are_aggregated_only_when_every_run_had_a_kill() -> None:
    kill = aggregate_runs(
        [run_summary(1.0, 1, failed=0, first=1.1), run_summary(1.0, 1, failed=2, first=3.0)]
    )
    assert kill["failover"]["failed_requests_after_kill"] == {"min": 0, "median": 1.0, "max": 2}
    assert kill["failover"]["seconds_to_first_success_from_other_provider"]["max"] == 3.0
    steady = [{k: v for k, v in run_summary(1.0, 1).items() if k != "failover"}]
    assert "failover" not in aggregate_runs(steady)


def test_a_run_that_never_saw_another_provider_is_not_averaged_away() -> None:
    runs = [run_summary(1.0, 1, first=1.1), run_summary(1.0, 1, first=None)]  # type: ignore[arg-type]
    result = aggregate_runs(runs)
    assert result["failover"]["runs_without_a_second_provider"] == 1
    assert result["failover"]["seconds_to_first_success_from_other_provider"]["max"] == 1.1


def test_aggregating_nothing_is_an_error() -> None:
    import pytest

    with pytest.raises(ValueError, match="no runs"):
        aggregate_runs([])


def test_the_killed_provider_is_taken_from_the_scenario_not_guessed_from_traffic() -> None:
    """Under a slow backend the traffic before the kill can be split evenly."""
    before = [
        rec(1.0, 100.0, provider="primary"),
        rec(2.0, 100.0, provider="secondary"),
        rec(3.0, 100.0, provider="secondary"),
    ]
    after = [rec(6.0, 400.0, provider="secondary")]
    guessed = summarise(before + after, duration_s=8.0, kill_at_s=4.0)
    assert guessed["failover"]["provider_killed"] == "secondary"  # the old guess, wrong here
    told = summarise(before + after, duration_s=8.0, kill_at_s=4.0, killed_provider="primary")
    assert told["failover"]["provider_killed"] == "primary"
    assert told["failover"]["seconds_to_first_success_from_other_provider"] == 2.0


def test_failover_is_marked_as_exercised_when_another_provider_answered_after_the_kill() -> None:
    records = [rec(1.0, 100.0, provider="primary"), rec(6.0, 400.0, provider="secondary")]
    failover = summarise(records, duration_s=8.0, kill_at_s=4.0, killed_provider="primary")[
        "failover"
    ]
    assert failover["exercised"] is True
    assert "note" not in failover


def test_a_run_where_nothing_completed_after_the_kill_says_it_proved_nothing() -> None:
    records = [rec(1.0, 16000.0, provider="primary"), rec(2.0, 20000.0, provider="secondary")]
    failover = summarise(records, duration_s=8.0, kill_at_s=4.0, killed_provider="primary")[
        "failover"
    ]
    assert failover["exercised"] is False
    assert failover["seconds_to_first_success_from_other_provider"] is None
    assert "not observed" in failover["note"]


def test_only_failures_after_the_kill_do_not_count_as_failover() -> None:
    records = [rec(1.0, 100.0, provider="primary"), rec(6.0, 9000.0, status=502, provider=None)]
    failover = summarise(records, duration_s=8.0, kill_at_s=4.0, killed_provider="primary")[
        "failover"
    ]
    assert failover["exercised"] is False
    assert failover["failed_requests_after_kill"] == 1

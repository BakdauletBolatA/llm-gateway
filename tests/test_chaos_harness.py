from __future__ import annotations

from chaos.metrics import Observation, RunAggregate, latency_block, percentile
from chaos.report import build_section
from chaos.workload import build_workload


def test_percentile_uses_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.5) == 50.0
    assert percentile(values, 0.95) == 95.0
    assert percentile(values, 0.99) == 99.0
    assert percentile([], 0.95) == 0.0


def test_latency_block_on_a_single_sample() -> None:
    block = latency_block([12.0])
    assert block["p50"] == block["p95"] == block["max"] == 12.0


def test_aggregate_counts_successes_and_failures_separately() -> None:
    aggregate = RunAggregate()
    aggregate.add(Observation(latency_ms=100, status=200, outcome="success", cost_usd=0.001))
    aggregate.add(
        Observation(latency_ms=9000, status=None, outcome="client_timeout", error_kind="timeout")
    )
    aggregate.add(
        Observation(
            latency_ms=200, status=502, outcome="http_502", error_kind="server_error", retries=2
        )
    )
    summary = aggregate.summarise()
    assert summary["requests"] == 3
    assert summary["successes"] == 1
    assert summary["success_rate"] == round(1 / 3, 4)
    assert summary["retries_total"] == 2
    assert summary["outcomes"] == {"client_timeout": 1, "http_502": 1, "success": 1}
    assert summary["error_kinds"] == {"server_error": 1, "timeout": 1}
    # Timeouts must not be hidden by averaging them with fast failures.
    assert summary["latency_ms_success_only"]["p95"] == 100.0
    assert summary["latency_ms"]["max"] == 9000.0


def test_workload_repeats_topics_so_a_cache_has_something_to_hit() -> None:
    prompts, stats = build_workload(200, seed=1)
    assert len(prompts) == 200
    assert stats.unique_prompts < 200
    assert stats.duplicate_rate > 0.3
    assert stats.paraphrase_rate > stats.duplicate_rate


def test_workload_is_reproducible_for_a_given_seed() -> None:
    first, _ = build_workload(50, seed=7)
    second, _ = build_workload(50, seed=7)
    third, _ = build_workload(50, seed=8)
    assert first == second
    assert first != third


def _fake_result(label: str, scenario: str, success_rate: float) -> dict:
    return {
        "label": label,
        "scenario": scenario,
        "duration_s": 10.0,
        "gateway_config": {
            "reliability": {
                key: {"enabled": key == "timeouts"}
                for key in ("timeouts", "retries", "circuit_breaker", "fallback", "cache")
            }
        },
        "results": {
            "requests": 100,
            "successes": int(success_rate * 100),
            "success_rate": success_rate,
            "latency_ms": {"p50": 100.0, "p95": 200.0, "p99": 300.0},
            "attempts_total": 100,
            "retries_total": 0,
            "fallbacks_total": 0,
            "breaker_skips_total": 0,
            "cache_hits": 0,
            "cost_usd_server": 0.01,
            "throughput_rps": 10.0,
        },
    }


def test_report_renders_a_matrix_over_iterations_and_scenarios() -> None:
    results = [
        _fake_result("01_baseline", "storm", 0.30),
        _fake_result("02_timeouts", "storm", 0.45),
        _fake_result("ablation_no_retries", "storm", 0.40),
    ]
    section = build_section(results)
    assert "01_baseline" in section and "02_timeouts" in section
    assert "30.0%" in section and "45.0%" in section
    assert "Ablation" in section
    assert section.startswith("<!-- BEGIN:GENERATED -->")
    assert section.rstrip().endswith("<!-- END:GENERATED -->")

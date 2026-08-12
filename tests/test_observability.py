"""Metrics exposition and structured logging.

The gateway reports the same per-request facts three times — response headers for
the chaos harness, Postgres for the ledger, Prometheus for an operator. These tests
pin the third one, including the part that is easy to get wrong: that a scrape
reflects state (breakers, budget) and not only counters.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from llm_gateway import observability
from tests.conftest import set_scenario

CHAT = "/v1/chat/completions"
PAYLOAD: dict[str, Any] = {
    "model": "chaos-default",
    "messages": [{"role": "user", "content": "метрики"}],
    "temperature": 0.0,
}


def sample(name: str, **labels: str) -> float:
    """Current value of one metric sample, or 0.0 if it has no observations yet."""
    value = observability.REGISTRY.get_sample_value(name, labels or None)
    return float(value) if value is not None else 0.0


async def test_metrics_endpoint_exposes_the_gateway_families(stack: dict[str, Any]) -> None:
    await stack["client"].post(CHAT, json=PAYLOAD)
    response = await stack["client"].get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    for family in (
        "llm_gateway_requests_total",
        "llm_gateway_request_duration_seconds",
        "llm_gateway_provider_calls_total",
        "llm_gateway_cost_usd_total",
        "llm_gateway_cache_lookups_total",
        "llm_gateway_circuit_breaker_state",
        "llm_gateway_budget_spent_usd",
        "llm_gateway_recorder_queue_depth",
    ):
        assert family in body, f"{family} missing from the exposition"


async def test_a_successful_request_moves_the_counters(stack: dict[str, Any]) -> None:
    before = sample("llm_gateway_requests_total", route="chaos-default", outcome="success")
    before_calls = sample(
        "llm_gateway_provider_calls_total", provider="mock_primary", outcome="success"
    )

    response = await stack["client"].post(CHAT, json=PAYLOAD)
    assert response.status_code == 200

    assert (
        sample("llm_gateway_requests_total", route="chaos-default", outcome="success") == before + 1
    )
    assert (
        sample("llm_gateway_provider_calls_total", provider="mock_primary", outcome="success")
        == before_calls + 1
    )
    assert sample("llm_gateway_cost_usd_total", provider="mock_primary") > 0


async def test_a_failed_request_is_counted_by_error_kind(stack: dict[str, Any]) -> None:
    set_scenario(stack["mock_app"], "total_outage")
    before = sample("llm_gateway_request_errors_total", route="chaos-default", kind="server_error")

    response = await stack["client"].post(CHAT, json=PAYLOAD)
    assert response.status_code == 502

    assert (
        sample("llm_gateway_request_errors_total", route="chaos-default", kind="server_error")
        == before + 1
    )
    assert sample("llm_gateway_provider_calls_total", provider="mock_primary", outcome="error") > 0


async def test_a_scrape_reflects_current_state_not_only_counters(stack: dict[str, Any]) -> None:
    """Gauges are filled in at scrape time, so they cannot drift from their source."""
    await stack["client"].post(CHAT, json=PAYLOAD)
    await stack["client"].get("/metrics")

    assert sample("llm_gateway_budget_limit_usd") == pytest.approx(
        stack["settings"].budget.limit_usd
    )
    assert sample("llm_gateway_budget_spent_usd") > 0
    # 0 == closed, and this stack has never failed a call.
    assert sample("llm_gateway_circuit_breaker_state", provider="mock_primary") == 0.0


def test_json_formatter_promotes_extra_fields() -> None:
    record = logging.LogRecord(
        name="llm_gateway.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="request failed: %s",
        args=("boom",),
        exc_info=None,
    )
    record.request_id = "abc123"
    record.attempts = 3

    payload = json.loads(observability.JsonFormatter().format(record))

    assert payload["message"] == "request failed: boom"
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "llm_gateway.test"
    assert payload["request_id"] == "abc123"
    assert payload["attempts"] == 3
    assert payload["ts"].endswith("+00:00")


def test_json_formatter_keeps_the_traceback() -> None:
    try:
        raise ValueError("upstream exploded")
    except ValueError:
        record = logging.LogRecord(
            name="llm_gateway.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="unhandled",
            args=(),
            exc_info=True,  # type: ignore[arg-type]
        )
        import sys

        record.exc_info = sys.exc_info()

    payload = json.loads(observability.JsonFormatter().format(record))
    assert "ValueError: upstream exploded" in payload["exception"]


def test_configure_logging_replaces_handlers_instead_of_stacking_them() -> None:
    observability.configure_logging("INFO", "json")
    observability.configure_logging("INFO", "json")

    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, observability.JsonFormatter)
    # uvicorn's own handlers are routed through ours rather than kept alongside.
    access = logging.getLogger("uvicorn.access")
    assert access.handlers == [] and access.propagate is True

    observability.configure_logging("INFO", "text")
    assert not isinstance(logging.getLogger().handlers[0].formatter, observability.JsonFormatter)

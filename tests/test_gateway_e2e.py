"""End-to-end tests through the real FastAPI app against the in-process mock."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text

from tests.conftest import set_scenario

CHAT = "/v1/chat/completions"
BODY: dict[str, Any] = {
    "model": "chaos-default",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "temperature": 0.0,
}


async def test_happy_path_returns_an_openai_shaped_response(stack: dict[str, Any]) -> None:
    response = await stack["client"].post(CHAT, json=BODY)
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"]
    assert payload["usage"]["total_tokens"] > 0
    assert response.headers["x-gateway-provider"] == "mock_primary"
    assert response.headers["x-gateway-attempts"] == "1"
    assert response.headers["x-gateway-cache"] == "miss"


async def test_unknown_route_is_a_client_error(stack: dict[str, Any]) -> None:
    response = await stack["client"].post(CHAT, json={**BODY, "model": "does-not-exist"})
    assert response.status_code == 400
    assert response.json()["error"]["kind"] == "bad_request"


async def test_malformed_payload_is_rejected_by_validation(stack: dict[str, Any]) -> None:
    response = await stack["client"].post(CHAT, json={"model": "chaos-default"})
    assert response.status_code == 422


async def test_provider_500_surfaces_as_a_typed_upstream_error(stack: dict[str, Any]) -> None:
    set_scenario(stack["mock_app"], "total_outage")
    response = await stack["client"].post(CHAT, json=BODY)
    assert response.status_code == 502
    assert response.json()["error"]["kind"] == "server_error"
    assert response.headers["x-gateway-error-kind"] == "server_error"


async def test_unparseable_body_is_classified_not_crashed(stack: dict[str, Any]) -> None:
    """A 200 with a truncated body must become a typed error, not a parser traceback."""
    set_scenario(stack["mock_app"], "bad_json")
    kinds = set()
    for _ in range(30):
        response = await stack["client"].post(CHAT, json=BODY)
        assert response.status_code in (200, 502)
        if response.status_code == 502:
            kinds.add(response.json()["error"]["kind"])
    assert kinds == {"invalid_response"}


async def test_rate_limit_is_passed_through_with_retry_after(stack: dict[str, Any]) -> None:
    set_scenario(stack["mock_app"], "rate_limited")
    statuses = set()
    for _ in range(20):
        response = await stack["client"].post(CHAT, json=BODY)
        statuses.add(response.status_code)
        if response.status_code == 429:
            assert response.headers["retry-after"] == "1"
    assert 429 in statuses


async def test_every_call_is_written_to_postgres(stack: dict[str, Any]) -> None:
    for _ in range(5):
        await stack["client"].post(CHAT, json=BODY)
    await stack["state"].recorder.drain()

    async with stack["state"].database.session() as session:
        calls = await session.scalar(text("SELECT count(*) FROM llm_calls"))
        attempts = await session.scalar(text("SELECT count(*) FROM llm_attempts"))
        cost = await session.scalar(text("SELECT sum(cost_usd) FROM llm_calls"))
    assert calls == 5
    assert attempts == 5
    assert float(cost) > 0


async def test_failed_calls_are_recorded_too(stack: dict[str, Any]) -> None:
    set_scenario(stack["mock_app"], "total_outage")
    await stack["client"].post(CHAT, json=BODY)
    await stack["state"].recorder.drain()

    async with stack["state"].database.session() as session:
        row = (
            await session.execute(text("SELECT outcome, error_kind, http_status FROM llm_calls"))
        ).one()
    assert row.outcome == "error"
    assert row.error_kind == "server_error"
    assert row.http_status == 502


async def test_usage_endpoint_reports_spend_grouped_by_provider(stack: dict[str, Any]) -> None:
    for _ in range(3):
        await stack["client"].post(CHAT, json=BODY)
    summary = (await stack["client"].get("/v1/usage?group_by=provider")).json()
    assert summary["totals"]["requests"] == 3
    assert summary["totals"]["successes"] == 3
    assert summary["totals"]["cost_usd"] > 0
    assert summary["groups"][0]["key"] == "mock_primary"
    assert summary["budget"]["limit_usd"] > 0


async def test_budget_exhaustion_refuses_with_402_and_stops_spending(
    stack: dict[str, Any],
) -> None:
    """The requirement is a refusal, not a silent bill."""
    state = stack["state"]
    state.budget._config.limit_usd = 0.0001  # noqa: SLF001
    state.budget.record_spend(0.001)

    response = await stack["client"].post(CHAT, json=BODY)
    assert response.status_code == 402
    body = response.json()["error"]
    assert body["kind"] == "budget_exceeded"
    assert body["details"]["limit_usd"] == 0.0001

    await state.recorder.drain()
    async with state.database.session() as session:
        providers = await session.scalar(
            text("SELECT count(*) FROM llm_attempts WHERE outcome = 'success'")
        )
    assert providers == 0, "a refused request must not reach a provider"


async def test_config_endpoint_exposes_the_effective_reliability_flags(
    stack: dict[str, Any],
) -> None:
    config = (await stack["client"].get("/v1/config")).json()
    assert set(config["reliability"]) == {
        "timeouts",
        "retries",
        "circuit_breaker",
        "fallback",
        "hedging",
        "cache",
    }
    assert config["default_route"] == "chaos-default"
    # No secrets in the introspection endpoint.
    assert "api_key" not in str(config)


async def test_health_and_readiness(stack: dict[str, Any]) -> None:
    assert (await stack["client"].get("/healthz")).json()["status"] == "ok"
    assert (await stack["client"].get("/readyz")).json()["status"] == "ready"


@pytest.mark.parametrize(
    ("route", "expected_dialect"),
    [("chaos-default", "mock_primary")],
)
async def test_route_selects_the_configured_provider(
    stack: dict[str, Any], route: str, expected_dialect: str
) -> None:
    response = await stack["client"].post(CHAT, json={**BODY, "model": route})
    assert response.headers["x-gateway-provider"] == expected_dialect

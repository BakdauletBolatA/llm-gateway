"""Semantic cache behaviour end to end, against a real pgvector table."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

CHAT = "/v1/chat/completions"


def body(prompt: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": "chaos-default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        **extra,
    }


async def ask(stack: dict[str, Any], prompt: str, **extra: Any) -> Any:
    response = await stack["client"].post(CHAT, json=body(prompt, **extra))
    await stack["state"].cache.drain()
    return response


async def test_the_same_question_twice_is_served_from_cache(
    cache_stack: dict[str, Any],
) -> None:
    first = await ask(cache_stack, "What is the capital of France?")
    second = await ask(cache_stack, "What is the capital of France?")

    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "hit"
    assert second.json()["choices"][0]["message"]["content"] == (
        first.json()["choices"][0]["message"]["content"]
    )
    assert float(second.headers["x-gateway-cost-usd"]) == 0.0
    assert second.headers["x-gateway-attempts"] == "0", "a hit must not call a provider"


async def test_a_paraphrase_above_the_threshold_hits(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "What is the capital of France?")
    response = await ask(cache_stack, "whats the capital of france")
    assert response.headers["x-gateway-cache"] == "hit"
    assert float(response.headers["x-gateway-cache-similarity"]) >= 0.60


async def test_a_different_question_is_a_miss(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "What is the capital of France?")
    response = await ask(cache_stack, "How do I set up a Postgres connection pool?")
    assert response.headers["x-gateway-cache"] == "miss"


async def test_a_creative_request_is_never_served_from_cache(
    cache_stack: dict[str, Any],
) -> None:
    """A caller asking for variety must not get a stored answer."""
    await ask(cache_stack, "Write a haiku about autumn rain.")
    response = await ask(cache_stack, "Write a haiku about autumn rain.", temperature=0.9)
    assert response.headers["x-gateway-cache"] == "miss"


async def test_a_client_can_opt_out(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "What is the capital of France?")
    response = await ask(cache_stack, "What is the capital of France?", cache=False)
    assert response.headers["x-gateway-cache"] == "miss"


async def test_a_hit_is_still_served_when_the_budget_is_exhausted(
    cache_stack: dict[str, Any],
) -> None:
    """A cached answer costs nothing, so refusing it would be pure loss."""
    await ask(cache_stack, "What is the capital of France?")

    budget = cache_stack["state"].budget
    budget._config.limit_usd = 0.0000001  # noqa: SLF001
    budget.record_spend(1.0)

    cached = await ask(cache_stack, "What is the capital of France?")
    assert cached.status_code == 200
    assert cached.headers["x-gateway-cache"] == "hit"

    fresh = await ask(cache_stack, "Explain what a database index is.")
    assert fresh.status_code == 402, "a miss must still be refused"


async def test_entries_are_scoped_and_counted(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "What is the capital of France?")
    await ask(cache_stack, "What is the capital of France?")

    async with cache_stack["state"].database.session() as session:
        row = (
            await session.execute(
                text("SELECT scope, hits, provider FROM semantic_cache LIMIT 1")
            )
        ).one()
    assert row.scope == "chaos-default:mock-gpt-4o-mini"
    assert row.hits >= 1
    assert row.provider == "mock_primary"


async def test_a_failed_request_is_not_cached(cache_stack: dict[str, Any]) -> None:
    from tests.conftest import set_scenario

    set_scenario(cache_stack["mock_app"], "total_outage")
    first = await ask(cache_stack, "What is the capital of France?")
    assert first.status_code == 502

    async with cache_stack["state"].database.session() as session:
        count = await session.scalar(text("SELECT count(*) FROM semantic_cache"))
    assert count == 0


async def test_reliability_state_reports_cache_statistics(
    cache_stack: dict[str, Any],
) -> None:
    await ask(cache_stack, "What is the capital of France?")
    await ask(cache_stack, "What is the capital of France?")
    state = (await cache_stack["client"].get("/v1/reliability/state")).json()
    assert state["cache"]["enabled"] is True
    assert state["cache"]["embedder"] == "hashing"
    assert state["cache"]["hits"] == 1
    assert state["cache"]["lookups"] == 2

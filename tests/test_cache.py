"""Semantic cache behaviour end to end, against a real pgvector table."""

from __future__ import annotations

import asyncio
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
    assert (
        second.json()["choices"][0]["message"]["content"]
        == (first.json()["choices"][0]["message"]["content"])
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
            await session.execute(text("SELECT scope, hits, provider FROM semantic_cache LIMIT 1"))
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


# -- sweeping expired entries --------------------------------------------------


async def test_the_sweeper_removes_only_expired_entries(cache_stack: dict[str, Any]) -> None:
    """The TTL filters expired rows out of a lookup; nothing removed them, so the
    table grew forever and the search slowed down with it (3.8 ms at zero expired
    rows against 12.8 ms at 20 000, measured on this stand)."""
    cache = cache_stack["state"].cache
    database = cache_stack["state"].database

    await cache.store(
        scope="sweep-test",
        prompt="этот ответ ещё живой",
        response_text="живой",
        provider="mock_primary",
        model="m",
        tokens_in=1,
        tokens_out=1,
        cost_usd=0.0,
    )
    async with database.session() as session:
        for index in range(5):
            await session.execute(
                text(
                    "INSERT INTO semantic_cache (scope, prompt_hash, prompt_text, embedding,"
                    " response_text, provider, model, tokens_in, tokens_out, cost_usd,"
                    " created_at, expires_at, hits) VALUES ('sweep-test', :h, 'dead', :v,"
                    " 'dead', 'mock_primary', 'm', 1, 1, 0, now(), now() - interval '1 second', 0)"
                ),
                {"h": f"dead{index}", "v": str([0.01 * index] * 256)},
            )
        await session.commit()

    removed = await cache.sweep()
    assert removed == 5, f"expected the five dead rows to go, {removed} went"

    async with database.session() as session:
        left = await session.scalar(
            text("SELECT count(*) FROM semantic_cache WHERE scope = 'sweep-test'")
        )
    assert left == 1, "the live entry must survive its own sweep"
    assert cache.stats()["swept"] == 5


async def test_a_second_sweep_finds_nothing_to_do(cache_stack: dict[str, Any]) -> None:
    cache = cache_stack["state"].cache
    await cache.sweep()
    assert await cache.sweep() == 0


async def test_the_sweeper_runs_in_the_background_and_stops_cleanly(
    cache_stack: dict[str, Any],
) -> None:
    """It is started from the app lifespan; a task that outlives shutdown would
    keep a database connection open after the pool is closed."""
    cache = cache_stack["state"].cache
    cache.config = cache.config.model_copy(update={"sweep_interval_s": 0.05})

    cache.start_sweeper()
    assert cache.stats()["sweeping"] is True
    await asyncio.sleep(0.15)

    await cache.stop_sweeper()
    assert cache.stats()["sweeping"] is False


async def test_the_sweeper_is_off_when_the_interval_is_zero(cache_stack: dict[str, Any]) -> None:
    """An operator has to be able to turn it off — and then the metric stays flat
    at zero, which is exactly what says the table is only growing."""
    cache = cache_stack["state"].cache
    cache.config = cache.config.model_copy(update={"sweep_interval_s": 0.0})
    cache.start_sweeper()
    assert cache.stats()["sweeping"] is False

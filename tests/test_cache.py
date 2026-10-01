"""Response cache behaviour end to end, against a real pgvector table.

The cache is the one mechanism in the gateway that can return a *wrong* answer
rather than a slow or failed one, so most of these tests pin what must never be
served: an answer to a different question, another tenant's answer, an answer
written for a different system prompt or conversation, or anything at all to a
client that did not ask for caching.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from llm_gateway.cache.embedder import HashingEmbedder, cosine_similarity
from llm_gateway.cache.store import CacheKey
from llm_gateway.settings import CacheConfig
from tests.conftest import CACHE_TENANT_KEYS

CHAT = "/v1/chat/completions"
SYSTEM_BANK = "You are a support assistant for Acme Bank. Answer concisely and politely."

#: Different questions that share most of their words. The hashing embedder scores
#: every pair above the 0.60 threshold the benchmark calibrated on its own workload.
COLLIDING_PAIRS = [
    ("What is the capital of France?", "What is the capital of Spain?"),
    ("Is it safe to take ibuprofen with alcohol?", "Is it safe to take ibuprofen without alcohol?"),
    ("Переведи на английский: я люблю кошек", "Переведи на английский: я ненавижу кошек"),
]


def body(prompt: str, *, system: str | None = None, **extra: Any) -> dict[str, Any]:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return {"model": "chaos-default", "messages": messages, "temperature": 0.0, **extra}


async def ask(
    stack: dict[str, Any],
    prompt: str,
    *,
    tenant: str | None = None,
    opt_in: bool | None = True,
    **extra: Any,
) -> Any:
    payload = body(prompt, **extra)
    if opt_in is not None:
        payload["cache"] = opt_in
    headers = {"authorization": f"Bearer {CACHE_TENANT_KEYS[tenant]}"} if tenant else {}
    response = await stack["client"].post(CHAT, json=payload, headers=headers)
    await stack["state"].cache.drain()
    return response


def served_from_cache(response: Any) -> bool:
    return bool(response.headers.get("x-gateway-cache") == "hit")


# -- configuration ---------------------------------------------------------------


def test_the_hashing_embedder_really_does_collide() -> None:
    """Why semantic matching on it is refused: these are different questions."""
    embedder = HashingEmbedder(256)
    for left, right in COLLIDING_PAIRS:
        similarity = cosine_similarity(embedder.embed_sync(left), embedder.embed_sync(right))
        assert similarity >= 0.60, f"{left!r} / {right!r} scored {similarity:.3f}"


def test_semantic_matching_on_the_hashing_embedder_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="allow_lexical_semantic"):
        CacheConfig(enabled=True, match="semantic", embedder="hashing")


def test_the_benchmark_can_still_measure_the_lexical_matcher_explicitly() -> None:
    config = CacheConfig(
        enabled=True, match="semantic", embedder="hashing", allow_lexical_semantic=True
    )
    assert config.match == "semantic"


def test_the_defaults_are_the_safe_ones() -> None:
    config = CacheConfig(enabled=True)
    assert config.match == "exact"
    assert config.require_opt_in is True


# -- exact matching (the default) ------------------------------------------------


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


async def test_case_punctuation_and_spacing_do_not_matter(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "What is the capital of France?")
    response = await ask(cache_stack, "  what is the CAPITAL of france  ")
    assert served_from_cache(response)


@pytest.mark.parametrize(("stored", "asked"), COLLIDING_PAIRS)
async def test_a_different_question_sharing_most_words_is_a_miss(
    cache_stack: dict[str, Any], stored: str, asked: str
) -> None:
    await ask(cache_stack, stored)
    assert not served_from_cache(await ask(cache_stack, asked))


async def test_a_paraphrase_is_a_miss_under_exact_matching(cache_stack: dict[str, Any]) -> None:
    """The price of never serving a wrong answer with a lexical embedder."""
    await ask(cache_stack, "What is the capital of France?")
    assert not served_from_cache(await ask(cache_stack, "whats the capital of france"))


# -- who may be served -----------------------------------------------------------


async def test_a_client_that_did_not_ask_is_never_served(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "What is the capital of France?")
    response = await ask(cache_stack, "What is the capital of France?", opt_in=None)
    assert not served_from_cache(response)


async def test_a_client_that_did_not_ask_is_not_stored_either(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "What is the capital of France?", opt_in=None)
    async with cache_stack["state"].database.session() as session:
        assert await session.scalar(text("SELECT count(*) FROM semantic_cache")) == 0


async def test_a_client_can_opt_out(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "What is the capital of France?")
    response = await ask(cache_stack, "What is the capital of France?", opt_in=False)
    assert not served_from_cache(response)


async def test_a_creative_request_is_never_served_from_cache(
    cache_stack: dict[str, Any],
) -> None:
    """A caller asking for variety must not get a stored answer."""
    await ask(cache_stack, "Write a haiku about autumn rain.")
    response = await ask(cache_stack, "Write a haiku about autumn rain.", temperature=0.9)
    assert not served_from_cache(response)


async def test_one_tenant_is_never_served_another_tenants_answer(
    tenant_cache_stack: dict[str, Any],
) -> None:
    await ask(tenant_cache_stack, "What is my account balance?", tenant="tenant-a")
    other = await ask(tenant_cache_stack, "What is my account balance?", tenant="tenant-b")
    same = await ask(tenant_cache_stack, "What is my account balance?", tenant="tenant-a")

    assert not served_from_cache(other), "tenant-b was served tenant-a's cached answer"
    assert served_from_cache(same)


async def test_a_different_system_prompt_is_a_different_scope(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "How do I reset my password?", system=SYSTEM_BANK)
    elsewhere = await ask(
        cache_stack, "How do I reset my password?", system="You are a support assistant for Globex."
    )
    again = await ask(cache_stack, "How do I reset my password?", system=SYSTEM_BANK)
    assert not served_from_cache(elsewhere)
    assert served_from_cache(again)


async def test_the_same_follow_up_in_a_different_conversation_is_a_miss(
    cache_stack: dict[str, Any],
) -> None:
    """'And what about the second one?' means nothing without the turns before it."""

    def dialogue(first_topic: str) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": f"Tell me about {first_topic}."},
            {"role": "assistant", "content": f"Here is an overview of {first_topic}."},
            {"role": "user", "content": "And what are the risks?"},
        ]

    client = cache_stack["client"]
    for topic, expect_hit in (
        ("index funds", False),
        ("crypto staking", False),
        ("index funds", True),
    ):
        response = await client.post(
            CHAT,
            json={
                "model": "chaos-default",
                "messages": dialogue(topic),
                "temperature": 0.0,
                "cache": True,
            },
        )
        await cache_stack["state"].cache.drain()
        assert served_from_cache(response) is expect_hit, topic


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"max_tokens": 16}, {"max_tokens": 512}),
        ({"max_tokens": 16}, {}),
        ({"stop": ["\n"]}, {"stop": ["."]}),
        ({"stop": ["\n"]}, {}),
        ({"temperature": 0.0}, {"temperature": 0.2}),
    ],
)
async def test_generation_parameters_that_change_the_answer_are_part_of_the_scope(
    cache_stack: dict[str, Any], first: dict[str, Any], second: dict[str, Any]
) -> None:
    """A 16-token answer is not the answer to the same question asked for 512."""
    prompt = "Explain how a hash map handles collisions."
    await ask(cache_stack, prompt, **first)
    other = await ask(cache_stack, prompt, **second)
    again = await ask(cache_stack, prompt, **first)
    assert not served_from_cache(other), f"{first} was served to a request with {second}"
    assert served_from_cache(again)


def test_the_key_scopes_the_context_and_matches_only_the_question() -> None:
    from llm_gateway.cache.store import SemanticCache

    base = {"route": "r", "model": "m", "tenant": "t", "query": "q"}
    one = SemanticCache.key_for(context="system: a", **base)
    two = SemanticCache.key_for(context="system: b", **base)
    assert one.scope != two.scope
    assert one.query == two.query == "q"
    assert ":-:none:" in SemanticCache.key_for(context="", **{**base, "tenant": None}).scope
    assert (
        SemanticCache.key_for(context="", params={"max_tokens": 8}, **base).scope
        != SemanticCache.key_for(context="", params={"max_tokens": 9}, **base).scope
    )


def test_an_oversized_scope_still_fits_the_column() -> None:
    from llm_gateway.cache.store import MAX_SCOPE_LENGTH, SemanticCache

    key = SemanticCache.key_for(route="r" * 90, model="m" * 120, tenant="t", context="c", query="q")
    assert len(key.scope) <= MAX_SCOPE_LENGTH


# -- the lexical matcher the benchmark measured ------------------------------------


async def test_the_lexical_matcher_hits_on_a_paraphrase(
    lexical_cache_stack: dict[str, Any],
) -> None:
    await ask(lexical_cache_stack, "What is the capital of France?")
    response = await ask(lexical_cache_stack, "whats the capital of france")
    assert served_from_cache(response)
    assert float(response.headers["x-gateway-cache-similarity"]) >= 0.60


async def test_the_lexical_matcher_also_hits_on_a_different_question(
    lexical_cache_stack: dict[str, Any],
) -> None:
    """Pinned on purpose: this is the defect that keeps it out of production."""
    await ask(lexical_cache_stack, "What is the capital of France?")
    assert served_from_cache(await ask(lexical_cache_stack, "What is the capital of Spain?"))


# -- budget, errors and statistics -----------------------------------------------


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
    assert served_from_cache(cached)

    fresh = await ask(cache_stack, "Explain what a database index is.")
    assert fresh.status_code == 402, "a miss must still be refused"


async def test_entries_are_scoped_and_counted(cache_stack: dict[str, Any]) -> None:
    await ask(cache_stack, "What is the capital of France?")
    await ask(cache_stack, "What is the capital of France?")

    async with cache_stack["state"].database.session() as session:
        row = (
            await session.execute(text("SELECT scope, hits, provider FROM semantic_cache LIMIT 1"))
        ).one()
    assert row.scope.startswith("chaos-default:mock-gpt-4o-mini:-:none:")
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
    assert state["cache"]["match"] == "exact"
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
        key=CacheKey(scope="sweep-test", query="этот ответ ещё живой"),
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

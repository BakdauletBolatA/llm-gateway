from __future__ import annotations

from llm_gateway.cache.embedder import HashingEmbedder, cosine_similarity, normalise

EMBEDDER = HashingEmbedder(dim=256)


def similarity(left: str, right: str) -> float:
    return cosine_similarity(EMBEDDER.embed_sync(left), EMBEDDER.embed_sync(right))


def test_normalisation_strips_punctuation_and_case() -> None:
    assert normalise("What's  the CAPITAL of France?") == "what s the capital of france"


def test_embedding_is_deterministic_across_calls() -> None:
    assert EMBEDDER.embed_sync("hello world") == EMBEDDER.embed_sync("hello world")


def test_embedding_is_unit_length() -> None:
    vector = EMBEDDER.embed_sync("what is the capital of France")
    assert abs(sum(component**2 for component in vector) ** 0.5 - 1.0) < 1e-9


def test_empty_input_does_not_explode() -> None:
    assert EMBEDDER.embed_sync("   ") == [0.0] * 256


def test_identical_text_is_maximally_similar() -> None:
    assert similarity("What is the capital of France?", "What is the capital of France?") > 0.999


def test_punctuation_and_case_differences_stay_above_the_configured_threshold() -> None:
    # config/gateway.yaml uses 0.60 (see scripts/calibrate_cache_threshold.py).
    assert similarity("What is the capital of France?", "whats the capital of france") > 0.60


def test_unrelated_questions_are_far_apart() -> None:
    score = similarity(
        "What is the capital of France?",
        "How do I set up a Postgres connection pool?",
    )
    assert score < 0.3, f"unrelated prompts must not collide, got {score}"


def test_configured_threshold_keeps_a_margin_over_the_worst_cross_topic_pair() -> None:
    """The cache must never answer a different question.

    This is the guard rail behind the threshold in config/gateway.yaml: if a
    change to the embedder brings two different questions closer together, this
    test fails before the cache starts serving wrong answers.
    """
    from itertools import combinations

    from chaos.workload import TOPICS
    from llm_gateway.settings import load_settings

    threshold = load_settings("config/gateway.yaml").reliability.cache.similarity_threshold
    vectors = [[EMBEDDER.embed_sync(variant) for variant in topic] for topic in TOPICS]
    worst_cross_topic = max(
        cosine_similarity(a, b)
        for (i, left), (j, right) in combinations(enumerate(vectors), 2)
        if i != j
        for a in left
        for b in right
    )
    assert worst_cross_topic < threshold, (
        f"two different questions score {worst_cross_topic:.3f}, "
        f"at or above the cache threshold {threshold}"
    )


def test_same_topic_different_phrasing_is_closer_than_a_different_topic() -> None:
    same_topic = similarity(
        "Explain what a database index is.",
        "Can you explain database indexes?",
    )
    other_topic = similarity(
        "Explain what a database index is.",
        "Write a haiku about autumn rain.",
    )
    assert same_topic > other_topic + 0.3

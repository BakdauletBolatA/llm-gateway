"""The real embedder: skipped unless `pip install -e .[embeddings]` and the model is cached."""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from llm_gateway.cache.embedder import SentenceTransformerEmbedder, cosine_similarity


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    try:
        return SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=384)
    except OSError as error:  # model not downloaded and no network
        pytest.skip(f"embedding model unavailable: {error}")


async def test_vectors_have_the_configured_width_and_unit_length(
    embedder: SentenceTransformerEmbedder,
) -> None:
    vector = await embedder.embed("How do I reset my password?")
    assert len(vector) == 384
    assert abs(sum(v * v for v in vector) ** 0.5 - 1.0) < 1e-4


async def test_a_paraphrase_is_closer_than_a_different_question_with_the_same_words(
    embedder: SentenceTransformerEmbedder,
) -> None:
    base = await embedder.embed("How do I reset my password?")
    paraphrase = await embedder.embed("I forgot my password, how can I change it?")
    different = await embedder.embed("How do I close my account?")
    assert cosine_similarity(base, paraphrase) > cosine_similarity(base, different)


async def test_a_wrong_configured_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match="dim"):
        SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=256)

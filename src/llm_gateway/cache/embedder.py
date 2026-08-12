"""Embedders for the semantic cache.

The default is a deterministic local hashing vectoriser: no model weights, no
network call, identical vectors in CI and in production, and it works with zero
paid keys. It captures lexical and paraphrase-level similarity ("What is the
capital of France?" vs "whats the capital of france") — not deep semantics.
That limitation is deliberate and is stated in RELIABILITY.md next to the cache
numbers, because a cache hit rate is meaningless without knowing what counts as
"similar".

`Embedder` is a protocol, and `OllamaEmbedder` is a drop-in that uses a real
embedding model when Ollama is running.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
CHAR_NGRAM = 4
WORD_WEIGHT = 2.0
BIGRAM_WEIGHT = 1.5
CHAR_WEIGHT = 1.0


class Embedder(Protocol):
    dim: int
    name: str

    async def embed(self, text: str) -> list[float]: ...


def normalise(text: str) -> str:
    lowered = text.lower()
    stripped = _NON_WORD.sub(" ", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()


def _bucket_and_sign(feature: str, dim: int) -> tuple[int, float]:
    # blake2b rather than hash(): Python's hash() is salted per process, which
    # would make cached vectors incomparable across restarts.
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder:
    """Signed hashing vectoriser over word unigrams, bigrams and character n-grams."""

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed_sync(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        normalised = normalise(text)
        if not normalised:
            return vector

        words = normalised.split(" ")
        for word in words:
            bucket, sign = _bucket_and_sign(f"w:{word}", self.dim)
            vector[bucket] += sign * WORD_WEIGHT
        for left, right in zip(words, words[1:], strict=False):
            bucket, sign = _bucket_and_sign(f"b:{left}_{right}", self.dim)
            vector[bucket] += sign * BIGRAM_WEIGHT

        padded = f" {normalised} "
        for index in range(len(padded) - CHAR_NGRAM + 1):
            gram = padded[index : index + CHAR_NGRAM]
            bucket, sign = _bucket_and_sign(f"c:{gram}", self.dim)
            vector[bucket] += sign * CHAR_WEIGHT

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]

    async def embed(self, text: str) -> list[float]:
        return self.embed_sync(text)


class OllamaEmbedder:
    """Real embeddings via a local Ollama model. Requires `--profile ollama`."""

    name = "ollama"

    def __init__(self, base_url: str, model: str, dim: int, client: httpx.AsyncClient) -> None:
        self.dim = dim
        self.model = model
        self._url = f"{base_url.rstrip('/')}/api/embeddings"
        self._client = client

    async def embed(self, text: str) -> list[float]:
        response = await self._client.post(
            self._url, json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        payload = response.json()
        vector = [float(value) for value in payload["embedding"]]
        if len(vector) != self.dim:
            raise ValueError(
                f"embedding model {self.model} returned dim={len(vector)}, "
                f"config expects {self.dim}"
            )
        norm = math.sqrt(sum(component * component for component in vector))
        return [component / norm for component in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)

"""Semantic cache backed by pgvector.

Lookup is a cosine-distance nearest-neighbour search inside a scope
(route + model), filtered by TTL, with an explicit similarity threshold — the
nearest neighbour is only a hit if it is close enough.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from llm_gateway.cache.embedder import Embedder
from llm_gateway.db.models import SemanticCacheEntry
from llm_gateway.db.session import Database
from llm_gateway.settings import CacheConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CacheHit:
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    similarity: float
    entry_id: int


class SemanticCache:
    def __init__(self, config: CacheConfig, embedder: Embedder, database: Database) -> None:
        self.config = config
        self.embedder = embedder
        self._db = database
        self._pending: set[asyncio.Task[None]] = set()
        self.lookups = 0
        self.hits = 0
        self.stores = 0
        self.errors = 0

    @staticmethod
    def scope_for(route: str, model: str) -> str:
        return f"{route}:{model}"

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:48]

    def enabled_for(self, temperature: float | None, client_opt_in: bool) -> bool:
        if not self.config.enabled or not client_opt_in:
            return False
        # A high temperature means the caller wants variety; serving a stored answer
        # would silently break that expectation.
        return (temperature or 0.0) <= self.config.max_temperature

    async def lookup(self, scope: str, prompt: str) -> CacheHit | None:
        self.lookups += 1
        try:
            vector = await self.embedder.embed(prompt)
            async with self._db.session() as session:
                distance = SemanticCacheEntry.embedding.cosine_distance(vector)
                rows = await session.execute(
                    select(SemanticCacheEntry, distance.label("distance"))
                    .where(
                        SemanticCacheEntry.scope == scope,
                        SemanticCacheEntry.expires_at > datetime.now(UTC),
                    )
                    .order_by(distance)
                    .limit(self.config.candidate_limit)
                )
                best = rows.first()
                if best is None:
                    return None
                entry, raw_distance = best
                similarity = 1.0 - float(raw_distance)
                if similarity < self.config.similarity_threshold:
                    return None

            # The hit counter is a metric, not part of the answer. Incrementing it
            # inline takes a row lock on the most popular entry — with a Zipf-shaped
            # workload that is exactly the row every concurrent request wants.
            self._spawn(self._bump_hits(entry.id))
            self.hits += 1
            return CacheHit(
                text=entry.response_text,
                provider=entry.provider,
                model=entry.model,
                tokens_in=entry.tokens_in,
                tokens_out=entry.tokens_out,
                cost_usd=float(entry.cost_usd),
                similarity=round(similarity, 6),
                entry_id=entry.id,
            )
        except Exception:
            # A broken cache must degrade to a cache miss, never to a failed request.
            self.errors += 1
            logger.exception("semantic cache lookup failed for scope=%s", scope)
            return None

    async def store(
        self,
        *,
        scope: str,
        prompt: str,
        response_text: str,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        try:
            vector = await self.embedder.embed(prompt)
            expires_at = datetime.now(UTC) + timedelta(seconds=self.config.ttl_s)
            prompt_hash = self._hash(prompt)
            async with self._db.session() as session:
                existing = await session.scalar(
                    select(SemanticCacheEntry.id).where(
                        SemanticCacheEntry.scope == scope,
                        SemanticCacheEntry.prompt_hash == prompt_hash,
                    )
                )
                if existing is not None:
                    await session.execute(
                        update(SemanticCacheEntry)
                        .where(SemanticCacheEntry.id == existing)
                        .values(
                            response_text=response_text,
                            expires_at=expires_at,
                            cost_usd=cost_usd,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                        )
                    )
                else:
                    session.add(
                        SemanticCacheEntry(
                            scope=scope,
                            prompt_hash=prompt_hash,
                            prompt_text=prompt[:8000],
                            embedding=vector,
                            response_text=response_text,
                            provider=provider,
                            model=model,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            cost_usd=cost_usd,
                            expires_at=expires_at,
                        )
                    )
                await session.commit()
            self.stores += 1
        except Exception:
            self.errors += 1
            logger.exception("semantic cache store failed for scope=%s", scope)

    async def _bump_hits(self, entry_id: int) -> None:
        async with self._db.session() as session:
            await session.execute(
                update(SemanticCacheEntry)
                .where(SemanticCacheEntry.id == entry_id)
                .values(hits=SemanticCacheEntry.hits + 1)
            )
            await session.commit()

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    def store_later(self, **kwargs: Any) -> None:
        """Write to the cache off the request's critical path.

        A client should not wait for a cache whose whole purpose is to make
        things faster. The trade-off is a short window in which a repeated
        prompt still misses because the first answer has not landed yet.
        """
        self._spawn(self.store(**kwargs))

    async def drain(self, timeout: float = 5.0) -> None:
        """Wait for in-flight writes — used on shutdown and between chaos runs."""
        pending = set(self._pending)
        if pending:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait(pending, timeout=timeout)

    def stats(self) -> dict[str, Any]:
        hit_rate = self.hits / self.lookups if self.lookups else 0.0
        return {
            "enabled": self.config.enabled,
            "embedder": self.embedder.name,
            "lookups": self.lookups,
            "hits": self.hits,
            "stores": self.stores,
            "errors": self.errors,
            "hit_rate": round(hit_rate, 4),
        }

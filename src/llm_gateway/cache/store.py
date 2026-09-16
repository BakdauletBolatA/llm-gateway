"""Response cache backed by pgvector.

Every entry lives in a scope: route, model, tenant and a hash of the conversation
context (system prompt and earlier turns). Inside a scope the last user message is
matched either exactly, after normalising case, punctuation and whitespace, or
semantically, as a cosine-distance nearest neighbour above an explicit threshold.

What goes into the scope is the correctness argument. Without the tenant, one
customer's answer is served to another; without the context, the same question
under a different system prompt — or halfway through a different conversation —
gets an answer written for someone else.

The TTL only filters expired rows out of the result; it does not remove them, so
they have to be swept. Left alone they are pure cost — nothing may ever be served
from them, and the search still has to walk past them. Measured on this stand with
an exact-match lookup: 3.8 ms at zero expired rows against 12.8 ms at 20 000, with
the table 29 MB larger. At a 15-minute TTL that is well under an hour of traffic.
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

from sqlalchemy import delete, select, update

from llm_gateway.cache.embedder import Embedder, normalise
from llm_gateway.db.models import SemanticCacheEntry
from llm_gateway.db.session import Database
from llm_gateway.settings import CacheConfig

logger = logging.getLogger(__name__)


#: The ``scope`` column is String(160); longer scopes are replaced by a digest.
MAX_SCOPE_LENGTH = 160
PUBLIC_TENANT = "-"


@dataclass(frozen=True, slots=True)
class CacheKey:
    scope: str
    query: str


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
        self._sweeper: asyncio.Task[None] | None = None
        self.lookups = 0
        self.hits = 0
        self.stores = 0
        self.errors = 0
        self.swept = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:48]

    @classmethod
    def key_for(
        cls,
        *,
        route: str,
        model: str,
        tenant: str | None,
        context: str,
        query: str,
    ) -> CacheKey:
        """Scope everything the answer depends on; match only the question itself."""
        context_digest = cls._hash(normalise(context))[:16] if context else "none"
        scope = f"{route}:{model}:{tenant or PUBLIC_TENANT}:{context_digest}"
        if len(scope) > MAX_SCOPE_LENGTH:
            scope = f"{route[:40]}:{cls._hash(scope)}"
        return CacheKey(scope=scope, query=query)

    def enabled_for(self, temperature: float | None, requested: bool | None) -> bool:
        if not self.config.enabled or requested is False:
            return False
        if requested is None and self.config.require_opt_in:
            return False
        # A high temperature means the caller wants variety; serving a stored answer
        # would silently break that expectation.
        return (temperature or 0.0) <= self.config.max_temperature

    async def lookup(self, key: CacheKey) -> CacheHit | None:
        self.lookups += 1
        try:
            async with self._db.session() as session:
                if self.config.match == "exact":
                    entry = await session.scalar(
                        select(SemanticCacheEntry).where(
                            SemanticCacheEntry.scope == key.scope,
                            SemanticCacheEntry.prompt_hash == self._hash(normalise(key.query)),
                            SemanticCacheEntry.expires_at > datetime.now(UTC),
                        )
                    )
                    if entry is None:
                        return None
                    similarity = 1.0
                else:
                    vector = await self.embedder.embed(key.query)
                    distance = SemanticCacheEntry.embedding.cosine_distance(vector)
                    rows = await session.execute(
                        select(SemanticCacheEntry, distance.label("distance"))
                        .where(
                            SemanticCacheEntry.scope == key.scope,
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
            logger.exception("semantic cache lookup failed for scope=%s", key.scope)
            return None

    async def store(
        self,
        *,
        key: CacheKey,
        response_text: str,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
    ) -> None:
        try:
            scope, prompt = key.scope, key.query
            vector = await self.embedder.embed(prompt)
            expires_at = datetime.now(UTC) + timedelta(seconds=self.config.ttl_s)
            # Normalised, so the exact matcher and the duplicate check below agree on
            # what "the same question" is.
            prompt_hash = self._hash(normalise(prompt))
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
            logger.exception("semantic cache store failed for scope=%s", key.scope)

    async def _bump_hits(self, entry_id: int) -> None:
        async with self._db.session() as session:
            await session.execute(
                update(SemanticCacheEntry)
                .where(SemanticCacheEntry.id == entry_id)
                .values(hits=SemanticCacheEntry.hits + 1)
            )
            await session.commit()

    async def sweep(self) -> int:
        """Delete every expired entry. Returns how many rows went."""
        async with self._db.session() as session:
            result = await session.execute(
                delete(SemanticCacheEntry).where(SemanticCacheEntry.expires_at <= datetime.now(UTC))
            )
            await session.commit()
        removed = int(getattr(result, "rowcount", 0) or 0)
        self.swept += removed
        return removed

    async def _sweep_forever(self) -> None:
        interval = self.config.sweep_interval_s
        while True:
            await asyncio.sleep(interval)
            try:
                removed = await self.sweep()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A sweep that cannot run is a slow leak, not an outage: lookups
                # still filter by TTL, so nothing stale is ever served.
                self.errors += 1
                logger.exception("semantic cache sweep failed")
            else:
                if removed:
                    logger.info("swept %d expired cache entries", removed)

    def start_sweeper(self) -> None:
        """Start the background sweep. Called once, from the app lifespan."""
        if not self.config.enabled or self.config.sweep_interval_s <= 0:
            return
        if self._sweeper is None or self._sweeper.done():
            self._sweeper = asyncio.create_task(self._sweep_forever())

    async def stop_sweeper(self) -> None:
        if self._sweeper is None:
            return
        self._sweeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._sweeper
        self._sweeper = None

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
            "match": self.config.match,
            "embedder": self.embedder.name,
            "lookups": self.lookups,
            "hits": self.hits,
            "stores": self.stores,
            "errors": self.errors,
            "hit_rate": round(hit_rate, 4),
            "swept": self.swept,
            "sweeping": self._sweeper is not None and not self._sweeper.done(),
        }

"""What a cache costs when nothing ever deletes its expired entries.

A TTL only filters expired rows out of a lookup; it is not obliged to remove them,
and without a sweep the table grows forever. The question this measures is what
that costs the search.

The lookup here is an exact match — the case the cache actually exists for — so
the answer is always findable and only the price of finding it is measured.

    python scripts/cache_sweep_probe.py               # no sweeping
    python scripts/cache_sweep_probe.py --sweep       # swept before each step

Writes bench/probes/cache_sweep_{off,on}.json, which RELIABILITY.md quotes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import text

from llm_gateway.cache.embedder import HashingEmbedder
from llm_gateway.cache.store import SemanticCache
from llm_gateway.db.session import Database
from llm_gateway.settings import CacheConfig, DatabaseConfig

DEFAULT_DSN = "postgresql+asyncpg://gateway:gateway@127.0.0.1:5432/llm_gateway"
SCOPE = "cache-sweep-probe"
PROMPT = "как обновить зависимость в проекте на питоне"
LADDER = (0, 1000, 5000, 20000)

_INSERT_DEAD = text(
    "INSERT INTO semantic_cache (scope, prompt_hash, prompt_text, embedding, response_text,"
    " provider, model, tokens_in, tokens_out, cost_usd, created_at, expires_at, hits)"
    " VALUES (:scope, :hash, 'dead', :vector, 'dead', 'mock_primary', 'm', 1, 1, 0,"
    " now(), :expires, 0)"
)


async def measure(dsn: str, sweep: bool, repeats: int) -> dict[str, Any]:
    database = Database(DatabaseConfig(dsn=dsn, run_migrations_on_startup=False))
    config = CacheConfig(enabled=True, similarity_threshold=0.90, ttl_s=900, candidate_limit=5)
    embedder = HashingEmbedder(config.embedding_dim)
    cache = SemanticCache(config, embedder, database)
    expired_at = datetime.now(UTC) - timedelta(seconds=1)

    async with database.session() as session:
        await session.execute(text("DELETE FROM semantic_cache WHERE scope = :s"), {"s": SCOPE})
        await session.commit()
    await cache.store(
        scope=SCOPE,
        prompt=PROMPT,
        response_text="живой ответ",
        provider="mock_primary",
        model="m",
        tokens_in=10,
        tokens_out=20,
        cost_usd=0.001,
    )

    steps: list[dict[str, Any]] = []
    inserted = 0
    for target in LADDER:
        async with database.session() as session:
            for index in range(inserted, target):
                vector = await embedder.embed(f"{PROMPT} вариант {index}")
                await session.execute(
                    _INSERT_DEAD,
                    {
                        "scope": SCOPE,
                        "hash": f"dead{index:06d}",
                        "vector": str(vector),
                        "expires": expired_at,
                    },
                )
            await session.commit()
        inserted = target

        removed = await cache.sweep() if sweep else 0
        async with database.session() as session:
            await session.execute(text("ANALYZE semantic_cache"))

        timings: list[float] = []
        served = False
        for _ in range(repeats):
            started = time.perf_counter()
            hit = await cache.lookup(SCOPE, PROMPT)
            timings.append((time.perf_counter() - started) * 1000)
            served = bool(hit and hit.text == "живой ответ")

        async with database.session() as session:
            table_bytes = await session.scalar(
                text("SELECT pg_total_relation_size('semantic_cache')")
            )
        steps.append(
            {
                "expired_rows_inserted": target,
                "swept": removed,
                "lookup_ms_p50": round(statistics.median(timings), 1),
                "lookup_ms_min": round(min(timings), 1),
                "table_bytes": int(table_bytes or 0),
                "answer_served": served,
            }
        )

    async with database.session() as session:
        await session.execute(text("DELETE FROM semantic_cache WHERE scope = :s"), {"s": SCOPE})
        await session.commit()
    await database.aclose()

    return {
        "sweep": sweep,
        "repeats_per_step": repeats,
        "candidate_limit": config.candidate_limit,
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="cache_sweep_probe")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--sweep", action="store_true", help="sweep before measuring")
    parser.add_argument("--repeats", type=int, default=25, help="lookups per rung")
    parser.add_argument("--out", default="bench/probes")
    args = parser.parse_args()

    result = asyncio.run(measure(args.dsn, args.sweep, args.repeats))
    path = Path(args.out) / f"cache_sweep_{'on' if args.sweep else 'off'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

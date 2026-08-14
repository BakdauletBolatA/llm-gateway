"""How many requests two gateway replicas let through with a limit of X rps each.

Answers one question: does the limit belong to the process or to the deployment?
Load is dealt to the replicas in turn and only response codes are counted — the
client's point of view, with no faith in the gateway's own counters.

    python scripts/shared_limit_probe.py \\
        --gateway http://127.0.0.1:8080 --gateway http://127.0.0.1:8082 \\
        --n 200 --concurrency 20

Expected: with `rate_limit.scope: local` two replicas let through about twice the
configured burst; with `shared`, exactly the configured burst between them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from typing import Any

import httpx

PAYLOAD: dict[str, Any] = {
    "model": "chaos-default",
    "messages": [{"role": "user", "content": "shared limit probe"}],
    # A high temperature disables the cache: a cache hit spends no provider token but
    # does spend a rate-limit token, and mixing the two effects in one measurement
    # would only muddy it.
    "temperature": 0.9,
    "max_tokens": 64,
}


async def _fire(
    client: httpx.AsyncClient,
    gateway: str,
    queue: asyncio.Queue[int],
    tally: Counter[str],
    per_gateway: Counter[str],
) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            response = await client.post(f"{gateway}/v1/chat/completions", json=PAYLOAD)
            status = str(response.status_code)
        except Exception as exc:  # pragma: no cover - diagnostic script
            status = type(exc).__name__
        tally[status] += 1
        per_gateway[f"{gateway} {status}"] += 1
        queue.task_done()


async def main_async(args: argparse.Namespace) -> int:
    queues: list[asyncio.Queue[int]] = []
    share = args.n // len(args.gateway)
    for _ in args.gateway:
        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in range(share):
            queue.put_nowait(index)
        queues.append(queue)

    tally: Counter[str] = Counter()
    per_gateway: Counter[str] = Counter()
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        workers = [
            asyncio.create_task(_fire(client, gateway, queues[index], tally, per_gateway))
            for index, gateway in enumerate(args.gateway)
            for _ in range(args.concurrency)
        ]
        await asyncio.gather(*workers)
    duration_s = time.perf_counter() - started

    served = tally.get("200", 0)
    throttled = tally.get("429", 0)
    summary = {
        "gateways": args.gateway,
        "requests": sum(tally.values()),
        "served_200": served,
        "throttled_429": throttled,
        "other": {code: count for code, count in tally.items() if code not in {"200", "429"}},
        "duration_s": round(duration_s, 2),
        "offered_rps": round(sum(tally.values()) / duration_s, 1) if duration_s else 0.0,
        "served_rps": round(served / duration_s, 1) if duration_s else 0.0,
        "per_gateway": dict(sorted(per_gateway.items())),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="shared_limit_probe")
    parser.add_argument("--gateway", action="append", required=True, help="repeatable")
    parser.add_argument("--n", type=int, default=200, help="requests across all replicas")
    parser.add_argument("--concurrency", type=int, default=20, help="workers per replica")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

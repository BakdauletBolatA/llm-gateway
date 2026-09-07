"""How many calls reach a dead provider when there is more than one replica.

A circuit breaker is a counter in one process, like the limits in iterations 10
and 11 — but the conclusion here may well be the opposite, which is why it is
measured. Every replica has to learn about the outage on its own, so every replica
pays for that lesson separately. The question is what the lesson costs, and whether
it costs enough to justify moving breaker state into shared storage.

The client load is identical in both runs (same request count, same concurrency);
only the number of gateway processes changes. The counting is done by the mock,
not by the gateway: `served` on its side is literally how many times something
knocked on an upstream that could not answer.

    scripts/dev_stack.sh up
    RUN_DIR=.run/replica-b GATEWAY_PORT=8082 scripts/dev_stack.sh up-gateway
    python scripts/breaker_replica_probe.py \\
        --gateway http://127.0.0.1:8080 --gateway http://127.0.0.1:8082
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

PAYLOAD: dict[str, Any] = {
    "model": "chaos-default",
    "messages": [{"role": "user", "content": "breaker replica probe"}],
    # Temperature turns the cache off: a cache hit never knocks on a provider, and
    # knocking is exactly what is being counted here.
    "temperature": 0.9,
    "max_tokens": 64,
}


async def _fire(
    client: httpx.AsyncClient,
    gateway: str,
    queue: asyncio.Queue[int],
    tally: Counter[str],
) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        try:
            response = await client.post(f"{gateway}/v1/chat/completions", json=PAYLOAD)
            tally[str(response.status_code)] += 1
        except Exception as exc:  # pragma: no cover - diagnostic script
            tally[type(exc).__name__] += 1
        queue.task_done()


async def measure(args: argparse.Namespace) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as admin:
        await admin.post(f"{args.mock}/admin/scenario", json={"scenario": args.scenario})
        await admin.post(f"{args.mock}/admin/reset")
        # Every replica starts naive, which is exactly what a deploy does.
        for gateway in args.gateway:
            await admin.post(f"{gateway}/v1/reliability/reset?cache=true")

    queues: list[asyncio.Queue[int]] = []
    share = args.n // len(args.gateway)
    for _ in args.gateway:
        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in range(share):
            queue.put_nowait(index)
        queues.append(queue)

    tally: Counter[str] = Counter()
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        workers = [
            asyncio.create_task(_fire(client, gateway, queues[index], tally))
            for index, gateway in enumerate(args.gateway)
            for _ in range(args.concurrency)
        ]
        await asyncio.gather(*workers)
    duration_s = time.perf_counter() - started

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as admin:
        mock_state = (await admin.get(f"{args.mock}/admin/state")).json()
        breakers = []
        for gateway in args.gateway:
            state = (await admin.get(f"{gateway}/v1/reliability/state")).json()
            breakers.append(
                {
                    "gateway": gateway,
                    "circuit_breakers": [
                        {
                            "provider": snapshot["provider"],
                            "state": snapshot["state"],
                            "failure_ratio": snapshot["failure_ratio"],
                        }
                        for snapshot in state["circuit_breakers"]
                    ],
                }
            )

    served = {
        str(upstream["upstream"]): int(upstream["served"]) for upstream in mock_state["upstreams"]
    }
    requests = sum(tally.values())
    return {
        "scenario": args.scenario,
        "replicas": len(args.gateway),
        "gateways": args.gateway,
        "requests": requests,
        "answered_200": tally.get("200", 0),
        "status_codes": dict(sorted(tally.items())),
        "duration_s": round(duration_s, 2),
        # The number this probe exists for: knocks on an upstream that could not
        # have answered any of them.
        "served_by_upstream": served,
        "wasted_calls_to_dead_upstream": served.get(args.dead, 0),
        "wasted_per_request": round(served.get(args.dead, 0) / requests, 3) if requests else 0.0,
        "breakers": breakers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="breaker_replica_probe")
    parser.add_argument("--gateway", action="append", required=True, help="repeatable")
    parser.add_argument("--mock", default="http://127.0.0.1:8081")
    parser.add_argument("--scenario", default="primary_outage")
    parser.add_argument("--dead", default="primary", help="the mock upstream that is down")
    parser.add_argument("--n", type=int, default=200, help="requests across all replicas")
    parser.add_argument("--concurrency", type=int, default=10, help="workers per replica")
    parser.add_argument("--out", default=None, help="where to write the JSON")
    args = parser.parse_args()

    result = asyncio.run(measure(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

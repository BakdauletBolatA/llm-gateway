"""What two gateway replicas let through when each one holds the same limit.

Answers one question, for both limits the gateway enforces: does the limit belong
to the process or to the deployment? Load is dealt to the replicas in turn and
response codes are counted from the client's side, with no faith in the gateway's
own counters; the money actually recorded is read back from `/v1/usage` afterwards,
which is the number the rate limit cannot fake.

    python scripts/shared_limit_probe.py \\
        --gateway http://127.0.0.1:8080 --gateway http://127.0.0.1:8082 \\
        --n 200 --concurrency 20

Expected with `scope: local` — the rate limit lets through about twice the
configured burst, and the budget overspends by about a factor of the replica count.
With `scope: shared` — one burst and one limit between them.
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


async def _usage(gateway: str, refusals: int) -> dict[str, Any]:
    """What the gateway itself recorded: cost from llm_calls, and the budget view.

    Read from a single replica on purpose — the recorded spend is a property of the
    period in the database, not of the process that happened to answer.
    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            body = (await client.get(f"{gateway}/v1/usage")).json()
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"error": type(exc).__name__}
    totals = body.get("totals", {})
    budget = body.get("budget", {})
    return {
        "cost_usd_recorded": totals.get("cost_usd"),
        "served_recorded": totals.get("successes"),
        "budget_scope": budget.get("scope"),
        "budget_limit_usd": budget.get("limit_usd"),
        "budget_spent_usd": budget.get("spent_usd"),
        # Overspend is measured against the money actually billed (the sum over
        # llm_calls), not against the gateway's own budget counter: the counter is
        # what is under test, so trusting it would beg the question. It is only
        # reported when the budget was the binding constraint — a run that never
        # saw a 402 was limited by something else, and "spent 0.02% of the limit"
        # would read like a result instead of an accident of configuration.
        "overspend_pct": (
            round((totals["cost_usd"] / budget["limit_usd"] - 1) * 100, 1)
            if refusals and budget.get("limit_usd") and totals.get("cost_usd") is not None
            else None
        ),
    }


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
    usage = await _usage(args.gateway[0], tally.get("402", 0))
    summary = {
        "gateways": args.gateway,
        "requests": sum(tally.values()),
        "served_200": served,
        "throttled_429": throttled,
        "other": {code: count for code, count in tally.items() if code not in {"200", "429"}},
        "duration_s": round(duration_s, 2),
        "offered_rps": round(sum(tally.values()) / duration_s, 1) if duration_s else 0.0,
        "served_rps": round(served / duration_s, 1) if duration_s else 0.0,
        "refused_402": tally.get("402", 0),
        "per_gateway": dict(sorted(per_gateway.items())),
        "usage": usage,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="shared_limit_probe")
    parser.add_argument("--gateway", action="append", required=True, help="repeatable")
    parser.add_argument("--n", type=int, default=200, help="requests across all replicas")
    parser.add_argument("--concurrency", type=int, default=20, help="workers per replica")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="max_tokens in the payload; 0 omits it, so the budget falls back to the "
        "configured (deliberately conservative) estimate_output_tokens",
    )
    args = parser.parse_args()
    if args.max_tokens:
        PAYLOAD["max_tokens"] = args.max_tokens
    else:
        PAYLOAD.pop("max_tokens", None)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

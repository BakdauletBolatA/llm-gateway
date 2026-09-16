"""Chaos harness.

Drives N requests through the gateway at a fixed concurrency while the mock
provider runs a given failure scenario, and writes one JSON result per run.

Everything the report needs is captured at measurement time — including the
gateway's effective reliability config — so a result file is self-describing and
does not depend on the repository being at the same revision later.

    python -m chaos.run --label 01_baseline --all
    python -m chaos.run --label 03_retries --scenario storm --n 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from chaos.metrics import Observation, RunAggregate
from chaos.workload import build_workload

DEFAULT_GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8080")
DEFAULT_MOCK = os.environ.get("MOCK_ADMIN_URL", "http://127.0.0.1:8081")
DEFAULT_SCENARIOS = [
    "healthy",
    "rate_limited",
    "flaky_500",
    "bad_json",
    "conn_reset",
    "slow",
    "hang",
    "storm",
    "capacity_limited",
    "primary_outage",
    "total_outage",
]


def _int_header(response: httpx.Response, name: str) -> int:
    try:
        return int(response.headers.get(name, "0"))
    except ValueError:
        return 0


def _float_header(response: httpx.Response, name: str) -> float:
    try:
        return float(response.headers.get(name, "0"))
    except ValueError:
        return 0.0


def observe(response: httpx.Response, latency_ms: float) -> Observation:
    status = response.status_code
    return Observation(
        latency_ms=latency_ms,
        status=status,
        outcome="success" if status == 200 else f"http_{status}",
        error_kind=response.headers.get("x-gateway-error-kind"),
        attempts=_int_header(response, "x-gateway-attempts"),
        retries=_int_header(response, "x-gateway-retries"),
        fallbacks=_int_header(response, "x-gateway-fallbacks"),
        breaker_skips=_int_header(response, "x-gateway-breaker-skips"),
        hedges=_int_header(response, "x-gateway-hedges"),
        cache_hit=response.headers.get("x-gateway-cache") == "hit",
        cost_usd=_float_header(response, "x-gateway-cost-usd"),
        wasted_cost_usd=_float_header(response, "x-gateway-cost-wasted-usd"),
        provider=response.headers.get("x-gateway-provider"),
    )


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    route: str,
    queue: asyncio.Queue[str],
    aggregate: RunAggregate,
) -> None:
    while True:
        try:
            prompt = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        payload = {
            "model": route,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.0,
            # The harness is a client that asks for caching: the gateway serves
            # nobody from the cache without an explicit opt-in.
            "cache": True,
        }
        started = time.perf_counter()
        try:
            response = await client.post(url, json=payload)
            latency_ms = (time.perf_counter() - started) * 1000
            aggregate.add(observe(response, latency_ms))
        except httpx.TimeoutException:
            # The client gave up before the gateway answered. For the naive
            # baseline this is the dominant failure mode and the whole point.
            latency_ms = (time.perf_counter() - started) * 1000
            aggregate.add(
                Observation(
                    latency_ms=latency_ms,
                    status=None,
                    outcome="client_timeout",
                    error_kind="client_timeout",
                )
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            aggregate.add(
                Observation(
                    latency_ms=latency_ms,
                    status=None,
                    outcome="transport_error",
                    error_kind=type(exc).__name__,
                )
            )
        finally:
            queue.task_done()


async def _post(client: httpx.AsyncClient, url: str, payload: dict[str, Any] | None = None) -> Any:
    response = await client.post(url, json=payload or {})
    response.raise_for_status()
    return response.json()


async def _get(client: httpx.AsyncClient, url: str) -> Any:
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


async def run_scenario(
    *,
    scenario: str,
    label: str,
    count: int,
    concurrency: int,
    timeout_s: float,
    route: str,
    gateway: str,
    mock: str,
    seed: int,
    note: str | None,
    api_key: str | None = None,
) -> dict[str, Any]:
    prompts, workload_stats = build_workload(count, seed=seed)
    # One header set for every client here: the workload, the setup calls and the
    # teardown calls all talk to the same gateway, and /v1/usage has always
    # required a key when auth is on.
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), headers=headers) as admin:
        await _post(admin, f"{mock}/admin/scenario", {"scenario": scenario})
        await _post(admin, f"{mock}/admin/reset")
        # Fresh breakers and an empty cache: every scenario starts from the same
        # state, otherwise the previous scenario's warm cache flatters this one.
        await _post(admin, f"{gateway}/v1/reliability/reset?cache=true")
        config = await _get(admin, f"{gateway}/v1/config")
        usage_before = await _get(admin, f"{gateway}/v1/usage")

    queue: asyncio.Queue[str] = asyncio.Queue()
    for prompt in prompts:
        queue.put_nowait(prompt)

    aggregate = RunAggregate()
    url = f"{gateway}/v1/chat/completions"
    started_wall = datetime.now(UTC)
    started = time.perf_counter()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_s),
        limits=httpx.Limits(max_connections=concurrency * 2),
        headers=headers,
    ) as client:
        workers = [
            asyncio.create_task(_worker(client, url, route, queue, aggregate))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)
    duration_s = time.perf_counter() - started

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), headers=headers) as admin:
        usage_after = await _get(admin, f"{gateway}/v1/usage")
        reliability = await _get(admin, f"{gateway}/v1/reliability/state")
        mock_state = await _get(admin, f"{mock}/admin/state")

    summary = aggregate.summarise()
    summary["throughput_rps"] = round(summary["requests"] / duration_s, 2) if duration_s else 0.0
    summary["cost_usd_server"] = round(
        float(usage_after["totals"]["cost_usd"]) - float(usage_before["totals"]["cost_usd"]), 6
    )

    return {
        "label": label,
        "scenario": scenario,
        "route": route,
        "note": note,
        "started_at": started_wall.isoformat(),
        "duration_s": round(duration_s, 2),
        "concurrency": concurrency,
        "client_timeout_s": timeout_s,
        "workload": {
            "seed": seed,
            **{
                key: getattr(workload_stats, key)
                for key in (
                    "requests",
                    "unique_prompts",
                    "unique_topics",
                    "duplicate_rate",
                    "paraphrase_rate",
                )
            },
        },
        "results": summary,
        "gateway_config": config,
        "gateway_reliability_state": reliability,
        "mock_state": mock_state,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
    }


def _write(result: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['label']}__{result['scenario']}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _print_line(result: dict[str, Any]) -> None:
    summary = result["results"]
    print(
        f"  {result['scenario']:<16} "
        f"success={summary['success_rate']:>7.1%}  "
        f"p50={summary['latency_ms']['p50']:>7.0f}ms  "
        f"p95={summary['latency_ms']['p95']:>8.0f}ms  "
        f"retries={summary['retries_total']:>4}  "
        f"fallbacks={summary['fallbacks_total']:>4}  "
        f"hedges={summary['hedges_total']:>4}  "
        f"cache={summary['cache_hits']:>4}  "
        f"${summary['cost_usd_server']:.4f}  "
        f"({result['duration_s']}s)"
    )


async def main_async(args: argparse.Namespace) -> int:
    scenarios = DEFAULT_SCENARIOS if args.all else args.scenario
    if not scenarios:
        print("nothing to run: pass --scenario NAME (repeatable) or --all")
        return 2

    print(
        f"chaos run label={args.label} scenarios={len(scenarios)} n={args.n} c={args.concurrency}"
    )
    out_dir = Path(args.out)
    for index, scenario in enumerate(scenarios):
        result = await run_scenario(
            scenario=scenario,
            label=args.label,
            count=args.n,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
            route=args.route,
            gateway=args.gateway,
            mock=args.mock,
            seed=args.seed,
            note=args.note,
            api_key=args.api_key,
        )
        _write(result, out_dir)
        _print_line(result)
        if index < len(scenarios) - 1 and args.settle > 0:
            # Let requests still in flight on the mock (hangs) finish before the
            # next scenario starts, so runs do not contaminate each other.
            await asyncio.sleep(args.settle)
    print(f"results written to {out_dir}/{args.label}__*.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chaos.run", description="Chaos harness for llm-gateway")
    parser.add_argument("--label", required=True, help="iteration label, e.g. 01_baseline")
    parser.add_argument(
        "--scenario", action="append", default=[], help="scenario name (repeatable)"
    )
    parser.add_argument("--all", action="store_true", help="run every default scenario")
    parser.add_argument("--n", type=int, default=150, help="requests per scenario")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=20.0, help="hard client-side cap, seconds")
    parser.add_argument("--route", default="chaos-default")
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY)
    parser.add_argument("--mock", default=DEFAULT_MOCK)
    parser.add_argument("--out", default="bench/results")
    parser.add_argument("--seed", type=int, default=42, help="workload seed")
    parser.add_argument("--settle", type=float, default=3.0, help="pause between scenarios")
    parser.add_argument("--note", default=None, help="free-form note stored in the result")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GATEWAY_API_KEY"),
        help="bearer token, when the gateway under test has auth enabled "
        "(default: $GATEWAY_API_KEY)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

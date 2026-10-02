#!/usr/bin/env python
"""Run a load-test scenario against the gateway and save a machine-readable report.

    python loadtest/run.py --target mock --scenario steady
    python loadtest/run.py --target live --scenario kill

Scenarios:
    steady  constant closed-loop load for --duration seconds
    kill    the same load, and halfway through the primary backend is killed:
            `docker compose stop ollama` for the live target, the mock's primary
            upstream switched to a total outage for the mock target. The gateway
            must fall back to the second hop.

Targets use routes of the same shape (two hops, same reliability overlay), so the
mock and live reports are comparable: the difference between them is the backend.

Writes reports/loadtest_<target>_<scenario>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarise import aggregate_runs, summarise  # noqa: E402

REPORTS = ROOT / "reports"
ROUTES = {"mock": "mock-two-hop", "live": "live-local"}
#: The provider each target's kill stops, as the gateway names it in X-Gateway-Provider.
KILLED_PROVIDER = {"mock": "mock_primary", "live": "ollama"}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def wait_until_up(url: str, seconds: int = 120) -> None:
    """The previous kill run restarts a server; give it time to come back."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/api/tags", timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise SystemExit(f"{url} did not come back within {seconds} s")


def warm_live_models(urls: list[str], model: str) -> None:
    """Load the model on every server first, so a cold start is not billed to latency."""
    for url in urls:
        wait_until_up(url)
        response = httpx.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 4},
            },
            timeout=300,
        )
        response.raise_for_status()


def kill_backend(target: str, mock_url: str, notes: dict[str, Any]) -> None:
    notes["killed_at_wall"] = datetime.now(UTC).isoformat()
    if target == "live":
        subprocess.run(["docker", "compose", "stop", "ollama"], cwd=ROOT, check=True)
    else:
        httpx.post(
            f"{mock_url}/admin/profile",
            json={"upstream": "primary", "profile": "total_outage"},
            timeout=10,
        ).raise_for_status()


def restore_backend(target: str, mock_url: str) -> None:
    if target == "live":
        subprocess.run(["docker", "compose", "start", "ollama"], cwd=ROOT, check=False)
    else:
        httpx.post(f"{mock_url}/admin/scenario", json={"scenario": "healthy"}, timeout=10)


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    route = ROUTES[args.target]
    kill_at_s = args.duration * args.kill_at if args.scenario == "kill" else None
    notes: dict[str, Any] = {}

    with httpx.Client(timeout=30) as http:
        http.post(f"{args.gateway}/v1/reliability/reset?cache=true").raise_for_status()
        if args.target == "mock":
            http.post(
                f"{args.mock}/admin/scenario", json={"scenario": "healthy"}
            ).raise_for_status()
            http.post(f"{args.mock}/admin/reset").raise_for_status()
    if args.target == "live":
        print(f"warming {args.model} on {len(args.ollama)} servers...", flush=True)
        warm_live_models(args.ollama, args.model)

    raw = Path(tempfile.mkdtemp()) / "raw.jsonl"
    env = {
        **os.environ,
        "LT_ROUTE": route,
        "LT_MAX_TOKENS": str(args.max_tokens),
        "LT_RAW": str(raw),
    }
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(ROOT / "loadtest" / "locustfile.py"),
        "--headless",
        "--only-summary",
        "--host",
        args.gateway,
        "-u",
        str(args.users),
        "-r",
        str(args.users),
        "-t",
        f"{args.duration}s",
    ]
    timer: threading.Timer | None = None
    started_wall = datetime.now(UTC)
    started = time.time()
    if kill_at_s is not None:
        timer = threading.Timer(kill_at_s, kill_backend, (args.target, args.mock, notes))
        timer.start()
    try:
        # Locust exits 1 when any request failed. That is a result, not a crash.
        completed = subprocess.run(command, env=env, cwd=ROOT, check=False)
        if completed.returncode not in (0, 1):
            raise SystemExit(f"locust failed with exit code {completed.returncode}")
    finally:
        if timer is not None:
            timer.cancel()
            restore_backend(args.target, args.mock)
    duration_s = time.time() - started

    records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    summary = summarise(
        records,
        duration_s=duration_s,
        kill_at_s=kill_at_s,
        killed_provider=KILLED_PROVIDER[args.target],
    )
    report = {
        "target": args.target,
        "scenario": args.scenario,
        "route": route,
        "backend": args.model if args.target == "live" else "mock provider",
        "users": args.users,
        "max_tokens": args.max_tokens,
        "duration_s": round(duration_s, 1),
        "kill_at_s": kill_at_s,
        "started_at": started_wall.isoformat(),
        "git_sha": git_sha(),
        "environment": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "notes": notes,
        "results": summary,
    }
    return report


def _warn_if_unproven(results: dict[str, Any], prefix: str = "") -> None:
    note = results.get("failover", {}).get("note")
    if note:
        print(f"\nWARNING {prefix}{note}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target", choices=["mock", "live"], required=True)
    parser.add_argument("--scenario", choices=["steady", "kill"], required=True)
    parser.add_argument("--users", type=int, default=4)
    parser.add_argument("--duration", type=int, default=90, help="seconds")
    parser.add_argument("--kill-at", type=float, default=0.5, help="fraction of the run")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gateway", default=os.environ.get("GATEWAY_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--mock", default=os.environ.get("MOCK_URL", "http://127.0.0.1:8081"))
    parser.add_argument(
        "--ollama",
        nargs="+",
        default=["http://127.0.0.1:11434", "http://127.0.0.1:11435"],
        help="Ollama servers to warm before a live run",
    )
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b"))
    parser.add_argument("--out", type=Path, default=REPORTS)
    parser.add_argument(
        "--repeat", type=int, default=1, help="run the scenario N times and report the spread"
    )
    parser.add_argument(
        "--pause", type=int, default=20, help="seconds to let the machine settle between runs"
    )
    args = parser.parse_args()

    return _main(args)


def _main(args: argparse.Namespace) -> int:
    args.out.mkdir(exist_ok=True)
    if args.repeat == 1:
        report = run_once(args)
        path = args.out / f"loadtest_{args.target}_{args.scenario}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["results"], indent=2))
        _warn_if_unproven(report["results"])
    else:
        reports = []
        for index in range(args.repeat):
            if index:
                time.sleep(args.pause)
            load_before = os.getloadavg()[0]
            print(f"run {index + 1}/{args.repeat}; host load average {load_before:.1f}", flush=True)
            report = run_once(args)
            report["host_load_average_1m_before"] = round(load_before, 2)
            reports.append(report)
        first = reports[0]
        report = {
            "target": first["target"],
            "scenario": first["scenario"],
            "route": first["route"],
            "backend": first["backend"],
            "users": first["users"],
            "max_tokens": first["max_tokens"],
            "repeats": len(reports),
            "pause_s": args.pause,
            "git_sha": first["git_sha"],
            "environment": first["environment"],
            "aggregate": aggregate_runs([r["results"] for r in reports]),
            "runs": [
                {
                    "started_at": r["started_at"],
                    "duration_s": r["duration_s"],
                    "host_load_average_1m_before": r["host_load_average_1m_before"],
                    "results": r["results"],
                }
                for r in reports
            ],
        }
        path = args.out / f"loadtest_{args.target}_{args.scenario}_repeats.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["aggregate"], indent=2))
        for index, run in enumerate(reports, 1):
            _warn_if_unproven(run["results"], f"run {index}: ")
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

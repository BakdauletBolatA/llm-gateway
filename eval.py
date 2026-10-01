#!/usr/bin/env python
"""Run every measurement in this repository and rebuild the results table in README.md.

    python eval.py                      # cache + routing (replayed) + load test on the mock
    python eval.py --only cache         # one measurement
    python eval.py --live               # also re-measure on live local models (slow)
    python eval.py --check              # only verify that README.md matches reports/

What each step needs:

    cache     the `embeddings` extra:            pip install -e ".[embeddings,loadtest]"
    routing   nothing, it replays the recorded answers in reports/. With --live it
              calls the models again, which needs the ollama profile (see README).
    load      a running stack. `--start-stack` runs `docker compose up -d --build`
              with the overlay the load test is defined against.

Numbers measured on the mock provider are labelled as such in the table; numbers
measured on a live model are measured on *your* hardware when you pass --live, and
will differ from the committed ones.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OVERLAY = "config/extras/live_local.yaml"
GATEWAY = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8080")

TUNING_NOTE_V1 = (
    "Rules v1 (as first run). Criteria for prompts 36 and 48 corrected after the first run; "
    "the original scoring gave always_large 47/50, routed 41/50."
)
TUNING_NOTE_V2 = (
    "Rules v2: percent sign, currency and quantity questions fixed after seeing v1 fail on "
    "this same set, and criteria 36/48 corrected. Optimistic, not held out; see "
    "routing_heldout.json."
)
HELDOUT_NOTE = (
    "Held-out: 30 prompts and criteria committed (49ab48b) before this run; router rules "
    "frozen at a13b860, not modified afterwards."
)


def run(title: str, command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n=== {title}\n$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, **(env or {})}, check=False)
    if result.returncode != 0:
        raise SystemExit(f"step failed: {title} (exit code {result.returncode})")


def gateway_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{GATEWAY}/readyz", timeout=3) as response:
            return bool(response.status == 200)
    except Exception:
        return False


def start_stack(live: bool) -> None:
    env = {"GATEWAY_CONFIG_OVERLAY": OVERLAY, "OLLAMA_ENABLED": "true" if live else "false"}
    command = ["docker", "compose"]
    if live:
        command += ["--profile", "ollama"]
    run("start the stack", [*command, "up", "-d", "--build"], env)
    for _ in range(90):
        if gateway_ready():
            return
        time.sleep(2)
    raise SystemExit(f"the gateway did not become ready at {GATEWAY}")


def cache() -> None:
    run("cache: all-MiniLM-L6-v2", [PY, "eval/cache_eval.py"])
    run("cache: hash n-gram baseline", [PY, "eval/cache_eval.py", "--embedder", "hashing"])


def routing(live: bool) -> None:
    heldout = "eval/data/routing_prompts_heldout.jsonl"
    answers = "reports/routing_answers.jsonl"
    held_answers = "reports/routing_heldout_answers.jsonl"
    script = [PY, "eval/routing_eval.py"]
    if live:
        run(
            "routing: make sure the large model is pulled",
            [
                "docker",
                "compose",
                "--profile",
                "ollama",
                "exec",
                "-T",
                "ollama",
                "ollama",
                "pull",
                os.environ.get("OLLAMA_LARGE_MODEL", "qwen2.5:3b"),
            ],
        )
        run(
            "routing: live answers, tuning set",
            [*script, "--out", "reports/routing_eval_rules_v2.json", "--note", TUNING_NOTE_V2],
        )
        run(
            "routing: live answers, held-out set",
            [
                *script,
                "--prompts",
                heldout,
                "--answers-out",
                held_answers,
                "--out",
                "reports/routing_heldout.json",
                "--note",
                HELDOUT_NOTE,
            ],
        )
        print(
            "\nnote: reports/routing_eval.json (rules as first run) is a frozen record "
            "and is not regenerated from new answers."
        )
        return
    run(
        "routing: replay, rules as first run",
        [
            *script,
            "--rescore",
            answers,
            "--out",
            "reports/routing_eval.json",
            "--note",
            TUNING_NOTE_V1,
        ],
    )
    run(
        "routing: replay, current rules, tuning set",
        [
            *script,
            "--rescore",
            answers,
            "--redecide",
            "--out",
            "reports/routing_eval_rules_v2.json",
            "--note",
            TUNING_NOTE_V2,
        ],
    )
    run(
        "routing: replay, current rules, held-out set",
        [
            *script,
            "--prompts",
            heldout,
            "--rescore",
            held_answers,
            "--redecide",
            "--out",
            "reports/routing_heldout.json",
            "--note",
            HELDOUT_NOTE,
        ],
    )


def load(live: bool, duration: int) -> None:
    if not gateway_ready():
        raise SystemExit(
            f"no gateway at {GATEWAY}. Start one with `python eval.py --start-stack`, or run "
            "`docker compose up -d --build` with GATEWAY_CONFIG_OVERLAY=" + OVERLAY
        )
    targets = ["mock", "live"] if live else ["mock"]
    for target in targets:
        for scenario in ("steady", "kill"):
            run(
                f"load test: {target} / {scenario}",
                [
                    PY,
                    "loadtest/run.py",
                    "--target",
                    target,
                    "--scenario",
                    scenario,
                    "--duration",
                    str(duration),
                ],
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=["cache", "routing", "load"])
    parser.add_argument("--live", action="store_true", help="also measure live local models")
    parser.add_argument("--start-stack", action="store_true", help="docker compose up first")
    parser.add_argument("--duration", type=int, default=60, help="seconds per load scenario")
    parser.add_argument("--check", action="store_true", help="only verify README.md is current")
    args = parser.parse_args()

    if args.check:
        run("check the README results block", [PY, "eval/readme_table.py", "--check"])
        return 0

    steps = args.only or ["cache", "routing", "load"]
    if args.start_stack:
        start_stack(args.live)
    if "cache" in steps:
        cache()
    if "routing" in steps:
        routing(args.live)
    if "load" in steps:
        load(args.live, args.duration)

    run("rebuild the README results block", [PY, "eval/readme_table.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

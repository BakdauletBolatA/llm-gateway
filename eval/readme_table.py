#!/usr/bin/env python
"""Build the results tables in README.md from the reports on disk.

    python eval/readme_table.py            # rewrite the block between the markers
    python eval/readme_table.py --check    # exit 1 if README.md is stale

Every number in the block comes from a JSON file written by a script in this
repository; nothing here is typed by hand. A row names the backend it was measured
on, because a mock number and a live-model number are not comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORTS = ROOT / "reports"
BENCH = ROOT / "bench" / "results"
BEGIN = "<!-- results:start -->"
END = "<!-- results:end -->"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def ms(value: float) -> str:
    return f"{value:,.0f} ms"


def chaos_table() -> str:
    def runs(label: str) -> list[dict[str, Any]]:
        return [load(p)["results"] for p in sorted(BENCH.glob(f"{label}__*.json"))]

    def scenario(label: str, name: str) -> dict[str, Any]:
        return load(BENCH / f"{label}__{name}.json")["results"]

    naive, final = "01_baseline", "09_backpressure"
    rows = []
    for label, title in ((naive, "naive gateway"), (final, "final build")):
        results = runs(label)
        requests = sum(r["requests"] for r in results)
        ok = sum(r["successes"] for r in results)
        full = sum(r["success_rate"] >= 0.9999 for r in results)
        rows.append(
            f"| {title} | {pct(ok / requests)} | {full} of {len(results)} | "
            f"{ms(scenario(label, 'storm')['latency_ms']['p95'])} | "
            f"{ms(scenario(label, 'hang')['latency_ms']['p95'])} | "
            f"${sum(r['cost_usd_server'] for r in results):.4f} |"
        )
    return "\n".join(
        [
            "| build (mock provider) | success, all scenarios | scenarios at 100% | "
            "p95, `storm` | p95, `hang` | cost, all requests |",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    )


def cache_table() -> str:
    rows = []
    for name, label in (
        ("hashing", "hash n-gram vectors (the old matcher)"),
        ("sentence-transformers", "all-MiniLM-L6-v2 (local, CPU)"),
    ):
        path = REPORTS / f"cache_eval_{name}.json"
        if not path.exists():
            continue
        report = load(path)
        five = next(r for r in report["recommendations"] if r["max_false_hit_rate"] == 0.05)
        worst = max(report["highest_similarity_of_a_different_pair"].values())
        if five["threshold"] is None:
            tail = "none qualifies | - | -"
        else:
            held = five["held_out"]
            tail = (
                f"{five['threshold']:.2f} | {pct(held['hit_rate'])} | "
                f"{held['false_hits']} of {report['pairs']['different'] // 2}"
            )
        rows.append(f"| {label} | {worst:.3f} | {tail} |")
    return "\n".join(
        [
            "| embedder | highest similarity of a *different* pair | threshold at 5% false hits "
            "| held-out hit rate | held-out false hits |",
            "|---|---|---|---|---|",
            *rows,
        ]
    )


def load_table() -> str:
    rows = []
    for target, backend in (("mock", "mock provider"), ("live", "live qwen2.5:0.5b, CPU")):
        for scenario in ("steady", "kill"):
            path = REPORTS / f"loadtest_{target}_{scenario}.json"
            if not path.exists():
                continue
            report = load(path)
            r = report["results"]
            lat = r["latency_ms"]
            extra = ""
            if scenario == "kill":
                f = r["failover"]
                extra = (
                    f"{f['failed_requests_after_kill']} failed after the kill; first answer from "
                    f"the other server after {f['seconds_to_first_success_from_other_provider']} s"
                )
            rows.append(
                f"| {backend} | {scenario} | {report['users']} | {r['requests']} | "
                f"{r['requests_per_s']} | {ms(lat['p50'])} | {ms(lat['p95'])} | "
                f"{ms(lat['p99'])} | {pct(r['success_rate'])} | {extra} |"
            )
    return "\n".join(
        [
            "| backend | scenario | users | requests | req/s | p50 | p95 | p99 | success "
            "| failover |",
            "|---|---|---|---|---|---|---|---|---|---|",
            *rows,
        ]
    )


def routing_table() -> str:
    sets = (
        ("routing_eval.json", "50 prompts, rules as first run"),
        ("routing_eval_rules_v2.json", "50 prompts, rules tuned on this set"),
        ("routing_heldout.json", "30 held-out prompts, rules frozen"),
    )
    rows = []
    for filename, title in sets:
        path = REPORTS / filename
        if not path.exists():
            continue
        report = load(path)
        n = report["prompts"]
        p = report["policies"]

        def cell(name: str, p: dict[str, Any] = p, n: int = n) -> str:
            return f"{p[name]['passed']}/{n}"

        rows.append(
            f"| {title} | {cell('always_large')} | {cell('always_small')} | {cell('routed')} | "
            f"${p['always_large']['modeled_cost_usd_per_100']:.4f} | "
            f"${p['routed']['modeled_cost_usd_per_100']:.4f} | "
            f"{pct(p['routed']['share_to_small'])} | "
            f"{pct(report['router_vs_labels']['accuracy'])} |"
        )
    return "\n".join(
        [
            "| set (live qwen2.5:0.5b and 3b) | correct, always large | correct, always small "
            "| correct, routed | modeled $/100, always large | modeled $/100, routed "
            "| sent to small | router vs hand labels |",
            "|---|---|---|---|---|---|---|---|",
            *rows,
        ]
    )


def render() -> str:
    return "\n".join(
        [
            BEGIN,
            "",
            "**Reliability under injected failures** — mock provider, 11 failure profiles "
            "([`chaos/run.py`](src/chaos/run.py), full report in "
            "[RELIABILITY.md](RELIABILITY.md)):",
            "",
            chaos_table(),
            "",
            "**Semantic cache** — 121 labelled query pairs, hand-written "
            "([`eval/cache_eval.py`](eval/cache_eval.py)). The threshold is chosen on half the "
            "pairs and scored on the other half:",
            "",
            cache_table(),
            "",
            "**Load test** — closed loop, `max_tokens` 64 "
            "([`loadtest/run.py`](loadtest/run.py)). *kill* stops the primary backend halfway:",
            "",
            load_table(),
            "",
            "**Routing by complexity** — correctness is a programmatic check, cost is modeled "
            "from measured tokens at gpt-4o-mini / gpt-4o prices "
            "([`eval/routing_eval.py`](eval/routing_eval.py)):",
            "",
            routing_table(),
            "",
            END,
        ]
    )


def update(text: str) -> str:
    start, end = text.index(BEGIN), text.index(END) + len(END)
    return text[:start] + render() + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = README.read_text(encoding="utf-8")
    updated = update(current)
    if args.check:
        if updated != current:
            print("README.md results block is stale: run python eval/readme_table.py")
            return 1
        return 0
    README.write_text(updated, encoding="utf-8")
    print("README.md results block rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())

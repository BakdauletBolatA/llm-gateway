#!/usr/bin/env python
"""Does routing by complexity save money without losing answers?

Sends 50 labelled prompts through the gateway twice, once to the small model and
once to the large one, scores every answer against a programmatic criterion, and
then replays three policies over those recorded answers:

    always_small   every request to the small model
    always_large   every request to the large model (the baseline)
    routed         the complexity router chooses per request (src/llm_gateway/complexity.py)

Replaying recorded answers, rather than calling the model a third time, means the
three policies are compared on identical outputs: any difference is the routing
decision, not sampling noise (temperature is 0, but local inference is not
bit-for-bit repeatable).

    python eval/routing_eval.py                      # needs the live stack, see README
    python eval/routing_eval.py --rescore reports/routing_answers.jsonl

What the numbers mean, and do not:
- Quality is "the answer contains the right fact, number or code construct". It is
  not "the answer is good". Code is checked statically and never executed.
- Cost is modeled: tokens measured on the local models, priced at the hosted
  reference prices in config/gateway.yaml (gpt-4o-mini for small, gpt-4o for large).
  Running locally costs $0, so a measured dollar figure would say nothing.
- The simple/complex labels are the author's judgement. Edit eval/data/routing_prompts.jsonl
  and re-run to apply yours.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_gateway.complexity import classify  # noqa: E402
from llm_gateway.schemas import ChatCompletionRequest  # noqa: E402
from llm_gateway.settings import load_settings  # noqa: E402

PROMPTS = ROOT / "eval" / "data" / "routing_prompts.jsonl"
REPORTS = ROOT / "reports"
SYSTEM = "Answer concisely."
REFERENCE_MODELS = {"small": "gpt-4o-mini", "large": "gpt-4o"}
_ANSWER = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)")


def passes(check: dict[str, Any], text: str) -> bool:
    lowered = text.lower()
    if "any" in check and not any(word.lower() in lowered for word in check["any"]):
        return False
    if "all" in check and not all(word.lower() in lowered for word in check["all"]):
        return False
    if "none" in check and any(word.lower() in lowered for word in check["none"]):
        return False
    if "regex" in check and not re.search(check["regex"], text):
        return False
    if "number" in check:
        found = _ANSWER.findall(text)
        if not found:
            return False
        try:
            value = float(found[-1].replace(",", ""))
        except ValueError:
            return False
        if not math.isclose(value, float(check["number"]), abs_tol=1e-6):
            return False
    return True


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denominator = 1 + z * z / n
    return (centre - spread) / denominator, (centre + spread) / denominator


def _cost(side: dict[str, Any], price: tuple[float, float]) -> float:
    return (side["tokens_in"] * price[0] + side["tokens_out"] * price[1]) / 1e6


def evaluate(
    records: list[dict[str, Any]], prices: dict[str, tuple[float, float]], requests: int
) -> dict[str, Any]:
    n = len(records)
    chooser = {
        "always_small": lambda r: "small",
        "always_large": lambda r: "large",
        "routed": lambda r: r["decision"],
    }
    policies: dict[str, Any] = {}
    for name, choose in chooser.items():
        picked = [(choose(r), r) for r in records]
        passed = sum(r[tier]["passed"] for tier, r in picked)
        low, high = wilson(passed, n)
        mean_cost = sum(_cost(r[tier], prices[tier]) for tier, r in picked) / n
        policies[name] = {
            "passed": passed,
            "pass_rate": round(passed / n, 4),
            "pass_rate_ci95": [round(low, 3), round(high, 3)],
            "modeled_cost_usd_per_100": mean_cost * requests,
            "mean_output_tokens": round(sum(r[t]["tokens_out"] for t, r in picked) / n, 1),
            "mean_latency_ms": round(sum(r[t]["latency_ms"] for t, r in picked) / n, 1),
            "share_to_small": round(sum(t == "small" for t, _ in picked) / n, 4),
        }

    labelled = {"simple": "small", "complex": "large"}
    correct = sum(r["decision"] == labelled[r["label"]] for r in records)
    router = {
        "accuracy": round(correct / n, 4),
        "complex_sent_to_small": sum(
            r["label"] == "complex" and r["decision"] == "small" for r in records
        ),
        "simple_sent_to_large": sum(
            r["label"] == "simple" and r["decision"] == "large" for r in records
        ),
    }
    paired = {
        "routed_wrong_large_right": sum(
            not r[r["decision"]]["passed"] and r["large"]["passed"] for r in records
        ),
        "routed_right_large_wrong": sum(
            r[r["decision"]]["passed"] and not r["large"]["passed"] for r in records
        ),
    }
    by_label = {
        label: {
            tier: sum(r[tier]["passed"] for r in records if r["label"] == label)
            for tier in ("small", "large")
        }
        | {"prompts": sum(r["label"] == label for r in records)}
        for label in ("simple", "complex")
    }
    return {
        "prompts": n,
        "requests_modeled": requests,
        "policies": policies,
        "router_vs_labels": router,
        "paired_routed_vs_always_large": paired,
        "passed_by_label_and_model": by_label,
    }


def collect(args: argparse.Namespace, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import httpx

    routes = {"small": args.small_route, "large": args.large_route}
    router_config = load_settings(str(ROOT / "config" / "gateway.yaml")).routing.complexity
    records: list[dict[str, Any]] = []
    with httpx.Client(base_url=args.gateway, timeout=args.timeout) as client:
        for route in routes.values():  # load each model before timing anything
            client.post(
                "/v1/chat/completions",
                json={
                    "model": route,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                },
            ).raise_for_status()
        for item in prompts:
            record: dict[str, Any] = {"id": item["id"], "label": item["label"]}
            request = ChatCompletionRequest.model_validate(
                {"messages": [{"role": "user", "content": item["prompt"]}]}
            )
            decision = classify(request, router_config)
            record["decision"] = decision.tier
            record["decision_reasons"] = decision.reasons
            for tier, route in routes.items():
                started = time.perf_counter()
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": route,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": item["prompt"]},
                        ],
                        "max_tokens": args.max_tokens,
                        "temperature": 0.0,
                    },
                )
                latency_ms = (time.perf_counter() - started) * 1000
                ok = response.status_code == 200
                body = response.json() if ok else {}
                text = body["choices"][0]["message"]["content"] if ok else ""
                usage = body.get("usage", {})
                record[tier] = {
                    "model": response.headers.get("x-gateway-model"),
                    "text": text,
                    "passed": ok and passes(item["check"], text),
                    "http_status": response.status_code,
                    "tokens_in": usage.get("prompt_tokens", 0),
                    "tokens_out": usage.get("completion_tokens", 0),
                    "latency_ms": round(latency_ms, 1),
                }
            print(
                f"{item['id']:>2} {item['label']:<8} router={record['decision']:<5} "
                f"small={'ok' if record['small']['passed'] else '--'} "
                f"large={'ok' if record['large']['passed'] else '--'}",
                flush=True,
            )
            records.append(record)
    return records


def rescore(
    path: Path, prompts: list[dict[str, Any]], redecide: bool = False
) -> list[dict[str, Any]]:
    """Re-score recorded answers; with `redecide`, also re-run the router on the prompts."""
    checks = {p["id"]: p["check"] for p in prompts}
    texts = {p["id"]: p["prompt"] for p in prompts}
    router_config = load_settings(str(ROOT / "config" / "gateway.yaml")).routing.complexity
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for record in records:
        if redecide:
            request = ChatCompletionRequest.model_validate(
                {"messages": [{"role": "user", "content": texts[record["id"]]}]}
            )
            decision = classify(request, router_config)
            record["decision"], record["decision_reasons"] = decision.tier, decision.reasons
        for tier in ("small", "large"):
            side = record[tier]
            side["passed"] = side["http_status"] == 200 and passes(
                checks[record["id"]], side["text"]
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gateway", default="http://127.0.0.1:8080")
    parser.add_argument("--small-route", default="small-local")
    parser.add_argument("--large-route", default="large-local")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--requests", type=int, default=100, help="cost is reported per this many")
    parser.add_argument("--rescore", type=Path, help="re-score recorded answers, no model calls")
    parser.add_argument(
        "--redecide", action="store_true", help="with --rescore: run the router again as it is now"
    )
    parser.add_argument("--out", type=Path, help="report path (default depends on the mode)")
    parser.add_argument("--note", help="free text stored in the report, e.g. how the rules changed")
    args = parser.parse_args()

    prompts = [json.loads(line) for line in PROMPTS.read_text().splitlines() if line.strip()]
    records = (
        rescore(args.rescore, prompts, args.redecide) if args.rescore else collect(args, prompts)
    )

    models = load_settings(str(ROOT / "config" / "gateway.yaml")).pricing.models
    prices = {
        tier: (models[name].input_per_mtok, models[name].output_per_mtok)
        for tier, name in REFERENCE_MODELS.items()
    }
    result = evaluate(records, prices, args.requests)
    result["reference_prices_usd_per_mtok"] = {
        tier: {"model": REFERENCE_MODELS[tier], "input": p[0], "output": p[1]}
        for tier, p in prices.items()
    }
    result["local_models"] = {
        tier: next((r[tier]["model"] for r in records if r[tier].get("model")), None)
        for tier in ("small", "large")
    }

    REPORTS.mkdir(exist_ok=True)
    if not args.rescore:
        with (REPORTS / "routing_answers.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if args.note:
        result["note"] = args.note
    default_name = "routing_eval_replay.json" if args.rescore else "routing_eval.json"
    out = args.out or REPORTS / default_name
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")

    print(f"\n{'policy':<14} {'pass':>9} {'95% CI':>14} {'$/100 req (modeled)':>21} {'small':>6}")
    for name, row in result["policies"].items():
        low, high = row["pass_rate_ci95"]
        print(
            f"{name:<14} {row['passed']:>3}/{result['prompts']:<3}  [{low:.2f}, {high:.2f}]"
            f" {row['modeled_cost_usd_per_100']:>20.4f} {row['share_to_small']:>6.0%}"
        )
    print(f"\nrouter vs hand labels: {result['router_vs_labels']}")
    print(f"routed vs always_large, paired: {result['paired_routed_vs_always_large']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

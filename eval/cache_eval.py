#!/usr/bin/env python
"""Threshold sweep for the semantic cache.

Reads labelled query pairs from eval/data/cache_pairs.jsonl: `same` pairs are two
phrasings of one question (the cache should answer the second from the first),
`different` pairs share words but ask something else (it must not). For each
similarity threshold it reports how many `same` pairs would hit and how many
`different` pairs would hit by mistake, then recommends, for several false-hit
budgets, the lowest threshold that stays within the budget.

The threshold is chosen on the even-numbered pairs and scored on the odd-numbered
ones, so the reported numbers are not measured on the data that picked the
threshold. With ~60 pairs per class that is a thin sample: a false-hit count of 0
only bounds the true rate below ~5% (95% confidence), and the output says so.

    python eval/cache_eval.py                                  # all-MiniLM-L6-v2
    python eval/cache_eval.py --embedder hashing               # the old matcher
    python eval/cache_eval.py --max-false-hit-rate 0 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_gateway.cache.embedder import (  # noqa: E402
    HashingEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
)

DATA = ROOT / "eval" / "data" / "cache_pairs.jsonl"
REPORTS = ROOT / "reports"
THRESHOLDS = [round(0.50 + 0.01 * step, 2) for step in range(50)]
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def load_pairs() -> list[dict[str, Any]]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def build_embedder(name: str) -> tuple[Any, str]:
    if name == "hashing":
        return HashingEmbedder(256), "hashing-256"
    return SentenceTransformerEmbedder(MINILM, dim=384), MINILM


def similarities(embedder: Any, pairs: list[dict[str, Any]]) -> list[float]:
    cache: dict[str, list[float]] = {}

    def vec(text: str) -> list[float]:
        if text not in cache:
            cache[text] = embedder.embed_sync(text)
        return cache[text]

    return [cosine_similarity(vec(p["a"]), vec(p["b"])) for p in pairs]


def wilson_upper(failures: int, n: int, z: float = 1.96) -> float:
    """Upper end of the 95% Wilson interval; honest about small samples."""
    if n == 0:
        return 1.0
    p = failures / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + spread) / (1 + z * z / n)


def sweep(pairs: list[dict[str, Any]], scores: list[float]) -> list[dict[str, Any]]:
    same = [s for p, s in zip(pairs, scores, strict=True) if p["label"] == "same"]
    different = [s for p, s in zip(pairs, scores, strict=True) if p["label"] == "different"]
    rows = []
    for t in THRESHOLDS:
        hits = sum(s >= t for s in same)
        false_hits = sum(s >= t for s in different)
        rows.append(
            {
                "threshold": t,
                "hit_rate": hits / len(same),
                "false_hit_rate": false_hits / len(different),
                "false_hits": false_hits,
                "false_hit_rate_upper_95": wilson_upper(false_hits, len(different)),
            }
        )
    return rows


def recommend(rows: list[dict[str, Any]], max_false_hit_rate: float) -> float | None:
    ok = [r for r in rows if r["false_hit_rate"] <= max_false_hit_rate]
    return min(ok, key=lambda r: r["threshold"])["threshold"] if ok else None


def at(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return next(r for r in rows if r["threshold"] == threshold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--embedder", choices=["sentence-transformers", "hashing"], default="sentence-transformers"
    )
    parser.add_argument(
        "--max-false-hit-rate",
        type=float,
        nargs="+",
        default=[0.0, 0.02, 0.05, 0.10],
        help="false-hit budgets to recommend a threshold for",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pairs = load_pairs()
    embedder, model_name = build_embedder(args.embedder)
    scores = similarities(embedder, pairs)

    tune = [(p, s) for p, s in zip(pairs, scores, strict=True) if p["id"] % 2 == 0]
    test = [(p, s) for p, s in zip(pairs, scores, strict=True) if p["id"] % 2 == 1]
    tune_rows = sweep([p for p, _ in tune], [s for _, s in tune])
    test_rows = sweep([p for p, _ in test], [s for _, s in test])
    all_rows = sweep(pairs, scores)

    labels = [p["label"] for p in pairs]
    recommendations = []
    for budget in args.max_false_hit_rate:
        chosen = recommend(tune_rows, budget)
        recommendations.append(
            {
                "max_false_hit_rate": budget,
                "threshold": chosen,
                "held_out": at(test_rows, chosen) if chosen is not None else None,
            }
        )

    by_category: dict[str, float] = {}
    for p, s in zip(pairs, scores, strict=True):
        if p["label"] == "different":
            by_category[p["category"]] = max(by_category.get(p["category"], 0.0), s)

    summary: dict[str, Any] = {
        "embedder": model_name,
        "pairs": {"same": labels.count("same"), "different": labels.count("different")},
        "selected_on": "even-numbered pair ids",
        "scored_on": "odd-numbered pair ids",
        "recommendations": recommendations,
        "highest_similarity_of_a_different_pair": {
            category: round(value, 4) for category, value in sorted(by_category.items())
        },
        "lowest_similarity_of_a_paraphrase": round(
            min(s for p, s in zip(pairs, scores, strict=True) if p["label"] == "same"), 4
        ),
        "curve_all_pairs": all_rows,
        "curve_held_out": test_rows,
    }

    print(f"embedder: {model_name}   pairs: {summary['pairs']}")
    print(f"{'threshold':>9} {'hit rate':>9} {'false hits':>11} {'false-hit rate (95% upper)':>28}")
    for r in all_rows:
        if round(r["threshold"] * 100) % 5 == 0:
            print(
                f"{r['threshold']:>9.2f} {r['hit_rate']:>9.1%} {r['false_hits']:>11} "
                f"{r['false_hit_rate']:>12.1%} ({r['false_hit_rate_upper_95']:.1%})"
            )
    print("\nhighest similarity of a *different* pair, by kind:")
    for category, value in summary["highest_similarity_of_a_different_pair"].items():
        print(f"  {category:<34} {value:.3f}")
    print("\nrecommended threshold per false-hit budget (chosen on tuning half, scored held-out):")
    for rec in recommendations:
        if rec["threshold"] is None:
            print(f"  budget {rec['max_false_hit_rate']:>5.0%}: no threshold qualifies")
            continue
        held = rec["held_out"]
        print(
            f"  budget {rec['max_false_hit_rate']:>5.0%}: threshold {rec['threshold']:.2f} -> "
            f"held-out hit rate {held['hit_rate']:.1%}, false hits {held['false_hits']} "
            f"of {summary['pairs']['different'] // 2} "
            f"(true rate below {held['false_hit_rate_upper_95']:.1%})"
        )

    out = args.out or REPORTS / f"cache_eval_{args.embedder}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

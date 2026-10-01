"""Every number the README states in prose is recomputed from something in the repo.

The results block is generated; the paragraphs around it are written by hand, and a
hand-written number goes stale the moment a measurement is repeated. Each figure the
prose quotes is pinned here to its source. When one moves, update the paragraph,
not this test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def load_module(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_results_block_is_what_the_reports_say() -> None:
    table = load_module("readme_table", ROOT / "eval" / "readme_table.py")
    assert table.update(README) == README, "run `python eval/readme_table.py`"


def test_the_hash_embedder_collisions_quoted_in_the_story() -> None:
    from llm_gateway.cache.embedder import HashingEmbedder, cosine_similarity

    embedder = HashingEmbedder(256)

    def score(a: str, b: str) -> str:
        return f"{cosine_similarity(embedder.embed_sync(a), embedder.embed_sync(b)):.3f}"

    assert score("What is the capital of France?", "What is the capital of Spain?") == "0.775"
    assert (
        score(
            "Is it safe to take ibuprofen with alcohol?",
            "Is it safe to take ibuprofen without alcohol?",
        )
        == "0.852"
    )
    assert "0.775" in README and "0.852" in README


def test_the_minilm_near_misses_quoted_in_the_story() -> None:
    pytest.importorskip("sentence_transformers")
    from llm_gateway.cache.embedder import SentenceTransformerEmbedder, cosine_similarity

    embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", dim=384)

    def score(a: str, b: str) -> str:
        return f"{cosine_similarity(embedder.embed_sync(a), embedder.embed_sync(b)):.3f}"

    assert (
        score("How do I convert Celsius to Fahrenheit?", "How do I convert Fahrenheit to Celsius?")
        == "0.995"
    )
    assert score("How do I enable dark mode?", "How do I disable dark mode?") == "0.927"
    assert "0.995" in README and "0.927" in README


def test_the_cache_sweep_latency_quoted_in_the_table() -> None:
    steps = json.loads((ROOT / "bench/probes/cache_sweep_off.json").read_text())["steps"]
    by_rows = {s["expired_rows_inserted"]: s["lookup_ms_p50"] for s in steps}
    assert f"{by_rows[0]} ms to {by_rows[20000]} ms" in README


@pytest.mark.parametrize(
    ("probe", "spelling"),
    [
        ("budget_local_1", "+4.9%"),
        ("budget_shared_2", "+5.6%"),
        ("budget_local_2", "+107.9%"),
        ("budget_shared_2_conservative", "−44.2%"),
    ],
)
def test_the_budget_overspend_figures_come_from_the_probes(probe: str, spelling: str) -> None:
    percent = json.loads((ROOT / f"bench/probes/{probe}.json").read_text())["usage"][
        "overspend_pct"
    ]
    written = f"+{percent}%" if percent >= 0 else f"−{abs(percent)}%"
    assert written == spelling
    assert spelling in README


def test_the_routing_story_matches_the_recorded_answers() -> None:
    answers = [
        json.loads(line)
        for line in (ROOT / "reports/routing_answers.jsonl").read_text().splitlines()
    ]
    prompts = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / "eval/data/routing_prompts.jsonl").read_text().splitlines()
    }
    sent_small = [a for a in answers if a["label"] == "complex" and a["decision"] == "small"]
    assert len(sent_small) == 6, "the story says 6 prompts went to the small model by mistake"
    assert all("Answer: <number>" in prompts[a["id"]]["prompt"] for a in sent_small)

    first = json.loads((ROOT / "reports/routing_eval.json").read_text())["policies"]
    tuned = json.loads((ROOT / "reports/routing_eval_rules_v2.json").read_text())["policies"]
    held = json.loads((ROOT / "reports/routing_heldout.json").read_text())["policies"]
    assert (first["routed"]["passed"], first["always_large"]["passed"]) == (43, 49)
    assert tuned["routed"]["passed"] == 47
    assert (held["routed"]["passed"], held["always_large"]["passed"]) == (28, 29)
    cheaper = (
        1
        - held["routed"]["modeled_cost_usd_per_100"]
        / held["always_large"]["modeled_cost_usd_per_100"]
    )
    assert 0.30 < cheaper < 0.36, "the story says 'a third lower'"
    for phrase in ("43 of 50", "49 of 50", "47 of 50", "28 of 30", "29 of 30"):
        assert phrase in README, phrase

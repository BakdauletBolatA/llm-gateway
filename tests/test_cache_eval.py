"""The cache evaluation is only worth quoting if its data and arithmetic are sound."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cache_eval", ROOT / "eval" / "cache_eval.py")
assert _spec is not None and _spec.loader is not None
cache_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache_eval)


def test_the_dataset_has_enough_pairs_of_both_kinds() -> None:
    pairs = cache_eval.load_pairs()
    labels = Counter(p["label"] for p in pairs)
    assert len(pairs) >= 100
    assert labels["same"] >= 50
    assert labels["different"] >= 50


def test_every_pair_is_distinct_and_ids_are_unique() -> None:
    pairs = cache_eval.load_pairs()
    assert len({p["id"] for p in pairs}) == len(pairs)
    assert all(p["a"].strip().lower() != p["b"].strip().lower() for p in pairs)
    keys = [frozenset((p["a"], p["b"])) for p in pairs]
    assert len(set(keys)) == len(keys), "a pair appears twice"


def test_the_threshold_search_takes_the_lowest_threshold_within_budget() -> None:
    rows = [
        {"threshold": 0.7, "false_hit_rate": 0.30},
        {"threshold": 0.8, "false_hit_rate": 0.04},
        {"threshold": 0.9, "false_hit_rate": 0.0},
    ]
    assert cache_eval.recommend(rows, 0.05) == 0.8
    assert cache_eval.recommend(rows, 0.0) == 0.9
    assert cache_eval.recommend(rows[:2], 0.0) is None


def test_zero_observed_false_hits_still_leaves_a_nonzero_upper_bound() -> None:
    bound = cache_eval.wilson_upper(0, 60)
    assert 0.05 < bound < 0.07
    assert cache_eval.wilson_upper(0, 0) == 1.0

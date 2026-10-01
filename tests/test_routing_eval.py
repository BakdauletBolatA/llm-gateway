"""The routing eval's scoring rules and arithmetic, pinned before any model is run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("routing_eval", ROOT / "eval" / "routing_eval.py")
assert _spec is not None and _spec.loader is not None
routing_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(routing_eval)
passes = routing_eval.passes
evaluate = routing_eval.evaluate


def test_any_requires_one_of_the_words_and_ignores_case() -> None:
    assert passes({"any": ["canberra"]}, "It is Canberra.")
    assert not passes({"any": ["canberra"]}, "Sydney")


def test_all_requires_every_word() -> None:
    assert passes({"all": ["tcp", "udp"]}, "TCP is reliable, UDP is not.")
    assert not passes({"all": ["tcp", "udp"]}, "TCP is reliable.")


def test_none_forbids_words() -> None:
    assert passes({"any": ["positive"], "none": ["negative"]}, "Positive.")
    assert not passes({"any": ["positive"], "none": ["negative"]}, "Positive, not negative.")


def test_regex_is_case_sensitive() -> None:
    assert passes({"regex": r"\bAu\b"}, "The symbol is Au.")
    assert not passes({"regex": r"\bAu\b"}, "au revoir")


def test_a_number_is_read_from_the_answer_line_not_from_the_working() -> None:
    reply = "Speed is 120 / 1.5 = 80.\nAnswer: 80"
    assert passes({"number": 80}, reply)
    assert not passes({"number": 120}, reply)


def test_the_last_answer_line_wins_and_commas_and_dollars_are_tolerated() -> None:
    assert passes({"number": 1200}, "Answer: 100\nOn reflection, Answer: $1,200")
    assert not passes({"number": 80}, "I think it is 80")


def record(
    pid: int, label: str, tier: str, small_ok: bool, large_ok: bool, **extra: Any
) -> dict[str, Any]:
    def side(ok: bool, tokens: int) -> dict[str, Any]:
        return {"passed": ok, "tokens_in": 10, "tokens_out": tokens, "latency_ms": 1000.0}

    return {
        "id": pid,
        "label": label,
        "decision": tier,
        "small": side(small_ok, 100),
        "large": side(large_ok, 100),
        **extra,
    }


PRICES = {"small": (1.0, 2.0), "large": (10.0, 20.0)}


def test_policies_pick_the_outcome_of_the_tier_they_choose() -> None:
    records = [
        record(1, "simple", "small", True, True),
        record(2, "complex", "large", False, True),
        record(3, "complex", "small", False, True),  # the router under-estimates this one
        record(4, "simple", "large", True, True),  # and over-estimates this one
    ]
    result = evaluate(records, PRICES, requests=100)
    assert result["policies"]["always_large"]["passed"] == 4
    assert result["policies"]["always_small"]["passed"] == 2
    assert result["policies"]["routed"]["passed"] == 3
    assert result["policies"]["routed"]["share_to_small"] == 0.5


def test_cost_is_scaled_to_the_requested_number_of_requests() -> None:
    records = [record(1, "simple", "small", True, True), record(2, "simple", "large", True, True)]
    result = evaluate(records, PRICES, requests=100)
    # per request: small = (10*1 + 100*2)/1e6, large = (10*10 + 100*20)/1e6
    small_cost, large_cost = 210 / 1e6, 2100 / 1e6
    assert (
        abs(result["policies"]["always_small"]["modeled_cost_usd_per_100"] - 100 * small_cost)
        < 1e-12
    )
    assert (
        abs(result["policies"]["always_large"]["modeled_cost_usd_per_100"] - 100 * large_cost)
        < 1e-12
    )
    assert (
        abs(
            result["policies"]["routed"]["modeled_cost_usd_per_100"]
            - 50 * (small_cost + large_cost)
        )
        < 1e-12
    )


def test_the_router_is_scored_against_the_hand_labels() -> None:
    records = [
        record(1, "simple", "small", True, True),
        record(2, "complex", "large", True, True),
        record(3, "complex", "small", True, True),
        record(4, "simple", "large", True, True),
    ]
    result = evaluate(records, PRICES, requests=100)
    assert result["router_vs_labels"] == {
        "accuracy": 0.5,
        "complex_sent_to_small": 1,
        "simple_sent_to_large": 1,
    }


def test_paired_counts_show_where_routing_changed_the_outcome() -> None:
    records = [
        record(1, "complex", "small", False, True),  # routing lost this one
        record(2, "simple", "small", True, False),  # routing happened to win this one
        record(3, "simple", "small", True, True),
    ]
    paired = evaluate(records, PRICES, requests=100)["paired_routed_vs_always_large"]
    assert paired == {"routed_wrong_large_right": 1, "routed_right_large_wrong": 1}


def test_the_prompt_file_is_what_the_eval_claims_it_is() -> None:
    prompts = [
        json.loads(line)
        for line in (ROOT / "eval/data/routing_prompts.jsonl").read_text().splitlines()
    ]
    assert len(prompts) == 50
    assert {p["label"] for p in prompts} == {"simple", "complex"}
    assert len({p["id"] for p in prompts}) == 50
    assert all(p["check"] for p in prompts)

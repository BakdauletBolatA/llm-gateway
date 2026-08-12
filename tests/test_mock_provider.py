from __future__ import annotations

from collections import Counter

import pytest

from mock_provider.injector import UpstreamState, allocate, build_deck
from mock_provider.profiles import Outcome, ProfileSpec, load_profiles


def spec(**overrides: object) -> ProfileSpec:
    base = {
        "name": "t",
        "outcomes": {Outcome.OK: 70, Outcome.HTTP_500: 30},
        "deck_size": 100,
        "latency_ms": (10, 20),
        "completion_tokens": (5, 10),
    }
    return ProfileSpec.model_validate(base | overrides)


def test_allocation_is_exact_and_sums_to_deck_size() -> None:
    counts = allocate({Outcome.OK: 70, Outcome.HTTP_500: 30}, 100)
    assert counts == {Outcome.OK: 70, Outcome.HTTP_500: 30}


def test_allocation_handles_weights_that_do_not_divide_evenly() -> None:
    counts = allocate({Outcome.OK: 1, Outcome.HTTP_429: 1, Outcome.HTTP_500: 1}, 100)
    assert sum(counts.values()) == 100
    assert max(counts.values()) - min(counts.values()) <= 1


def test_deck_proportions_are_exact_over_one_pass() -> None:
    deck = build_deck(spec(), seed=1)
    counts = Counter(card.outcome for card in deck)
    assert counts[Outcome.OK] == 70
    assert counts[Outcome.HTTP_500] == 30


def test_deck_is_shuffled_but_reproducible() -> None:
    first = [card.outcome for card in build_deck(spec(), seed=1)]
    second = [card.outcome for card in build_deck(spec(), seed=1)]
    third = [card.outcome for card in build_deck(spec(), seed=2)]
    assert first == second, "same seed must deal the same cards in the same order"
    assert first != third, "a different seed must deal a different order"
    assert first != sorted(first, key=str), "the deck must actually be shuffled"


def test_warmup_window_is_always_healthy() -> None:
    state = UpstreamState("primary", base_seed=1)
    state.set_profile(spec(outcomes={Outcome.HTTP_500: 100}, warmup_ok=5))
    outcomes = [state.next_card().outcome for _ in range(8)]
    assert outcomes[:5] == [Outcome.OK] * 5
    assert outcomes[5:] == [Outcome.HTTP_500] * 3


def test_deck_cycles_when_more_requests_than_cards_arrive() -> None:
    state = UpstreamState("primary", base_seed=1)
    state.set_profile(spec(deck_size=10))
    counts = Counter(state.next_card().outcome for _ in range(30))
    assert counts[Outcome.OK] == 21
    assert counts[Outcome.HTTP_500] == 9


def test_shipped_profiles_and_scenarios_are_valid() -> None:
    library = load_profiles("config/failure_profiles.yaml")
    assert library.seed != 0
    assert "storm" in library.profiles
    assert set(library.upstreams()) == {"primary", "secondary", "tertiary"}
    for name, assignment in library.scenarios.items():
        assert assignment, f"scenario {name} assigns no profiles"


def test_profile_with_no_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        ProfileSpec.model_validate({"name": "bad", "outcomes": {}})

"""Deterministic failure injection.

Independent random draws per request would make every chaos run differ by a few
percent, which is exactly the noise that makes an iteration table unreadable.
Instead each upstream holds a shuffled *deck* of outcomes built from the profile
weights: over a full pass the proportions are exact, and only the pairing of a
particular outcome to a particular request varies with concurrency.

The deck is seeded from (global seed, upstream, profile), so re-running a
scenario a month later deals the same cards in the same order.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from mock_provider.profiles import Outcome, ProfileSpec


@dataclass(slots=True)
class Card:
    outcome: Outcome
    latency_ms: int
    completion_tokens: int


def allocate(weights: dict[Outcome, int], deck_size: int) -> dict[Outcome, int]:
    """Largest-remainder allocation of `deck_size` cards across weighted outcomes."""
    total = sum(weights.values())
    exact = {outcome: weight * deck_size / total for outcome, weight in weights.items()}
    counts = {outcome: int(value) for outcome, value in exact.items()}
    shortfall = deck_size - sum(counts.values())
    if shortfall > 0:
        by_remainder = sorted(
            exact, key=lambda outcome: (exact[outcome] - counts[outcome], str(outcome)), reverse=True
        )
        for outcome in by_remainder[:shortfall]:
            counts[outcome] += 1
    return counts


def build_deck(spec: ProfileSpec, seed: int) -> list[Card]:
    rng = random.Random(seed)
    counts = allocate(spec.outcomes, spec.deck_size)
    deck = [
        Card(
            outcome=outcome,
            latency_ms=rng.randint(*spec.latency_ms),
            completion_tokens=rng.randint(*spec.completion_tokens),
        )
        for outcome, count in sorted(counts.items(), key=lambda item: str(item[0]))
        for _ in range(count)
    ]
    rng.shuffle(deck)
    return deck


class UpstreamState:
    """One named upstream of the mock provider, currently running one profile."""

    def __init__(self, name: str, base_seed: int) -> None:
        self.name = name
        self._base_seed = base_seed
        self.profile: ProfileSpec | None = None
        self._deck: list[Card] = []
        self._cursor = 0
        self._served = 0
        self.counts: dict[str, int] = {}

    def set_profile(self, spec: ProfileSpec) -> None:
        self.profile = spec
        self._deck = build_deck(spec, self._seed_for(spec.name))
        self._cursor = 0
        self._served = 0
        self.counts = {}

    def _seed_for(self, profile_name: str) -> int:
        return stable_seed(f"{self._base_seed}:{self.name}:{profile_name}")

    def next_card(self) -> Card:
        if self.profile is None or not self._deck:
            return Card(outcome=Outcome.OK, latency_ms=0, completion_tokens=64)

        self._served += 1
        if self._served <= self.profile.warmup_ok:
            # Warm-up window: always healthy, so a circuit breaker can observe a
            # provider that works and only then starts failing.
            card = Card(outcome=Outcome.OK, latency_ms=self.profile.latency_ms[0],
                        completion_tokens=self.profile.completion_tokens[0])
        else:
            card = self._deck[self._cursor % len(self._deck)]
            self._cursor += 1
        self.counts[str(card.outcome)] = self.counts.get(str(card.outcome), 0) + 1
        return card

    def snapshot(self) -> dict[str, object]:
        return {
            "upstream": self.name,
            "profile": self.profile.name if self.profile else None,
            "served": self._served,
            "deck_size": len(self._deck),
            "cursor": self._cursor,
            "counts": dict(sorted(self.counts.items())),
        }


def stable_seed(text: str) -> int:
    """Process-independent seed (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "big")


@dataclass
class MockState:
    base_seed: int
    upstreams: dict[str, UpstreamState] = field(default_factory=dict)
    scenario: str | None = None

    def upstream(self, name: str) -> UpstreamState:
        state = self.upstreams.get(name)
        if state is None:
            state = UpstreamState(name, self.base_seed)
            self.upstreams[name] = state
        return state

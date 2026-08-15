"""The hand-written prose in RELIABILITY.md must agree with the committed results.

The tables in the report are generated, so they cannot drift. The prose around them
is written by hand, and it did drift: ablation runs were re-measured twice as the
final build changed, and four numbers quoted in the iteration-7 analysis silently
became numbers from a build that no longer exists.

This pins the specific figures the prose quotes. When a re-measurement moves one of
them, the test fails and names the paragraph that has to be rewritten — instead of
the report quietly claiming something the JSON next to it contradicts.

Only the hand-written part of the document is searched: a number that appears in a
generated table proves nothing about the sentence that quotes it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORT = Path("RELIABILITY.md")
RESULTS = Path("bench/results")
BEGIN_MARKER = "<!-- BEGIN:GENERATED -->"
END_MARKER = "<!-- END:GENERATED -->"

#: (label, scenario, field, where the prose quotes it)
CLAIMS: list[tuple[str, str, str, str]] = [
    ("09_backpressure", "capacity_limited", "success", "итерация 9: 43.3% → 100%"),
    ("09_backpressure", "capacity_limited", "p95", "итерация 9: цена доступности"),
    ("09_backpressure", "capacity_limited", "cost", "итерация 9 и ablation no_bulkhead"),
    ("09_backpressure", "rate_limited", "cost", "итерация 9: на 24% дешевле"),
    ("08_admission", "capacity_limited", "success", "итерация 8 и 9: было 43.3%"),
    ("08_admission", "capacity_limited", "p95", "итерация 9: p95 вырос с 1503"),
    ("08_admission", "rate_limited", "cost", "итерация 9: было $0.0285"),
    ("07_hedging", "capacity_limited", "success", "итерация 8: полная сборка 38.7%"),
    ("07_hedging", "slow", "p95", "итерация 7: хвост с хеджем"),
    ("07_hedging", "slow", "cost", "итерация 7: цена хеджа на slow"),
    ("03_retries", "capacity_limited", "success", "итерация 8: лучший результат колонки"),
    ("04_circuit_breaker", "capacity_limited", "success", "итерация 8: брейкер сделал хуже"),
    ("ablation_no_hedging", "hang", "success", "итерация 7 и ablation: без хеджа"),
    ("ablation_no_hedging", "hang", "p95", "итерация 7 и ablation: хвост без хеджа"),
    ("ablation_no_hedging", "slow", "p95", "итерация 7 и ablation: хвост без хеджа"),
    ("ablation_no_retries", "capacity_limited", "success", "итерация 9: ретраи несущие"),
    ("ablation_no_fallback", "storm", "success", "ablation: без fallback"),
    ("ablation_no_bulkhead", "capacity_limited", "cost", "итерация 9: без лимита дороже"),
    ("ablation_no_cache", "storm", "cost", "ablation: кэш — механизм стоимости"),
    ("extra_capacity_bulkhead", "capacity_limited", "success", "итерация 8: лимит по квоте"),
    ("extra_capacity_no_bulkhead", "capacity_limited", "success", "итерация 8: без лимита"),
]


def prose() -> str:
    """The document without its generated section."""
    text = REPORT.read_text(encoding="utf-8")
    head, _, rest = text.partition(BEGIN_MARKER)
    _, _, tail = rest.partition(END_MARKER)
    return head + tail


def value(label: str, scenario: str, field: str) -> float:
    summary = json.loads((RESULTS / f"{label}__{scenario}.json").read_text())["results"]
    return {
        "success": summary["success_rate"] * 100,
        "p50": summary["latency_ms"]["p50"],
        "p95": summary["latency_ms"]["p95"],
        "cost": summary["cost_usd_server"],
    }[field]


def spellings(field: str, number: float) -> list[str]:
    """How the prose is allowed to write this number."""
    if field == "success":
        return [f"{number:.1f}%"]
    if field == "cost":
        return [f"${number:.4f}"]
    rounded = f"{number:.0f}"
    # Thousands are written with a space in prose and without one in tables.
    spaced = f"{rounded[:-3]} {rounded[-3:]}" if len(rounded) > 3 else rounded
    return [rounded, spaced]


@pytest.mark.parametrize(("label", "scenario", "field", "where"), CLAIMS)
def test_the_prose_quotes_the_measured_number(
    label: str, scenario: str, field: str, where: str
) -> None:
    if not (RESULTS / f"{label}__{scenario}.json").exists():
        pytest.skip(f"{label}__{scenario} has not been measured")
    measured = value(label, scenario, field)
    allowed = spellings(field, measured)
    document = prose()

    assert any(spelling in document for spelling in allowed), (
        f"{label}__{scenario} {field} = {allowed[0]}, but RELIABILITY.md does not say so.\n"
        f"Where it is quoted: {where}.\n"
        "Either the prose is stale after a re-measurement, or the claim moved — "
        "update the paragraph, not this test."
    )

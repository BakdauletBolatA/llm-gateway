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


# -- the two-replica probes ----------------------------------------------------

#: (probe file, field, spelling, where the prose quotes it)
PROBE_CLAIMS: list[tuple[str, str, str, str]] = [
    ("budget_local_1", "served_200", "89", "итерация 11: одна реплика, local"),
    ("budget_local_1", "billed", "$0.005410", "итерация 11: одна реплика, local"),
    ("budget_local_1", "overspend", "+8.2%", "итерация 11: одна реплика, local"),
    ("budget_local_2", "served_200", "179", "итерация 11: две реплики, local"),
    ("budget_local_2", "billed", "$0.011023", "итерация 11: две реплики, local"),
    ("budget_local_2", "overspend", "+120.5%", "итерация 11 и рецепт в README"),
    ("budget_shared_1", "served_200", "84", "итерация 11: одна реплика, shared"),
    ("budget_shared_1", "billed", "$0.005156", "итерация 11: одна реплика, shared"),
    ("budget_shared_1", "overspend", "+3.1%", "итерация 11: одна реплика, shared"),
    ("budget_shared_2", "served_200", "87", "итерация 11: две реплики, shared"),
    ("budget_shared_2", "billed", "$0.005329", "итерация 11: две реплики, shared"),
    ("budget_shared_2", "overspend", "+6.6%", "итерация 11 и рецепт в README"),
    ("budget_shared_2_conservative", "served_200", "51", "итерация 11: консервативная оценка"),
    ("budget_shared_2_conservative", "billed", "$0.003165", "итерация 11: консервативная оценка"),
    ("budget_shared_2_conservative", "overspend", "−36.7%", "итерация 11: цена запаса"),
    ("ratelimit_local_2", "served_200", "103", "лимит на нескольких репликах: local"),
    ("ratelimit_local_2", "throttled_429", "497", "лимит на нескольких репликах: local"),
    ("ratelimit_shared_2", "served_200", "63", "лимит на нескольких репликах: shared"),
    ("ratelimit_shared_2", "throttled_429", "537", "лимит на нескольких репликах: shared"),
]


def probe_spelling(name: str, field: str) -> str:
    """How the report is allowed to write this number.

    Negative overspend is written with the typographic minus, because that is what
    the prose uses; Python formats an ASCII hyphen, and the two are different
    characters, so the substring search would silently never match.
    """
    probe = json.loads((Path("bench/probes") / f"{name}.json").read_text())
    if field == "billed":
        return f"${probe['usage']['cost_usd_recorded']:.6f}"
    if field == "overspend":
        percent = probe["usage"]["overspend_pct"]
        return f"+{percent}%" if percent >= 0 else f"−{abs(percent)}%"
    return str(probe[field])


@pytest.mark.parametrize(("name", "field", "expected", "where"), PROBE_CLAIMS)
def test_the_prose_quotes_the_measured_probe(
    name: str, field: str, expected: str, where: str
) -> None:
    """The multi-replica numbers are hand-copied into the report from a probe run.

    They cannot be regenerated by the report builder — the probes need two gateway
    processes — so the only thing keeping them honest is this test.
    """
    if not (Path("bench/probes") / f"{name}.json").exists():
        pytest.skip(f"{name} has not been measured")
    measured = probe_spelling(name, field)
    assert measured == expected, (
        f"{name}.{field} is {measured}, but this test expects {expected}: "
        f"re-run scripts/run_probes.sh and update both the report and this list."
    )
    assert measured in prose(), (
        f"{name}.{field} = {measured}, but RELIABILITY.md does not say so.\n"
        f"Where it is quoted: {where}."
    )

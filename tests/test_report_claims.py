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
    ("budget_local_1", "served_200", "85", "итерация 11: одна реплика, local"),
    ("budget_local_1", "billed", "$0.005246", "итерация 11: одна реплика, local"),
    ("budget_local_1", "overspend", "+4.9%", "итерация 11: одна реплика, local"),
    ("budget_local_2", "served_200", "168", "итерация 11: две реплики, local"),
    ("budget_local_2", "billed", "$0.010396", "итерация 11: две реплики, local"),
    ("budget_local_2", "overspend", "+107.9%", "итерация 11 и рецепт в README"),
    ("budget_shared_1", "served_200", "84", "итерация 11: одна реплика, shared"),
    ("budget_shared_1", "billed", "$0.005156", "итерация 11: одна реплика, shared"),
    ("budget_shared_1", "overspend", "+3.1%", "итерация 11: одна реплика, shared"),
    ("budget_shared_2", "served_200", "86", "итерация 11: две реплики, shared"),
    ("budget_shared_2", "billed", "$0.005280", "итерация 11: две реплики, shared"),
    ("budget_shared_2", "overspend", "+5.6%", "итерация 11 и рецепт в README"),
    ("budget_shared_2_conservative", "served_200", "45", "итерация 11: консервативная оценка"),
    ("budget_shared_2_conservative", "billed", "$0.002788", "итерация 11: консервативная оценка"),
    ("budget_shared_2_conservative", "overspend", "−44.2%", "итерация 11: цена запаса"),
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


# -- the generated section ------------------------------------------------------


def test_the_generated_tables_match_the_results_on_disk() -> None:
    """The tables between the markers are built from bench/results, not written.

    Nothing stops someone from editing a generated table by hand, or from adding a
    result file and forgetting to rebuild — and either way the document would keep
    looking right. This rebuilds the section in memory and compares.
    """
    from chaos.report import BEGIN_MARKER as BUILD_BEGIN
    from chaos.report import build_section, load_results

    results = load_results(RESULTS)
    assert results, "bench/results is empty, so the report cannot be verified"

    expected = build_section(results)
    document = REPORT.read_text(encoding="utf-8")
    start = document.index(BUILD_BEGIN)
    end = document.index(END_MARKER) + len(END_MARKER)
    actual = document[start:end]

    assert actual == expected, (
        f"the generated section is stale against {len(results)} result files.\n"
        "Run `python -m chaos.report` — do not edit the tables by hand."
    )


# -- the cache sweep probe ------------------------------------------------------

#: (probe file, expired rows inserted, how the prose writes the p50)
SWEEP_CLAIMS: list[tuple[str, int, str]] = [
    ("cache_sweep_off", 0, "3.6 мс"),
    ("cache_sweep_off", 1000, "3.3 мс"),
    ("cache_sweep_off", 5000, "6.2 мс"),
    ("cache_sweep_off", 20000, "13.5 мс"),
    ("cache_sweep_on", 0, "3.9 мс"),
    ("cache_sweep_on", 1000, "3.6 мс"),
    ("cache_sweep_on", 5000, "4.2 мс"),
    ("cache_sweep_on", 20000, "3.9 мс"),
]


@pytest.mark.parametrize(("name", "rows", "expected"), SWEEP_CLAIMS)
def test_the_prose_quotes_the_measured_sweep(name: str, rows: int, expected: str) -> None:
    path = Path("bench/probes") / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name} has not been measured")
    steps = json.loads(path.read_text())["steps"]
    step = next(s for s in steps if s["expired_rows_inserted"] == rows)
    measured = f"{step['lookup_ms_p50']} мс"

    assert measured == expected, (
        f"{name} at {rows} expired rows is {measured}, this test expects {expected}: "
        "re-run scripts/cache_sweep_probe.py and update the table in RELIABILITY.md."
    )
    assert measured in prose(), (
        f"{name} at {rows} expired rows = {measured}, but RELIABILITY.md does not say so."
    )


def test_the_answer_is_served_at_every_step_of_the_sweep_probe() -> None:
    """The finding is about cost, not correctness — and the report says so. If a
    lookup ever came back empty the claim would have to change, not the sentence."""
    for name in ("cache_sweep_off", "cache_sweep_on"):
        path = Path("bench/probes") / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{name} has not been measured")
        steps = json.loads(path.read_text())["steps"]
        assert all(step["answer_served"] for step in steps), (
            f"{name}: expired rows hid a live answer, which would make this a "
            "correctness finding and not a latency one"
        )


# -- the breaker replica probe --------------------------------------------------

#: (probe file, field, how the prose writes it)
BREAKER_CLAIMS: list[tuple[str, str]] = [
    ("breaker_1replica_n200", "22 стука"),
    ("breaker_2replicas_n200", "29 стуков"),
    ("breaker_1replica_n600", "23 стука"),
    ("breaker_2replicas_n600", "30 стуков"),
]


@pytest.mark.parametrize(("name", "expected"), BREAKER_CLAIMS)
def test_the_prose_quotes_the_measured_breaker_probe(name: str, expected: str) -> None:
    path = Path("bench/probes") / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{name} has not been measured")
    wasted = json.loads(path.read_text())["wasted_calls_to_dead_upstream"]

    assert expected.startswith(f"{wasted} "), (
        f"{name} wasted {wasted} calls, the report says {expected!r}: "
        "re-run ONLY=breaker scripts/run_probes.sh and update the table."
    )
    assert expected in prose(), f"{name} = {expected}, but RELIABILITY.md does not say so."


def test_the_cost_of_a_second_replica_does_not_grow_with_traffic() -> None:
    """The whole argument for not building a shared breaker. If this ever stops
    holding, the section is wrong and the decision has to be revisited — so it is
    an assertion, not a sentence."""
    probes = {}
    for name in (
        "breaker_1replica_n200",
        "breaker_2replicas_n200",
        "breaker_1replica_n600",
        "breaker_2replicas_n600",
    ):
        path = Path("bench/probes") / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{name} has not been measured")
        probes[name] = json.loads(path.read_text())

    small = (
        probes["breaker_2replicas_n200"]["wasted_calls_to_dead_upstream"]
        - probes["breaker_1replica_n200"]["wasted_calls_to_dead_upstream"]
    )
    large = (
        probes["breaker_2replicas_n600"]["wasted_calls_to_dead_upstream"]
        - probes["breaker_1replica_n600"]["wasted_calls_to_dead_upstream"]
    )
    assert small == large == 7, (
        f"the second replica cost {small} extra calls at n=200 and {large} at n=600. "
        "The report claims this is a constant equal to min_calls; if it now scales "
        "with traffic, a shared breaker is back on the table."
    )
    assert "**+7**" in prose()


def test_every_breaker_probe_still_answered_every_request() -> None:
    """The finding is about wasted calls, not availability: fallback carried all of
    them. A drop here would make it a different finding."""
    for name, _ in BREAKER_CLAIMS:
        path = Path("bench/probes") / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{name} has not been measured")
        probe = json.loads(path.read_text())
        assert probe["answered_200"] == probe["requests"], (
            f"{name}: {probe['answered_200']} of {probe['requests']} answered"
        )


# -- totals across scenarios, and the README headline ---------------------------

ORIGINAL_SCENARIOS_EXCLUDE = ("capacity_limited",)


def total_success(label: str, exclude: tuple[str, ...] = ()) -> float:
    """Pooled success rate of a build across its scenarios, as the prose reports it."""
    requests = successes = 0
    for path in RESULTS.glob(f"{label}__*.json"):
        run = json.loads(path.read_text())
        if run["scenario"] in exclude:
            continue
        requests += run["results"]["requests"]
        successes += run["results"]["successes"]
    return 100 * successes / requests


def readme() -> str:
    return Path("README.md").read_text(encoding="utf-8")


#: (label, excluded scenarios, which document, where it is quoted)
TOTAL_CLAIMS: list[tuple[str, tuple[str, ...], str, str]] = [
    ("01_baseline", (), "readme", "README: заглавная таблица, наивный шлюз"),
    ("09_backpressure", (), "readme", "README: заглавная таблица, финальная сборка"),
    ("01_baseline", ORIGINAL_SCENARIOS_EXCLUDE, "report", "итерация 1: итог по десяти сценариям"),
    ("03_retries", ORIGINAL_SCENARIOS_EXCLUDE, "report", "итерации 3 и 4: суммарно"),
    ("04_circuit_breaker", ORIGINAL_SCENARIOS_EXCLUDE, "report", "итерации 4 и 5: суммарно"),
    ("05_fallback", ORIGINAL_SCENARIOS_EXCLUDE, "report", "итерация 5: суммарно"),
    ("03_retries", (), "report", "итерация 3: на всех одиннадцати"),
]


@pytest.mark.parametrize(("label", "exclude", "document", "where"), TOTAL_CLAIMS)
def test_totals_are_quoted_over_the_scenarios_they_were_computed_on(
    label: str, exclude: tuple[str, ...], document: str, where: str
) -> None:
    """The README once quoted a ten-scenario total next to an eleven-scenario table.

    Both documents now say which set a total covers, and each total is recomputed
    here from the committed results over exactly that set.
    """
    spelling = f"{total_success(label, exclude):.1f}%"
    text = readme() if document == "readme" else prose()
    assert spelling in text, f"{label} total over {where} is {spelling}, not quoted"


def test_the_readme_headline_table_matches_the_results() -> None:
    """Independent of eval/readme_table.py: recompute the chaos row from the raw results."""
    text = readme()
    naive, final = "01_baseline", "09_backpressure"

    def full_marks(label: str) -> int:
        return sum(
            json.loads(p.read_text())["results"]["success_rate"] >= 0.9999
            for p in RESULTS.glob(f"{label}__*.json")
        )

    def cost(label: str) -> float:
        return sum(
            json.loads(p.read_text())["results"]["cost_usd_server"]
            for p in RESULTS.glob(f"{label}__*.json")
        )

    n = len(list(RESULTS.glob(f"{final}__*.json")))
    assert f"| {full_marks(naive)} of {n} |" in text
    assert f"| {full_marks(final)} of {n} |" in text
    for label, scenario in ((naive, "storm"), (final, "storm"), (naive, "hang"), (final, "hang")):
        assert f"{value(label, scenario, 'p95'):,.0f} ms" in text, f"{label} {scenario} p95"
    assert f"${cost(naive):.4f}" in text and f"${cost(final):.4f}" in text

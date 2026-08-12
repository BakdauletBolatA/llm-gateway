"""Turn bench/results/*.json into the tables inside RELIABILITY.md.

The report is generated, never hand-written: every number in the document comes
from a result file that is committed next to it.

    python -m chaos.report                 # rewrite RELIABILITY.md in place
    python -m chaos.report --stdout        # print the generated section
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

BEGIN_MARKER = "<!-- BEGIN:GENERATED -->"
END_MARKER = "<!-- END:GENERATED -->"
ABLATION_PREFIX = "ablation"
#: Прогоны, которые отвечают на отдельный вопрос и не должны попадать в матрицу
#: итераций (другой N, другая конкурентность и т.п.).
EXTRA_PREFIX = "extra"


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results


def _sort_key(label: str) -> tuple[int, str]:
    head = label.split("_", 1)[0]
    return (int(head), label) if head.isdigit() else (999, label)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_нет данных_\n"
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _matrix(
    results: list[dict[str, Any]],
    labels: list[str],
    scenarios: list[str],
    extract: Any,
) -> str:
    index = {(r["label"], r["scenario"]): r for r in results}
    rows = []
    for scenario in scenarios:
        row = [f"`{scenario}`"]
        for label in labels:
            result = index.get((label, scenario))
            row.append(extract(result) if result else "—")
        rows.append(row)
    return _table(["сценарий", *labels], rows)


def _totals_row(label: str, runs: list[dict[str, Any]]) -> list[str]:
    requests = sum(r["results"]["requests"] for r in runs)
    successes = sum(r["results"]["successes"] for r in runs)
    worst = min((r["results"]["success_rate"] for r in runs), default=0.0)
    return [
        f"**{label}**",
        str(len(runs)),
        str(requests),
        _pct(successes / requests if requests else 0.0),
        _pct(worst),
        str(sum(r["results"]["retries_total"] for r in runs)),
        str(sum(r["results"]["fallbacks_total"] for r in runs)),
        str(sum(r["results"]["breaker_skips_total"] for r in runs)),
        str(sum(r["results"]["cache_hits"] for r in runs)),
        f"{sum(r['results']['cost_usd_server'] for r in runs):.4f}",
        f"{sum(r['duration_s'] for r in runs):.0f}",
    ]


#: Mechanisms in the order the iterations introduced them; the letter is used by the
#: compact "вкл" column.
MECHANISMS = [
    ("timeouts", "T", "таймауты"),
    ("retries", "R", "ретраи"),
    ("circuit_breaker", "B", "circuit breaker"),
    ("fallback", "F", "fallback"),
    ("cache", "C", "семкэш"),
    ("hedging", "H", "хедж"),
    ("bulkhead", "L", "лимит"),
    ("rate_limit", "A", "rate limit"),
]


def _enabled(run: dict[str, Any], key: str) -> bool:
    """A run made before a mechanism existed simply carries no flag for it."""
    section = run["gateway_config"]["reliability"].get(key) or {}
    return bool(section.get("enabled"))


def _flags(run: dict[str, Any]) -> str:
    enabled = [letter for key, letter, _ in MECHANISMS if _enabled(run, key)]
    return "".join(enabled) or "—"


def _hedging_table(results: list[dict[str, Any]]) -> str | None:
    """What hedging cost: duplicates launched, answers paid for and dropped, p95 against
    the same build with hedging off."""
    hedged = [
        run
        for run in results
        if _enabled(run, "hedging") and not run["label"].startswith((ABLATION_PREFIX, EXTRA_PREFIX))
    ]
    if not hedged:
        return None
    baseline = {
        run["scenario"]: run for run in results if run["label"] == f"{ABLATION_PREFIX}_no_hedging"
    }
    rows = []
    for run in sorted(hedged, key=lambda r: (_sort_key(r["label"]), r["scenario"])):
        summary = run["results"]
        without = baseline.get(run["scenario"])
        rows.append(
            [
                f"`{run['scenario']}`",
                f"{summary.get('hedged_requests', 0)}/{summary['requests']}",
                str(summary.get("hedges_total", 0)),
                f"{summary['latency_ms']['p95']:.0f}",
                f"{without['results']['latency_ms']['p95']:.0f}" if without else "—",
                f"{summary['cost_usd_server']:.4f}",
                f"{without['results']['cost_usd_server']:.4f}" if without else "—",
                f"{summary.get('cost_usd_wasted', 0.0):.4f}",
            ]
        )
    return _table(
        [
            "сценарий",
            "запросов с хеджем",
            "хеджей",
            "p95 с хеджем",
            "p95 без",
            "$ с хеджем",
            "$ без",
            "из них впустую $",
        ],
        rows,
    )


def _capacity_table(results: list[dict[str, Any]]) -> str | None:
    """The capacity-limited runs, with what the *provider* saw next to what we did."""
    runs = [
        run
        for run in results
        if run["scenario"] == "capacity_limited" and run["label"].startswith("extra_capacity")
    ]
    if not runs:
        return None
    rows = []
    for run in sorted(runs, key=lambda r: r["results"]["success_rate"]):
        summary = run["results"]
        state = run.get("gateway_reliability_state") or {}
        bulkheads = state.get("bulkheads") or []
        breakers = state.get("circuit_breakers") or []
        rejected = sum(
            int(upstream.get("rejected_overload") or 0)
            for upstream in (run.get("mock_state") or {}).get("upstreams", [])
        )
        queued = sum(int(b.get("queued") or 0) for b in bulkheads)
        waits = [float(b.get("avg_queue_wait_ms") or 0) for b in bulkheads if b.get("queued")]
        open_breakers = sum(1 for b in breakers if b.get("state") == "open")
        rows.append(
            [
                f"`{run['label'].removeprefix('extra_')}`",
                _pct(summary["success_rate"]),
                f"{summary['latency_ms']['p95']:.0f}",
                str(rejected),
                f"{open_breakers} из {len(breakers)}",
                str(queued),
                f"{max(waits, default=0.0):.0f}",
                f"{run['duration_s']:.1f}",
            ]
        )
    return _table(
        [
            "конфигурация",
            "success",
            "p95, мс",
            "503 от провайдера",
            "брейкеров открыто",
            "запросов в очереди",
            "макс. ожидание слота, мс",
            "прогон, с",
        ],
        rows,
    )


def build_section(results: list[dict[str, Any]]) -> str:
    iterations = [r for r in results if not r["label"].startswith((ABLATION_PREFIX, EXTRA_PREFIX))]
    ablations = [r for r in results if r["label"].startswith(ABLATION_PREFIX)]
    extras = [r for r in results if r["label"].startswith(EXTRA_PREFIX)]

    labels = sorted({r["label"] for r in iterations}, key=_sort_key)
    scenarios = sorted({r["scenario"] for r in iterations})
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in iterations:
        by_label[run["label"]].append(run)

    parts: list[str] = [BEGIN_MARKER, ""]
    parts.append("<!-- Сгенерировано `python -m chaos.report`. Руками не править. -->")
    parts.append("")

    # Which mechanisms were on in each iteration.
    parts.append("### Что включено в каждой итерации\n")
    flag_rows = []
    for label in labels:
        run = by_label[label][0]
        flag_rows.append(
            [
                f"**{label}**",
                *("да" if _enabled(run, key) else "—" for key, _, _ in MECHANISMS),
            ]
        )
    parts.append(_table(["итерация", *(title for _, _, title in MECHANISMS)], flag_rows))

    parts.append("\n### Сводка по итерациям\n")
    parts.append(
        _table(
            [
                "итерация",
                "сценариев",
                "запросов",
                "success (все)",
                "success (худший сценарий)",
                "ретраев",
                "fallback",
                "CB-пропусков",
                "кэш-хитов",
                "стоимость $",
                "время, с",
            ],
            [_totals_row(label, by_label[label]) for label in labels],
        )
    )

    parts.append("\n### Success rate по сценариям\n")
    parts.append(
        _matrix(iterations, labels, scenarios, lambda r: _pct(r["results"]["success_rate"]))
    )

    parts.append("\n### p95 latency, мс\n")
    parts.append(
        _matrix(iterations, labels, scenarios, lambda r: f"{r['results']['latency_ms']['p95']:.0f}")
    )

    parts.append("\n### p50 latency, мс\n")
    parts.append(
        _matrix(iterations, labels, scenarios, lambda r: f"{r['results']['latency_ms']['p50']:.0f}")
    )

    parts.append("\n### Стоимость прогона, $\n")
    parts.append(
        _matrix(iterations, labels, scenarios, lambda r: f"{r['results']['cost_usd_server']:.4f}")
    )

    hedging_table = _hedging_table(results)
    if hedging_table is not None:
        parts.append("\n### Хеджирование: сколько дублей и во что они обошлись\n")
        parts.append(
            "Столбцы «без» взяты из ablation-прогона `ablation_no_hedging` — "
            "та же сборка с выключенным хеджем.\n"
        )
        parts.append(hedging_table)

    capacity_table = _capacity_table(results)
    if capacity_table is not None:
        parts.append("\n### Провайдер с квотой конкурентности\n")
        parts.append(
            "Сценарий `capacity_limited`: каждый апстрим обслуживает не больше двух "
            "запросов одновременно, всё сверх — 503. Слева направо растёт то, "
            "насколько шлюз уважает чужую квоту сам.\n"
        )
        parts.append(capacity_table)

    if ablations:
        parts.append("\n### Ablation: выключаем по одному механизму\n")
        parts.append(
            "Базой служит финальная конфигурация; в каждой строке выключен ровно один механизм.\n"
        )
        ablation_scenarios = sorted({r["scenario"] for r in ablations})
        ablation_labels = sorted({r["label"] for r in ablations})
        rows = []
        for label in ablation_labels:
            for scenario in ablation_scenarios:
                matches = [
                    r for r in ablations if r["label"] == label and r["scenario"] == scenario
                ]
                if not matches:
                    continue
                run = matches[0]
                summary = run["results"]
                rows.append(
                    [
                        f"`{label}`",
                        f"`{scenario}`",
                        _flags(run),
                        _pct(summary["success_rate"]),
                        f"{summary['latency_ms']['p95']:.0f}",
                        str(summary["retries_total"]),
                        str(summary["fallbacks_total"]),
                        str(summary["cache_hits"]),
                        f"{summary['cost_usd_server']:.4f}",
                    ]
                )
        parts.append(
            _table(
                [
                    "конфигурация",
                    "сценарий",
                    "включено",
                    "success",
                    "p95, мс",
                    "ретраев",
                    "fallback",
                    "кэш",
                    "$",
                ],
                rows,
            )
        )

    if extras:
        parts.append("\n### Дополнительные замеры\n")
        parts.append("Прогоны с другими параметрами нагрузки — не часть матрицы итераций.\n")
        rows = []
        for run in sorted(extras, key=lambda r: (r["label"], r["scenario"])):
            summary = run["results"]
            rows.append(
                [
                    f"`{run['label']}`",
                    f"`{run['scenario']}`",
                    f"{summary['requests']} x{run['concurrency']}",
                    _pct(summary["success_rate"]),
                    f"{summary['latency_ms']['p50']:.0f}",
                    f"{summary['latency_ms']['p95']:.0f}",
                    f"{summary['latency_ms']['max']:.0f}",
                    str(summary.get("hedges_total", 0)),
                    str(summary["cache_hits"]),
                    f"{summary['cost_usd_server']:.4f}",
                    run.get("note") or "",
                ]
            )
        parts.append(
            _table(
                [
                    "прогон",
                    "сценарий",
                    "нагрузка",
                    "success",
                    "p50",
                    "p95",
                    "max",
                    "хеджей",
                    "кэш",
                    "$",
                    "зачем",
                ],
                rows,
            )
        )

    parts.append("\n### Полная детализация\n")
    detail_rows = []
    detail_source = iterations + ablations
    for run in sorted(detail_source, key=lambda r: (_sort_key(r["label"]), r["scenario"])):
        summary = run["results"]
        detail_rows.append(
            [
                f"`{run['label']}`",
                f"`{run['scenario']}`",
                _flags(run),
                str(summary["requests"]),
                _pct(summary["success_rate"]),
                f"{summary['latency_ms']['p50']:.0f}",
                f"{summary['latency_ms']['p95']:.0f}",
                f"{summary['latency_ms']['p99']:.0f}",
                str(summary["attempts_total"]),
                str(summary["retries_total"]),
                str(summary["fallbacks_total"]),
                str(summary["breaker_skips_total"]),
                str(summary["cache_hits"]),
                f"{summary['cost_usd_server']:.4f}",
                f"{summary['throughput_rps']:.1f}",
            ]
        )
    parts.append(
        _table(
            [
                "итерация",
                "сценарий",
                "вкл",
                "N",
                "success",
                "p50",
                "p95",
                "p99",
                "попыток",
                "ретраев",
                "fallback",
                "CB",
                "кэш",
                "$",
                "rps",
            ],
            detail_rows,
        )
    )

    parts.append("")
    parts.append(END_MARKER)
    return "\n".join(parts)


def inject(document: Path, section: str) -> None:
    text = document.read_text(encoding="utf-8") if document.exists() else ""
    if BEGIN_MARKER in text and END_MARKER in text:
        head = text.split(BEGIN_MARKER)[0]
        tail = text.split(END_MARKER, 1)[1]
        document.write_text(head + section + tail, encoding="utf-8")
    else:
        document.write_text(text.rstrip() + "\n\n" + section + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(prog="chaos.report")
    parser.add_argument("--results", default="bench/results")
    parser.add_argument("--document", default="RELIABILITY.md")
    parser.add_argument("--stdout", action="store_true", help="print instead of writing the file")
    args = parser.parse_args()

    results = load_results(Path(args.results))
    if not results:
        print(f"no result files in {args.results}")
        return 1
    section = build_section(results)
    if args.stdout:
        print(section)
    else:
        inject(Path(args.document), section)
        print(f"updated {args.document} from {len(results)} result file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

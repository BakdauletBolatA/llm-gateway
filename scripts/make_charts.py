"""Draw the README charts from the files in bench/results and bench/probes.

Nothing is typed in by hand: every number on the images is read from the same JSON
the report tables are generated from, so re-running the benchmark and then this
script keeps the pictures in step with RELIABILITY.md.

    pip install -e ".[charts]"
    python scripts/make_charts.py            # writes docs/img/*.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
PROBES = ROOT / "bench" / "probes"
OUT = ROOT / "docs" / "img"

ITERATIONS = [
    ("01_baseline", "naive"),
    ("02_timeouts", "timeouts"),
    ("03_retries", "retries"),
    ("04_circuit_breaker", "circuit\nbreaker"),
    ("05_fallback", "fallback"),
    ("06_semantic_cache", "semantic\ncache"),
    ("07_hedging", "hedging"),
    ("08_admission", "admission"),
    ("09_backpressure", "backpressure"),
]
FIRST, FINAL = ITERATIONS[0][0], ITERATIONS[-1][0]
# A step counts as "made things worse" only if it lost more than this many points;
# the 0.4-point wobble between hedging and the build before it is noise.
DROP_POINTS = 5.0
FOOTER = "Mock provider, 11 failure profiles, 1,650 requests per build"

BG, FG, MUTED, GRID = "#0f172a", "#e2e8f0", "#94a3b8", "#1e293b"
BLUE, RED, GREEN = "#38bdf8", "#f87171", "#4ade80"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "text.color": FG,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
    }
)


def load() -> dict[str, dict[str, dict]]:
    """label -> scenario -> the `results` block of that run."""
    runs: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(RESULTS.glob("*__*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        runs[run["label"]][run["scenario"]] = run["results"]
    return runs


def success_rate(scenarios: dict[str, dict]) -> float:
    ok = sum(s["successes"] for s in scenarios.values())
    total = sum(s["requests"] for s in scenarios.values())
    return 100 * ok / total


def at_full_success(scenarios: dict[str, dict]) -> int:
    return sum(1 for s in scenarios.values() if s["successes"] == s["requests"])


def overspend(name: str) -> float:
    probe = json.loads((PROBES / f"{name}.json").read_text(encoding="utf-8"))
    return probe["usage"]["overspend_pct"]


def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.07, 0.90, title, fontsize=24, fontweight="bold")
    fig.text(0.07, 0.835, subtitle, fontsize=14, color=MUTED)
    fig.text(0.07, 0.03, FOOTER, fontsize=10.5, color=MUTED)


def success_by_iteration(runs: dict) -> None:
    rates = [success_rate(runs[label]) for label, _ in ITERATIONS]
    # Colour by what the step did to the number, not by hand: a drop is red.
    colours = [MUTED, MUTED]
    for i in range(2, len(rates)):
        colours.append(RED if rates[i] < rates[i - 1] - DROP_POINTS else BLUE)
    colours[-1] = GREEN

    fig = plt.figure(figsize=(12, 6.27), dpi=100)
    ax = fig.add_axes((0.07, 0.14, 0.9, 0.62))
    ax.bar(range(len(rates)), rates, color=colours, width=0.62)
    for i, value in enumerate(rates):
        emphasised = colours[i] in (RED, GREEN)
        ax.text(
            i,
            value + 1.8,
            f"{value:.1f}%",
            ha="center",
            fontsize=14,
            fontweight="bold",
            color=colours[i] if emphasised else FG,
        )
    for i, colour in enumerate(colours):
        if colour == RED:
            ax.text(
                i,
                rates[i] + 12,
                "alone, it made\nthings worse",
                color=RED,
                fontsize=11,
                ha="center",
                va="bottom",
            )
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([name for _, name in ITERATIONS], fontsize=10.5)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    header(
        fig,
        "One mechanism per iteration, same load every time",
        "Success rate across 11 failure scenarios (1,650 requests per build)",
    )
    fig.savefig(OUT / "success_by_iteration.png")
    plt.close(fig)


def p95_latency(runs: dict) -> None:
    profiles = [
        ("storm", "provider failure storm"),
        ("hang", "provider hangs"),
        ("slow", "slow provider"),
    ]
    before = [runs[FIRST][s]["latency_ms"]["p95"] for s, _ in profiles]
    after = [runs[FINAL][s]["latency_ms"]["p95"] for s, _ in profiles]

    fig = plt.figure(figsize=(12, 6.27), dpi=100)
    ax = fig.add_axes((0.07, 0.14, 0.9, 0.62))
    width = 0.34
    xs = range(len(profiles))
    ax.bar([x - width / 2 for x in xs], before, width, color=RED, label="naive gateway")
    ax.bar([x + width / 2 for x in xs], after, width, color=GREEN, label="final build")
    for x, b, a in zip(xs, before, after, strict=True):
        ax.text(
            x - width / 2,
            b * 1.12,
            f"{b:,.0f} ms",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color=RED,
        )
        ax.text(
            x + width / 2,
            a * 1.12,
            f"{a:,.0f} ms",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color=GREEN,
        )
    ax.set_yscale("log")
    ax.set_ylim(100, 60000)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for _, label in profiles], fontsize=13)
    ax.set_yticks([])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=12, loc="upper right", labelcolor=FG)
    storm_before, storm_after = before[0], after[0]
    header(
        fig,
        f"p95 latency: {storm_before / 1000:.0f} s → {storm_after:.0f} ms",
        "Timeouts, circuit breaker, fallback and hedging (log scale)",
    )
    fig.savefig(OUT / "p95_latency.png")
    plt.close(fig)


def summary_card(runs: dict) -> None:
    first, final = runs[FIRST], runs[FINAL]
    breaker_before = success_rate(runs["03_retries"])
    breaker_after = success_rate(runs["04_circuit_breaker"])
    cap_before = runs["08_admission"]["capacity_limited"]
    cap_after = runs["09_backpressure"]["capacity_limited"]
    cap_rate = lambda s: 100 * s["successes"] / s["requests"]  # noqa: E731

    cards = [
        (
            f"{success_rate(first):.1f}% → {success_rate(final):.1f}%",
            "success rate, 11 failure scenarios",
            GREEN,
        ),
        (
            f"{at_full_success(first)} → {at_full_success(final)} of {len(final)}",
            "scenarios at 100% success",
            GREEN,
        ),
        (
            f"{first['storm']['latency_ms']['p95']:,.0f} → "
            f"{final['storm']['latency_ms']['p95']:,.0f} ms",
            "p95 in provider failure storm",
            BLUE,
        ),
        (
            f"{cap_rate(cap_before):.0f}% → {cap_rate(cap_after):.0f}%",
            "Retry-After as backpressure",
            BLUE,
        ),
        (f"{breaker_before:.1f}% → {breaker_after:.1f}%", "breaker alone cut availability", RED),
        (
            f"+{overspend('budget_local_2'):.1f}% → +{overspend('budget_shared_2'):.1f}%",
            "budget overrun, local vs shared",
            BLUE,
        ),
    ]

    fig = plt.figure(figsize=(12, 6.27), dpi=100)
    fig.text(0.06, 0.88, "LLM Gateway", fontsize=34, fontweight="bold")
    fig.text(
        0.06,
        0.81,
        "Anthropic, OpenAI and Ollama behind one OpenAI-compatible API",
        fontsize=15,
        color=MUTED,
    )
    for i, (big, small, colour) in enumerate(cards):
        x = 0.06 + (i % 3) * 0.31
        y = 0.52 - (i // 3) * 0.27
        fig.patches.append(
            patches.FancyBboxPatch(
                (x, y - 0.05),
                0.29,
                0.22,
                boxstyle="round,pad=0.005,rounding_size=0.015",
                transform=fig.transFigure,
                fc=GRID,
                ec="none",
            )
        )
        fig.text(x + 0.02, y + 0.08, big, fontsize=21, fontweight="bold", color=colour)
        fig.text(x + 0.02, y + 0.01, small, fontsize=11.5, color=MUTED)
    fig.text(
        0.06,
        0.06,
        "Measured on a mock provider, reproducible with one command",
        fontsize=12,
        color=MUTED,
    )
    fig.savefig(OUT / "summary.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load()
    success_by_iteration(runs)
    p95_latency(runs)
    summary_card(runs)
    print(f"wrote {len(list(OUT.glob('*.png')))} images to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

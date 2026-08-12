#!/usr/bin/env bash
# Короткий хаос-прогон для CI: поднимает мок и шлюз, гоняет несколько сценариев
# и проверяет инварианты, которые не должны ломаться никогда.
#
# Проверяется не «стало лучше», а то, что стенд и шлюз в целом живы:
#   healthy        — 100% успеха (иначе сломан happy path);
#   total_outage   — 0% успеха и быстрый отказ (иначе шлюз висит вместо отказа);
#   storm          — хоть какая-то доступность и учёт стоимости сходится.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-bench/ci}"
N="${N:-40}"
CONCURRENCY="${CONCURRENCY:-8}"

./scripts/dev_stack.sh up

cleanup() { ./scripts/dev_stack.sh down >/dev/null 2>&1 || true; }
trap cleanup EXIT

.venv/bin/python -m chaos.run \
  --label ci_smoke \
  --scenario healthy --scenario storm --scenario total_outage \
  --n "$N" --concurrency "$CONCURRENCY" --timeout 15 --settle 1 \
  --out "$OUT" --note "CI smoke"

.venv/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
failures: list[str] = []


def load(scenario: str) -> dict:
    return json.loads((out / f"ci_smoke__{scenario}.json").read_text())


healthy = load("healthy")["results"]
if healthy["success_rate"] != 1.0:
    failures.append(f"healthy success_rate={healthy['success_rate']} (expected 1.0)")
if healthy["latency_ms"]["p95"] > 5000:
    failures.append(f"healthy p95={healthy['latency_ms']['p95']}ms is implausibly high")

outage = load("total_outage")["results"]
if outage["success_rate"] != 0.0:
    failures.append(f"total_outage success_rate={outage['success_rate']} (expected 0.0)")
if outage["latency_ms"]["p95"] > 5000:
    failures.append(
        f"total_outage p95={outage['latency_ms']['p95']}ms: the gateway is hanging "
        "instead of failing fast"
    )

storm = load("storm")["results"]
if storm["success_rate"] <= 0.0:
    failures.append("storm produced no successful answers at all")

for scenario in ("healthy", "storm"):
    result = load(scenario)["results"]
    client, server = result["cost_usd_client"], result["cost_usd_server"]
    if abs(client - server) > 1e-6:
        failures.append(
            f"{scenario}: cost accounting disagrees, client={client} server={server}"
        )

if failures:
    print("CI smoke checks FAILED:")
    for line in failures:
        print(f"  - {line}")
    raise SystemExit(1)

print("CI smoke checks passed:")
for scenario in ("healthy", "storm", "total_outage"):
    result = load(scenario)["results"]
    print(
        f"  {scenario:<14} success={result['success_rate']:.1%} "
        f"p95={result['latency_ms']['p95']:.0f}ms cost=${result['cost_usd_server']:.4f}"
    )
PY

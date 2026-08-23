#!/usr/bin/env bash
# Короткий хаос-прогон для CI: поднимает мок и шлюз, гоняет несколько сценариев
# и проверяет инварианты, которые не должны ломаться никогда.
#
# Проверяется не «стало лучше», а то, что стенд и шлюз в целом живы:
#   healthy        — 100% успеха (иначе сломан happy path);
#   total_outage   — 0% успеха и быстрый отказ (иначе шлюз висит вместо отказа);
#   storm          — хоть какая-то доступность и учёт стоимости сходится.
# Отдельным блоком в конце — проводка общего бюджета: она ломается тихо, поэтому
# проверяется на настоящем процессе, а не только в тестах под ASGI-транспортом.
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

# Метрики проверяются здесь, а не только в тестах: в тестах шлюз живёт под
# ASGI-транспортом, а тут — настоящий процесс, который только что отработал нагрузку.
echo "checking the Prometheus exposition"
metrics="$(curl -fsS "${GATEWAY_URL:-http://127.0.0.1:8080}/metrics")"
for family in \
  llm_gateway_requests_total \
  llm_gateway_request_duration_seconds_bucket \
  llm_gateway_provider_calls_total \
  llm_gateway_circuit_breaker_state \
  llm_gateway_budget_spent_usd; do
  if ! grep -q "^${family}" <<<"$metrics"; then
    echo "  $family is missing from /metrics" >&2
    exit 1
  fi
done
echo "  ok: $(grep -c '^llm_gateway_' <<<"$metrics") gateway samples exposed"

# -- общий бюджет: проверяется проводка, а не поведение под отказами ------------
#
# Замер перерасхода живёт в scripts/run_probes.sh и требует двух реплик. Здесь
# другая задача: поймать сломанную проводку — не применилась миграция, scope не
# доехал до трекера, резервация не коммитится. Всё это снаружи выглядит как
# «шлюз работает», поэтому нужен явный тест, а не наблюдение.
echo "checking the shared budget end to end"
./scripts/dev_stack.sh down >/dev/null 2>&1

# Период обнуляется через SQLAlchemy, а не через psql: клиента может не быть на
# раннере, а зависимость приложения есть всегда.
.venv/bin/python - <<'PY'
import asyncio
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DEFAULT = "postgresql+asyncpg://gateway:gateway@127.0.0.1:5432/llm_gateway"
DSN = os.environ.get("DATABASE_URL", DEFAULT)


async def main() -> None:
    engine = create_async_engine(DSN)
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE llm_attempts, llm_calls"))
        await connection.execute(text("DELETE FROM budget_periods"))
    await engine.dispose()


asyncio.run(main())
PY

GATEWAY_CONFIG_OVERLAY=config/extras/budget_tiny.yaml BUDGET_SCOPE=shared \
  ./scripts/dev_stack.sh up >/dev/null
.venv/bin/python scripts/shared_limit_probe.py \
  --gateway "${GATEWAY_URL:-http://127.0.0.1:8080}" --n 120 --concurrency 8 \
  >"$OUT/budget_shared.json"

metrics="$(curl -fsS "${GATEWAY_URL:-http://127.0.0.1:8080}/metrics")"
.venv/bin/python - "$OUT/budget_shared.json" <<'PY'
import json
import sys
from pathlib import Path

probe = json.loads(Path(sys.argv[1]).read_text())
usage = probe["usage"]
failures: list[str] = []

if usage["budget_scope"] != "shared":
    failures.append(f"budget scope is {usage['budget_scope']!r}, not 'shared'")
if probe["refused_402"] == 0:
    failures.append("nothing was refused: a $0.005 limit must run out within 120 requests")
if probe["served_200"] == 0:
    failures.append("everything was refused: the reservation never settles back")

limit, billed = usage["budget_limit_usd"], usage["cost_usd_recorded"]
# Порог намеренно щедрый: точная величина перерасхода — предмет замера в
# run_probes.sh, здесь достаточно поймать «резервация вообще не держит».
if billed > limit * 1.5:
    failures.append(f"billed ${billed:.6f} against a ${limit} limit: the reservation is not holding")

if failures:
    print("shared budget checks FAILED:")
    for line in failures:
        print(f"  - {line}")
    raise SystemExit(1)

print(
    f"  ok: served={probe['served_200']} refused={probe['refused_402']} "
    f"billed=${billed:.6f} of ${limit} (overspend {usage['overspend_pct']}%)"
)
PY

for check in 'llm_gateway_budget_shared 1.0' 'llm_gateway_budget_backend_errors 0.0'; do
  if ! grep -qx "$check" <<<"$metrics"; then
    echo "  expected '$check' in /metrics, got: $(grep "^${check%% *}" <<<"$metrics")" >&2
    exit 1
  fi
done
echo "  ok: the shared counter is the one actually in use, with no fallbacks"

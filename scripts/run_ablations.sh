#!/usr/bin/env bash
# Ablation: от финальной конфигурации выключаем по одному механизму и смотрим,
# что именно ломается. Каждый прогон — отдельный файл в bench/results.
#
# Требует поднятого Postgres и мока; шлюз перезапускается на каждой конфигурации.
#
#   scripts/run_ablations.sh              # сценарии по умолчанию
#   SCENARIOS="storm hang" scripts/run_ablations.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N="${N:-150}"
CONCURRENCY="${CONCURRENCY:-15}"
SCENARIOS="${SCENARIOS:-storm hang primary_outage}"

declare -a ABLATIONS=(
  "no_timeouts:выключены таймауты"
  "no_retries:выключены ретраи"
  "no_breaker:выключен circuit breaker"
  "no_fallback:выключен fallback"
  "no_cache:выключен семантический кэш"
  "breaker_ratio_09:circuit breaker открывается только при 90% отказов"
)

scenario_args=()
for scenario in $SCENARIOS; do
  scenario_args+=(--scenario "$scenario")
done

for entry in "${ABLATIONS[@]}"; do
  name="${entry%%:*}"
  note="${entry#*:}"
  echo "=== ablation: $name ($note) ==="
  GATEWAY_CONFIG_OVERLAY="config/ablations/${name}.yaml" ./scripts/dev_stack.sh down >/dev/null 2>&1 || true
  sleep 1
  GATEWAY_CONFIG_OVERLAY="config/ablations/${name}.yaml" ./scripts/dev_stack.sh up >/dev/null
  sleep 1
  .venv/bin/python -m chaos.run \
    --label "ablation_${name}" \
    "${scenario_args[@]}" \
    --n "$N" --concurrency "$CONCURRENCY" \
    --note "$note"
done

echo "=== restoring the full configuration ==="
./scripts/dev_stack.sh down >/dev/null 2>&1 || true
sleep 1
./scripts/dev_stack.sh up >/dev/null
echo "done; run 'python -m chaos.report' to refresh the tables"

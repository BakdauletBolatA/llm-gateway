#!/usr/bin/env bash
# Пересобрать ВЕСЬ отчёт с нуля: каждая итерация, ablation, дополнительные прогоны,
# и в конце — таблицы в RELIABILITY.md.
#
# Главное утверждение отчёта — «любую строку можно перепрогнать на текущем коде» —
# должно проверяться одной командой, а не знанием, какой оверлей к чему относится.
#
#   scripts/reproduce_report.sh                      # полный прогон, ~20 минут
#   ONLY=09 scripts/reproduce_report.sh              # перепрогнать одну итерацию
#   OUT=/tmp/probe N=10 SETTLE=0 scripts/reproduce_report.sh   # проверить сам скрипт
#
# ВНИМАНИЕ: по умолчанию скрипт ПЕРЕЗАПИСЫВАЕТ bench/results — в этом и состоит
# «пересобрать отчёт». Если нужно просто проверить, что скрипт живой, задайте OUT:
# с коротким N числа получатся другие, и коммитить их нельзя.
#
# Нужны поднятый PostgreSQL с pgvector и .venv (make install). Мок и шлюз скрипт
# поднимает сам, перезапуская шлюз на каждой конфигурации.
#
# ВАЖНО: во время прогона на машине не должно идти ничего тяжёлого. Один комплект
# замеров уже пришлось выбросить из-за фоновой установки пакета — см. раздел
# «Ограничения» в RELIABILITY.md.
set -euo pipefail

# Отчёт в RELIABILITY.md измерял лексический семантический кэш (хеш-эмбеддер,
# порог 0.60) на воркладе из 22 заведомо разных тем. В config/gateway.yaml по
# умолчанию стоит безопасный exact, поэтому воспроизведение включает измеренный
# режим явно. Переопределите переменные, чтобы измерить другой.
export GW__RELIABILITY__CACHE__ENABLED="${GW__RELIABILITY__CACHE__ENABLED:-true}"
export GW__RELIABILITY__CACHE__SIMILARITY_THRESHOLD="${GW__RELIABILITY__CACHE__SIMILARITY_THRESHOLD:-0.60}"
export GW__RELIABILITY__CACHE__MATCH="${GW__RELIABILITY__CACHE__MATCH:-semantic}"
export GW__RELIABILITY__CACHE__ALLOW_LEXICAL_SEMANTIC="${GW__RELIABILITY__CACHE__ALLOW_LEXICAL_SEMANTIC:-true}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N="${N:-150}"
CONCURRENCY="${CONCURRENCY:-15}"
OUT="${OUT:-bench/results}"
# Пауза между сценариями: зависшие на моке запросы должны догореть, иначе следующий
# сценарий стартует с хвостом предыдущего. Нулевая — только для проверки скрипта.
SETTLE="${SETTLE:-3.0}"
# Подстрока метки: ONLY=09 перепрогонит только девятую итерацию.
ONLY="${ONLY:-}"
PY="${PY:-.venv/bin/python}"

started_at="$(date +%s)"

run() {
  local overlay="$1" label="$2" note="$3"
  shift 3
  echo
  echo "=== $label ==="
  ./scripts/dev_stack.sh down >/dev/null 2>&1 || true
  if [[ -n "$overlay" ]]; then
    GATEWAY_CONFIG_OVERLAY="$overlay" ./scripts/dev_stack.sh up >/dev/null
  else
    ./scripts/dev_stack.sh up >/dev/null
  fi
  "$PY" -m chaos.run --label "$label" --n "$N" --concurrency "$CONCURRENCY" \
    --out "$OUT" --settle "$SETTLE" --note "$note" "$@"
}

echo "воспроизведение отчёта: N=$N, concurrency=$CONCURRENCY"

# 1. Матрица итераций: конфигурация каждой лежит в config/iterations/*.yaml.
for overlay in config/iterations/*.yaml; do
  label="$(basename "$overlay" .yaml)"
  [[ -n "$ONLY" && "$label" != *"$ONLY"* ]] && continue
  note="$(grep -m1 '^# Итерация' "$overlay" | cut -c3- || echo "$label")"
  run "$overlay" "$label" "$note" --all
done

# 2. Ablation: от финальной сборки выключаем по одному механизму.
if [[ -z "$ONLY" ]]; then
  SCENARIOS="storm hang primary_outage slow capacity_limited" OUT="$OUT" ./scripts/run_ablations.sh
fi

if [[ -z "${SKIP_EXTRAS:-}" && -z "$ONLY" ]]; then
  # 3. Прогоны, отвечающие на отдельные вопросы: разогрев кэша, задержка хеджа,
  #    провайдер с квотой конкурентности, rate limit под нагрузкой.
  run "" extra_cache_warmup "N=600: тот же кэш после разогрева" \
    --scenario healthy --scenario storm --n 600

  for delay in 50 150 400 1000; do
    echo
    echo "=== extra_hedge_delay_${delay}ms ==="
    ./scripts/dev_stack.sh down >/dev/null 2>&1 || true
    GW__RELIABILITY__HEDGING__DELAY_MS="$delay" \
      GATEWAY_CONFIG_OVERLAY=config/iterations/07_hedging.yaml \
      ./scripts/dev_stack.sh up >/dev/null
    "$PY" -m chaos.run --label "extra_hedge_delay_${delay}ms" \
      --scenario healthy --scenario slow --scenario hang \
      --n "$N" --concurrency "$CONCURRENCY" --out "$OUT" --settle "$SETTLE" \
      --note "hedging.delay_ms=${delay}"
  done

  run config/extras/capacity_no_bulkhead.yaml extra_capacity_no_bulkhead \
    "без лимита конкурентности" --scenario capacity_limited
  run config/extras/capacity_bulkhead.yaml extra_capacity_bulkhead \
    "лимит шлюза = квота провайдера (2)" --scenario capacity_limited
  run config/extras/capacity_bulkhead_no_hedge.yaml extra_capacity_bulkhead_no_hedge \
    "лимит = квота провайдера (2), хедж выключен" --scenario capacity_limited

  echo
  echo "=== extra_rate_limit_25rps ==="
  ./scripts/dev_stack.sh down >/dev/null 2>&1 || true
  GW__RELIABILITY__RATE_LIMIT__REQUESTS_PER_SECOND=25 GW__RELIABILITY__RATE_LIMIT__BURST=25 \
    ./scripts/dev_stack.sh up >/dev/null
  "$PY" -m chaos.run --label extra_rate_limit_25rps --scenario healthy \
    --n "$N" --concurrency "$CONCURRENCY" --out "$OUT" --settle "$SETTLE" \
    --note "rate limit 25 rps, burst 25 (стенд отдаёт ~100 rps)"
fi

echo
echo "=== восстанавливаем полную конфигурацию и пересобираем таблицы ==="
./scripts/dev_stack.sh down >/dev/null 2>&1 || true
./scripts/dev_stack.sh up >/dev/null
if [[ "$OUT" == "bench/results" ]]; then
  "$PY" -m chaos.report
else
  echo "OUT=$OUT — таблицы не пересобираю, чтобы не смешать пробный прогон с отчётом"
fi

echo
echo "готово за $(( ($(date +%s) - started_at) / 60 )) мин."
echo "проверьте git diff RELIABILITY.md: числа должны отличаться от коммита только шумом."

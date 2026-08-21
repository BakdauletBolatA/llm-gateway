#!/usr/bin/env bash
# Замеры, которые нельзя сделать хаос-харнессом: они про ДВЕ реплики шлюза.
#
# Харнесс гоняет один процесс и отвечает на вопрос «как ведёт себя механизм под
# отказами». Здесь другой вопрос: принадлежит ли лимит процессу или деплойменту.
# Ответ виден только тогда, когда процессов больше одного.
#
# Результаты пишутся в bench/probes/*.json; на них ссылается RELIABILITY.md, и
# tests/test_report_claims.py проверяет, что проза цитирует именно эти числа.
#
# Требуется: поднятый PostgreSQL (DATABASE_URL) и .venv. Docker не нужен.
#   scripts/run_probes.sh            # все замеры
#   OUT=/tmp/probes scripts/run_probes.sh   # не трогая опубликованные результаты
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-bench/probes}"
VENV="${VENV:-$ROOT/.venv}"
PSQL_DSN="${PSQL_DSN:-postgresql://gateway:gateway@127.0.0.1:5432/llm_gateway}"
N="${N:-200}"
CONCURRENCY="${CONCURRENCY:-10}"

mkdir -p "$OUT"

stop_all() {
  ./scripts/dev_stack.sh down >/dev/null 2>&1 || true
  RUN_DIR=.run/replica-b ./scripts/dev_stack.sh down >/dev/null 2>&1 || true
}
trap stop_all EXIT

reset_period() {
  # Замер начинается с нуля: и потраченные деньги, и резервации, и токены.
  psql "$PSQL_DSN" -q \
    -c "TRUNCATE llm_attempts, llm_calls;" \
    -c "DELETE FROM budget_periods;" \
    -c "DELETE FROM rate_limit_buckets;" >/dev/null 2>&1 || true
}

# probe <файл> <оверлей> <реплик> <n> <concurrency> [доп. аргументы пробы...]
probe() {
  local name="$1" overlay="$2" replicas="$3" n="$4" concurrency="$5"
  shift 5
  stop_all
  reset_period
  export GATEWAY_CONFIG_OVERLAY="$overlay"
  ./scripts/dev_stack.sh up >/dev/null 2>&1
  local args=(--gateway http://127.0.0.1:8080)
  if [[ "$replicas" == "2" ]]; then
    RUN_DIR=.run/replica-b GATEWAY_PORT=8082 ./scripts/dev_stack.sh up-gateway >/dev/null 2>&1
    args+=(--gateway http://127.0.0.1:8082)
  fi
  sleep 3
  "$VENV/bin/python" scripts/shared_limit_probe.py "${args[@]}" \
    --n "$n" --concurrency "$concurrency" "$@" >"$OUT/$name.json"
  echo "  $name -> $OUT/$name.json"
}

echo "== rate limit: 25 rps / burst 25, две реплики =="
RATE_LIMIT_SCOPE=local  probe ratelimit_local_2  config/extras/ratelimit_tiny.yaml 2 600 20
RATE_LIMIT_SCOPE=shared probe ratelimit_shared_2 config/extras/ratelimit_tiny.yaml 2 600 20

echo "== бюджет: лимит \$0.005 =="
BUDGET_SCOPE=local  probe budget_local_1  config/extras/budget_tiny.yaml 1 "$N" "$CONCURRENCY"
BUDGET_SCOPE=local  probe budget_local_2  config/extras/budget_tiny.yaml 2 "$N" "$CONCURRENCY"
BUDGET_SCOPE=shared probe budget_shared_1 config/extras/budget_tiny.yaml 1 "$N" "$CONCURRENCY"
BUDGET_SCOPE=shared probe budget_shared_2 config/extras/budget_tiny.yaml 2 "$N" "$CONCURRENCY"
# Тот же прогон, но клиент не задаёт max_tokens: в оценку идёт консервативное
# estimate_output_tokens, и видно вторую половину размена — недоиспользование.
BUDGET_SCOPE=shared probe budget_shared_2_conservative \
  config/extras/budget_tiny.yaml 2 "$N" "$CONCURRENCY" --max-tokens 0

echo "готово: $OUT"

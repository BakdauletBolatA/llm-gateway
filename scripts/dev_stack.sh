#!/usr/bin/env bash
# Start / stop the stack natively, without Docker.
#
# Expects a reachable PostgreSQL with the pgvector extension available, e.g.
#   docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=gateway -e POSTGRES_USER=gateway \
#     -e POSTGRES_DB=llm_gateway pgvector/pgvector:pg16
# or a local server with postgresql-16-pgvector installed.
#
# Usage: scripts/dev_stack.sh {up|down|status|logs}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/.run}"
VENV="${VENV:-$ROOT/.venv}"

export DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://gateway:gateway@127.0.0.1:5432/llm_gateway}"
export MOCK_BASE_URL="${MOCK_BASE_URL:-http://127.0.0.1:8081}"
export GATEWAY_PORT="${GATEWAY_PORT:-8080}"
export MOCK_PORT="${MOCK_PORT:-8081}"

mkdir -p "$RUN_DIR"

start_one() {
  local name="$1" app="$2" port="$3"
  if [[ -f "$RUN_DIR/$name.pid" ]] && kill -0 "$(cat "$RUN_DIR/$name.pid")" 2>/dev/null; then
    echo "$name already running (pid $(cat "$RUN_DIR/$name.pid"))"
    return
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  nohup uvicorn "$app" --factory --host 127.0.0.1 --port "$port" \
    >"$RUN_DIR/$name.log" 2>&1 &
  echo $! >"$RUN_DIR/$name.pid"
  echo "$name started on :$port (pid $!)"
}

wait_for() {
  local url="$1" tries="${2:-60}"
  for _ in $(seq "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "timed out waiting for $url" >&2
  return 1
}

case "${1:-up}" in
  up)
    start_one mock "mock_provider.main:create_app" "$MOCK_PORT"
    start_one gateway "llm_gateway.main:create_app" "$GATEWAY_PORT"
    wait_for "http://127.0.0.1:$MOCK_PORT/healthz"
    wait_for "http://127.0.0.1:$GATEWAY_PORT/healthz"
    echo "stack is up: gateway http://127.0.0.1:$GATEWAY_PORT  mock http://127.0.0.1:$MOCK_PORT"
    ;;
  down)
    for name in gateway mock; do
      if [[ -f "$RUN_DIR/$name.pid" ]]; then
        pid="$(cat "$RUN_DIR/$name.pid")"
        kill "$pid" 2>/dev/null || true
        rm -f "$RUN_DIR/$name.pid"
        echo "$name stopped"
      fi
    done
    ;;
  status)
    for name in gateway mock; do
      if [[ -f "$RUN_DIR/$name.pid" ]] && kill -0 "$(cat "$RUN_DIR/$name.pid")" 2>/dev/null; then
        echo "$name: running (pid $(cat "$RUN_DIR/$name.pid"))"
      else
        echo "$name: stopped"
      fi
    done
    ;;
  logs)
    tail -n 50 "$RUN_DIR"/*.log
    ;;
  *)
    echo "usage: $0 {up|down|status|logs}" >&2
    exit 2
    ;;
esac

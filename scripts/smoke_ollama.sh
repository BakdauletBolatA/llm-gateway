#!/usr/bin/env bash
# Проверка живого провайдера: тот же шлюз, но запрос уходит в локальную Ollama,
# а не в мок. Нужна, чтобы «работает без единого платного ключа» было проверяемым
# утверждением, а не обещанием.
#
#   docker compose --profile ollama up -d --build
#   scripts/smoke_ollama.sh
#
# Отчёт RELIABILITY.md сознательно построен на моке: живая модель даёт разброс
# латентности, зависящий от железа, и таблицу итераций стало бы нельзя сравнивать.
set -euo pipefail

GATEWAY="${GATEWAY_URL:-http://127.0.0.1:8080}"
OLLAMA="${OLLAMA_URL:-http://127.0.0.1:11434}"
MODEL="${OLLAMA_MODEL:-llama3.2}"

echo "1/4 Ollama отвечает?"
if ! curl -fsS "$OLLAMA/api/tags" >/dev/null; then
  echo "  Ollama недоступна на $OLLAMA." >&2
  echo "  Поднимите её: docker compose --profile ollama up -d" >&2
  exit 1
fi
echo "  ok"

echo "2/4 модель $MODEL загружена?"
if ! curl -fsS "$OLLAMA/api/tags" | grep -q "\"$MODEL"; then
  echo "  модели $MODEL нет; тяну (это долго при первом запуске)" >&2
  curl -fsS "$OLLAMA/api/pull" -d "{\"model\":\"$MODEL\"}" >/dev/null
fi
echo "  ok"

echo "3/4 шлюз видит провайдера ollama?"
if ! curl -fsS "$GATEWAY/v1/config" | grep -q '"ollama": *{ *"type": *"ollama", *"enabled": *true'; then
  echo "  провайдер ollama выключен в конфиге шлюза." >&2
  echo "  Поставьте OLLAMA_ENABLED=true в .env и перезапустите gateway." >&2
  exit 1
fi
echo "  ok"

echo "4/4 запрос через шлюз в живую модель"
response="$(curl -fsS -X POST "$GATEWAY/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"local-only\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":32}")"

echo "$response" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
text = payload["choices"][0]["message"]["content"]
usage = payload["usage"]
gw = payload["gateway"]
print(f"  ответ: {text.strip()[:80]!r}")
print(f"  провайдер: {gw[\"provider\"]}, попыток: {gw[\"attempts\"]}, latency: {gw[\"latency_ms\"]}ms")
print(f"  токены: {usage[\"prompt_tokens\"]}/{usage[\"completion_tokens\"]}, стоимость: ${gw[\"cost_usd\"]}")
assert text.strip(), "модель вернула пустой ответ"
assert usage["completion_tokens"] > 0, "провайдер не отчитался о токенах"
'

echo
echo "живой провайдер работает; расходы видны в $GATEWAY/v1/usage?group_by=provider"

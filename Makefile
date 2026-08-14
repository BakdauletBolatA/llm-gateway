.DEFAULT_GOAL := help
SHELL := /bin/bash
VENV ?= .venv
PY   := $(VENV)/bin/python
N    ?= 150
C    ?= 15
LABEL ?= manual

.PHONY: help
help: ## Показать список целей
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(VENV)/bin/activate: pyproject.toml
	uv venv --python 3.12 $(VENV)
	VIRTUAL_ENV=$(VENV) uv pip install -e ".[dev]"
	@touch $(VENV)/bin/activate

.PHONY: install
install: $(VENV)/bin/activate ## Создать venv и поставить зависимости

.PHONY: up
up: ## docker compose up (postgres + mock + gateway)
	docker compose up -d --build
	@echo "gateway: http://127.0.0.1:$${GATEWAY_PORT:-8080}  mock: http://127.0.0.1:$${MOCK_PORT:-8081}"

.PHONY: up-ollama
up-ollama: ## То же самое плюс локальная Ollama
	docker compose --profile ollama up -d --build

.PHONY: down
down: ## Остановить compose
	docker compose down

.PHONY: native-up
native-up: install ## Поднять стенд без Docker (нужен внешний Postgres с pgvector)
	./scripts/dev_stack.sh up

.PHONY: native-down
native-down: ## Остановить нативный стенд
	./scripts/dev_stack.sh down

.PHONY: test
test: install ## Юнит-тесты
	$(PY) -m pytest -q

.PHONY: lint
lint: install ## ruff + mypy
	$(VENV)/bin/ruff check src tests scripts
	$(VENV)/bin/ruff format --check src tests scripts
	$(VENV)/bin/mypy

.PHONY: fmt
fmt: install ## Автоформат
	$(VENV)/bin/ruff format src tests scripts
	$(VENV)/bin/ruff check --fix src tests scripts

.PHONY: bench
bench: ## Прогнать все сценарии хаоса: make bench LABEL=03_retries
	$(PY) -m chaos.run --label $(LABEL) --all --n $(N) --concurrency $(C)

.PHONY: report
report: ## Пересобрать таблицы в RELIABILITY.md из bench/results
	$(PY) -m chaos.report

.PHONY: smoke-ollama
smoke-ollama: ## Проверить, что живой провайдер Ollama отвечает через шлюз
	./scripts/smoke_ollama.sh

.PHONY: reproduce
reproduce: ## Пересобрать весь отчёт с нуля (все итерации, ablation, extras) — ~20 мин
	./scripts/reproduce_report.sh

.PHONY: ablations
ablations: ## Ablation-прогоны: выключить по одному механизму из финальной сборки
	./scripts/run_ablations.sh

.PHONY: ci-smoke
ci-smoke: ## Короткий хаос-прогон с проверкой инвариантов (то же, что в CI)
	./scripts/ci_smoke.sh

"""Prometheus metrics and structured logging.

The gateway already measures itself for the chaos harness (response headers) and
for the ledger (Postgres). This module adds the third consumer — an operator with
Prometheus — from the *same* per-request data, so the three never disagree.

Metrics live in a dedicated registry rather than the global default one: the app
factory can be called many times in one process (every end-to-end test builds its
own app), and a dedicated registry keeps that from turning into duplicate
registration errors or metrics leaking between test cases.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Literal

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.platform_collector import PlatformCollector
from prometheus_client.process_collector import ProcessCollector

from llm_gateway.errors import GatewayError
from llm_gateway.router import ExecutionResult

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: Buckets are chosen for this service, not copied from a default: normal answers
#: land at 40–150 ms, the request deadline is 12 s, and the interesting question is
#: "how much traffic is past the hedge delay", so the ladder is dense in 0.1–1 s.
LATENCY_BUCKETS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, float("inf"))

REGISTRY = CollectorRegistry()
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)


def _counter(name: str, doc: str, labels: tuple[str, ...] = ()) -> Counter:
    return Counter(name, doc, labelnames=labels, registry=REGISTRY)


def _gauge(name: str, doc: str, labels: tuple[str, ...] = ()) -> Gauge:
    return Gauge(name, doc, labelnames=labels, registry=REGISTRY)


REQUESTS = _counter(
    "llm_gateway_requests_total", "Requests served, by route and outcome", ("route", "outcome")
)
REQUEST_ERRORS = _counter(
    "llm_gateway_request_errors_total", "Failed requests, by error kind", ("route", "kind")
)
REQUEST_DURATION = Histogram(
    "llm_gateway_request_duration_seconds",
    "End-to-end request latency as the client sees it",
    labelnames=("route",),
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
PROVIDER_CALLS = _counter(
    "llm_gateway_provider_calls_total",
    "Provider calls, by outcome: success, error, cancelled (lost a hedge race), "
    "discarded (answered too late to be used), skipped_breaker",
    ("provider", "outcome"),
)
PROVIDER_ERRORS = _counter(
    "llm_gateway_provider_errors_total", "Failed provider calls by error kind", ("provider", "kind")
)
PROVIDER_DURATION = Histogram(
    "llm_gateway_provider_call_duration_seconds",
    "Latency of a single provider call",
    labelnames=("provider",),
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)
RETRIES = _counter("llm_gateway_retries_total", "Retries against the same provider", ("route",))
FALLBACKS = _counter(
    "llm_gateway_fallbacks_total", "Hops to the next provider in the chain", ("route",)
)
HEDGES = _counter(
    "llm_gateway_hedges_total", "Duplicate calls launched while a hop was still live", ("route",)
)
BREAKER_SKIPS = _counter(
    "llm_gateway_breaker_skips_total", "Calls short-circuited by an open breaker", ("provider",)
)
CACHE_LOOKUPS = _counter(
    "llm_gateway_cache_lookups_total", "Semantic cache lookups by result", ("result",)
)
COST = _counter("llm_gateway_cost_usd_total", "Money spent, by provider", ("provider",))
COST_WASTED = _counter(
    "llm_gateway_cost_wasted_usd_total", "Money spent on answers that were not served"
)
TOKENS = _counter("llm_gateway_tokens_total", "Tokens billed", ("provider", "direction"))

BREAKER_STATE = _gauge(
    "llm_gateway_circuit_breaker_state",
    "Circuit breaker state: 0 closed, 1 half-open, 2 open",
    ("provider",),
)
BREAKER_FAILURE_RATIO = _gauge(
    "llm_gateway_circuit_breaker_failure_ratio",
    "Failure ratio inside the breaker's current window",
    ("provider",),
)
BUDGET_LIMIT = _gauge("llm_gateway_budget_limit_usd", "Configured budget limit for the period")
BUDGET_SPENT = _gauge("llm_gateway_budget_spent_usd", "Spend recorded for the current period")
RECORDER_QUEUE = _gauge("llm_gateway_recorder_queue_depth", "Call records waiting to be written")
RECORDER_WRITTEN = _gauge(
    "llm_gateway_recorder_records_written", "Call records written to Postgres"
)
RECORDER_DROPPED = _gauge(
    "llm_gateway_recorder_records_dropped",
    "Call records dropped because the queue was full — logging must never block a request",
)
RECORDER_FAILURES = _gauge("llm_gateway_recorder_write_failures", "Failed batch writes")
BULKHEAD_IN_FLIGHT = _gauge(
    "llm_gateway_bulkhead_in_flight", "Calls currently in flight to a provider", ("provider",)
)
BULKHEAD_LIMIT = _gauge(
    "llm_gateway_bulkhead_limit", "Concurrency limit per provider, 0 when unlimited", ("provider",)
)
BULKHEAD_QUEUED = _gauge(
    "llm_gateway_bulkhead_queued", "Calls that had to wait for a slot", ("provider",)
)
BULKHEAD_SHED = _gauge(
    "llm_gateway_bulkhead_shed",
    "Calls shed because no slot came free before the deadline",
    ("provider",),
)

_BREAKER_STATE_VALUES = {"closed": 0.0, "half_open": 1.0, "open": 2.0}


def observe_request(
    *,
    route: str,
    latency_s: float,
    result: ExecutionResult | None = None,
    error: GatewayError | None = None,
) -> None:
    """Record one finished request. Called once per response, success or failure."""
    outcome = "success" if result is not None else "error"
    REQUESTS.labels(route=route, outcome=outcome).inc()
    REQUEST_DURATION.labels(route=route).observe(latency_s)

    if error is not None:
        REQUEST_ERRORS.labels(route=route, kind=str(error.kind)).inc()

    counters = result or error
    if counters is None:  # pragma: no cover - one of the two is always given
        return
    if counters.retries:
        RETRIES.labels(route=route).inc(counters.retries)
    if counters.fallbacks:
        FALLBACKS.labels(route=route).inc(counters.fallbacks)
    if counters.hedges:
        HEDGES.labels(route=route).inc(counters.hedges)

    for attempt in counters.attempt_records:
        PROVIDER_CALLS.labels(provider=attempt.provider, outcome=attempt.outcome).inc()
        if attempt.outcome == "skipped_breaker":
            BREAKER_SKIPS.labels(provider=attempt.provider).inc()
        elif attempt.outcome == "error" and attempt.error_kind:
            PROVIDER_ERRORS.labels(provider=attempt.provider, kind=attempt.error_kind).inc()
        if attempt.latency_ms:
            PROVIDER_DURATION.labels(provider=attempt.provider).observe(attempt.latency_ms / 1000)
        if attempt.cost_usd:
            COST.labels(provider=attempt.provider).inc(attempt.cost_usd)
        if attempt.tokens_in:
            TOKENS.labels(provider=attempt.provider, direction="in").inc(attempt.tokens_in)
        if attempt.tokens_out:
            TOKENS.labels(provider=attempt.provider, direction="out").inc(attempt.tokens_out)

    if result is not None:
        CACHE_LOOKUPS.labels(result="hit" if result.cache_hit else "miss").inc()
        if result.wasted_cost_usd:
            COST_WASTED.inc(result.wasted_cost_usd)


def refresh_gauges(
    *,
    breakers: list[dict[str, Any]],
    bulkheads: list[dict[str, Any]],
    budget: dict[str, Any],
    recorder: dict[str, Any],
) -> None:
    """Copy current state into gauges. Called on scrape, not on every request.

    Gauges mirror state that already lives elsewhere (breaker registry, budget
    tracker, recorder queue), so reading it at scrape time keeps one source of
    truth instead of two counters that can drift apart.
    """
    for snapshot in breakers:
        provider = str(snapshot["provider"])
        state = _BREAKER_STATE_VALUES.get(str(snapshot["state"]), 0.0)
        BREAKER_STATE.labels(provider=provider).set(state)
        BREAKER_FAILURE_RATIO.labels(provider=provider).set(float(snapshot["failure_ratio"]))
    for snapshot in bulkheads:
        provider = str(snapshot["provider"])
        BULKHEAD_IN_FLIGHT.labels(provider=provider).set(float(snapshot["in_flight"]))
        BULKHEAD_LIMIT.labels(provider=provider).set(float(snapshot["limit"]))
        BULKHEAD_QUEUED.labels(provider=provider).set(float(snapshot["queued"]))
        BULKHEAD_SHED.labels(provider=provider).set(float(snapshot["shed"]))
    BUDGET_LIMIT.set(float(budget.get("limit_usd", 0.0)))
    BUDGET_SPENT.set(float(budget.get("spent_usd", 0.0)))
    RECORDER_QUEUE.set(float(recorder.get("queued", 0)))
    RECORDER_WRITTEN.set(float(recorder.get("written", 0)))
    RECORDER_DROPPED.set(float(recorder.get("dropped", 0)))
    RECORDER_FAILURES.set(float(recorder.get("write_failures", 0)))


def render() -> bytes:
    payload: bytes = generate_latest(REGISTRY)
    return payload


# -- logging ------------------------------------------------------------------

#: Attributes every LogRecord carries. Anything else was passed as `extra=` and is
#: worth putting into the JSON line.
_STANDARD_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with `extra=` fields promoted to top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str, log_format: Literal["text", "json"]) -> None:
    """Install the root handler. `force=True` so a second app in one process
    (every end-to-end test) replaces the handler instead of stacking another one."""
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)

    # uvicorn installs its own handlers with propagate=False, so its access log would
    # stay plain text while everything else became JSON. One process, one log format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        server_logger = logging.getLogger(name)
        server_logger.handlers.clear()
        server_logger.propagate = True

    # One "HTTP Request: POST ..." line per provider call doubles the log volume of a
    # gateway whose entire job is making provider calls, and says nothing the request
    # log does not already say.
    logging.getLogger("httpx").setLevel(logging.WARNING)

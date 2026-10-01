"""Locust scenario: closed-loop chat completions through the gateway.

Each virtual user sends one request, waits for the answer, and sends the next. That
is the honest model for a backend whose speed is the thing being measured: the load
adapts to how fast the system answers instead of piling up an unbounded queue.

Every response is recorded (time, latency, status, provider) and written to
LT_RAW when the run ends; loadtest/run.py turns that into the report. Locust's own
percentile buckets are not used, so the mock and live runs share one definition.

Driven by loadtest/run.py; settings come from the environment:
    LT_ROUTE        route to call              (default: live-local)
    LT_MAX_TOKENS   completion cap per request (default: 64)
    LT_RAW          where to write raw records (required)
    LT_TIMEOUT_S    client-side timeout        (default: 90)
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

from locust import HttpUser, constant, events, task

ROUTE = os.environ.get("LT_ROUTE", "live-local")
MAX_TOKENS = int(os.environ.get("LT_MAX_TOKENS", "64"))
TIMEOUT_S = float(os.environ.get("LT_TIMEOUT_S", "90"))
RAW_PATH = os.environ.get("LT_RAW")

PROMPTS = [
    "Explain what a database index is in two sentences.",
    "Give me three tips for writing readable Python.",
    "What is the difference between TCP and UDP?",
    "Summarise how exponential backoff works.",
    "Why do people use a circuit breaker in a distributed system?",
    "Write a one-line commit message for a bug fix in a retry loop.",
    "What does HTTP status 429 mean?",
    "List four causes of a slow SQL query.",
    "Describe what a load balancer does.",
    "How is a thread different from a process?",
    "What is idempotency and why does it matter for retries?",
    "Name two trade-offs of caching API responses.",
]

_records: list[dict[str, Any]] = []
_started = time.time()


@events.test_start.add_listener
def _reset(**_: Any) -> None:
    global _started
    _started = time.time()
    _records.clear()


class ChatUser(HttpUser):
    wait_time = constant(0)

    @task
    def chat(self) -> None:
        payload = {
            "model": ROUTE,
            "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        }
        sent = time.time()
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT_S,
            catch_response=True,
            name=f"POST {ROUTE}",
        ) as response:
            elapsed_ms = (time.time() - sent) * 1000
            status = response.status_code or 0
            provider = response.headers.get("x-gateway-provider") if status else None
            _records.append(
                {
                    "t": round(sent - _started, 3),
                    "latency_ms": round(elapsed_ms, 1),
                    "status": status,
                    "provider": provider,
                    "fallbacks": int(response.headers.get("x-gateway-fallbacks", 0) or 0)
                    if status
                    else 0,
                }
            )
            if status != 200:
                response.failure(f"HTTP {status}")


@events.quitting.add_listener
def _dump(**_: Any) -> None:
    if not RAW_PATH:
        return
    with open(RAW_PATH, "w") as handle:
        for record in _records:
            handle.write(json.dumps(record) + "\n")

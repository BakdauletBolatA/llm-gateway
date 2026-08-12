"""Orchestration behaviour with stub providers.

The end-to-end tests prove the wiring; these prove the decisions — how many
attempts, in what order, against which provider, and when the gateway gives up.
Providers here are stubs, so a test can express "fail twice then succeed"
exactly and assert on the call log.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from llm_gateway.errors import ErrorKind, ProviderError
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.registry import build_timeout
from llm_gateway.reliability.breaker import BreakerRegistry
from llm_gateway.router import DeadlineExceeded, Orchestrator
from llm_gateway.schemas import ChatCompletionRequest, ChatMessage
from llm_gateway.settings import Settings, load_settings


@dataclass
class StubAdapter:
    """Replays a scripted sequence of outcomes and records every call."""

    name: str
    script: list[Any] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    delay_s: float = 0.0

    async def complete(
        self, request: ChatCompletionRequest, model: str, client: httpx.AsyncClient
    ) -> ProviderResponse:
        self.calls.append(model)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        outcome = self.script.pop(0) if self.script else "ok"
        if isinstance(outcome, BaseException):
            raise outcome
        return ProviderResponse(
            text=f"answer from {self.name}", model=model, tokens_in=10, tokens_out=20
        )


class StubRegistry:
    def __init__(self, adapters: dict[str, StubAdapter]) -> None:
        self._adapters = adapters

    def adapter(self, name: str) -> StubAdapter:
        return self._adapters[name]

    def client(self, name: str) -> Any:
        return None

    def enabled_providers(self) -> list[str]:
        return sorted(self._adapters)


class StubBudget:
    def __init__(self) -> None:
        self.spent = 0.0

    async def refresh(self, force: bool = False) -> None:
        return None

    def check(self, estimated_cost_usd: float, api_key_id: str | None = None) -> None:
        return None

    def record_spend(self, cost_usd: float, api_key_id: str | None = None) -> None:
        self.spent += cost_usd


class DisabledCache:
    """Cache that is never consulted — the cache path has its own tests."""

    def enabled_for(self, temperature: float | None, client_opt_in: bool) -> bool:
        return False

    @staticmethod
    def scope_for(route: str, model: str) -> str:
        return f"{route}:{model}"


def provider_error(kind: ErrorKind, retry_after_s: float | None = None) -> ProviderError:
    return ProviderError(
        f"stub {kind}",
        kind=kind,
        provider="mock_primary",
        model="mock-gpt-4o-mini",
        status_code=500,
        retry_after_s=retry_after_s,
    )


def settings_with(**reliability: dict[str, Any]) -> Settings:
    """Shipped config with every mechanism forced off, then the requested patches.

    Starting from "all off" rather than from whatever config/gateway.yaml
    currently ships keeps these tests meaningful as iterations flip flags: a test
    that does not mention fallback must not silently acquire a three-provider
    chain later.
    """
    tree = load_settings("config/gateway.yaml").model_dump()
    for section in (
        "timeouts",
        "retries",
        "circuit_breaker",
        "fallback",
        "hedging",
        "bulkhead",
        "rate_limit",
        "cache",
    ):
        tree["reliability"][section]["enabled"] = False
    for section, patch in reliability.items():
        tree["reliability"][section].update(patch)
    return Settings.model_validate(tree)


def build(settings: Settings, adapters: dict[str, StubAdapter]) -> tuple[Orchestrator, StubBudget]:
    registry = StubRegistry(adapters)
    breakers = BreakerRegistry(settings.reliability.circuit_breaker, list(adapters))
    budget = StubBudget()
    orchestrator = Orchestrator(
        settings,
        registry,  # type: ignore[arg-type]
        breakers,
        DisabledCache(),  # type: ignore[arg-type]
        budget,  # type: ignore[arg-type]
        rng=random.Random(1),
    )
    return orchestrator, budget


REQUEST = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")], temperature=0.0)


# -- timeouts -----------------------------------------------------------------


def test_disabled_timeouts_mean_no_timeout_at_all() -> None:
    """The naive baseline really has no limit, not a large one."""
    settings = settings_with(timeouts={"enabled": False})
    timeout = build_timeout(settings.reliability.timeouts)
    assert timeout.read is None and timeout.connect is None


def test_enabled_timeouts_are_applied_per_phase() -> None:
    settings = settings_with(
        timeouts={"enabled": True, "connect_s": 1.5, "read_s": 4.0, "write_s": 2.0}
    )
    timeout = build_timeout(settings.reliability.timeouts)
    assert (timeout.connect, timeout.read, timeout.write) == (1.5, 4.0, 2.0)


async def test_deadline_cuts_the_retry_ladder_and_reports_the_real_cause() -> None:
    """A per-call timeout is not enough: attempts must share one wall-clock budget.

    When the budget runs out the caller gets the actual upstream failure rather
    than a synthetic timeout — that is the more useful error to log.
    """
    settings = settings_with(
        timeouts={"enabled": True, "total_s": 0.15},
        retries={"enabled": True, "max_attempts": 5, "base_delay_ms": 60, "jitter": "none"},
    )
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)] * 5)
    orchestrator, _ = build(settings, {"mock_primary": adapter})

    with pytest.raises(ProviderError) as excinfo:
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert excinfo.value.kind is ErrorKind.SERVER_ERROR
    assert len(adapter.calls) < 5, "the deadline must cut the retry ladder short"


async def test_deadline_stops_the_chain_before_trying_the_next_provider() -> None:
    settings = settings_with(
        timeouts={"enabled": True, "total_s": 0.1},
        retries={"enabled": False},
        fallback={"enabled": True},
    )
    slow = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)], delay_s=0.2)
    spare = StubAdapter("mock_secondary")
    third = StubAdapter("mock_tertiary")
    orchestrator, _ = build(
        settings,
        {"mock_primary": slow, "mock_secondary": spare, "mock_tertiary": third},
    )

    with pytest.raises(DeadlineExceeded):
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert spare.calls == [], "no time left means the fallback is not even attempted"
    assert third.calls == []


async def test_without_a_deadline_every_attempt_is_used() -> None:
    settings = settings_with(
        timeouts={"enabled": False},
        retries={"enabled": True, "max_attempts": 3, "base_delay_ms": 1, "jitter": "none"},
    )
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)] * 3)
    orchestrator, _ = build(settings, {"mock_primary": adapter})

    with pytest.raises(ProviderError):
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert len(adapter.calls) == 3


# -- retries ------------------------------------------------------------------


def retry_settings(**overrides: Any) -> Settings:
    return settings_with(
        timeouts={"enabled": True, "total_s": 30.0},
        retries={
            "enabled": True,
            "max_attempts": 3,
            "base_delay_ms": 1,
            "jitter": "none",
            **overrides,
        },
    )


async def test_a_transient_failure_is_retried_and_the_request_succeeds() -> None:
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)])
    orchestrator, _ = build(retry_settings(), {"mock_primary": adapter})

    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert result.attempts == 2
    assert result.retries == 1
    assert result.fallbacks == 0
    assert len(adapter.calls) == 2


async def test_retries_stop_at_max_attempts() -> None:
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)] * 5)
    orchestrator, _ = build(retry_settings(max_attempts=3), {"mock_primary": adapter})

    with pytest.raises(ProviderError) as excinfo:
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert len(adapter.calls) == 3
    assert excinfo.value.attempts == 3
    assert excinfo.value.retries == 2


async def test_a_client_error_is_not_retried() -> None:
    """Retrying a malformed request just multiplies the same 400."""
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.BAD_REQUEST)])
    orchestrator, _ = build(retry_settings(), {"mock_primary": adapter})

    with pytest.raises(ProviderError) as excinfo:
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert excinfo.value.kind is ErrorKind.BAD_REQUEST
    assert len(adapter.calls) == 1


async def test_an_error_kind_absent_from_retry_on_is_not_retried() -> None:
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.AUTH)])
    orchestrator, _ = build(retry_settings(retry_on=["timeout"]), {"mock_primary": adapter})

    with pytest.raises(ProviderError):
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert len(adapter.calls) == 1


async def test_disabled_retries_make_exactly_one_attempt() -> None:
    settings = settings_with(retries={"enabled": False}, timeouts={"enabled": True})
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)])
    orchestrator, _ = build(settings, {"mock_primary": adapter})

    with pytest.raises(ProviderError):
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert len(adapter.calls) == 1


async def test_a_healthy_provider_is_called_once_and_billed_once() -> None:
    adapter = StubAdapter("mock_primary")
    orchestrator, budget = build(retry_settings(), {"mock_primary": adapter})

    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert result.attempts == 1 and result.retries == 0
    assert result.cost_usd > 0
    assert budget.spent == pytest.approx(result.cost_usd)


# -- circuit breaker ----------------------------------------------------------


def breaker_settings(**overrides: Any) -> Settings:
    return settings_with(
        timeouts={"enabled": True, "total_s": 30.0},
        retries={"enabled": False},
        circuit_breaker={
            "enabled": True,
            "window_s": 60.0,
            "min_calls": 3,
            "failure_ratio": 0.5,
            "cooldown_s": 60.0,
            "half_open_max_calls": 1,
            **overrides,
        },
    )


async def test_an_open_breaker_stops_calling_the_provider_at_all() -> None:
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)] * 20)
    orchestrator, _ = build(breaker_settings(), {"mock_primary": adapter})

    for _ in range(3):
        with pytest.raises(ProviderError):
            await orchestrator.execute(REQUEST, route_name="chaos-default")
    calls_before = len(adapter.calls)

    with pytest.raises(ProviderError) as excinfo:
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert excinfo.value.kind is ErrorKind.CIRCUIT_OPEN
    assert excinfo.value.breaker_skips == 1
    assert len(adapter.calls) == calls_before, "an open breaker must not touch the network"


async def test_a_breaker_open_on_one_provider_does_not_affect_another() -> None:
    settings = breaker_settings()
    broken = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)] * 10)
    healthy = StubAdapter("mock_secondary")
    orchestrator, _ = build(settings, {"mock_primary": broken, "mock_secondary": healthy})

    for _ in range(3):
        with pytest.raises(ProviderError):
            await orchestrator.execute(REQUEST, route_name="chaos-default")

    # A different route pointing at the healthy provider still works.
    result = await orchestrator.execute(REQUEST, route_name="chaos-default-secondary")
    assert result.provider == "mock_secondary"


async def test_a_client_error_does_not_open_the_breaker() -> None:
    """A 400 proves the provider is answering; it is our payload that is wrong."""
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.BAD_REQUEST)] * 10)
    orchestrator, _ = build(breaker_settings(min_calls=2), {"mock_primary": adapter})

    for _ in range(5):
        with pytest.raises(ProviderError) as excinfo:
            await orchestrator.execute(REQUEST, route_name="chaos-default")
        assert excinfo.value.kind is ErrorKind.BAD_REQUEST
    assert len(adapter.calls) == 5, "every request must still reach the provider"


async def test_a_disabled_breaker_never_short_circuits() -> None:
    adapter = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)] * 10)
    settings = settings_with(circuit_breaker={"enabled": False}, retries={"enabled": False})
    orchestrator, _ = build(settings, {"mock_primary": adapter})

    for _ in range(6):
        with pytest.raises(ProviderError) as excinfo:
            await orchestrator.execute(REQUEST, route_name="chaos-default")
        assert excinfo.value.kind is ErrorKind.SERVER_ERROR
    assert len(adapter.calls) == 6


# -- fallback chain -----------------------------------------------------------


def fallback_settings(**overrides: Any) -> Settings:
    return settings_with(
        timeouts={"enabled": True, "total_s": 30.0},
        retries={"enabled": False},
        circuit_breaker={"enabled": False},
        fallback={"enabled": True, "max_providers": 3, **overrides},
    )


async def test_the_chain_moves_to_the_next_provider_on_failure() -> None:
    first = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)])
    second = StubAdapter("mock_secondary")
    third = StubAdapter("mock_tertiary")
    orchestrator, _ = build(
        fallback_settings(),
        {"mock_primary": first, "mock_secondary": second, "mock_tertiary": third},
    )

    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert result.provider == "mock_secondary"
    assert result.fallbacks == 1
    assert third.calls == [], "the chain must stop at the first provider that answers"


async def test_the_chain_walks_all_the_way_down() -> None:
    adapters = {
        "mock_primary": StubAdapter("mock_primary", [provider_error(ErrorKind.SERVER_ERROR)]),
        "mock_secondary": StubAdapter("mock_secondary", [provider_error(ErrorKind.OVERLOADED)]),
        "mock_tertiary": StubAdapter("mock_tertiary"),
    }
    orchestrator, _ = build(fallback_settings(), adapters)

    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert result.provider == "mock_tertiary"
    assert result.fallbacks == 2
    assert result.cost_usd == 0.0, "the local model is priced at zero in the shipped config"


async def test_when_every_provider_fails_the_last_error_is_returned() -> None:
    adapters = {
        name: StubAdapter(name, [provider_error(ErrorKind.SERVER_ERROR)])
        for name in ("mock_primary", "mock_secondary", "mock_tertiary")
    }
    orchestrator, _ = build(fallback_settings(), adapters)

    with pytest.raises(ProviderError) as excinfo:
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert excinfo.value.fallbacks == 2
    assert excinfo.value.attempts == 3


async def test_max_providers_bounds_the_chain() -> None:
    adapters = {
        "mock_primary": StubAdapter("mock_primary", [provider_error(ErrorKind.SERVER_ERROR)]),
        "mock_secondary": StubAdapter("mock_secondary", [provider_error(ErrorKind.SERVER_ERROR)]),
        "mock_tertiary": StubAdapter("mock_tertiary"),
    }
    orchestrator, _ = build(fallback_settings(max_providers=2), adapters)

    with pytest.raises(ProviderError):
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert adapters["mock_tertiary"].calls == []


async def test_a_hop_skipped_by_an_open_breaker_still_counts_as_a_fallback() -> None:
    """The metric must reflect where the traffic actually went.

    An earlier version counted only hops it had called, so a request served by
    the second provider because the first one's breaker was open was reported as
    "no fallback" — which understated the shift by an order of magnitude.
    """
    settings = settings_with(
        timeouts={"enabled": True, "total_s": 30.0},
        retries={"enabled": False},
        circuit_breaker={
            "enabled": True,
            "min_calls": 2,
            "failure_ratio": 0.5,
            "window_s": 60.0,
            "cooldown_s": 60.0,
        },
        fallback={"enabled": True, "max_providers": 3},
    )
    broken = StubAdapter("mock_primary", [provider_error(ErrorKind.SERVER_ERROR)] * 10)
    spare = StubAdapter("mock_secondary")
    orchestrator, _ = build(
        settings,
        {"mock_primary": broken, "mock_secondary": spare, "mock_tertiary": StubAdapter("t")},
    )

    for _ in range(2):
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    calls_to_broken = len(broken.calls)

    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert result.provider == "mock_secondary"
    assert result.breaker_skips == 1
    assert result.fallbacks == 1, "traffic moved to another provider: that is a fallback"
    assert len(broken.calls) == calls_to_broken, "the dead provider was not called again"


# -- hedging ------------------------------------------------------------------


def hedging_settings(**overrides: Any) -> Settings:
    return settings_with(
        timeouts={"enabled": True, "total_s": 30.0},
        retries={"enabled": False},
        circuit_breaker={"enabled": False},
        fallback={"enabled": True, "max_providers": 3},
        hedging={"enabled": True, "delay_ms": 100, "max_in_flight": 2, **overrides},
    )


async def test_a_slow_provider_is_raced_and_the_faster_answer_wins() -> None:
    """The failure mode hedging exists for: a provider that works, slowly.

    Nothing else in the stack helps here. There is no error to retry, nothing for
    the breaker to count, and no failure to trigger a fallback.
    """
    slow = StubAdapter("mock_primary", delay_s=2.0)
    fast = StubAdapter("mock_secondary")
    orchestrator, _ = build(
        hedging_settings(),
        {"mock_primary": slow, "mock_secondary": fast, "mock_tertiary": StubAdapter("t")},
    )

    started = time.perf_counter()
    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    elapsed_s = time.perf_counter() - started

    assert result.provider == "mock_secondary"
    assert result.hedges == 1
    assert result.fallbacks == 1, "the answer came from the second link of the chain"
    assert slow.calls, "the slow provider was still given its chance first"
    assert elapsed_s < 1.0, f"waited {elapsed_s:.2f}s instead of racing the slow provider"


async def test_a_fast_provider_is_never_hedged() -> None:
    """The delay is the whole design: below it, hedging costs nothing."""
    fast = StubAdapter("mock_primary")
    spare = StubAdapter("mock_secondary")
    orchestrator, budget = build(
        hedging_settings(),
        {"mock_primary": fast, "mock_secondary": spare, "mock_tertiary": StubAdapter("t")},
    )

    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert result.provider == "mock_primary"
    assert result.hedges == 0
    assert result.wasted_cost_usd == 0.0
    assert spare.calls == [], "a healthy provider must not be duplicated"
    assert budget.spent == pytest.approx(result.cost_usd)


async def test_the_loser_of_a_hedge_race_is_cancelled_and_logged() -> None:
    """A cancelled call still went out, so it is logged as spent work.

    Without the record the attempt log would claim one provider call where two
    were actually made, and the report would understate the load hedging creates.
    """
    slow = StubAdapter("mock_primary", delay_s=2.0)
    orchestrator, _ = build(
        hedging_settings(),
        {
            "mock_primary": slow,
            "mock_secondary": StubAdapter("mock_secondary"),
            "mock_tertiary": StubAdapter("t"),
        },
    )

    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    outcomes = {(record.provider, record.outcome) for record in result.attempt_records}
    assert ("mock_primary", "cancelled") in outcomes
    assert ("mock_secondary", "success") in outcomes
    assert result.attempts == 2, "both calls were really made"
    assert [record.attempt_no for record in result.attempt_records] == [1, 2]


async def test_a_failure_moves_on_immediately_and_is_not_counted_as_a_hedge() -> None:
    """Launching after a failure is a plain fallback: nothing runs in parallel."""
    broken = StubAdapter("mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)])
    spare = StubAdapter("mock_secondary")
    orchestrator, _ = build(
        hedging_settings(delay_ms=5000),
        {"mock_primary": broken, "mock_secondary": spare, "mock_tertiary": StubAdapter("t")},
    )

    started = time.perf_counter()
    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    elapsed_s = time.perf_counter() - started

    assert result.provider == "mock_secondary"
    assert result.fallbacks == 1
    assert result.hedges == 0, "nothing was in flight, so this is a fallback, not a hedge"
    assert elapsed_s < 1.0, "a failure must not wait for the hedge delay"


async def test_without_hedging_the_slow_provider_is_waited_out() -> None:
    """Same providers, hedging off: this is the row hedging is compared against."""
    slow = StubAdapter("mock_primary", delay_s=0.6)
    fast = StubAdapter("mock_secondary")
    settings = hedging_settings()
    settings.reliability.hedging.enabled = False
    orchestrator, _ = build(
        settings,
        {"mock_primary": slow, "mock_secondary": fast, "mock_tertiary": StubAdapter("t")},
    )

    started = time.perf_counter()
    result = await orchestrator.execute(REQUEST, route_name="chaos-default")
    elapsed_s = time.perf_counter() - started

    assert result.provider == "mock_primary"
    assert result.hedges == 0
    assert fast.calls == []
    assert elapsed_s >= 0.5, "without a hedge the client pays the slow provider's latency"


async def test_the_deadline_still_bounds_a_hedged_request() -> None:
    """Two providers in flight must not extend the wall-clock budget."""
    settings = hedging_settings()
    settings.reliability.timeouts.total_s = 0.4
    adapters = {
        name: StubAdapter(name, delay_s=5.0)
        for name in ("mock_primary", "mock_secondary", "mock_tertiary")
    }
    orchestrator, _ = build(settings, adapters)

    started = time.perf_counter()
    with pytest.raises(DeadlineExceeded) as excinfo:
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    elapsed_s = time.perf_counter() - started

    assert excinfo.value.hedges == 1
    assert excinfo.value.attempts == 2
    assert elapsed_s < 1.5, f"the deadline of 0.4s was not enforced (took {elapsed_s:.2f}s)"
    assert adapters["mock_tertiary"].calls == [], "max_in_flight=2 bounds the parallel calls"


async def test_a_hedge_is_not_launched_when_there_is_no_time_left_for_it() -> None:
    """A hedge that cannot finish before the deadline is not worth the money."""
    settings = hedging_settings(delay_ms=300)
    settings.reliability.timeouts.total_s = 0.2
    slow = StubAdapter("mock_primary", delay_s=5.0)
    spare = StubAdapter("mock_secondary")
    orchestrator, _ = build(
        settings,
        {"mock_primary": slow, "mock_secondary": spare, "mock_tertiary": StubAdapter("t")},
    )

    with pytest.raises(DeadlineExceeded):
        await orchestrator.execute(REQUEST, route_name="chaos-default")
    assert spare.calls == [], "the deadline arrived before the hedge delay did"

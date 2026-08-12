"""Orchestration behaviour with stub providers.

The end-to-end tests prove the wiring; these prove the decisions — how many
attempts, in what order, against which provider, and when the gateway gives up.
Providers here are stubs, so a test can express "fail twice then succeed"
exactly and assert on the call log.
"""

from __future__ import annotations

import asyncio
import random
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
    base = load_settings("config/gateway.yaml")
    tree = base.model_dump()
    for section, patch in reliability.items():
        tree["reliability"][section].update(patch)
    return Settings.model_validate(tree)


def build(
    settings: Settings, adapters: dict[str, StubAdapter]
) -> tuple[Orchestrator, StubBudget]:
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


REQUEST = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="hi")], temperature=0.0
)


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
    slow = StubAdapter(
        "mock_primary", script=[provider_error(ErrorKind.SERVER_ERROR)], delay_s=0.2
    )
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

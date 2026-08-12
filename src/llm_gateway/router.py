"""Request orchestration.

One place composes every reliability mechanism, in this order:

    semantic cache -> budget -> [ per provider in chain: breaker -> retries -> call ]

Each mechanism is switched on by config, which is what makes the iteration table in
RELIABILITY.md possible: the same code path runs for the naive baseline and for the
final build, with different flags.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

from llm_gateway.budget import BudgetTracker
from llm_gateway.cache.store import SemanticCache
from llm_gateway.cost import compute_cost_usd, estimate_tokens
from llm_gateway.db.recorder import AttemptRecord
from llm_gateway.errors import (
    ErrorKind,
    GatewayError,
    NoProviderAvailableError,
    ProviderError,
)
from llm_gateway.providers.registry import ProviderRegistry
from llm_gateway.reliability.breaker import BreakerRegistry
from llm_gateway.reliability.retry import compute_backoff, should_retry
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.settings import Settings

logger = logging.getLogger(__name__)

#: Provider errors that are the caller's fault. They are not retried, they do not
#: trigger a fallback, and for the circuit breaker they count as a *healthy* call:
#: a 400 means the provider answered and rejected our payload.
CLIENT_FAULT_KINDS = frozenset({ErrorKind.BAD_REQUEST})


class DeadlineExceeded(GatewayError):
    kind = ErrorKind.DEADLINE_EXCEEDED


@dataclass
class ExecutionResult:
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    finish_reason: str
    attempts: int = 0
    retries: int = 0
    fallbacks: int = 0
    breaker_skips: int = 0
    cache_hit: bool = False
    cache_similarity: float | None = None
    provider_latency_ms: int = 0
    attempt_records: list[AttemptRecord] = field(default_factory=list)


@dataclass
class _Progress:
    """Mutable counters shared between the attempt loop and the error path."""

    attempts: int = 0
    retries: int = 0
    hop_index: int = 0
    breaker_skips: int = 0
    provider_latency_ms: int = 0
    records: list[AttemptRecord] = field(default_factory=list)

    @property
    def fallbacks(self) -> int:
        """How many providers were passed over before the one that answered.

        Counted by position in the chain rather than by "hops actually called",
        because a hop skipped by an open circuit breaker is still a fallback from
        the client's point of view — that is precisely when the traffic moves.
        """
        return self.hop_index

    def apply_to(self, error: GatewayError) -> GatewayError:
        error.attempts = self.attempts
        error.retries = self.retries
        error.fallbacks = self.fallbacks
        error.breaker_skips = self.breaker_skips
        error.provider_latency_ms = self.provider_latency_ms
        error.attempt_records = list(self.records)
        return error


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        registry: ProviderRegistry,
        breakers: BreakerRegistry,
        cache: SemanticCache,
        budget: BudgetTracker,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.breakers = breakers
        self.cache = cache
        self.budget = budget
        self._rng = rng or random.Random()

    # -- helpers -----------------------------------------------------------

    def _max_attempts(self) -> int:
        retries = self.settings.reliability.retries
        return retries.max_attempts if retries.enabled else 1

    def _deadline(self, started: float) -> float | None:
        """Wall-clock budget for the whole request, retries and fallbacks included.

        Without it, `max_attempts x read_timeout` can exceed any per-call timeout
        and the client still waits far too long.
        """
        timeouts = self.settings.reliability.timeouts
        return started + timeouts.total_s if timeouts.enabled else None

    @staticmethod
    def _remaining(deadline: float | None) -> float | None:
        return None if deadline is None else deadline - time.monotonic()

    def _estimate_cost(self, request: ChatCompletionRequest, provider: str, model: str) -> float:
        tokens_in = estimate_tokens(request.prompt_text())
        tokens_out = request.max_tokens or self.settings.budget.estimate_output_tokens
        return compute_cost_usd(self.settings.pricing, provider, model, tokens_in, tokens_out)

    # -- main entry point --------------------------------------------------

    async def execute(
        self,
        request: ChatCompletionRequest,
        *,
        route_name: str,
        api_key_id: str | None = None,
    ) -> ExecutionResult:
        started = time.monotonic()
        deadline = self._deadline(started)
        chain = self.settings.resolve_chain(route_name)
        if not chain:
            raise NoProviderAvailableError(
                f"route {route_name!r} has no enabled providers; "
                "check providers.*.enabled in the config"
            )

        prompt = request.prompt_text()
        scope = SemanticCache.scope_for(route_name, chain[0].model)
        cacheable = self.cache.enabled_for(request.temperature, request.cache)

        # 1. Cache. A hit costs nothing, so it is served even when the budget is spent.
        if cacheable:
            hit = await self.cache.lookup(scope, prompt)
            if hit is not None:
                return ExecutionResult(
                    text=hit.text,
                    provider=hit.provider,
                    model=hit.model,
                    tokens_in=hit.tokens_in,
                    tokens_out=hit.tokens_out,
                    cost_usd=0.0,
                    finish_reason="stop",
                    cache_hit=True,
                    cache_similarity=hit.similarity,
                    provider_latency_ms=int((time.monotonic() - started) * 1000),
                )

        # 2. Budget. A refusal before spending, not an invoice afterwards.
        await self.budget.refresh()
        estimate = self._estimate_cost(request, chain[0].provider, chain[0].model)
        self.budget.check(estimate, api_key_id)

        # 3. Provider chain.
        progress = _Progress()
        last_error: GatewayError | None = None
        max_attempts = self._max_attempts()

        for hop_index, hop in enumerate(chain):
            adapter = self.registry.adapter(hop.provider)
            client = self.registry.client(hop.provider)
            breaker = self.breakers.get(hop.provider)
            progress.hop_index = hop_index
            hop_entered = False

            for attempt_in_hop in range(1, max_attempts + 1):
                remaining = self._remaining(deadline)
                if remaining is not None and remaining <= 0:
                    raise progress.apply_to(
                        DeadlineExceeded(
                            f"request deadline of "
                            f"{self.settings.reliability.timeouts.total_s}s exceeded "
                            f"after {progress.attempts} attempt(s)"
                        )
                    )

                if not breaker.allow():
                    progress.breaker_skips += 1
                    progress.records.append(
                        AttemptRecord(
                            attempt_no=progress.attempts + 1,
                            hop_index=hop_index,
                            provider=hop.provider,
                            model=hop.model,
                            outcome="skipped_breaker",
                            error_kind=str(ErrorKind.CIRCUIT_OPEN),
                        )
                    )
                    last_error = ProviderError(
                        f"circuit breaker is open for provider {hop.provider!r}",
                        kind=ErrorKind.CIRCUIT_OPEN,
                        provider=hop.provider,
                        model=hop.model,
                    )
                    break  # try the next provider in the chain

                progress.attempts += 1
                if hop_entered:
                    progress.retries += 1
                hop_entered = True

                call_started = time.monotonic()
                try:
                    response = await adapter.complete(request, hop.model, client)
                except ProviderError as error:
                    latency_ms = int((time.monotonic() - call_started) * 1000)
                    progress.provider_latency_ms += latency_ms
                    client_fault = error.kind in CLIENT_FAULT_KINDS
                    breaker.record(ok=client_fault)
                    progress.records.append(
                        AttemptRecord(
                            attempt_no=progress.attempts,
                            hop_index=hop_index,
                            provider=hop.provider,
                            model=hop.model,
                            outcome="error",
                            error_kind=str(error.kind),
                            http_status=error.status_code,
                            latency_ms=latency_ms,
                        )
                    )
                    last_error = error

                    if client_fault:
                        raise progress.apply_to(error) from None

                    if attempt_in_hop < max_attempts and should_retry(
                        error.kind, self.settings.reliability.retries
                    ):
                        delay = compute_backoff(
                            attempt_in_hop,
                            self.settings.reliability.retries,
                            retry_after_s=error.retry_after_s,
                            rng=self._rng,
                        )
                        remaining = self._remaining(deadline)
                        if remaining is not None and delay >= remaining:
                            break  # no time left for another attempt here
                        progress.records[-1].backoff_ms = int(delay * 1000)
                        await asyncio.sleep(delay)
                        continue
                    break  # out of attempts for this provider: fall back

                latency_ms = int((time.monotonic() - call_started) * 1000)
                progress.provider_latency_ms += latency_ms
                breaker.record(ok=True)
                cost = compute_cost_usd(
                    self.settings.pricing,
                    hop.provider,
                    hop.model,
                    response.tokens_in,
                    response.tokens_out,
                )
                progress.records.append(
                    AttemptRecord(
                        attempt_no=progress.attempts,
                        hop_index=hop_index,
                        provider=hop.provider,
                        model=hop.model,
                        outcome="success",
                        http_status=200,
                        latency_ms=latency_ms,
                        tokens_in=response.tokens_in,
                        tokens_out=response.tokens_out,
                        cost_usd=cost,
                    )
                )
                self.budget.record_spend(cost, api_key_id)

                if cacheable:
                    self.cache.store_later(
                        scope=scope,
                        prompt=prompt,
                        response_text=response.text,
                        provider=hop.provider,
                        model=hop.model,
                        tokens_in=response.tokens_in,
                        tokens_out=response.tokens_out,
                        cost_usd=cost,
                    )

                return ExecutionResult(
                    text=response.text,
                    provider=hop.provider,
                    model=hop.model,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    cost_usd=cost,
                    finish_reason=response.finish_reason,
                    attempts=progress.attempts,
                    retries=progress.retries,
                    fallbacks=progress.fallbacks,
                    breaker_skips=progress.breaker_skips,
                    provider_latency_ms=progress.provider_latency_ms,
                    attempt_records=progress.records,
                )

        raise progress.apply_to(
            last_error
            or NoProviderAvailableError("no provider in the chain accepted the request")
        )

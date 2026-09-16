"""Request orchestration.

One place composes every reliability mechanism, in this order:

    semantic cache -> budget -> per provider in the chain:
        circuit breaker -> concurrency slot -> retries -> call

(The rate limit sits one layer up, in the HTTP handler: it refuses before any of
this happens, including before the cache lookup.)

The chain is walked one provider at a time, or — with hedging on — the next
provider is launched in parallel after a delay and the first answer wins.

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
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.registry import ProviderRegistry
from llm_gateway.reliability.breaker import BreakerRegistry
from llm_gateway.reliability.bulkhead import BulkheadRegistry
from llm_gateway.reliability.retry import compute_backoff, should_retry
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.settings import RouteTarget, Settings

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
    hedges: int = 0
    wasted_cost_usd: float = 0.0
    cache_hit: bool = False
    cache_similarity: float | None = None
    provider_latency_ms: int = 0
    attempt_records: list[AttemptRecord] = field(default_factory=list)


@dataclass
class _HopProgress:
    """Counters for one provider in the chain: one hop is up to max_attempts calls."""

    hop_index: int
    attempts: int = 0
    retries: int = 0
    breaker_skips: int = 0
    provider_latency_ms: int = 0
    records: list[AttemptRecord] = field(default_factory=list)


@dataclass
class _HopOutcome:
    """What one hop produced: an answer, or the error that ended it."""

    hop: RouteTarget
    progress: _HopProgress
    response: ProviderResponse | None = None
    cost_usd: float = 0.0
    error: GatewayError | None = None
    #: Stop the whole request instead of trying the next provider: either the
    #: caller's payload is wrong, or the wall-clock budget is gone.
    fatal: bool = False

    @property
    def ok(self) -> bool:
        return self.response is not None


@dataclass
class _RequestProgress:
    """Everything accumulated while serving one request, across every hop.

    Hops are separate objects rather than one shared counter because with hedging
    two of them run at the same time; totals are sums, and the per-hop attempt log
    stays attributable to its provider.
    """

    hops: list[_HopProgress] = field(default_factory=list)
    hedges: int = 0
    winner_hop: int | None = None
    #: Answers that arrived after the request was already served by someone else.
    #: They were generated upstream, so they are billed and logged.
    discarded: list[_HopOutcome] = field(default_factory=list)

    def hop(self, hop_index: int) -> _HopProgress:
        progress = _HopProgress(hop_index=hop_index)
        self.hops.append(progress)
        return progress

    @property
    def attempts(self) -> int:
        return sum(hop.attempts for hop in self.hops)

    @property
    def retries(self) -> int:
        return sum(hop.retries for hop in self.hops)

    @property
    def breaker_skips(self) -> int:
        return sum(hop.breaker_skips for hop in self.hops)

    @property
    def provider_latency_ms(self) -> int:
        """Provider time consumed, summed over hops.

        With hedging this exceeds the request's wall clock on purpose: two calls
        ran at once, and both of them cost the upstream real work.
        """
        return sum(hop.provider_latency_ms for hop in self.hops)

    @property
    def fallbacks(self) -> int:
        """How many providers were passed over before the one that answered.

        Counted by position in the chain rather than by "hops actually called",
        because a hop skipped by an open circuit breaker is still a fallback from
        the client's point of view — that is precisely when the traffic moves.
        """
        if self.winner_hop is not None:
            return self.winner_hop
        return max((hop.hop_index for hop in self.hops), default=0)

    def attempt_records(self) -> list[AttemptRecord]:
        merged = [
            record for hop in sorted(self.hops, key=lambda h: h.hop_index) for record in hop.records
        ]
        for number, record in enumerate(merged, start=1):
            record.attempt_no = number
        return merged

    def apply_to(self, error: GatewayError) -> GatewayError:
        error.attempts = self.attempts
        error.retries = self.retries
        error.fallbacks = self.fallbacks
        error.breaker_skips = self.breaker_skips
        error.hedges = self.hedges
        error.provider_latency_ms = self.provider_latency_ms
        error.attempt_records = self.attempt_records()
        return error


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        registry: ProviderRegistry,
        breakers: BreakerRegistry,
        cache: SemanticCache,
        budget: BudgetTracker,
        bulkheads: BulkheadRegistry | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.breakers = breakers
        self.cache = cache
        self.budget = budget
        self.bulkheads = bulkheads or BulkheadRegistry(
            settings.reliability.bulkhead, registry.enabled_providers()
        )
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

    def _deadline_error(self, attempts: int, provider: str | None = None) -> DeadlineExceeded:
        where = f" on {provider}" if provider else ""
        return DeadlineExceeded(
            f"request deadline of {self.settings.reliability.timeouts.total_s}s "
            f"exceeded after {attempts} attempt(s){where}"
        )

    def _slot_wait_s(self, deadline: float | None) -> float | None:
        """How long a request may wait for a provider slot.

        The configured queue timeout, further clipped by the request deadline: there
        is no point queueing for a slot the request will not live long enough to use.
        """
        bulkhead = self.settings.reliability.bulkhead
        wait = bulkhead.queue_timeout_s if bulkhead.enabled else None
        remaining = self._remaining(deadline)
        if remaining is None:
            return wait
        remaining = max(remaining, 0.0)
        return remaining if wait is None else min(wait, remaining)

    def _is_backpressure(self, error: ProviderError) -> bool:
        """Did the provider answer "I am alive, come back in N seconds"?

        A `Retry-After` header is the provider telling us its own terms, which is
        not the same signal as a failure. When this is on, such a call is left out
        of the breaker's window entirely: it says nothing about whether the provider
        is healthy, only that it is busy right now.
        """
        return (
            self.settings.reliability.circuit_breaker.retry_after_is_backpressure
            and error.retry_after_s is not None
        )

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

        cache_key = SemanticCache.key_for(
            route=route_name,
            model=chain[0].model,
            tenant=api_key_id,
            context=request.context_text(),
            query=request.user_text(),
        )
        cacheable = self.cache.enabled_for(request.temperature, request.cache)

        # 1. Cache. A hit costs nothing, so it is served even when the budget is spent.
        if cacheable:
            hit = await self.cache.lookup(cache_key)
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

        # 2. Budget. A refusal before spending, not an invoice afterwards. In shared
        # mode the estimate is committed in Postgres here and corrected below, so a
        # second replica cannot spend the same dollar while this request is in flight.
        await self.budget.refresh()
        estimate = self._estimate_cost(request, chain[0].provider, chain[0].model)
        reservation = await self.budget.reserve(estimate, api_key_id)
        spent = 0.0
        try:
            # 3. Provider chain, sequentially or with the next provider raced in.
            progress = _RequestProgress()
            hedging = self.settings.reliability.hedging
            if hedging.enabled and len(chain) > 1:
                decisive = await self._walk_hedged(request, chain, deadline, progress)
            else:
                decisive = await self._walk_chain(request, chain, deadline, progress)

            # A hedge can produce a second answer that arrived too late to be used. The
            # upstream still did the work, so it is billed and logged as "discarded" —
            # otherwise the report would understate what hedging costs.
            wasted = 0.0
            for extra in progress.discarded:
                wasted += extra.cost_usd
                if extra.progress.records:
                    extra.progress.records[-1].outcome = "discarded"

            spent = wasted

            response = decisive.response
            if response is None:
                raise progress.apply_to(
                    decisive.error
                    or NoProviderAvailableError("no provider in the chain accepted the request")
                )
            hop = decisive.hop
            spent += decisive.cost_usd

            if cacheable:
                self.cache.store_later(
                    key=cache_key,
                    response_text=response.text,
                    provider=hop.provider,
                    model=hop.model,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    cost_usd=decisive.cost_usd,
                )

            return ExecutionResult(
                text=response.text,
                provider=hop.provider,
                model=hop.model,
                tokens_in=response.tokens_in,
                tokens_out=response.tokens_out,
                cost_usd=decisive.cost_usd + wasted,
                wasted_cost_usd=wasted,
                finish_reason=response.finish_reason,
                attempts=progress.attempts,
                retries=progress.retries,
                fallbacks=progress.fallbacks,
                breaker_skips=progress.breaker_skips,
                hedges=progress.hedges,
                provider_latency_ms=progress.provider_latency_ms,
                attempt_records=progress.attempt_records(),
            )
        finally:
            # Settle to what the request really cost. A failed request spent nothing
            # and gets its reservation back — otherwise one outage would eat the day's
            # budget in refusals for money that was never billed.
            await self.budget.settle(reservation, spent)

    # -- walking the chain -------------------------------------------------

    async def _walk_chain(
        self,
        request: ChatCompletionRequest,
        chain: list[RouteTarget],
        deadline: float | None,
        progress: _RequestProgress,
    ) -> _HopOutcome:
        """One provider at a time: the next one is called only after this one failed."""
        outcome: _HopOutcome | None = None
        for hop_index, hop in enumerate(chain):
            outcome = await self._run_hop(
                request,
                hop,
                deadline=deadline,
                hop_progress=progress.hop(hop_index),
                totals=progress,
            )
            if outcome.ok:
                progress.winner_hop = hop_index
                return outcome
            if outcome.fatal:
                return outcome
        if outcome is None:  # pragma: no cover - resolve_chain never returns an empty chain
            return _HopOutcome(
                hop=chain[0],
                progress=progress.hop(0),
                error=NoProviderAvailableError("the provider chain is empty"),
            )
        return outcome

    async def _walk_hedged(
        self,
        request: ChatCompletionRequest,
        chain: list[RouteTarget],
        deadline: float | None,
        progress: _RequestProgress,
    ) -> _HopOutcome:
        """Race the chain: launch the next provider when the current one goes quiet.

        A hop is launched either because the previous one failed (a plain fallback)
        or because it has been silent for longer than the hedge delay (a hedge).
        The first answer wins and the rest are cancelled.
        """
        hedging = self.settings.reliability.hedging
        pending: set[asyncio.Task[_HopOutcome]] = set()
        last_failure: _HopOutcome | None = None
        next_hop = 0

        def launch() -> None:
            """Start the next provider in the chain."""
            nonlocal next_hop
            if pending:
                # Something is still in flight, so this call duplicates work rather
                # than replacing it: that is what makes it a hedge and not a fallback.
                progress.hedges += 1
            pending.add(
                asyncio.create_task(
                    self._run_hop(
                        request,
                        chain[next_hop],
                        deadline=deadline,
                        hop_progress=progress.hop(next_hop),
                        totals=progress,
                    ),
                    name=f"hop-{next_hop}",
                )
            )
            next_hop += 1

        async def cancel_losers() -> None:
            for task in pending:
                task.cancel()
            # Awaiting the cancellation is what lets each hop log the call it had
            # already sent upstream before it lost the race.
            await asyncio.gather(*pending, return_exceptions=True)
            pending.clear()

        launch()
        while pending:
            room_for_a_hedge = next_hop < len(chain) and len(pending) < hedging.max_in_flight
            done, still_running = await asyncio.wait(
                pending,
                timeout=self._hedge_window(hedging.delay_s if room_for_a_hedge else None, deadline),
                return_when=asyncio.FIRST_COMPLETED,
            )
            pending = still_running

            if not done:
                remaining = self._remaining(deadline)
                if remaining is not None and remaining <= 0:
                    hop_progress = progress.hops[-1]
                    await cancel_losers()
                    return _HopOutcome(
                        hop=chain[hop_progress.hop_index],
                        progress=hop_progress,
                        error=self._deadline_error(progress.attempts),
                        fatal=True,
                    )
                launch()  # the current provider has gone quiet: race the next one
                continue

            # Prefer an answer over an error, and an error that ends the whole
            # request over one that merely ends a hop.
            completed = sorted(
                (task.result() for task in done),
                key=lambda outcome: (not outcome.ok, not outcome.fatal),
            )
            head = completed[0]
            if head.ok or head.fatal:
                for extra in completed[1:]:
                    if extra.ok:
                        progress.discarded.append(extra)
                    else:
                        last_failure = extra
                if head.ok:
                    progress.winner_hop = head.progress.hop_index
                await cancel_losers()
                return head

            last_failure = completed[-1]
            if next_hop < len(chain) and len(pending) < hedging.max_in_flight:
                launch()  # a provider failed: move on to the next one

        if last_failure is None:  # pragma: no cover - the loop only ends after a failure
            return _HopOutcome(
                hop=chain[0],
                progress=progress.hops[0],
                error=NoProviderAvailableError("no provider in the chain accepted the request"),
            )
        return last_failure

    def _hedge_window(self, window_s: float | None, deadline: float | None) -> float | None:
        """How long to wait before launching the next provider.

        Bounded by the request deadline: there is no point starting a hedge that
        cannot finish in time.
        """
        remaining = self._remaining(deadline)
        if remaining is None:
            return window_s
        remaining = max(remaining, 0.0)
        return remaining if window_s is None else min(window_s, remaining)

    async def _run_hop(
        self,
        request: ChatCompletionRequest,
        hop: RouteTarget,
        *,
        deadline: float | None,
        hop_progress: _HopProgress,
        totals: _RequestProgress,
    ) -> _HopOutcome:
        """One provider with its retry ladder. Never raises for a provider failure.

        `totals` is only read for error messages; the counters this call produces go
        into `hop_progress`, which is per-provider so that hedged hops running at the
        same time do not overwrite each other.
        """
        progress = hop_progress
        adapter = self.registry.adapter(hop.provider)
        client = self.registry.client(hop.provider)
        breaker = self.breakers.get(hop.provider)
        outcome = _HopOutcome(hop=hop, progress=progress)
        max_attempts = self._max_attempts()

        for attempt_in_hop in range(1, max_attempts + 1):
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 0:
                outcome.error = self._deadline_error(totals.attempts, hop.provider)
                outcome.fatal = True
                return outcome

            if not breaker.allow():
                progress.breaker_skips += 1
                progress.records.append(
                    AttemptRecord(
                        attempt_no=progress.attempts + 1,
                        hop_index=progress.hop_index,
                        provider=hop.provider,
                        model=hop.model,
                        outcome="skipped_breaker",
                        error_kind=str(ErrorKind.CIRCUIT_OPEN),
                    )
                )
                outcome.error = ProviderError(
                    f"circuit breaker is open for provider {hop.provider!r}",
                    kind=ErrorKind.CIRCUIT_OPEN,
                    provider=hop.provider,
                    model=hop.model,
                )
                return outcome  # try the next provider in the chain

            bulkhead = self.bulkheads.get(hop.provider)
            if not await bulkhead.acquire(self._slot_wait_s(deadline)):
                # Our own queue to this provider is full. Nothing was sent, so the
                # breaker learns nothing — a full queue is our problem, not the
                # provider's — and the chain moves on to the next one.
                progress.records.append(
                    AttemptRecord(
                        attempt_no=progress.attempts + 1,
                        hop_index=progress.hop_index,
                        provider=hop.provider,
                        model=hop.model,
                        outcome="shed_bulkhead",
                        error_kind=str(ErrorKind.CAPACITY),
                    )
                )
                outcome.error = ProviderError(
                    f"no capacity for provider {hop.provider!r}: "
                    f"{self.settings.reliability.bulkhead.max_concurrent_per_provider} "
                    "concurrent calls already in flight",
                    kind=ErrorKind.CAPACITY,
                    provider=hop.provider,
                    model=hop.model,
                )
                return outcome

            progress.attempts += 1
            if attempt_in_hop > 1:
                progress.retries += 1

            call_started = time.monotonic()
            try:
                try:
                    response = await adapter.complete(request, hop.model, client)
                finally:
                    # The slot goes back the moment the call ends: a backoff sleep
                    # must not sit on provider capacity it is not using.
                    bulkhead.release()
            except asyncio.CancelledError:
                # Lost a hedge race. The call was already sent upstream, so it is
                # logged as spent work; the breaker learns nothing from it, because
                # we never found out whether the provider was going to answer.
                latency_ms = int((time.monotonic() - call_started) * 1000)
                progress.provider_latency_ms += latency_ms
                progress.records.append(
                    AttemptRecord(
                        attempt_no=progress.attempts,
                        hop_index=progress.hop_index,
                        provider=hop.provider,
                        model=hop.model,
                        outcome="cancelled",
                        latency_ms=latency_ms,
                    )
                )
                raise
            except ProviderError as error:
                latency_ms = int((time.monotonic() - call_started) * 1000)
                progress.provider_latency_ms += latency_ms
                client_fault = error.kind in CLIENT_FAULT_KINDS
                if not self._is_backpressure(error):
                    breaker.record(ok=client_fault)
                progress.records.append(
                    AttemptRecord(
                        attempt_no=progress.attempts,
                        hop_index=progress.hop_index,
                        provider=hop.provider,
                        model=hop.model,
                        outcome="error",
                        error_kind=str(error.kind),
                        http_status=error.status_code,
                        latency_ms=latency_ms,
                    )
                )
                outcome.error = error

                if client_fault:
                    outcome.fatal = True
                    return outcome

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
                        return outcome  # no time left for another attempt here
                    progress.records[-1].backoff_ms = int(delay * 1000)
                    await asyncio.sleep(delay)
                    continue
                return outcome  # out of attempts for this provider: fall back

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
                    hop_index=progress.hop_index,
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
            outcome.response = response
            outcome.cost_usd = cost
            return outcome

        return outcome

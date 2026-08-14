"""FastAPI application: OpenAI-compatible chat completions plus ops endpoints."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text as sql_text

from llm_gateway import observability
from llm_gateway.budget import BudgetTracker, period_start
from llm_gateway.cache.embedder import Embedder, HashingEmbedder, OllamaEmbedder
from llm_gateway.cache.store import SemanticCache
from llm_gateway.db import migrate
from llm_gateway.db.recorder import CallRecord, CallRecorder
from llm_gateway.db.session import Database
from llm_gateway.errors import AuthenticationError, ErrorKind, GatewayError, ThrottledError
from llm_gateway.providers.registry import ProviderRegistry
from llm_gateway.reliability.breaker import BreakerRegistry
from llm_gateway.reliability.bulkhead import BulkheadRegistry
from llm_gateway.reliability.ratelimit import RateLimiter
from llm_gateway.router import ExecutionResult, Orchestrator
from llm_gateway.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorBody,
    ErrorResponse,
    new_request_id,
)
from llm_gateway.settings import Settings, load_settings
from llm_gateway.usage import GroupBy, usage_summary

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    settings: Settings
    database: Database
    registry: ProviderRegistry
    breakers: BreakerRegistry
    bulkheads: BulkheadRegistry
    limiter: RateLimiter
    cache: SemanticCache
    budget: BudgetTracker
    recorder: CallRecorder
    orchestrator: Orchestrator
    embed_client: httpx.AsyncClient | None = None
    started_at: float = 0.0


def _build_embedder(
    settings: Settings,
) -> tuple[Embedder, httpx.AsyncClient | None]:
    cache_config = settings.reliability.cache
    if cache_config.embedder == "ollama":
        provider = settings.providers.get("ollama")
        if provider is None:
            raise ValueError("cache.embedder=ollama but no 'ollama' provider is configured")
        client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        embedder = OllamaEmbedder(
            provider.base_url, cache_config.ollama_embed_model, cache_config.embedding_dim, client
        )
        return embedder, client
    return HashingEmbedder(cache_config.embedding_dim), None


async def build_state(settings: Settings) -> AppState:
    database = Database(settings.database)
    if settings.database.run_migrations_on_startup:
        await migrate.upgrade(settings.database.dsn)

    registry = ProviderRegistry(settings)
    breakers = BreakerRegistry(settings.reliability.circuit_breaker, registry.enabled_providers())
    bulkheads = BulkheadRegistry(
        settings.reliability.bulkhead,
        registry.enabled_providers(),
        {name: provider.max_concurrent for name, provider in settings.providers.items()},
    )
    limiter = RateLimiter(settings.reliability.rate_limit, database)
    embedder, embed_client = _build_embedder(settings)
    cache = SemanticCache(settings.reliability.cache, embedder, database)
    budget = BudgetTracker(settings.budget, settings.auth, database)
    await budget.refresh(force=True)
    recorder = CallRecorder(database, settings.database)
    recorder.start()
    orchestrator = Orchestrator(settings, registry, breakers, cache, budget, bulkheads)

    return AppState(
        settings=settings,
        database=database,
        registry=registry,
        breakers=breakers,
        bulkheads=bulkheads,
        limiter=limiter,
        cache=cache,
        budget=budget,
        recorder=recorder,
        orchestrator=orchestrator,
        embed_client=embed_client,
        started_at=time.time(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    observability.configure_logging(settings.app.log_level, settings.app.log_format)
    state = await build_state(settings)
    app.state.core = state
    logger.info(
        "gateway ready: providers=%s routes=%s reliability=%s",
        registry_summary(state),
        sorted(settings.routes.definitions),
        {
            "timeouts": settings.reliability.timeouts.enabled,
            "retries": settings.reliability.retries.enabled,
            "circuit_breaker": settings.reliability.circuit_breaker.enabled,
            "fallback": settings.reliability.fallback.enabled,
            "hedging": settings.reliability.hedging.enabled,
            "bulkhead": settings.reliability.bulkhead.enabled,
            "rate_limit": settings.reliability.rate_limit.enabled,
            "cache": settings.reliability.cache.enabled,
        },
    )
    try:
        yield
    finally:
        await state.cache.drain()
        await state.recorder.stop()
        await state.registry.aclose()
        if state.embed_client is not None:
            await state.embed_client.aclose()
        await state.database.aclose()


def registry_summary(state: AppState) -> list[str]:
    return state.registry.enabled_providers()


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.core
    return state


def authenticate(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> str | None:
    """Returns the API key id, or None when auth is disabled."""
    state: AppState = request.app.state.core
    auth = state.settings.auth
    if not auth.enabled:
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthenticationError("missing or malformed Authorization header")
    presented = authorization.split(" ", 1)[1].strip()
    key = auth.lookup(presented)
    if key is None:
        raise AuthenticationError("unknown API key")
    return key.id


def _gateway_headers(
    request_id: str,
    *,
    result: ExecutionResult | None = None,
    error: GatewayError | None = None,
    latency_ms: int = 0,
) -> dict[str, str]:
    """Per-request telemetry in headers.

    The chaos harness aggregates from these, which keeps the measurement client
    independent of the database.
    """
    headers = {
        "X-Gateway-Request-Id": request_id,
        "X-Gateway-Latency-Ms": str(latency_ms),
    }
    if result is not None:
        headers.update(
            {
                "X-Gateway-Provider": result.provider,
                "X-Gateway-Model": result.model,
                "X-Gateway-Attempts": str(result.attempts),
                "X-Gateway-Retries": str(result.retries),
                "X-Gateway-Fallbacks": str(result.fallbacks),
                "X-Gateway-Breaker-Skips": str(result.breaker_skips),
                "X-Gateway-Hedges": str(result.hedges),
                "X-Gateway-Cache": "hit" if result.cache_hit else "miss",
                "X-Gateway-Cost-Usd": f"{result.cost_usd:.6f}",
                "X-Gateway-Tokens": f"{result.tokens_in}/{result.tokens_out}",
            }
        )
        if result.cache_similarity is not None:
            headers["X-Gateway-Cache-Similarity"] = f"{result.cache_similarity:.4f}"
        if result.wasted_cost_usd:
            # Paid for, not served: an answer that lost a hedge race.
            headers["X-Gateway-Cost-Wasted-Usd"] = f"{result.wasted_cost_usd:.6f}"
    if error is not None:
        headers.update(
            {
                "X-Gateway-Error-Kind": str(error.kind),
                "X-Gateway-Attempts": str(error.attempts),
                "X-Gateway-Retries": str(error.retries),
                "X-Gateway-Fallbacks": str(error.fallbacks),
                "X-Gateway-Breaker-Skips": str(error.breaker_skips),
                "X-Gateway-Hedges": str(error.hedges),
                "X-Gateway-Cache": "miss",
                "X-Gateway-Cost-Usd": "0.000000",
            }
        )
    return headers


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="llm-gateway",
        version="0.1.0",
        summary="Reliability-focused gateway over Anthropic, OpenAI and Ollama",
        lifespan=lifespan,
    )
    app.state.settings = settings or load_settings()

    # -- chat completions ---------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(
        payload: ChatCompletionRequest,
        state: Annotated[AppState, Depends(get_state)],
        api_key_id: Annotated[str | None, Depends(authenticate)],
    ) -> JSONResponse:
        request_id = new_request_id()
        route_name = payload.model or state.settings.routes.default
        started = time.monotonic()

        # Admission control comes first, before the cache and before the route is
        # even resolved: the point of a rate limit is to refuse work cheaply.
        retry_after_s = await state.limiter.check(api_key_id)
        if retry_after_s is not None:
            error: GatewayError = ThrottledError(
                f"rate limit of {state.settings.reliability.rate_limit.requests_per_second}"
                " req/s exceeded",
                retry_after_s=retry_after_s,
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            observability.observe_request(
                route=route_name, latency_s=latency_ms / 1000, error=error
            )
            state.recorder.submit(
                CallRecord(
                    request_id=request_id,
                    route=route_name,
                    api_key_id=api_key_id,
                    outcome="error",
                    error_kind=str(error.kind),
                    http_status=error.http_status,
                    latency_ms=latency_ms,
                    prompt_chars=len(payload.prompt_text()),
                )
            )
            body = ErrorResponse(
                error=ErrorBody(
                    type="rate_limit_error",
                    message=error.message,
                    kind=str(error.kind),
                    request_id=request_id,
                )
            )
            headers = _gateway_headers(request_id, error=error, latency_ms=latency_ms)
            headers["Retry-After"] = str(max(1, math.ceil(retry_after_s)))
            return JSONResponse(status_code=429, content=body.model_dump(), headers=headers)

        if route_name not in state.settings.routes.definitions:
            known = ", ".join(sorted(state.settings.routes.definitions))
            body = ErrorResponse(
                error=ErrorBody(
                    type="invalid_request_error",
                    message=f"unknown route {route_name!r}; configured routes: {known}",
                    kind=str(ErrorKind.BAD_REQUEST),
                    request_id=request_id,
                )
            )
            return JSONResponse(
                status_code=400,
                content=body.model_dump(),
                headers=_gateway_headers(request_id),
            )

        try:
            result = await state.orchestrator.execute(
                payload, route_name=route_name, api_key_id=api_key_id
            )
        except GatewayError as error:
            latency_ms = int((time.monotonic() - started) * 1000)
            observability.observe_request(
                route=route_name, latency_s=latency_ms / 1000, error=error
            )
            logger.warning(
                "request failed: %s",
                error.message,
                extra={
                    "request_id": request_id,
                    "route": route_name,
                    "error_kind": str(error.kind),
                    "http_status": error.http_status,
                    "attempts": error.attempts,
                    "retries": error.retries,
                    "fallbacks": error.fallbacks,
                    "hedges": error.hedges,
                    "latency_ms": latency_ms,
                },
            )
            state.recorder.submit(
                CallRecord(
                    request_id=request_id,
                    route=route_name,
                    api_key_id=api_key_id,
                    outcome="error",
                    error_kind=str(error.kind),
                    http_status=error.http_status,
                    attempts=error.attempts,
                    retries=error.retries,
                    fallbacks=error.fallbacks,
                    breaker_skips=error.breaker_skips,
                    hedges=error.hedges,
                    latency_ms=latency_ms,
                    provider_latency_ms=error.provider_latency_ms,
                    prompt_chars=len(payload.prompt_text()),
                    attempt_records=list(error.attempt_records),
                )
            )
            body = ErrorResponse(
                error=ErrorBody(
                    type=_error_type(error),
                    message=error.message,
                    kind=str(error.kind),
                    request_id=request_id,
                    attempts=error.attempts,
                    details=_error_details(error),
                )
            )
            headers = _gateway_headers(request_id, error=error, latency_ms=latency_ms)
            if error.kind is ErrorKind.RATE_LIMITED:
                headers["Retry-After"] = "1"
            return JSONResponse(
                status_code=error.http_status, content=body.model_dump(), headers=headers
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        observability.observe_request(route=route_name, latency_s=latency_ms / 1000, result=result)
        if latency_ms >= state.settings.app.slow_request_ms:
            logger.warning(
                "slow request: %dms",
                latency_ms,
                extra={
                    "request_id": request_id,
                    "route": route_name,
                    "provider": result.provider,
                    "attempts": result.attempts,
                    "retries": result.retries,
                    "fallbacks": result.fallbacks,
                    "hedges": result.hedges,
                    "cache_hit": result.cache_hit,
                    "latency_ms": latency_ms,
                },
            )

        state.recorder.submit(
            CallRecord(
                request_id=request_id,
                route=route_name,
                api_key_id=api_key_id,
                outcome="success",
                http_status=200,
                provider=result.provider,
                model=result.model,
                attempts=result.attempts,
                retries=result.retries,
                fallbacks=result.fallbacks,
                breaker_skips=result.breaker_skips,
                hedges=result.hedges,
                cache_hit=result.cache_hit,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                cost_usd=result.cost_usd,
                latency_ms=latency_ms,
                provider_latency_ms=result.provider_latency_ms,
                prompt_chars=len(payload.prompt_text()),
                attempt_records=result.attempt_records,
            )
        )

        response = ChatCompletionResponse.build(
            request_id=request_id,
            model=result.model,
            text=result.text,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            finish_reason=result.finish_reason,
            gateway={
                "route": route_name,
                "provider": result.provider,
                "attempts": result.attempts,
                "retries": result.retries,
                "fallbacks": result.fallbacks,
                "breaker_skips": result.breaker_skips,
                "hedges": result.hedges,
                "cache_hit": result.cache_hit,
                "cache_similarity": result.cache_similarity,
                "cost_usd": result.cost_usd,
                "latency_ms": latency_ms,
            },
        )
        return JSONResponse(
            status_code=200,
            content=response.model_dump(),
            headers=_gateway_headers(request_id, result=result, latency_ms=latency_ms),
        )

    # -- usage and budget ---------------------------------------------------

    @app.get("/v1/usage")
    async def usage(
        state: Annotated[AppState, Depends(get_state)],
        api_key_id: Annotated[str | None, Depends(authenticate)],
        from_: Annotated[datetime | None, Query(alias="from")] = None,
        to: Annotated[datetime | None, Query()] = None,
        group_by: Annotated[GroupBy, Query()] = "none",
    ) -> dict[str, Any]:
        until = to or datetime.now(UTC) + timedelta(seconds=1)
        since = from_ or period_start(state.settings.budget.period)
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        if until.tzinfo is None:
            until = until.replace(tzinfo=UTC)

        # Make sure everything already answered is on disk before summing it.
        await state.recorder.drain(timeout=2.0)
        summary = await usage_summary(
            state.database,
            since=since,
            until=until,
            group_by=group_by,
            api_key_id=api_key_id,
        )
        await state.budget.refresh(force=True)
        summary["budget"] = state.budget.snapshot().__dict__
        return summary

    # -- introspection ------------------------------------------------------

    @app.get("/v1/config")
    async def config(state: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
        """Effective, secret-free configuration. Bench results embed this verbatim."""
        settings = state.settings
        return {
            "reliability": settings.reliability.summary(),
            "budget": settings.budget.model_dump(),
            "routes": {
                name: [hop.model_dump() for hop in route.chain]
                for name, route in settings.routes.definitions.items()
            },
            "default_route": settings.routes.default,
            "providers": {
                name: {"type": provider.type, "enabled": provider.enabled}
                for name, provider in settings.providers.items()
            },
            "auth_enabled": settings.auth.enabled,
        }

    @app.get("/v1/reliability/state")
    async def reliability_state(state: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
        return {
            "circuit_breakers": state.breakers.snapshot(),
            "bulkheads": state.bulkheads.snapshot(),
            "rate_limit": state.limiter.snapshot(),
            "cache": state.cache.stats(),
            "budget": state.budget.snapshot().__dict__,
            "recorder": state.recorder.stats(),
        }

    @app.post("/v1/reliability/reset")
    async def reliability_reset(
        state: Annotated[AppState, Depends(get_state)],
        cache: Annotated[bool, Query()] = False,
    ) -> dict[str, Any]:
        """Reset breaker state (and optionally the cache) between chaos runs.

        The harness calls this before every scenario so runs cannot inherit a warm
        cache or a half-open breaker from the previous one.
        """
        state.breakers.reset()
        state.bulkheads.reset()
        state.limiter.reset()
        cleared = 0
        if cache:
            await state.cache.drain()
            async with state.database.session() as session:
                result = await session.execute(sql_text("DELETE FROM semantic_cache"))
                await session.commit()
                cleared = int(getattr(result, "rowcount", 0) or 0)
            state.cache.lookups = 0
            state.cache.hits = 0
            state.cache.stores = 0
            state.cache.errors = 0
        return {"status": "ok", "cache_entries_cleared": cleared}

    @app.get("/metrics")
    async def metrics(state: Annotated[AppState, Depends(get_state)]) -> Response:
        """Prometheus exposition.

        Unauthenticated, like the other ops endpoints — in a real deployment this
        belongs behind a network policy, not behind an API key, because the scraper
        is infrastructure and not a tenant.
        """
        observability.refresh_gauges(
            breakers=state.breakers.snapshot(),
            bulkheads=state.bulkheads.snapshot(),
            budget=state.budget.snapshot().__dict__,
            recorder=state.recorder.stats(),
        )
        return Response(content=observability.render(), media_type=observability.CONTENT_TYPE)

    @app.get("/healthz")
    async def healthz(state: Annotated[AppState, Depends(get_state)]) -> dict[str, Any]:
        return {
            "status": "ok",
            "uptime_s": round(time.time() - state.started_at, 1),
            "providers": state.registry.enabled_providers(),
            "recorder": state.recorder.stats(),
        }

    @app.get("/readyz")
    async def readyz(state: Annotated[AppState, Depends(get_state)]) -> JSONResponse:
        try:
            async with state.database.session() as session:
                await session.execute(sql_text("SELECT 1"))
        except Exception as exc:  # pragma: no cover - only on a broken database
            return JSONResponse(
                status_code=503, content={"status": "database unavailable", "detail": str(exc)}
            )
        return JSONResponse(status_code=200, content={"status": "ready"})

    @app.exception_handler(AuthenticationError)
    async def auth_error_handler(_: Request, exc: AuthenticationError) -> JSONResponse:
        request_id = new_request_id()
        body = ErrorResponse(
            error=ErrorBody(
                type="authentication_error",
                message=exc.message,
                kind=str(exc.kind),
                request_id=request_id,
            )
        )
        return JSONResponse(status_code=401, content=body.model_dump())

    return app


def _error_type(error: GatewayError) -> str:
    mapping = {
        ErrorKind.RATE_LIMITED: "rate_limit_error",
        ErrorKind.BUDGET_EXCEEDED: "budget_exceeded_error",
        ErrorKind.BAD_REQUEST: "invalid_request_error",
        ErrorKind.CIRCUIT_OPEN: "circuit_open_error",
        ErrorKind.TIMEOUT: "timeout_error",
        ErrorKind.DEADLINE_EXCEEDED: "timeout_error",
        ErrorKind.NO_PROVIDER: "no_provider_error",
    }
    return mapping.get(error.kind, "upstream_error")


def _error_details(error: GatewayError) -> dict[str, Any]:
    details: dict[str, Any] = {
        "retries": error.retries,
        "fallbacks": error.fallbacks,
        "breaker_skips": error.breaker_skips,
    }
    for field_name in ("spent_usd", "limit_usd", "period", "provider", "status_code"):
        value = getattr(error, field_name, None)
        if value is not None:
            details[field_name] = value
    return details


# Started with `uvicorn llm_gateway.main:create_app --factory`, so importing this
# module never reads the config file as a side effect (tests build their own app).

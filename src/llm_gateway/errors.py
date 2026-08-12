"""Error taxonomy.

Everything a provider can do wrong is normalised into one `ErrorKind` so that the
retry policy, the circuit breaker, the HTTP layer and the chaos report all speak
the same language.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorKind(StrEnum):
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMITED = "rate_limited"
    SERVER_ERROR = "server_error"
    OVERLOADED = "overloaded"
    INVALID_RESPONSE = "invalid_response"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"
    CIRCUIT_OPEN = "circuit_open"
    #: Наш собственный лимит, а не провайдерский: очередь к провайдеру полна.
    CAPACITY = "capacity"
    #: Наш собственный rate limit на входе.
    THROTTLED = "throttled"
    BUDGET_EXCEEDED = "budget_exceeded"
    NO_PROVIDER = "no_provider"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    UNKNOWN = "unknown"


#: HTTP status returned to the client for each error kind.
HTTP_STATUS_BY_KIND: dict[ErrorKind, int] = {
    ErrorKind.TIMEOUT: 504,
    ErrorKind.DEADLINE_EXCEEDED: 504,
    ErrorKind.CONNECTION: 502,
    ErrorKind.RATE_LIMITED: 429,
    ErrorKind.SERVER_ERROR: 502,
    ErrorKind.OVERLOADED: 502,
    ErrorKind.INVALID_RESPONSE: 502,
    ErrorKind.AUTH: 502,
    ErrorKind.BAD_REQUEST: 400,
    ErrorKind.CIRCUIT_OPEN: 503,
    ErrorKind.CAPACITY: 503,
    ErrorKind.THROTTLED: 429,
    ErrorKind.BUDGET_EXCEEDED: 402,
    ErrorKind.NO_PROVIDER: 503,
    ErrorKind.UNKNOWN: 502,
}


class GatewayError(Exception):
    """Base class for everything the gateway turns into a non-2xx response.

    The `attempts`/`retries`/`fallbacks`/`breaker_skips` fields are filled in by the
    orchestrator before the error propagates, so a failed request is as measurable
    as a successful one: the HTTP layer reports them in headers and the recorder
    writes them to Postgres.
    """

    kind: ErrorKind = ErrorKind.UNKNOWN

    def __init__(self, message: str, *, kind: ErrorKind | None = None) -> None:
        super().__init__(message)
        self.message = message
        if kind is not None:
            self.kind = kind
        self.attempts: int = 0
        self.retries: int = 0
        self.fallbacks: int = 0
        self.breaker_skips: int = 0
        self.hedges: int = 0
        self.provider_latency_ms: int = 0
        self.attempt_records: list[Any] = []

    @property
    def http_status(self) -> int:
        return HTTP_STATUS_BY_KIND.get(self.kind, 502)


class ProviderError(GatewayError):
    """A single provider call failed.

    `retry_after_s` is populated from the upstream Retry-After header when present,
    so the retry policy can honour the provider's own backpressure signal.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind,
        provider: str,
        model: str,
        status_code: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message, kind=kind)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retry_after_s = retry_after_s

    def __str__(self) -> str:
        status = f" status={self.status_code}" if self.status_code is not None else ""
        return f"[{self.provider}/{self.model}] {self.kind}{status}: {self.message}"


class BudgetExceededError(GatewayError):
    kind = ErrorKind.BUDGET_EXCEEDED

    def __init__(self, message: str, *, spent_usd: float, limit_usd: float, period: str) -> None:
        super().__init__(message)
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        self.period = period


class ThrottledError(GatewayError):
    """The gateway's own rate limit, not a provider's."""

    kind = ErrorKind.THROTTLED

    def __init__(self, message: str, *, retry_after_s: float) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class NoProviderAvailableError(GatewayError):
    kind = ErrorKind.NO_PROVIDER


class BadRequestError(GatewayError):
    kind = ErrorKind.BAD_REQUEST


class AuthenticationError(GatewayError):
    kind = ErrorKind.AUTH

    @property
    def http_status(self) -> int:
        return 401

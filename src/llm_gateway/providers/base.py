"""Provider adapter protocol and shared HTTP error mapping.

Adapters speak raw HTTP via httpx rather than vendor SDKs: it keeps the three
dialects uniform, and it lets every provider be pointed at the mock by changing
one `base_url` in config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from llm_gateway.errors import ErrorKind, ProviderError
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.settings import ProviderConfig


@dataclass(slots=True)
class ProviderResponse:
    """Normalised successful answer from any provider dialect."""

    text: str
    model: str
    tokens_in: int
    tokens_out: int
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderAdapter(Protocol):
    name: str
    config: ProviderConfig

    async def complete(
        self,
        request: ChatCompletionRequest,
        model: str,
        client: httpx.AsyncClient,
    ) -> ProviderResponse: ...


def parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        # HTTP-date form: not worth parsing for our purposes, treat as absent.
        return None


def map_status_error(
    response: httpx.Response,
    *,
    provider: str,
    model: str,
) -> ProviderError:
    """Turn a non-2xx upstream response into a typed ProviderError."""
    status = response.status_code
    if status == 401 or status == 403:
        kind = ErrorKind.AUTH
    elif status == 429:
        kind = ErrorKind.RATE_LIMITED
    elif status in (502, 503, 529):
        kind = ErrorKind.OVERLOADED
    elif status >= 500:
        kind = ErrorKind.SERVER_ERROR
    elif status == 408:
        kind = ErrorKind.TIMEOUT
    elif status >= 400:
        kind = ErrorKind.BAD_REQUEST
    else:
        kind = ErrorKind.UNKNOWN

    snippet = response.text[:200].replace("\n", " ") if response.text else ""
    return ProviderError(
        f"upstream returned {status}: {snippet}",
        kind=kind,
        provider=provider,
        model=model,
        status_code=status,
        retry_after_s=parse_retry_after(response),
    )


def map_transport_error(exc: Exception, *, provider: str, model: str) -> ProviderError:
    """Turn an httpx transport exception into a typed ProviderError."""
    if isinstance(exc, httpx.TimeoutException):
        kind = ErrorKind.TIMEOUT
        message = f"request timed out: {type(exc).__name__}"
    elif isinstance(exc, httpx.RemoteProtocolError):
        # The mock's `conn_abort` outcome and real mid-body resets land here.
        kind = ErrorKind.CONNECTION
        message = f"connection broken mid-response: {exc}"
    elif isinstance(exc, httpx.TransportError):
        kind = ErrorKind.CONNECTION
        message = f"transport error: {type(exc).__name__}: {exc}"
    else:
        kind = ErrorKind.UNKNOWN
        message = f"unexpected error: {type(exc).__name__}: {exc}"
    return ProviderError(message, kind=kind, provider=provider, model=model)


def decode_json(response: httpx.Response, *, provider: str, model: str) -> dict[str, Any]:
    """Parse a 200 response body, classifying unparseable payloads as invalid_response."""
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        snippet = response.text[:160].replace("\n", " ")
        raise ProviderError(
            f"response body is not valid JSON ({exc}); body starts with {snippet!r}",
            kind=ErrorKind.INVALID_RESPONSE,
            provider=provider,
            model=model,
            status_code=response.status_code,
        ) from exc
    if not isinstance(payload, dict):
        raise ProviderError(
            f"expected a JSON object, got {type(payload).__name__}",
            kind=ErrorKind.INVALID_RESPONSE,
            provider=provider,
            model=model,
            status_code=response.status_code,
        )
    return payload


def schema_error(message: str, *, provider: str, model: str) -> ProviderError:
    return ProviderError(
        message,
        kind=ErrorKind.INVALID_RESPONSE,
        provider=provider,
        model=model,
        status_code=200,
    )

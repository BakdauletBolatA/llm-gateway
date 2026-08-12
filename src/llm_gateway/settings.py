"""Configuration loading: YAML base + optional overlay + environment overrides.

Every reliability knob lives here, so a chaos run can be reproduced from a config
file alone rather than from a git revision.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from llm_gateway.errors import ErrorKind

DEFAULT_CONFIG_PATH = "config/gateway.yaml"
ENV_PREFIX = "GW__"

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class TimeoutConfig(BaseModel):
    enabled: bool = False
    connect_s: float = 2.0
    read_s: float = 8.0
    write_s: float = 5.0
    total_s: float = 12.0


class RetryConfig(BaseModel):
    enabled: bool = False
    max_attempts: int = Field(default=3, ge=1)
    base_delay_ms: int = Field(default=120, ge=0)
    max_delay_ms: int = Field(default=2000, ge=0)
    jitter: Literal["none", "full", "equal"] = "full"
    respect_retry_after: bool = True
    max_retry_after_s: float = 3.0
    retry_on: list[ErrorKind] = Field(
        default_factory=lambda: [
            ErrorKind.TIMEOUT,
            ErrorKind.CONNECTION,
            ErrorKind.RATE_LIMITED,
            ErrorKind.SERVER_ERROR,
            ErrorKind.OVERLOADED,
            ErrorKind.INVALID_RESPONSE,
        ]
    )

    @property
    def retryable(self) -> frozenset[ErrorKind]:
        return frozenset(self.retry_on)


class CircuitBreakerConfig(BaseModel):
    enabled: bool = False
    window_s: float = 10.0
    min_calls: int = Field(default=8, ge=1)
    failure_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    cooldown_s: float = 4.0
    half_open_max_calls: int = Field(default=2, ge=1)


class FallbackConfig(BaseModel):
    enabled: bool = False
    max_providers: int = Field(default=3, ge=1)


class HedgingConfig(BaseModel):
    """Tail-latency hedging: race the next provider instead of waiting it out.

    Retries help when a provider *fails*; they do nothing when it answers, slowly.
    A hedge starts the next provider in the chain after `delay_ms` and takes
    whichever answer arrives first. The delay is the whole design: set above the
    normal p95, only the slow tail gets duplicated.
    """

    enabled: bool = False
    delay_ms: int = Field(default=400, ge=1)
    max_in_flight: int = Field(default=2, ge=2)

    @property
    def delay_s(self) -> float:
        return self.delay_ms / 1000.0


class BulkheadConfig(BaseModel):
    """Limit on calls in flight to one provider.

    The breaker watches for a provider that fails; this watches for one that is
    full. `queue_timeout_ms` is how long a request may wait for a slot before it is
    shed to the next provider — always additionally bounded by the request deadline.
    """

    enabled: bool = False
    max_concurrent_per_provider: int = Field(default=8, ge=1)
    queue_timeout_ms: int = Field(default=2000, ge=0)

    @property
    def queue_timeout_s(self) -> float:
        return self.queue_timeout_ms / 1000.0


class RateLimitConfig(BaseModel):
    """Token bucket per API key, or one shared bucket when auth is off."""

    enabled: bool = False
    requests_per_second: float = Field(default=50.0, gt=0.0)
    burst: int = Field(default=100, ge=1)


class CacheConfig(BaseModel):
    enabled: bool = False
    similarity_threshold: float = Field(default=0.93, gt=0.0, le=1.0)
    ttl_s: int = Field(default=900, ge=1)
    max_temperature: float = 0.3
    embedding_dim: int = Field(default=256, ge=16, le=2000)
    candidate_limit: int = Field(default=5, ge=1)
    embedder: Literal["hashing", "ollama"] = "hashing"
    ollama_embed_model: str = "nomic-embed-text"


class ReliabilityConfig(BaseModel):
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    retries: RetryConfig = Field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    hedging: HedgingConfig = Field(default_factory=HedgingConfig)
    bulkhead: BulkheadConfig = Field(default_factory=BulkheadConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @model_validator(mode="after")
    def _hedging_needs_something_to_hedge_with(self) -> ReliabilityConfig:
        if self.hedging.enabled and not self.fallback.enabled:
            raise ValueError(
                "reliability.hedging requires reliability.fallback: without a chain "
                "there is no second provider to race against the first"
            )
        deadline_ms = self.timeouts.total_s * 1000
        if self.hedging.enabled and self.timeouts.enabled and self.hedging.delay_ms >= deadline_ms:
            raise ValueError(
                f"reliability.hedging.delay_ms={self.hedging.delay_ms} is not below "
                f"reliability.timeouts.total_s={self.timeouts.total_s}s: the hedge "
                "would never be launched before the request deadline"
            )
        return self

    def summary(self) -> dict[str, Any]:
        """Compact, secret-free view used by /v1/config and stored in bench results."""
        return {
            "timeouts": self.timeouts.model_dump(),
            "retries": self.retries.model_dump(mode="json"),
            "circuit_breaker": self.circuit_breaker.model_dump(),
            "fallback": self.fallback.model_dump(),
            "hedging": self.hedging.model_dump(),
            "bulkhead": self.bulkhead.model_dump(),
            "rate_limit": self.rate_limit.model_dump(),
            "cache": self.cache.model_dump(),
        }


class ProviderConfig(BaseModel):
    type: Literal["openai", "anthropic", "ollama"]
    base_url: str
    api_key_env: str | None = None
    enabled: bool = True
    #: Квота конкурентности именно этого провайдера. У разных провайдеров она разная,
    #: поэтому один общий лимит — компромисс: здесь его можно переопределить.
    #: None — берётся reliability.bulkhead.max_concurrent_per_provider.
    max_concurrent: int | None = Field(default=None, ge=1)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class RouteTarget(BaseModel):
    provider: str
    model: str


class RouteConfig(BaseModel):
    chain: list[RouteTarget] = Field(min_length=1)


class RoutesConfig(BaseModel):
    default: str
    definitions: dict[str, RouteConfig]

    @model_validator(mode="after")
    def _default_exists(self) -> RoutesConfig:
        if self.default not in self.definitions:
            raise ValueError(f"routes.default={self.default!r} is not among routes.definitions")
        return self


class ModelPrice(BaseModel):
    input_per_mtok: float = 0.0
    output_per_mtok: float = 0.0


class PricingConfig(BaseModel):
    default: ModelPrice = Field(default_factory=ModelPrice)
    models: dict[str, ModelPrice] = Field(default_factory=dict)

    def price_for(self, provider: str, model: str) -> ModelPrice:
        return self.models.get(f"{provider}:{model}") or self.models.get(model) or self.default


class ApiKeyConfig(BaseModel):
    id: str
    key: str
    budget_limit_usd: float | None = None


class AuthConfig(BaseModel):
    keys: list[ApiKeyConfig] = Field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return bool(self.keys)

    def lookup(self, presented: str) -> ApiKeyConfig | None:
        return next((k for k in self.keys if k.key and k.key == presented), None)


class BudgetConfig(BaseModel):
    enabled: bool = True
    period: Literal["day", "month"] = "day"
    limit_usd: float = Field(default=25.0, ge=0.0)
    refresh_interval_s: float = 5.0
    estimate_output_tokens: int = 512


class DatabaseConfig(BaseModel):
    dsn: str
    pool_size: int = 10
    max_overflow: int = 10
    recorder_queue_size: int = 10000
    recorder_batch_size: int = 50
    recorder_flush_interval_s: float = 0.25
    run_migrations_on_startup: bool = True


class AppConfig(BaseModel):
    name: str = "llm-gateway"
    log_level: str = "INFO"
    #: json — по одной строке-объекту на запись, для сбора логов; text — для человека.
    log_format: Literal["text", "json"] = "text"
    slow_request_ms: int = 5000


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig
    auth: AuthConfig = Field(default_factory=AuthConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    providers: dict[str, ProviderConfig]
    routes: RoutesConfig
    pricing: PricingConfig = Field(default_factory=PricingConfig)

    @model_validator(mode="after")
    def _routes_reference_known_providers(self) -> Settings:
        for route_name, route in self.routes.definitions.items():
            for hop in route.chain:
                if hop.provider not in self.providers:
                    raise ValueError(
                        f"route {route_name!r} references unknown provider {hop.provider!r}"
                    )
        return self

    def resolve_chain(self, route_name: str) -> list[RouteTarget]:
        """Chain of enabled providers for a route, truncated by the fallback config."""
        route = self.routes.definitions[route_name]
        chain = [hop for hop in route.chain if self.providers[hop.provider].enabled]
        limit = self.reliability.fallback.max_providers if self.reliability.fallback.enabled else 1
        return chain[:limit]


def _expand_env(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        env_value = os.environ.get(name)
        if env_value is not None and env_value != "":
            return env_value
        return default if default is not None else ""

    return _ENV_REF.sub(repl, value)


def _expand_tree(node: Any) -> Any:
    if isinstance(node, str):
        return _expand_env(node)
    if isinstance(node, dict):
        return {key: _expand_tree(val) for key, val in node.items()}
    if isinstance(node, list):
        return [_expand_tree(item) for item in node]
    return node


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_env_overrides(tree: dict[str, Any]) -> dict[str, Any]:
    """GW__RELIABILITY__RETRIES__ENABLED=true -> tree['reliability']['retries']['enabled']."""
    for env_name, raw in sorted(os.environ.items()):
        if not env_name.startswith(ENV_PREFIX):
            continue
        path = [part.lower() for part in env_name[len(ENV_PREFIX) :].split("__") if part]
        if not path:
            continue
        cursor: dict[str, Any] = tree
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce_scalar(raw)
    return tree


def load_settings(
    path: str | Path | None = None,
    overlay_path: str | Path | None = None,
) -> Settings:
    """Load base YAML, apply an optional overlay, expand ${VARS}, apply GW__ overrides."""
    base_path = Path(path or os.environ.get("GATEWAY_CONFIG", DEFAULT_CONFIG_PATH))
    with base_path.open(encoding="utf-8") as handle:
        tree: dict[str, Any] = yaml.safe_load(handle) or {}

    overlay = overlay_path or os.environ.get("GATEWAY_CONFIG_OVERLAY")
    if overlay:
        with Path(overlay).open(encoding="utf-8") as handle:
            tree = _deep_merge(tree, yaml.safe_load(handle) or {})

    tree = _expand_tree(tree)
    tree = _apply_env_overrides(tree)
    return Settings.model_validate(tree)

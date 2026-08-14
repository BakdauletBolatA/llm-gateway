"""Database schema.

`llm_calls` is one row per client request (the billing and reporting unit);
`llm_attempts` is one row per provider call, so a single request that retried
twice and then fell back to another provider leaves a readable trail.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LlmCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    route: Mapped[str] = mapped_column(String(128))
    api_key_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    provider: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    outcome: Mapped[str] = mapped_column(String(32), index=True)  # success | error
    error_kind: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    http_status: Mapped[int] = mapped_column(Integer)

    attempts: Mapped[int] = mapped_column(Integer, default=0)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    fallbacks: Mapped[int] = mapped_column(Integer, default=0)
    breaker_skips: Mapped[int] = mapped_column(Integer, default=0)
    hedges: Mapped[int] = mapped_column(Integer, default=0, server_default=text("0"))
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(14, 6), default=0)

    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    provider_latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    prompt_chars: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (Index("ix_llm_calls_created_at_provider", "created_at", "provider"),)


class LlmAttempt(Base):
    __tablename__ = "llm_attempts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    attempt_no: Mapped[int] = mapped_column(Integer)  # 1-based, across the whole request
    hop_index: Mapped[int] = mapped_column(Integer)  # position in the provider chain
    provider: Mapped[str] = mapped_column(String(64), index=True)
    model: Mapped[str] = mapped_column(String(128))

    outcome: Mapped[str] = mapped_column(String(32))  # success | error | skipped_breaker
    error_kind: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    backoff_ms: Mapped[int] = mapped_column(Integer, default=0)

    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(14, 6), default=0)


class RateLimitBucket(Base):
    """Token bucket shared by every replica (`rate_limit.scope: shared`).

    Taken with a single atomic statement — see reliability/ratelimit.py — so the
    row lock, not the application, is what keeps two replicas from spending the
    same token twice.
    """

    __tablename__ = "rate_limit_buckets"

    scope: Mapped[str] = mapped_column(String(128), primary_key=True)
    tokens: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SemanticCacheEntry(Base):
    __tablename__ = "semantic_cache"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(160), index=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt_text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(256))

    response_text: Mapped[str] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(14, 6), default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    hits: Mapped[int] = mapped_column(Integer, default=0)

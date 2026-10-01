"""initial schema: call log, attempt log, semantic cache

Revision ID: 0001
Revises:
Create Date: 2026-08-12
"""

from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 256


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "llm_calls",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("route", sa.String(length=128), nullable=False),
        sa.Column("api_key_id", sa.String(length=128), nullable=True),
        sa.Column("provider", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("error_kind", sa.String(length=32), nullable=True),
        sa.Column("http_status", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("retries", sa.Integer(), nullable=False),
        sa.Column("fallbacks", sa.Integer(), nullable=False),
        sa.Column("breaker_skips", sa.Integer(), nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=14, scale=6), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("provider_latency_ms", sa.Integer(), nullable=False),
        sa.Column("prompt_chars", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id"),
    )
    op.create_index("ix_llm_calls_created_at", "llm_calls", ["created_at"])
    op.create_index("ix_llm_calls_api_key_id", "llm_calls", ["api_key_id"])
    op.create_index("ix_llm_calls_provider", "llm_calls", ["provider"])
    op.create_index("ix_llm_calls_model", "llm_calls", ["model"])
    op.create_index("ix_llm_calls_outcome", "llm_calls", ["outcome"])
    op.create_index("ix_llm_calls_error_kind", "llm_calls", ["error_kind"])
    op.create_index("ix_llm_calls_cache_hit", "llm_calls", ["cache_hit"])
    op.create_index("ix_llm_calls_request_id", "llm_calls", ["request_id"])
    op.create_index("ix_llm_calls_created_at_provider", "llm_calls", ["created_at", "provider"])

    op.create_table(
        "llm_attempts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("hop_index", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("error_kind", sa.String(length=32), nullable=True),
        sa.Column("http_status", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("backoff_ms", sa.Integer(), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=14, scale=6), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_attempts_request_id", "llm_attempts", ["request_id"])
    op.create_index("ix_llm_attempts_created_at", "llm_attempts", ["created_at"])
    op.create_index("ix_llm_attempts_provider", "llm_attempts", ["provider"])
    op.create_index("ix_llm_attempts_error_kind", "llm_attempts", ["error_kind"])

    op.create_table(
        "semantic_cache",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("scope", sa.String(length=160), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("response_text", sa.Text(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=14, scale=6), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hits", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_semantic_cache_scope", "semantic_cache", ["scope"])
    op.create_index("ix_semantic_cache_prompt_hash", "semantic_cache", ["prompt_hash"])
    op.create_index("ix_semantic_cache_expires_at", "semantic_cache", ["expires_at"])
    # HNSW rather than IVFFlat: it needs no training pass, so recall is sane from
    # the first inserted row — which matters for a cache that starts empty.
    op.execute(
        "CREATE INDEX ix_semantic_cache_embedding_hnsw "
        "ON semantic_cache USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("semantic_cache")
    op.drop_table("llm_attempts")
    op.drop_table("llm_calls")

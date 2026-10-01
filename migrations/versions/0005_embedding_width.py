"""widen the cache embedding column to 384

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-01

all-MiniLM-L6-v2 produces 384 dimensions; the hashing embedder's 256 are
zero-padded, which leaves cosine similarity unchanged. Cached answers are derived
data with a 15-minute TTL, so the table is emptied rather than converted: vectors
from different embedders are not comparable anyway.
"""

from __future__ import annotations

import pgvector.sqlalchemy
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def _resize(width: int) -> None:
    op.execute("DROP INDEX IF EXISTS ix_semantic_cache_embedding_hnsw")
    op.execute("TRUNCATE semantic_cache")
    op.alter_column(
        "semantic_cache",
        "embedding",
        type_=pgvector.sqlalchemy.Vector(width),
        existing_type=pgvector.sqlalchemy.Vector(),
        existing_nullable=False,
        postgresql_using=f"embedding::vector({width})",
    )
    op.execute(
        "CREATE INDEX ix_semantic_cache_embedding_hnsw "
        "ON semantic_cache USING hnsw (embedding vector_cosine_ops)"
    )


def upgrade() -> None:
    _resize(384)


def downgrade() -> None:
    _resize(256)

"""shared rate-limit buckets

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-14

One row per scope (API key, or the global scope when auth is off). The token count
lives in the database so that every replica spends from the same allowance instead
of each one handing out the full limit.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "rate_limit_buckets",
        sa.Column("scope", sa.String(length=128), primary_key=True),
        sa.Column("tokens", sa.Float(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("rate_limit_buckets")

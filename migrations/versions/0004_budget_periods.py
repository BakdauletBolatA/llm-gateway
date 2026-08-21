"""shared budget periods

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-21

One row per (period, scope): the money already committed for this day or month.
The row is the deployment's single counter, so a limit of $25 stays $25 no matter
how many replicas are serving — instead of $25 per process.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "budget_periods",
        sa.Column("period_key", sa.String(length=32), primary_key=True),
        sa.Column("scope", sa.String(length=128), primary_key=True),
        sa.Column("spent_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("budget_periods")

"""hedged requests: count parallel calls per request

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-12

A hedge is visible in `llm_attempts` already — a hedged request leaves an extra row
with outcome 'cancelled' or 'discarded' — but the summary endpoints group by
`llm_calls`, so the count belongs there next to fallbacks and breaker_skips.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_calls",
        sa.Column("hedges", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("llm_calls", "hedges")

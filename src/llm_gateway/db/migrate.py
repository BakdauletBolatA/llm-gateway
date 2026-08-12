"""Run Alembic migrations from application startup.

Keeping this in-process means `docker compose up` needs no separate migration
step and a fresh database is usable on first boot.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _alembic_config(dsn: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", dsn)
    # Leave the app's logging alone: see the note in migrations/env.py.
    config.attributes["configure_logging"] = False
    return config


def upgrade_sync(dsn: str) -> None:
    command.upgrade(_alembic_config(dsn), "head")


async def upgrade(dsn: str) -> None:
    """Alembic is synchronous and spins up its own loop, so it runs in a thread."""
    logger.info("running database migrations")
    await asyncio.to_thread(upgrade_sync, dsn)
    logger.info("database migrations are up to date")

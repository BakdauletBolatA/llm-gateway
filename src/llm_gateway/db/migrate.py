"""Run Alembic migrations from application startup.

Keeping this in-process means `docker compose up` needs no separate migration
step and a fresh database is usable on first boot.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)


def find_project_root(starts: Iterable[Path]) -> Path:
    """The directory holding alembic.ini and migrations/, searched upward from each start.

    Counting parents of this file works in a checkout and breaks in the Docker
    image, where `pip install .` puts the code in site-packages and the migrations
    stay in the working directory.
    """
    for start in starts:
        for directory in (start, *start.parents):
            if (directory / "alembic.ini").is_file() and (directory / "migrations").is_dir():
                return directory
    raise FileNotFoundError("alembic.ini and migrations/ not found; run from the project root")


def _alembic_config(dsn: str) -> Config:
    root = find_project_root([Path(__file__).resolve(), Path.cwd()])
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
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

"""Fixtures for end-to-end tests.

The gateway app is wired to an in-process mock provider through an ASGI
transport, so the whole path — routing, adapters, error classification, retry,
breaker, cache, budget, recorder — runs for real without any network or paid key.
A PostgreSQL with pgvector is the only external dependency; tests that need it
skip when it is not reachable.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import text

from llm_gateway.main import build_state, create_app
from llm_gateway.settings import Settings, load_settings
from mock_provider.main import apply_scenario
from mock_provider.main import create_app as create_mock_app

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://gateway:gateway@127.0.0.1:5432/llm_gateway_test",
)
CONFIG = "config/gateway.yaml"
PROFILES = "config/failure_profiles.yaml"

#: End-to-end tests pin the reliability flags on purpose: they assert the wiring
#: (routing, classification, logging, budget), and must not change meaning every
#: time an iteration flips a flag in config/gateway.yaml. Mechanism behaviour is
#: covered by tests/test_orchestrator.py, which builds its own settings per case.
ALL_MECHANISMS_OFF = {
    section: {"enabled": False}
    for section in ("timeouts", "retries", "circuit_breaker", "fallback", "cache")
}


async def _database_available(dsn: str) -> bool:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(dsn)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
    finally:
        await engine.dispose()


def build_settings(reliability: dict[str, dict[str, Any]] | None = None) -> Settings:
    tree = load_settings(CONFIG).model_dump()
    tree["database"].update({"dsn": TEST_DSN, "run_migrations_on_startup": True})
    for name in ("openai", "anthropic", "ollama"):
        tree["providers"][name]["enabled"] = False
    for name in ("primary", "secondary", "tertiary"):
        tree["providers"][f"mock_{name}"]["base_url"] = f"http://mock/p/{name}"
    for section, patch in (reliability or ALL_MECHANISMS_OFF).items():
        tree["reliability"][section].update(patch)
    return Settings.model_validate(tree)


@pytest.fixture
def gateway_settings() -> Settings:
    return build_settings()


@pytest.fixture
def mock_app() -> FastAPI:
    return create_mock_app(PROFILES)


@asynccontextmanager
async def running_stack(
    settings: Settings, mock_app: FastAPI
) -> AsyncIterator[dict[str, Any]]:
    if not await _database_available(TEST_DSN):
        pytest.skip(f"PostgreSQL not reachable at {TEST_DSN}")

    apply_scenario(mock_app.state.mock, mock_app.state.library, "healthy")

    app = create_app(settings)
    state = await build_state(settings)
    app.state.core = state
    # Point every provider client at the in-process mock.
    for name in state.registry.enabled_providers():
        await state.registry._clients[name].aclose()  # noqa: SLF001
        state.registry._clients[name] = httpx.AsyncClient(  # noqa: SLF001
            transport=httpx.ASGITransport(app=mock_app),
            timeout=state.registry._clients[name].timeout,  # noqa: SLF001
        )

    async with state.database.session() as session:
        await session.execute(text("TRUNCATE llm_calls, llm_attempts, semantic_cache"))
        await session.commit()

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            yield {
                "client": client,
                "mock_app": mock_app,
                "state": state,
                "settings": settings,
            }
    finally:
        await state.cache.drain()
        await state.recorder.stop()
        await state.registry.aclose()
        await state.database.aclose()


@pytest_asyncio.fixture
async def stack(gateway_settings: Settings, mock_app: FastAPI) -> AsyncIterator[dict[str, Any]]:
    async with running_stack(gateway_settings, mock_app) as running:
        yield running


@pytest_asyncio.fixture
async def cache_stack(mock_app: FastAPI) -> AsyncIterator[dict[str, Any]]:
    """Only the semantic cache is on, so hits and misses are unambiguous."""
    settings = build_settings(
        {
            **ALL_MECHANISMS_OFF,
            "cache": {"enabled": True, "similarity_threshold": 0.60, "ttl_s": 900},
        }
    )
    async with running_stack(settings, mock_app) as running:
        yield running


def set_scenario(mock_app: FastAPI, scenario: str) -> None:
    apply_scenario(mock_app.state.mock, mock_app.state.library, scenario)

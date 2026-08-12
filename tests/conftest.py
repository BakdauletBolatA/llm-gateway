"""Fixtures for end-to-end tests.

The gateway app is wired to an in-process mock provider through an ASGI
transport, so the whole path — routing, adapters, error classification, retry,
breaker, budget, recorder — runs for real without any network or paid key.
A PostgreSQL with pgvector is the only external dependency; tests that need it
skip when it is not reachable.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from llm_gateway.main import build_state, create_app
from llm_gateway.settings import Settings, load_settings
from mock_provider.main import create_app as create_mock_app

TEST_DSN = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://gateway:gateway@127.0.0.1:5432/llm_gateway_test",
)


async def _database_available(dsn: str) -> bool:
    from sqlalchemy import text
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


@pytest.fixture
def gateway_settings() -> Settings:
    """Base config with the mock as the only provider and every mechanism off.

    The reliability flags are pinned here on purpose: end-to-end tests assert the
    wiring (routing, classification, logging, budget), and they must not change
    meaning every time an iteration flips a flag in config/gateway.yaml.
    Mechanism behaviour is covered by tests/test_orchestrator.py, which builds
    its own settings per case.
    """
    overrides = {
        "database": {"dsn": TEST_DSN, "run_migrations_on_startup": True},
        "reliability": {
            "timeouts": {"enabled": False},
            "retries": {"enabled": False},
            "circuit_breaker": {"enabled": False},
            "fallback": {"enabled": False},
            "cache": {"enabled": False},
        },
        "providers": {
            "openai": {"enabled": False},
            "anthropic": {"enabled": False},
            "ollama": {"enabled": False},
            "mock_primary": {"base_url": "http://mock/p/primary"},
            "mock_secondary": {"base_url": "http://mock/p/secondary"},
            "mock_tertiary": {"base_url": "http://mock/p/tertiary"},
        },
    }
    settings = load_settings("config/gateway.yaml")
    merged = settings.model_dump()
    for section, values in overrides.items():
        if section in ("providers", "reliability"):
            for name, patch in values.items():  # type: ignore[union-attr]
                merged[section][name].update(patch)
        else:
            merged[section].update(values)  # type: ignore[union-attr]
    return Settings.model_validate(merged)


@pytest.fixture
def mock_app() -> FastAPI:
    return create_mock_app("config/failure_profiles.yaml")


@pytest_asyncio.fixture
async def stack(
    gateway_settings: Settings, mock_app: FastAPI
) -> AsyncIterator[dict[str, Any]]:
    if not await _database_available(TEST_DSN):
        pytest.skip(f"PostgreSQL not reachable at {TEST_DSN}")

    # Bring the mock up through its own lifespan so profiles are loaded.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock_app), base_url="http://mock"
    ) as mock_client:
        from mock_provider.main import apply_scenario

        apply_scenario(mock_app.state.mock, mock_app.state.library, "healthy")

        app = create_app(gateway_settings)
        state = await build_state(gateway_settings)
        app.state.core = state
        # Point every provider client at the in-process mock.
        for name in state.registry.enabled_providers():
            await state.registry._clients[name].aclose()  # noqa: SLF001
            state.registry._clients[name] = httpx.AsyncClient(  # noqa: SLF001
                transport=httpx.ASGITransport(app=mock_app),
                timeout=state.registry._clients[name].timeout,  # noqa: SLF001
            )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            async with state.database.session() as session:
                from sqlalchemy import text

                await session.execute(text("TRUNCATE llm_calls, llm_attempts, semantic_cache"))
                await session.commit()
            yield {
                "client": client,
                "mock_client": mock_client,
                "mock_app": mock_app,
                "state": state,
                "settings": gateway_settings,
            }

        await state.recorder.stop()
        await state.registry.aclose()
        await state.database.aclose()


def set_scenario(mock_app: FastAPI, scenario: str) -> None:
    from mock_provider.main import apply_scenario

    apply_scenario(mock_app.state.mock, mock_app.state.library, scenario)

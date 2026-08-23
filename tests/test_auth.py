"""API keys: the check itself, and what an unauthenticated caller can still reach.

Auth was the one shipped feature with no tests at all — which matters more here
than for most features, because its failure mode is silent. A comparison that
leaks and an admin endpoint that is open both look exactly like a working gateway.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI

from llm_gateway.settings import AuthConfig, Settings
from tests.conftest import build_settings, running_stack

CHAT = "/v1/chat/completions"
PAYLOAD: dict[str, Any] = {
    "model": "chaos-default",
    "messages": [{"role": "user", "content": "auth test"}],
    "temperature": 0.9,
}

TENANT_KEY = "sk-tenant-aaaaaaaaaaaaaaaaaaaa"
OTHER_KEY = "sk-other-bbbbbbbbbbbbbbbbbbbbb"


def auth_config(*keys: tuple[str, str]) -> AuthConfig:
    return AuthConfig.model_validate({"keys": [{"id": i, "key": k} for i, k in keys]})


# -- the comparison ------------------------------------------------------------


def test_the_right_key_is_found() -> None:
    auth = auth_config(("tenant", TENANT_KEY), ("other", OTHER_KEY))
    key = auth.lookup(TENANT_KEY)
    assert key is not None and key.id == "tenant"


def test_the_last_configured_key_also_works() -> None:
    """The loop deliberately does not stop at the first match; a bug there would
    show up as only the first key ever authenticating."""
    auth = auth_config(("tenant", TENANT_KEY), ("other", OTHER_KEY))
    key = auth.lookup(OTHER_KEY)
    assert key is not None and key.id == "other"


@pytest.mark.parametrize(
    "presented",
    [
        "",
        "sk-",
        TENANT_KEY[:-1],  # correct prefix, one character short
        TENANT_KEY + "x",  # correct prefix, one character too long
        TENANT_KEY.upper(),
        " " + TENANT_KEY,
    ],
)
def test_a_near_miss_is_not_a_match(presented: str) -> None:
    assert auth_config(("tenant", TENANT_KEY)).lookup(presented) is None


def test_an_empty_configured_key_matches_nothing() -> None:
    """A key left blank in the config must not turn into a wildcard."""
    auth = AuthConfig.model_validate({"keys": [{"id": "blank", "key": ""}]})
    assert auth.lookup("") is None
    assert auth.lookup("anything") is None


def test_auth_is_off_until_a_key_is_configured() -> None:
    assert AuthConfig.model_validate({"keys": []}).enabled is False
    assert auth_config(("tenant", TENANT_KEY)).enabled is True


# -- the endpoints -------------------------------------------------------------


def authenticated_settings() -> Settings:
    settings = build_settings()
    return Settings.model_validate(
        {
            **settings.model_dump(),
            "auth": {"keys": [{"id": "tenant", "key": TENANT_KEY}]},
        }
    )


@pytest_asyncio.fixture
async def authed_stack(mock_app: FastAPI) -> AsyncIterator[dict[str, Any]]:
    async with running_stack(authenticated_settings(), mock_app) as running:
        yield running


async def test_a_request_without_a_key_is_refused(authed_stack: dict[str, Any]) -> None:
    response = await authed_stack["client"].post(CHAT, json=PAYLOAD)
    assert response.status_code == 401


async def test_a_request_with_a_wrong_key_is_refused(authed_stack: dict[str, Any]) -> None:
    response = await authed_stack["client"].post(
        CHAT, json=PAYLOAD, headers={"authorization": f"Bearer {OTHER_KEY}"}
    )
    assert response.status_code == 401


async def test_a_malformed_authorization_header_is_refused(authed_stack: dict[str, Any]) -> None:
    for header in (TENANT_KEY, f"Basic {TENANT_KEY}", "Bearer"):
        response = await authed_stack["client"].post(
            CHAT, json=PAYLOAD, headers={"authorization": header}
        )
        assert response.status_code == 401, f"{header!r} was accepted"


async def test_the_right_key_gets_an_answer(authed_stack: dict[str, Any]) -> None:
    response = await authed_stack["client"].post(
        CHAT, json=PAYLOAD, headers={"authorization": f"Bearer {TENANT_KEY}"}
    )
    assert response.status_code == 200, response.text


async def test_the_scheme_is_case_insensitive(authed_stack: dict[str, Any]) -> None:
    """`bearer` and `Bearer` are the same scheme per RFC 7235; the key is not."""
    response = await authed_stack["client"].post(
        CHAT, json=PAYLOAD, headers={"authorization": f"bearer {TENANT_KEY}"}
    )
    assert response.status_code == 200, response.text


async def test_usage_is_scoped_to_the_presented_key(authed_stack: dict[str, Any]) -> None:
    client = authed_stack["client"]
    headers = {"authorization": f"Bearer {TENANT_KEY}"}
    await client.post(CHAT, json=PAYLOAD, headers=headers)

    assert (await client.get("/v1/usage")).status_code == 401
    usage = await client.get("/v1/usage", headers=headers)
    assert usage.status_code == 200
    assert usage.json()["totals"]["requests"] >= 1


async def test_the_config_endpoint_never_exposes_a_key(authed_stack: dict[str, Any]) -> None:
    """Introspection stays open on purpose, so it has to stay secret-free."""
    response = await authed_stack["client"].get("/v1/config")
    assert response.status_code == 200
    assert TENANT_KEY not in response.text
    assert response.json()["auth_enabled"] is True

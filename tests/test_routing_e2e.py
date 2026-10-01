"""The router end to end: which route answers, and what the response says about why."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest_asyncio
from fastapi import FastAPI

from llm_gateway.observability import REGISTRY
from llm_gateway.settings import Settings
from tests.conftest import build_settings, running_stack

CHAT = "/v1/chat/completions"
SIMPLE = "What is the capital of France?"
HARD = "Write a Python function that merges two sorted lists."


def routed_settings(*, enabled: bool = True) -> Settings:
    tree = build_settings().model_dump()
    tree["routing"] = {
        "complexity": {
            "enabled": enabled,
            "small_route": "chaos-default-secondary",
            "large_route": "chaos-default",
        }
    }
    return Settings.model_validate(tree)


@pytest_asyncio.fixture
async def routed(mock_app: FastAPI) -> AsyncIterator[dict[str, Any]]:
    async with running_stack(routed_settings(), mock_app) as running:
        yield running


async def ask(stack: dict[str, Any], prompt: str, model: str = "auto") -> Any:
    return await stack["client"].post(
        CHAT, json={"model": model, "messages": [{"role": "user", "content": prompt}]}
    )


def decisions(tier: str) -> float:
    value = REGISTRY.get_sample_value("llm_gateway_routing_decisions_total", {"tier": tier})
    return value or 0.0


async def test_a_simple_request_is_answered_by_the_small_route(routed: dict[str, Any]) -> None:
    response = await ask(routed, SIMPLE)
    assert response.status_code == 200
    assert response.headers["x-gateway-route"] == "chaos-default-secondary"
    assert response.headers["x-gateway-route-decision"].startswith("small; score=0")
    assert response.json()["gateway"]["routing"] == {
        "tier": "small",
        "score": 0,
        "reasons": [],
        "route": "chaos-default-secondary",
    }


async def test_a_hard_request_is_answered_by_the_large_route(routed: dict[str, Any]) -> None:
    response = await ask(routed, HARD)
    assert response.headers["x-gateway-route"] == "chaos-default"
    assert response.headers["x-gateway-route-decision"].startswith("large;")
    assert response.json()["gateway"]["routing"]["reasons"]


async def test_a_named_route_is_never_rerouted(routed: dict[str, Any]) -> None:
    response = await ask(routed, HARD, model="chaos-default-secondary")
    assert response.headers["x-gateway-route"] == "chaos-default-secondary"
    assert "x-gateway-route-decision" not in response.headers
    assert "routing" not in response.json()["gateway"]


async def test_every_decision_is_counted(routed: dict[str, Any]) -> None:
    small, large = decisions("small"), decisions("large")
    await ask(routed, SIMPLE)
    await ask(routed, HARD)
    assert decisions("small") == small + 1
    assert decisions("large") == large + 1


async def test_the_decision_is_stored_with_the_call(routed: dict[str, Any]) -> None:
    response = await ask(routed, HARD)
    await routed["state"].recorder.drain(timeout=2.0)
    usage = await routed["client"].get("/v1/usage?group_by=route")
    assert any(row["key"] == "chaos-default" for row in usage.json()["groups"]), response


async def test_with_the_router_off_the_virtual_route_does_not_exist(
    mock_app: FastAPI,
) -> None:
    async with running_stack(routed_settings(enabled=False), mock_app) as off:
        response = await ask(off, SIMPLE)
    assert response.status_code == 400

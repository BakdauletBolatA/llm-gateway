"""Mock LLM provider with config-driven failure injection.

Speaks three dialects — OpenAI (`/v1/chat/completions`), Anthropic (`/v1/messages`)
and Ollama (`/api/chat`) — on independently configured named upstreams
(`/p/primary`, `/p/secondary`, ...). That is what lets a fallback chain cross API
dialects without a single paid key.

Failure behaviour comes entirely from config/failure_profiles.yaml; this module
only knows *how* to fail, never *when*.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from mock_provider.injector import Card, MockState, UpstreamState
from mock_provider.profiles import Outcome, ProfileLibrary, load_profiles

logger = logging.getLogger(__name__)

DEFAULT_SCENARIO = "healthy"

_SENTENCES = [
    "The short answer is that it depends on the constraints you care about most.",
    "Start by measuring the current behaviour, then change one thing at a time.",
    "There are three common approaches, and the trade-off is mostly operational.",
    "In practice the bottleneck is rarely the model itself.",
    "A worked example makes this clearer than an abstract description.",
    "The failure mode to watch for is silent degradation rather than a hard error.",
    "This is well covered by the standard tooling, so you rarely need custom code.",
    "Keep the fallback path simple: it runs exactly when everything else is broken.",
]


class ProfileRequest(BaseModel):
    upstream: str
    profile: str


class ScenarioRequest(BaseModel):
    scenario: str


def deterministic_answer(prompt: str) -> str:
    """Same prompt in, same answer out — so cache behaviour is testable."""
    digest = hashlib.blake2b(prompt.strip().lower().encode(), digest_size=8).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    picked = rng.sample(_SENTENCES, k=rng.randint(2, 4))
    tag = digest.hex()[:8]
    return " ".join(picked) + f" [mock:{tag}]"


def estimate_prompt_tokens(prompt: str) -> int:
    return max(1, len(prompt) // 4)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    library: ProfileLibrary = app.state.library
    state: MockState = app.state.mock
    apply_scenario(state, library, os.environ.get("MOCK_SCENARIO", DEFAULT_SCENARIO))
    logger.info(
        "mock provider ready: upstreams=%s scenario=%s",
        library.upstreams(),
        state.scenario,
    )
    yield


def apply_scenario(state: MockState, library: ProfileLibrary, scenario: str) -> None:
    if scenario not in library.scenarios:
        raise KeyError(scenario)
    for upstream, profile_name in library.scenarios[scenario].items():
        state.upstream(upstream).set_profile(library.profiles[profile_name])
    state.scenario = scenario


def create_app(profiles_path: str | None = None) -> FastAPI:
    library = load_profiles(profiles_path or os.environ.get("FAILURE_PROFILES"))
    app = FastAPI(title="mock-provider", version="0.1.0", lifespan=lifespan)
    app.state.library = library
    app.state.mock = MockState(base_seed=library.seed)

    # -- admin --------------------------------------------------------------

    @app.post("/admin/scenario")
    async def set_scenario(payload: ScenarioRequest) -> dict[str, Any]:
        try:
            apply_scenario(app.state.mock, library, payload.scenario)
        except KeyError:
            return JSONResponse(  # type: ignore[return-value]
                status_code=400,
                content={
                    "error": f"unknown scenario {payload.scenario!r}",
                    "known": sorted(library.scenarios),
                },
            )
        return {"scenario": payload.scenario, "state": _state_payload(app)}

    @app.post("/admin/profile")
    async def set_profile(payload: ProfileRequest) -> dict[str, Any]:
        if payload.profile not in library.profiles:
            return JSONResponse(  # type: ignore[return-value]
                status_code=400,
                content={
                    "error": f"unknown profile {payload.profile!r}",
                    "known": sorted(library.profiles),
                },
            )
        app.state.mock.upstream(payload.upstream).set_profile(library.profiles[payload.profile])
        return {"upstream": payload.upstream, "profile": payload.profile}

    @app.post("/admin/reset")
    async def reset() -> dict[str, Any]:
        """Re-deal every deck and zero the counters, keeping the current scenario."""
        state: MockState = app.state.mock
        if state.scenario:
            apply_scenario(state, library, state.scenario)
        return {"status": "ok", "state": _state_payload(app)}

    @app.get("/admin/state")
    async def get_state() -> dict[str, Any]:
        return _state_payload(app)

    @app.get("/admin/profiles")
    async def list_profiles() -> dict[str, Any]:
        return {
            "seed": library.seed,
            "profiles": {
                name: {
                    "description": spec.description,
                    "outcomes": {str(k): v for k, v in spec.outcomes.items()},
                    "latency_ms": list(spec.latency_ms),
                    "hang_s": spec.hang_s,
                    "warmup_ok": spec.warmup_ok,
                }
                for name, spec in library.profiles.items()
            },
            "scenarios": library.scenarios,
        }

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # -- provider dialects --------------------------------------------------

    @app.post("/p/{upstream}/v1/chat/completions")
    async def openai_chat(upstream: str, request: Request) -> Response:
        body = await _read_json(request)
        prompt = _openai_prompt(body)
        return await _serve(app, upstream, prompt, body.get("model", "mock-model"), "openai")

    @app.post("/p/{upstream}/v1/messages")
    async def anthropic_messages(upstream: str, request: Request) -> Response:
        body = await _read_json(request)
        prompt = _anthropic_prompt(body)
        return await _serve(app, upstream, prompt, body.get("model", "mock-model"), "anthropic")

    @app.post("/p/{upstream}/api/chat")
    async def ollama_chat(upstream: str, request: Request) -> Response:
        body = await _read_json(request)
        prompt = _openai_prompt(body)
        return await _serve(app, upstream, prompt, body.get("model", "mock-model"), "ollama")

    return app


def _state_payload(app: FastAPI) -> dict[str, Any]:
    state: MockState = app.state.mock
    return {
        "scenario": state.scenario,
        "upstreams": [state.upstreams[name].snapshot() for name in sorted(state.upstreams)],
    }


async def _read_json(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _openai_prompt(body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    parts = [
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages if isinstance(m, dict)
    ]
    return "\n".join(parts)


def _anthropic_prompt(body: dict[str, Any]) -> str:
    parts = []
    if system := body.get("system"):
        parts.append(f"system: {system}")
    parts.append(_openai_prompt(body))
    return "\n".join(part for part in parts if part)


async def _serve(app: FastAPI, upstream: str, prompt: str, model: str, dialect: str) -> Response:
    state: MockState = app.state.mock
    upstream_state = state.upstream(upstream)
    spec = upstream_state.profile

    if not upstream_state.admit():
        # The upstream is at its concurrency ceiling. A real provider answers this
        # with 503 (or 429) and a Retry-After rather than queueing forever.
        return JSONResponse(
            status_code=503,
            headers={"Retry-After": str(int(spec.retry_after_s if spec else 1))},
            content={
                "error": {
                    "message": (
                        f"too many concurrent requests: limit {spec.max_concurrency if spec else 0}"
                    ),
                    "type": "overloaded_error",
                }
            },
        )
    try:
        return await _serve_card(upstream_state, prompt, model, dialect)
    finally:
        upstream_state.release()


async def _serve_card(
    upstream_state: UpstreamState, prompt: str, model: str, dialect: str
) -> Response:
    spec = upstream_state.profile
    card = upstream_state.next_card()

    if card.outcome is Outcome.HANG:
        hang_s = spec.hang_s if spec else 30.0
        await asyncio.sleep(hang_s)
        return _success(card, prompt, model, dialect)

    if card.latency_ms:
        await asyncio.sleep(card.latency_ms / 1000)

    match card.outcome:
        case Outcome.OK:
            return _success(card, prompt, model, dialect)
        case Outcome.HTTP_429:
            retry_after = spec.retry_after_s if spec else 1.0
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(int(retry_after))},
                content={
                    "error": {
                        "message": "Rate limit reached for requests",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
            )
        case Outcome.HTTP_500:
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "internal server error", "type": "server_error"}},
            )
        case Outcome.HTTP_503:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "overloaded", "type": "overloaded_error"}},
            )
        case Outcome.BAD_JSON:
            # Valid content-type, truncated body: the classic "provider returned
            # HTML/partial JSON" failure that naive clients turn into a 500.
            return Response(
                status_code=200,
                media_type="application/json",
                content='{"id": "chatcmpl-mock", "object": "chat.completion", "choices": [{"in',
            )
        case Outcome.BAD_SCHEMA:
            return JSONResponse(status_code=200, content=_empty_payload(model, dialect))
        case Outcome.CONN_ABORT:
            return StreamingResponse(_abort_stream(), media_type="application/json")
        case _:  # pragma: no cover - exhaustive above
            return _success(card, prompt, model, dialect)


async def _abort_stream() -> AsyncIterator[bytes]:
    """Emit a partial body then abort, so the client sees a broken connection."""
    yield b'{"id": "chatcmpl-mock", "object": "chat.completion", "choices": [{"index": 0, "mes'
    await asyncio.sleep(0.01)
    raise RuntimeError("mock provider: connection aborted mid-response")


def _empty_payload(model: str, dialect: str) -> dict[str, Any]:
    if dialect == "anthropic":
        return {"id": "msg_mock", "type": "message", "model": model, "content": []}
    if dialect == "ollama":
        return {"model": model, "done": True}
    return {"id": "chatcmpl-mock", "object": "chat.completion", "model": model, "choices": []}


def _success(card: Card, prompt: str, model: str, dialect: str) -> Response:
    text = deterministic_answer(prompt)
    tokens_in = estimate_prompt_tokens(prompt)
    tokens_out = card.completion_tokens
    created = int(time.time())

    if dialect == "anthropic":
        payload: dict[str, Any] = {
            "id": "msg_mock",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
        }
    elif dialect == "ollama":
        payload = {
            "model": model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created)),
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": tokens_in,
            "eval_count": tokens_out,
        }
    else:
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "total_tokens": tokens_in + tokens_out,
            },
        }
    return JSONResponse(status_code=200, content=payload)

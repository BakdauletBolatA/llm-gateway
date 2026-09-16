"""Provider adapters against a scripted HTTP transport.

The chaos harness exercises the adapters only through the mock provider, which
speaks each dialect the way this repository wrote it. These tests pin the wire
format itself — what each adapter sends and how it reads what comes back — and
the error classification the retry policy and the breaker depend on. A wrong
ErrorKind here silently changes whether a failure is retried, counted against a
provider, or treated as backpressure.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from llm_gateway.errors import ErrorKind, ProviderError
from llm_gateway.providers.anthropic import DEFAULT_MAX_TOKENS, AnthropicAdapter
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.ollama import OllamaAdapter
from llm_gateway.providers.openai import OpenAIAdapter
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.settings import ProviderConfig

Handler = Callable[[httpx.Request], httpx.Response]

CONVERSATION = ChatCompletionRequest.model_validate(
    {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Name a prime."},
            {"role": "assistant", "content": "Seven."},
            {"role": "user", "content": "Another one?"},
        ],
        "max_tokens": 64,
        "temperature": 0.2,
        "stop": ["\n\n"],
    }
)


class Recorder:
    """A transport that answers from a handler and remembers what it was sent."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    @property
    def body(self) -> dict[str, Any]:
        return dict(json.loads(self.requests[-1].content))


def adapter(kind: str, *, api_key_env: str | None = None) -> Any:
    config = ProviderConfig(type=kind, base_url="https://upstream.test/", api_key_env=api_key_env)
    return {"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, "ollama": OllamaAdapter}[kind](
        f"{kind}_test", config
    )


async def call(
    kind: str, handler: Handler, request: ChatCompletionRequest = CONVERSATION, **kw: Any
) -> tuple[ProviderResponse, Recorder]:
    recorder = Recorder(handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(recorder)) as client:
        response = await adapter(kind, **kw).complete(request, "model-x", client)
    return response, recorder


async def failure(kind: str, handler: Handler) -> ProviderError:
    with pytest.raises(ProviderError) as excinfo:
        await call(kind, handler)
    return excinfo.value


def reply(payload: Any, status: int = 200, headers: dict[str, str] | None = None) -> Handler:
    return lambda _request: httpx.Response(status, json=payload, headers=headers)


OK_BODIES: dict[str, dict[str, Any]] = {
    "openai": {
        "model": "model-x-2026",
        "choices": [
            {"message": {"role": "assistant", "content": "Eleven."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 21, "completion_tokens": 3},
    },
    "anthropic": {
        "model": "model-x-2026",
        "content": [{"type": "text", "text": "Eleven."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 21, "output_tokens": 3},
    },
    "ollama": {
        "model": "model-x-2026",
        "message": {"role": "assistant", "content": "Eleven."},
        "done_reason": "stop",
        "prompt_eval_count": 21,
        "eval_count": 3,
    },
}
DIALECTS = list(OK_BODIES)


# -- what goes out ---------------------------------------------------------------


async def test_openai_sends_the_conversation_unchanged() -> None:
    _, sent = await call("openai", reply(OK_BODIES["openai"]))
    assert str(sent.requests[-1].url) == "https://upstream.test/v1/chat/completions"
    assert sent.body["messages"] == [m.model_dump() for m in CONVERSATION.messages]
    assert (sent.body["max_tokens"], sent.body["temperature"], sent.body["stop"]) == (
        64,
        0.2,
        ["\n\n"],
    )


async def test_anthropic_lifts_the_system_prompt_out_of_the_turns() -> None:
    _, sent = await call("anthropic", reply(OK_BODIES["anthropic"]))
    assert str(sent.requests[-1].url) == "https://upstream.test/v1/messages"
    assert sent.body["system"] == "Be brief."
    assert [t["role"] for t in sent.body["messages"]] == ["user", "assistant", "user"]
    assert sent.body["stop_sequences"] == ["\n\n"]
    assert sent.requests[-1].headers["anthropic-version"] == "2023-06-01"


async def test_anthropic_always_sends_max_tokens_because_the_api_requires_it() -> None:
    bare = ChatCompletionRequest.model_validate({"messages": [{"role": "user", "content": "hi"}]})
    _, sent = await call("anthropic", reply(OK_BODIES["anthropic"]), request=bare)
    assert sent.body["max_tokens"] == DEFAULT_MAX_TOKENS
    assert "temperature" not in sent.body, "an unset temperature must not be invented"


async def test_anthropic_promotes_a_lone_system_message_to_a_user_turn() -> None:
    only_system = ChatCompletionRequest.model_validate(
        {"messages": [{"role": "system", "content": "Say hello."}]}
    )
    _, sent = await call("anthropic", reply(OK_BODIES["anthropic"]), request=only_system)
    assert sent.body["messages"] == [{"role": "user", "content": "Say hello."}]
    assert "system" not in sent.body


async def test_ollama_puts_sampling_options_where_ollama_reads_them() -> None:
    _, sent = await call("ollama", reply(OK_BODIES["ollama"]))
    assert str(sent.requests[-1].url) == "https://upstream.test/api/chat"
    assert sent.body["stream"] is False
    assert sent.body["options"] == {"temperature": 0.2, "num_predict": 64, "stop": ["\n\n"]}


@pytest.mark.parametrize(
    ("kind", "header", "expected"),
    [("openai", "authorization", "Bearer sk-live-123"), ("anthropic", "x-api-key", "sk-live-123")],
)
async def test_the_key_goes_in_the_dialects_own_header(
    monkeypatch: pytest.MonkeyPatch, kind: str, header: str, expected: str
) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "sk-live-123")
    _, sent = await call(kind, reply(OK_BODIES[kind]), api_key_env="TEST_PROVIDER_KEY")
    assert sent.requests[-1].headers[header] == expected


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
async def test_no_key_configured_means_no_auth_header(kind: str) -> None:
    _, sent = await call(kind, reply(OK_BODIES[kind]))
    headers = sent.requests[-1].headers
    assert "authorization" not in headers and "x-api-key" not in headers


# -- what comes back -------------------------------------------------------------


@pytest.mark.parametrize("kind", DIALECTS)
async def test_every_dialect_normalises_to_the_same_response(kind: str) -> None:
    response, _ = await call(kind, reply(OK_BODIES[kind]))
    assert (response.text, response.model, response.tokens_in, response.tokens_out) == (
        "Eleven.",
        "model-x-2026",
        21,
        3,
    )


async def test_anthropic_joins_text_blocks_and_skips_the_rest() -> None:
    body = {
        **OK_BODIES["anthropic"],
        "content": [
            {"type": "text", "text": "Eleven"},
            {"type": "tool_use", "id": "t1", "name": "calc", "input": {}},
            {"type": "text", "text": " and thirteen."},
        ],
    }
    response, _ = await call("anthropic", reply(body))
    assert response.text == "Eleven and thirteen."


@pytest.mark.parametrize(
    ("kind", "body"),
    [
        ("openai", {"choices": []}),
        ("openai", {"choices": [{"message": {"content": None}}]}),
        ("anthropic", {"content": []}),
        ("anthropic", {"content": [{"type": "tool_use", "id": "t1"}]}),
        ("ollama", {"message": {"content": ""}}),
        ("ollama", {"done": True}),
    ],
)
async def test_a_200_without_an_answer_is_an_invalid_response(kind: str, body: Any) -> None:
    """A well-formed HTTP success that carries no text must not be served as one."""
    error = await failure(kind, reply(body))
    assert error.kind is ErrorKind.INVALID_RESPONSE
    assert error.status_code == 200


@pytest.mark.parametrize("kind", DIALECTS)
async def test_a_body_that_is_not_json_is_an_invalid_response(kind: str) -> None:
    error = await failure(kind, lambda _r: httpx.Response(200, text="<html>bad gateway</html>"))
    assert error.kind is ErrorKind.INVALID_RESPONSE


# -- error classification --------------------------------------------------------


@pytest.mark.parametrize("kind", DIALECTS)
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ErrorKind.AUTH),
        (403, ErrorKind.AUTH),
        (400, ErrorKind.BAD_REQUEST),
        (408, ErrorKind.TIMEOUT),
        (429, ErrorKind.RATE_LIMITED),
        (500, ErrorKind.SERVER_ERROR),
        (503, ErrorKind.OVERLOADED),
        (529, ErrorKind.OVERLOADED),
    ],
)
async def test_status_codes_map_to_the_kinds_the_policies_act_on(
    kind: str, status: int, expected: ErrorKind
) -> None:
    error = await failure(kind, reply({"error": "nope"}, status=status))
    assert error.kind is expected
    assert error.status_code == status
    assert error.provider == f"{kind}_test"


@pytest.mark.parametrize("kind", DIALECTS)
async def test_retry_after_in_seconds_is_carried_to_the_retry_policy(kind: str) -> None:
    """This field is what the breaker reads to tell backpressure from failure."""
    error = await failure(kind, reply({}, status=429, headers={"retry-after": "2"}))
    assert error.retry_after_s == 2.0


async def test_an_http_date_retry_after_is_treated_as_absent() -> None:
    error = await failure(
        "openai", reply({}, status=503, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    )
    assert error.retry_after_s is None


@pytest.mark.parametrize("kind", DIALECTS)
@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (httpx.ReadTimeout("slow"), ErrorKind.TIMEOUT),
        (httpx.ConnectError("refused"), ErrorKind.CONNECTION),
        (httpx.RemoteProtocolError("peer closed"), ErrorKind.CONNECTION),
    ],
)
async def test_transport_failures_are_classified(
    kind: str, exception: Exception, expected: ErrorKind
) -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise exception

    error = await failure(kind, explode)
    assert error.kind is expected
    assert error.status_code is None

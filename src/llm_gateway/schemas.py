"""Public API contract: an OpenAI-compatible, non-streaming chat completion."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    # `model` names a route from config (routes.definitions), not a provider model.
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=32000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stop: list[str] | None = None
    # Opt-in to the response cache. None means "not asked for": with the default
    # reliability.cache.require_opt_in the request is never served from the cache.
    cache: bool | None = None

    def prompt_text(self) -> str:
        """Flattened prompt used for token estimation."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)

    def generation_params(self) -> dict[str, Any]:
        """Parameters that change what the model writes, as opposed to who asked."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": self.stop,
        }

    def _last_user_index(self) -> int:
        for index in range(len(self.messages) - 1, -1, -1):
            if self.messages[index].role == "user":
                return index
        return len(self.messages) - 1

    def user_text(self) -> str:
        return self.messages[self._last_user_index()].content

    def context_text(self) -> str:
        """Everything except the last user message: system prompt and dialogue so far.

        The cache matches the question, but the answer also depends on this. It is
        hashed into the cache scope, so the same question under a different system
        prompt or after a different conversation is never a hit.
        """
        last = self._last_user_index()
        return "\n".join(f"{m.role}: {m.content}" for i, m in enumerate(self.messages) if i != last)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    # Non-standard but useful: how the gateway produced this answer.
    gateway: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        model: str,
        text: str,
        tokens_in: int,
        tokens_out: int,
        finish_reason: str,
        gateway: dict[str, Any],
    ) -> ChatCompletionResponse:
        return cls(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                total_tokens=tokens_in + tokens_out,
            ),
            gateway=gateway,
        )


class ErrorBody(BaseModel):
    type: str
    message: str
    kind: str
    request_id: str
    attempts: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]

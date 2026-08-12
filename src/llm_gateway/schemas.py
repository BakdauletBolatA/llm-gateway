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
    # Escape hatch for clients that must not be served from the semantic cache.
    cache: bool = True

    def prompt_text(self) -> str:
        """Flattened prompt used for cache keys and token estimation."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)

    def user_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return self.messages[-1].content


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

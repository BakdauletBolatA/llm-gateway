"""OpenAI-dialect adapter (`POST /v1/chat/completions`).

Also covers any OpenAI-compatible endpoint, including the mock's `openai` upstreams.
"""

from __future__ import annotations

from typing import Any

import httpx

from llm_gateway.providers.base import (
    ProviderResponse,
    decode_json,
    map_status_error,
    map_transport_error,
    schema_error,
)
from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.settings import ProviderConfig


class OpenAIAdapter:
    dialect = "openai"

    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config
        self._url = f"{config.base_url.rstrip('/')}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if key := self.config.api_key:
            headers["authorization"] = f"Bearer {key}"
        return headers

    def _payload(self, request: ChatCompletionRequest, model: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop"] = request.stop
        return payload

    async def complete(
        self,
        request: ChatCompletionRequest,
        model: str,
        client: httpx.AsyncClient,
    ) -> ProviderResponse:
        try:
            response = await client.post(
                self._url, json=self._payload(request, model), headers=self._headers()
            )
        except Exception as exc:
            raise map_transport_error(exc, provider=self.name, model=model) from exc

        if response.status_code >= 400:
            raise map_status_error(response, provider=self.name, model=model)

        payload = decode_json(response, provider=self.name, model=model)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise schema_error("response has no choices[]", provider=self.name, model=model)
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise schema_error(
                "choices[0].message.content is missing or not a string",
                provider=self.name,
                model=model,
            )

        usage = payload.get("usage") or {}
        return ProviderResponse(
            text=message["content"],
            model=str(payload.get("model") or model),
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            finish_reason=str(choices[0].get("finish_reason") or "stop"),
            raw=payload,
        )

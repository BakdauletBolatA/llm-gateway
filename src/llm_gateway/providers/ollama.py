"""Ollama-dialect adapter (`POST /api/chat`, non-streaming).

Ollama reports token counts as prompt_eval_count / eval_count and has no notion
of cost — the pricing table maps local models to $0, which is exactly the point
of keeping it as the last hop in a fallback chain.
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


class OllamaAdapter:
    dialect = "ollama"

    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config
        self._url = f"{config.base_url.rstrip('/')}/api/chat"

    def _payload(self, request: ChatCompletionRequest, model: str) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if request.stop:
            options["stop"] = request.stop

        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
            "stream": False,
        }
        if options:
            payload["options"] = options
        return payload

    async def complete(
        self,
        request: ChatCompletionRequest,
        model: str,
        client: httpx.AsyncClient,
    ) -> ProviderResponse:
        try:
            response = await client.post(
                self._url,
                json=self._payload(request, model),
                headers={"content-type": "application/json"},
            )
        except Exception as exc:
            raise map_transport_error(exc, provider=self.name, model=model) from exc

        if response.status_code >= 400:
            raise map_status_error(response, provider=self.name, model=model)

        payload = decode_json(response, provider=self.name, model=model)
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise schema_error(
                "message.content is missing or not a string", provider=self.name, model=model
            )
        if not message["content"]:
            raise schema_error("message.content is empty", provider=self.name, model=model)

        return ProviderResponse(
            text=message["content"],
            model=str(payload.get("model") or model),
            tokens_in=int(payload.get("prompt_eval_count") or 0),
            tokens_out=int(payload.get("eval_count") or 0),
            finish_reason=str(payload.get("done_reason") or "stop"),
            raw=payload,
        )

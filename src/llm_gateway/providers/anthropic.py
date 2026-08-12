"""Anthropic-dialect adapter (`POST /v1/messages`).

Kept as a separate dialect on purpose: the system prompt is a top-level field
rather than a message, and usage is reported as input_tokens/output_tokens.
A fallback chain that crosses this boundary is a real translation, not a re-host.
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

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024


class AnthropicAdapter:
    dialect = "anthropic"

    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config
        self._url = f"{config.base_url.rstrip('/')}/v1/messages"

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if key := self.config.api_key:
            headers["x-api-key"] = key
        return headers

    def _payload(self, request: ChatCompletionRequest, model: str) -> dict[str, Any]:
        system_parts = [m.content for m in request.messages if m.role == "system"]
        turns = [
            {"role": m.role, "content": m.content} for m in request.messages if m.role != "system"
        ]
        if not turns:
            # Anthropic requires at least one turn; promote the system text.
            turns = [{"role": "user", "content": system_parts.pop() if system_parts else ""}]

        payload: dict[str, Any] = {
            "model": model,
            "messages": turns,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop_sequences"] = request.stop
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
        blocks = payload.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise schema_error("response has no content[]", provider=self.name, model=model)

        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            raise schema_error("content[] contains no text blocks", provider=self.name, model=model)

        usage = payload.get("usage") or {}
        return ProviderResponse(
            text=text,
            model=str(payload.get("model") or model),
            tokens_in=int(usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
            finish_reason=str(payload.get("stop_reason") or "end_turn"),
            raw=payload,
        )

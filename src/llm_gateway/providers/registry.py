"""Provider registry: one adapter and one HTTP client per configured provider.

The client owns the timeout policy, so switching timeouts on or off is a config
change rather than a code change.
"""

from __future__ import annotations

import httpx

from llm_gateway.providers.anthropic import AnthropicAdapter
from llm_gateway.providers.base import ProviderAdapter
from llm_gateway.providers.ollama import OllamaAdapter
from llm_gateway.providers.openai import OpenAIAdapter
from llm_gateway.settings import ProviderConfig, Settings, TimeoutConfig

_ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "ollama": OllamaAdapter,
}


def build_timeout(config: TimeoutConfig) -> httpx.Timeout:
    """`enabled: false` means literally no timeout — that is the naive baseline."""
    if not config.enabled:
        return httpx.Timeout(None)
    return httpx.Timeout(
        connect=config.connect_s,
        read=config.read_s,
        write=config.write_s,
        pool=config.connect_s,
    )


def build_adapter(name: str, config: ProviderConfig) -> ProviderAdapter:
    adapter_cls = _ADAPTERS[config.type]
    return adapter_cls(name, config)  # type: ignore[return-value]


class ProviderRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._adapters: dict[str, ProviderAdapter] = {}
        self._clients: dict[str, httpx.AsyncClient] = {}

        timeout = build_timeout(settings.reliability.timeouts)
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        for name, provider in settings.providers.items():
            if not provider.enabled:
                continue
            self._adapters[name] = build_adapter(name, provider)
            self._clients[name] = httpx.AsyncClient(timeout=timeout, limits=limits)

    def adapter(self, name: str) -> ProviderAdapter:
        return self._adapters[name]

    def client(self, name: str) -> httpx.AsyncClient:
        return self._clients[name]

    def enabled_providers(self) -> list[str]:
        return sorted(self._adapters)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()

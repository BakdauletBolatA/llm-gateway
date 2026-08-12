from __future__ import annotations

from pathlib import Path

import pytest

from llm_gateway.cost import compute_cost_usd, estimate_tokens
from llm_gateway.settings import PricingConfig, load_settings

CONFIG = "config/gateway.yaml"


def test_shipped_config_loads_and_is_internally_consistent() -> None:
    settings = load_settings(CONFIG)
    assert settings.routes.default in settings.routes.definitions
    for route in settings.routes.definitions.values():
        for hop in route.chain:
            assert hop.provider in settings.providers


def test_env_placeholders_use_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOCK_BASE_URL", raising=False)
    settings = load_settings(CONFIG)
    assert settings.providers["mock_primary"].base_url == "http://mock:8081/p/primary"


def test_env_placeholders_are_substituted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_BASE_URL", "http://127.0.0.1:9999")
    settings = load_settings(CONFIG)
    assert settings.providers["mock_primary"].base_url == "http://127.0.0.1:9999/p/primary"


def test_env_override_reaches_a_nested_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GW__RELIABILITY__RETRIES__ENABLED", "true")
    monkeypatch.setenv("GW__RELIABILITY__RETRIES__MAX_ATTEMPTS", "7")
    settings = load_settings(CONFIG)
    assert settings.reliability.retries.enabled is True
    assert settings.reliability.retries.max_attempts == 7


def test_overlay_is_merged_on_top_of_the_base(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        "reliability:\n  timeouts:\n    enabled: true\n    read_s: 3.5\n", encoding="utf-8"
    )
    settings = load_settings(CONFIG, overlay_path=overlay)
    assert settings.reliability.timeouts.enabled is True
    assert settings.reliability.timeouts.read_s == 3.5
    # Untouched keys survive the merge.
    assert settings.reliability.timeouts.connect_s == 2.0


def test_fallback_disabled_truncates_the_chain_to_one_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GW__RELIABILITY__FALLBACK__ENABLED", "false")
    # Hedging is meaningless without a chain, and the config refuses the combination.
    monkeypatch.setenv("GW__RELIABILITY__HEDGING__ENABLED", "false")
    settings = load_settings(CONFIG)
    assert len(settings.resolve_chain("chaos-default")) == 1


def test_hedging_without_a_fallback_chain_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail at startup, not silently at runtime.

    With a single-provider chain there is no second provider to race, so a config
    that asks for hedging without fallback promises something it cannot deliver.
    """
    monkeypatch.setenv("GW__RELIABILITY__HEDGING__ENABLED", "true")
    monkeypatch.setenv("GW__RELIABILITY__FALLBACK__ENABLED", "false")
    with pytest.raises(ValueError, match=r"hedging requires reliability\.fallback"):
        load_settings(CONFIG)


def test_a_hedge_delay_past_the_deadline_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GW__RELIABILITY__HEDGING__DELAY_MS", "60000")
    with pytest.raises(ValueError, match="is not below"):
        load_settings(CONFIG)


def test_fallback_enabled_uses_the_whole_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GW__RELIABILITY__FALLBACK__ENABLED", "true")
    settings = load_settings(CONFIG)
    assert len(settings.resolve_chain("chaos-default")) == 3


def test_disabled_providers_are_excluded_from_the_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GW__RELIABILITY__FALLBACK__ENABLED", "true")
    monkeypatch.setenv("GW__PROVIDERS__MOCK_SECONDARY__ENABLED", "false")
    settings = load_settings(CONFIG)
    providers = [hop.provider for hop in settings.resolve_chain("chaos-default")]
    assert providers == ["mock_primary", "mock_tertiary"]


def test_token_estimate_is_a_rough_quarter_of_the_characters() -> None:
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("") == 1


def test_pricing_prefers_the_provider_qualified_entry() -> None:
    pricing = PricingConfig.model_validate(
        {
            "default": {"input_per_mtok": 9.0, "output_per_mtok": 9.0},
            "models": {
                "gpt-4o-mini": {"input_per_mtok": 0.15, "output_per_mtok": 0.60},
                "openai:gpt-4o-mini": {"input_per_mtok": 0.10, "output_per_mtok": 0.40},
            },
        }
    )
    assert pricing.price_for("openai", "gpt-4o-mini").input_per_mtok == 0.10
    assert pricing.price_for("azure", "gpt-4o-mini").input_per_mtok == 0.15
    assert pricing.price_for("azure", "unknown-model").input_per_mtok == 9.0


def test_cost_is_computed_per_million_tokens() -> None:
    pricing = PricingConfig.model_validate(
        {"models": {"m": {"input_per_mtok": 3.0, "output_per_mtok": 15.0}}}
    )
    assert compute_cost_usd(pricing, "p", "m", 1_000_000, 0) == 3.0
    assert compute_cost_usd(pricing, "p", "m", 0, 1_000_000) == 15.0
    assert compute_cost_usd(pricing, "p", "m", 1000, 2000) == round(0.003 + 0.03, 6)


def test_local_models_cost_nothing() -> None:
    settings = load_settings(CONFIG)
    assert compute_cost_usd(settings.pricing, "ollama", "llama3.2", 10_000, 10_000) == 0.0

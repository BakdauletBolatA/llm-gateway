"""Token accounting and cost calculation."""

from __future__ import annotations

from llm_gateway.settings import PricingConfig

#: Rough characters-per-token ratio, used only for the pre-flight budget estimate.
#: Real accounting always uses the token counts reported by the provider.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def compute_cost_usd(
    pricing: PricingConfig,
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float:
    price = pricing.price_for(provider, model)
    cost = (tokens_in / 1_000_000) * price.input_per_mtok
    cost += (tokens_out / 1_000_000) * price.output_per_mtok
    # Six decimals ≈ a micro-dollar; matches the NUMERIC(14, 6) column in Postgres.
    return round(cost, 6)

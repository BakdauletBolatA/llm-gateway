"""Choose between a small and a large model from how hard the request looks.

The rules are deliberately plain: a handful of surface features, each worth a few
points, summed and compared with one threshold. Nothing here is learned, so every
decision can be explained by pointing at the line that produced it, and returned
with the response as `reasons`.

What it cannot see is difficulty that is not visible on the surface: a short,
innocent-looking question can be hard. The routing eval measures how often that
costs quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from llm_gateway.schemas import ChatCompletionRequest
from llm_gateway.settings import ComplexityRouterConfig

Tier = Literal["small", "large"]
_NUMBERED_ITEM = re.compile(r"^\s*(\d+[.)]|[-*])\s+", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Decision:
    tier: Tier
    score: int
    reasons: list[str] = field(default_factory=list)
    route: str = ""

    def header(self) -> str:
        why = ",".join(self.reasons) or "none"
        return f"{self.tier}; score={self.score}; reasons={why}"


def _matches(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            return found.group(0).strip().lower()
    return None


def classify(request: ChatCompletionRequest, config: ComplexityRouterConfig) -> Decision:
    text = request.user_text()
    score = 0
    reasons: list[str] = []

    def add(points: int, label: str) -> None:
        nonlocal score
        if points:
            score += points
            reasons.append(f"{label}:+{points}")

    if len(text) >= config.long_chars:
        add(config.long_points, f"length>={config.long_chars}")
    elif len(text) >= config.medium_chars:
        add(config.medium_points, f"length>={config.medium_chars}")

    if (marker := _matches(config.code_markers, text)) is not None:
        add(config.code_points, f"code({marker})")
    if (marker := _matches(config.reasoning_markers, text)) is not None:
        add(config.reasoning_points, f"reasoning({marker})")
    if (marker := _matches(config.math_markers, text)) is not None:
        add(config.math_points, f"math({marker})")

    if text.count("?") >= 2 or len(_NUMBERED_ITEM.findall(text)) >= 2:
        add(config.multi_part_points, "multi_part")
    if len(request.messages) >= config.long_conversation_turns:
        add(config.turns_points, f"turns>={config.long_conversation_turns}")

    tier: Tier = "large" if score >= config.large_threshold else "small"
    route = config.large_route if tier == "large" else config.small_route
    return Decision(tier=tier, score=score, reasons=reasons, route=route)

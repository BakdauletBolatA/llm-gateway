"""Spend and traffic summary over a period."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from sqlalchemy import Float, Integer, cast, func, select

from llm_gateway.db.models import LlmCall
from llm_gateway.db.session import Database

GroupBy = Literal["none", "provider", "model", "route", "api_key", "day"]

_GROUPING: dict[str, Any] = {
    "none": None,
    "provider": LlmCall.provider,
    "model": LlmCall.model,
    "route": LlmCall.route,
    "api_key": LlmCall.api_key_id,
    "day": func.date_trunc("day", LlmCall.created_at),
}


def _metrics() -> list[Any]:
    success = func.sum(cast(LlmCall.outcome == "success", Integer))
    return [
        func.count().label("requests"),
        success.label("successes"),
        func.sum(cast(LlmCall.cache_hit, Integer)).label("cache_hits"),
        func.sum(LlmCall.attempts).label("attempts"),
        func.sum(LlmCall.retries).label("retries"),
        func.sum(LlmCall.fallbacks).label("fallbacks"),
        func.sum(LlmCall.tokens_in).label("tokens_in"),
        func.sum(LlmCall.tokens_out).label("tokens_out"),
        func.coalesce(func.sum(LlmCall.cost_usd), 0).label("cost_usd"),
        func.percentile_cont(0.5)
        .within_group(cast(LlmCall.latency_ms, Float))
        .label("p50_latency_ms"),
        func.percentile_cont(0.95)
        .within_group(cast(LlmCall.latency_ms, Float))
        .label("p95_latency_ms"),
    ]


def _row_to_dict(row: Any, key: str | None) -> dict[str, Any]:
    requests = int(row.requests or 0)
    successes = int(row.successes or 0)
    payload: dict[str, Any] = {
        "requests": requests,
        "successes": successes,
        "errors": requests - successes,
        "success_rate": round(successes / requests, 4) if requests else 0.0,
        "cache_hits": int(row.cache_hits or 0),
        "attempts": int(row.attempts or 0),
        "retries": int(row.retries or 0),
        "fallbacks": int(row.fallbacks or 0),
        "tokens_in": int(row.tokens_in or 0),
        "tokens_out": int(row.tokens_out or 0),
        "cost_usd": round(float(row.cost_usd or 0), 6),
        "p50_latency_ms": round(float(row.p50_latency_ms or 0), 1),
        "p95_latency_ms": round(float(row.p95_latency_ms or 0), 1),
    }
    if key is not None:
        payload["key"] = key
    return payload


async def usage_summary(
    database: Database,
    *,
    since: datetime,
    until: datetime,
    group_by: GroupBy = "none",
    api_key_id: str | None = None,
) -> dict[str, Any]:
    grouping = _GROUPING[group_by]
    filters = [LlmCall.created_at >= since, LlmCall.created_at < until]
    if api_key_id is not None:
        filters.append(LlmCall.api_key_id == api_key_id)

    async with database.session() as session:
        total_row = (await session.execute(select(*_metrics()).where(*filters))).one()
        totals = _row_to_dict(total_row, None)

        groups: list[dict[str, Any]] = []
        if grouping is not None:
            rows = await session.execute(
                select(grouping.label("key"), *_metrics())
                .where(*filters)
                .group_by(grouping)
                .order_by(func.coalesce(func.sum(LlmCall.cost_usd), 0).desc())
            )
            for row in rows.all():
                key = row.key
                if isinstance(key, datetime):
                    key = key.date().isoformat()
                groups.append(_row_to_dict(row, str(key) if key is not None else "unknown"))

    return {
        "from": since.isoformat(),
        "to": until.isoformat(),
        "group_by": group_by,
        "totals": totals,
        "groups": groups,
    }

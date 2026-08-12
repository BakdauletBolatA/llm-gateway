from __future__ import annotations

from datetime import UTC, datetime

import pytest

from llm_gateway.budget import BudgetTracker, period_start
from llm_gateway.errors import BudgetExceededError
from llm_gateway.settings import AuthConfig, BudgetConfig


class _NoDatabase:
    """The tracker only touches the database on refresh(); these tests never do."""


def tracker(limit: float = 1.0, keys: list[dict[str, object]] | None = None) -> BudgetTracker:
    auth = AuthConfig.model_validate({"keys": keys or []})
    config = BudgetConfig(enabled=True, period="day", limit_usd=limit)
    return BudgetTracker(config, auth, _NoDatabase())  # type: ignore[arg-type]


def test_day_period_starts_at_utc_midnight() -> None:
    now = datetime(2026, 8, 12, 15, 30, tzinfo=UTC)
    assert period_start("day", now) == datetime(2026, 8, 12, 0, 0, tzinfo=UTC)


def test_month_period_starts_on_the_first() -> None:
    now = datetime(2026, 8, 12, 15, 30, tzinfo=UTC)
    assert period_start("month", now) == datetime(2026, 8, 1, 0, 0, tzinfo=UTC)


def test_spending_under_the_limit_is_allowed() -> None:
    budget = tracker(limit=1.0)
    budget.record_spend(0.4)
    budget.check(0.5)


def test_crossing_the_limit_is_refused_before_the_provider_is_called() -> None:
    budget = tracker(limit=1.0)
    budget.record_spend(0.8)
    with pytest.raises(BudgetExceededError) as excinfo:
        budget.check(0.3)
    assert excinfo.value.limit_usd == 1.0
    assert excinfo.value.spent_usd == pytest.approx(0.8)
    assert excinfo.value.http_status == 402


def test_a_disabled_budget_never_refuses() -> None:
    budget = tracker(limit=0.0)
    budget._config.enabled = False  # noqa: SLF001
    budget.record_spend(100.0)
    budget.check(100.0)


def test_per_key_limit_applies_on_top_of_the_global_one() -> None:
    budget = tracker(
        limit=100.0,
        keys=[{"id": "team-a", "key": "secret", "budget_limit_usd": 0.5}],
    )
    budget.record_spend(0.4, "team-a")
    budget.check(0.05, "team-a")
    with pytest.raises(BudgetExceededError) as excinfo:
        budget.check(0.2, "team-a")
    assert "team-a" in str(excinfo.value)


def test_one_key_running_out_does_not_block_another() -> None:
    budget = tracker(
        limit=100.0,
        keys=[
            {"id": "team-a", "key": "a", "budget_limit_usd": 0.5},
            {"id": "team-b", "key": "b", "budget_limit_usd": 0.5},
        ],
    )
    budget.record_spend(0.6, "team-a")
    with pytest.raises(BudgetExceededError):
        budget.check(0.01, "team-a")
    budget.check(0.01, "team-b")


def test_snapshot_reports_what_is_left() -> None:
    budget = tracker(limit=2.0)
    budget.record_spend(0.5)
    snapshot = budget.snapshot()
    assert snapshot.spent_usd == pytest.approx(0.5)
    assert snapshot.remaining_usd == pytest.approx(1.5)


def test_database_view_wins_when_it_is_ahead_of_the_local_counter() -> None:
    # After a restart the local counter is zero but the period is already spent.
    budget = tracker(limit=1.0)
    budget._global.from_db = 0.95  # noqa: SLF001
    with pytest.raises(BudgetExceededError):
        budget.check(0.1)

"""The files that only the benchmark and the dashboard ever touch.

Twenty-odd config overlays, the scenario list the harness runs, and the Grafana
dashboard are all outside the request path: nothing imports them, so a typo in one
surfaces twenty minutes into a measurement run, or — worse — as an empty panel that
nobody notices. These tests load every one of them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaos.run import DEFAULT_SCENARIOS
from llm_gateway import observability
from llm_gateway.settings import load_settings
from mock_provider.profiles import load_profiles

CONFIG = "config/gateway.yaml"
PROFILES = "config/failure_profiles.yaml"
DASHBOARD = Path("ops/grafana-dashboard.json")

OVERLAYS = sorted(
    path
    for directory in ("config/iterations", "config/ablations", "config/extras")
    for path in Path(directory).glob("*.yaml")
)


def test_there_are_overlays_to_check() -> None:
    """A glob that silently matches nothing would make every test below vacuous."""
    assert len(OVERLAYS) >= 20, f"expected the full set of overlays, found {len(OVERLAYS)}"


@pytest.mark.parametrize("overlay", OVERLAYS, ids=lambda path: path.stem)
def test_every_overlay_loads_on_top_of_the_base_config(overlay: Path) -> None:
    """An overlay is only exercised when someone re-runs that row of the report.

    Loading them here means a bad key fails in seconds instead of in the middle of
    a twenty-minute reproduction run.
    """
    settings = load_settings(CONFIG, overlay_path=overlay)
    assert settings.routes.default in settings.routes.definitions
    assert settings.resolve_chain(settings.routes.default), "the default route has no providers"


@pytest.mark.parametrize("scenario", DEFAULT_SCENARIOS)
def test_every_default_scenario_exists_in_the_profile_library(scenario: str) -> None:
    library = load_profiles(PROFILES)
    assert scenario in library.scenarios, (
        f"chaos.run would ask the mock for {scenario!r}, which config/failure_profiles.yaml "
        "does not define"
    )


def test_the_report_covers_every_scenario_the_harness_runs() -> None:
    """The matrix in RELIABILITY.md is only complete if every scenario was measured."""
    measured = {
        path.name.split("__", 1)[1].removesuffix(".json")
        for path in Path("bench/results").glob("0*__*.json")
    }
    missing = set(DEFAULT_SCENARIOS) - measured
    assert not missing, f"no committed results for {sorted(missing)}"


# -- the Grafana dashboard ----------------------------------------------------


def exported_metric_names() -> set[str]:
    """Family names the gateway actually exposes, read from a live exposition."""
    names = set()
    for line in observability.render().decode().splitlines():
        if line.startswith("# TYPE "):
            names.add(line.split()[2])
    return names


def dashboard_metric_names() -> set[str]:
    import re

    return set(re.findall(r"llm_gateway_[a-z_]+", DASHBOARD.read_text(encoding="utf-8")))


def test_the_dashboard_is_valid_json_with_panels() -> None:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    assert dashboard["title"] and dashboard["uid"]
    panels = [panel for panel in dashboard["panels"] if panel["type"] != "row"]
    assert len(panels) >= 8, "a dashboard this small is not worth shipping"
    for panel in panels:
        assert panel.get("targets"), f"panel {panel['title']!r} queries nothing"


def test_every_metric_on_the_dashboard_is_actually_exported() -> None:
    """A renamed metric leaves an empty panel, which is worse than no panel."""
    exported = exported_metric_names()
    unknown = []
    for name in sorted(dashboard_metric_names()):
        # Histograms are queried through their _bucket/_sum/_count samples, and
        # counters through the _total sample; the family is what gets registered.
        base = name.removesuffix("_bucket").removesuffix("_sum").removesuffix("_count")
        if base not in exported and base.removesuffix("_total") not in exported:
            unknown.append(name)
    assert not unknown, f"the dashboard queries metrics the gateway does not export: {unknown}"


# -- a typo must not measure a different build --------------------------------


@pytest.mark.parametrize(
    ("overlay_yaml", "typo"),
    [
        ("reliability:\n  hedgin:\n    enabled: false\n", "reliability.hedgin"),
        ("reliability:\n  hedging:\n    delya_ms: 999\n", "reliability.hedging.delya_ms"),
        ("budgt:\n  limit_usd: 1.0\n", "budgt"),
    ],
    ids=["misspelled section", "misspelled field", "misspelled top level"],
)
def test_an_unknown_key_is_refused(tmp_path: Path, overlay_yaml: str, typo: str) -> None:
    """Config models forbid extras on purpose.

    Pydantic ignores unknown keys by default, and for this project that is the worst
    possible default: `hedgin:` instead of `hedging:` would load, do nothing, and the
    chaos run would attribute the result to a configuration the file does not
    actually describe.
    """
    overlay = tmp_path / "typo.yaml"
    overlay.write_text(overlay_yaml, encoding="utf-8")

    with pytest.raises(ValueError, match=typo.split(".")[-1]):
        load_settings(CONFIG, overlay_path=overlay)


def test_an_unknown_environment_override_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GW__RELIABILITY__HEDGING__DELYA_MS", "999")
    with pytest.raises(ValueError, match="delya_ms"):
        load_settings(CONFIG)

"""Failure-profile configuration for the mock provider."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_PROFILE_PATH = "config/failure_profiles.yaml"


class Outcome(StrEnum):
    OK = "ok"
    HTTP_429 = "http_429"
    HTTP_500 = "http_500"
    HTTP_503 = "http_503"
    CONN_ABORT = "conn_abort"
    BAD_JSON = "bad_json"
    BAD_SCHEMA = "bad_schema"
    HANG = "hang"


class ProfileDefaults(BaseModel):
    deck_size: int = Field(default=100, ge=1)
    latency_ms: tuple[int, int] = (40, 120)
    completion_tokens: tuple[int, int] = (48, 160)
    retry_after_s: float = 1.0
    hang_s: float = 30.0
    warmup_ok: int = 0


class ProfileSpec(BaseModel):
    name: str = ""
    description: str = ""
    outcomes: dict[Outcome, int]
    deck_size: int = 100
    latency_ms: tuple[int, int] = (40, 120)
    completion_tokens: tuple[int, int] = (48, 160)
    retry_after_s: float = 1.0
    hang_s: float = 30.0
    warmup_ok: int = 0

    @model_validator(mode="after")
    def _validate(self) -> ProfileSpec:
        if not self.outcomes or sum(self.outcomes.values()) <= 0:
            raise ValueError(f"profile {self.name!r} has no positive outcome weights")
        if self.latency_ms[0] > self.latency_ms[1]:
            raise ValueError(f"profile {self.name!r} has an inverted latency_ms range")
        return self


class ProfileLibrary(BaseModel):
    seed: int = 0
    profiles: dict[str, ProfileSpec]
    scenarios: dict[str, dict[str, str]]

    @model_validator(mode="after")
    def _scenarios_reference_known_profiles(self) -> ProfileLibrary:
        for scenario, assignment in self.scenarios.items():
            for upstream, profile in assignment.items():
                if profile not in self.profiles:
                    raise ValueError(
                        f"scenario {scenario!r} assigns unknown profile "
                        f"{profile!r} to upstream {upstream!r}"
                    )
        return self

    def upstreams(self) -> list[str]:
        names: set[str] = set()
        for assignment in self.scenarios.values():
            names.update(assignment)
        return sorted(names)


def load_profiles(path: str | Path | None = None) -> ProfileLibrary:
    profile_path = Path(path or DEFAULT_PROFILE_PATH)
    with profile_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    defaults = ProfileDefaults.model_validate(raw.get("defaults") or {})
    profiles: dict[str, ProfileSpec] = {}
    for name, spec in (raw.get("profiles") or {}).items():
        merged = defaults.model_dump() | dict(spec)
        merged["name"] = name
        profiles[name] = ProfileSpec.model_validate(merged)

    return ProfileLibrary(
        seed=int(raw.get("seed") or 0),
        profiles=profiles,
        scenarios=raw.get("scenarios") or {},
    )

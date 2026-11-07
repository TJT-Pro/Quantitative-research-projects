"""Research configuration: every choice lives in configs/*.yaml and is validated here.

Models are strict and frozen, so a typo or a wrong type fails on load. The resolved
configuration is fingerprinted and the fingerprint is written next to every output.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SourcesConfig(_Strict):
    fred_api: str
    fred_key_env: str
    gdpnow: tuple[str, ...] = Field(min_length=1)
    official_hosts: tuple[str, ...]


class DownloadConfig(_Strict):
    retries: int = Field(ge=0)
    backoff_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    chunk_bytes: int = Field(gt=0)
    pause_seconds: float = Field(ge=0)
    page_limit: int = Field(ge=1, le=100_000)


Transform = Literal["lin", "chg", "pch", "pca"]


class SeriesConfig(_Strict):
    id: str
    name: str
    frequency: Literal["m", "q"]
    transform: Transform
    blocks: tuple[str, ...]
    category: str


class PanelConfig(_Strict):
    spec_source: str
    observation_start: str
    estimation_start: str
    blocks: tuple[str, ...]
    series: tuple[SeriesConfig, ...]
    target: str

    @model_validator(mode="after")
    def _consistent(self) -> PanelConfig:
        ids = [s.id for s in self.series]
        if len(set(ids)) != len(ids):
            raise ValueError("series ids must be unique")
        if self.target not in ids:
            raise ValueError(f"target {self.target} is not in the panel")
        target = next(s for s in self.series if s.id == self.target)
        if target.frequency != "q":
            raise ValueError("the target must be quarterly")
        first = self.blocks[0]
        for s in self.series:
            unknown = set(s.blocks) - set(self.blocks)
            if unknown:
                raise ValueError(f"{s.id}: unknown blocks {sorted(unknown)}")
            if first not in s.blocks:
                raise ValueError(f"{s.id}: every series loads on the {first} block")
            if s.frequency == "q" and s.transform != "pca":
                raise ValueError(f"{s.id}: quarterly series enter as annualized growth (pca)")
        return self

    @property
    def ordered(self) -> tuple[SeriesConfig, ...]:
        """Monthly series first, then quarterly, each in file order (as the model expects)."""
        return tuple(s for s in self.series if s.frequency == "m") + tuple(
            s for s in self.series if s.frequency == "q"
        )


class ModelConfig(_Strict):
    threshold: float = Field(gt=0)
    max_iter: int = Field(ge=1)
    measurement_variance: float = Field(gt=0)
    reestimate: Literal["monthly"]
    warm_start: bool
    accelerate: bool
    pandemic_free: tuple[str, ...]
    rho_start: float = Field(gt=0, lt=1)


class EvaluationConfig(_Strict):
    first_quarter: str
    horizon_edges: tuple[int, ...]
    excluded_years: tuple[int, ...]
    example_quarter: str

    @model_validator(mode="after")
    def _edges(self) -> EvaluationConfig:
        e = self.horizon_edges
        if len(e) < 2 or any(b <= a for a, b in itertools.pairwise(e)) or e[0] < 0:
            raise ValueError("horizon_edges must be increasing and start at 0 or later")
        return self


class MarketConfig(_Strict):
    yields: tuple[str, ...]
    primary: str


class PlotConfig(_Strict):
    dpi: int = Field(ge=72)
    palette: dict[str, str]


class Config(_Strict):
    sources: SourcesConfig
    download: DownloadConfig
    panel: PanelConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    market: MarketConfig
    plot: PlotConfig

    @model_validator(mode="after")
    def _consistent(self) -> Config:
        if self.market.primary not in self.market.yields:
            raise ValueError("the primary yield must be one of the yields")
        return self

    def fingerprint(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    if isinstance(value, dict):
        return {k: _tuples(v) for k, v in value.items()}
    return value


def load(*, path: Path, override: Path | None = None) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if override is not None:
        raw = _deep_merge(raw, yaml.safe_load(override.read_text(encoding="utf-8")) or {})
    return Config.model_validate(_tuples(raw))


def write_resolved(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": config.fingerprint(), "config": config.model_dump(mode="json")}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

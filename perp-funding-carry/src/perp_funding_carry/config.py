"""Research configuration: every choice lives in configs/*.yaml and is validated here.

Models are strict and frozen, so a typo or a wrong type fails on load. The resolved
configuration is fingerprinted and the fingerprint is written next to every output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SourcesConfig(_Strict):
    vision_base: str
    vision_list: str
    binance_fapi: str
    binance_spot_api: str
    bybit_api: str
    okx_api: str
    official_hosts: tuple[str, ...]
    venue_names: dict[str, str]  # venue key -> display name; the order is the display order
    full_history_venues: tuple[str, ...]


class DownloadConfig(_Strict):
    retries: int = Field(ge=0)
    backoff_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    chunk_bytes: int = Field(gt=0)
    workers: int = Field(ge=1, le=32)
    rest_pause_seconds: float = Field(ge=0)


class UniverseConfig(_Strict):
    quote: str
    first_month: str
    last_month: str | None  # None: the last complete month at fetch time
    min_months: int = Field(ge=1)
    bar_interval: str
    core_symbols: tuple[str, ...]
    core_bar_interval: str
    multiplier_prefixes: dict[str, int]
    stablecoin_pair: str
    oi_window_days: int = Field(ge=7)


class RebuildConfig(_Strict):
    symbols: tuple[str, ...]
    months: int = Field(ge=1)
    interest_8h: float
    clamp: float = Field(gt=0)
    tolerance: float = Field(gt=0)
    equal_weight_interval_hours: tuple[int, ...]
    conventions: tuple[Literal["faq", "per_interval"], ...]
    cap_test_threshold: float = Field(gt=0)


class CostConfig(_Strict):
    retrieved: str
    source: str
    spot_taker: float = Field(ge=0)
    perp_taker: float = Field(ge=0)
    spread_clip_quantiles: tuple[float, float]
    cost_multiples: tuple[float, ...]

    @model_validator(mode="after")
    def _check(self) -> CostConfig:
        lo, hi = self.spread_clip_quantiles
        if not 0.0 <= lo < hi <= 1.0:
            raise ValueError("spread_clip_quantiles must satisfy 0 <= low < high <= 1")
        if 1.0 not in self.cost_multiples or any(m <= 0 for m in self.cost_multiples):
            raise ValueError("cost_multiples must be positive and include 1.0")
        return self


class MarginConfig(_Strict):
    maintenance_rate: dict[str, float]
    default_maintenance_rate: float = Field(gt=0, lt=1)
    source: str


class CarryConfig(_Strict):
    leverages: tuple[float, ...]
    headline_leverage: float
    reserve_share: float = Field(ge=0, lt=1)
    topup_leverage_multiple: float = Field(gt=1)

    @model_validator(mode="after")
    def _headline_listed(self) -> CarryConfig:
        if self.headline_leverage not in self.leverages:
            raise ValueError("carry.headline_leverage must be one of carry.leverages")
        return self


class ForecastConfig(_Strict):
    horizons: tuple[int, ...]
    har_lags: tuple[int, ...]
    momentum_hours: tuple[int, ...]
    min_train_months: int = Field(ge=1)
    winsor: float = Field(ge=0, lt=0.5)
    ewc_coefficient: float = Field(gt=0)


class PortfolioConfig(_Strict):
    rebalance_hours: int = Field(ge=1)
    exit_rank_multiple: float = Field(ge=1)
    top_k: tuple[int, ...]
    headline_top_k: int
    min_quote_volume_usd: tuple[float, ...]
    headline_min_quote_volume_usd: float
    volume_window_days: int = Field(ge=1)
    models: tuple[Literal["persistence", "har", "har_plus"], ...]
    headline_model: Literal["persistence", "har", "har_plus"]
    participation: float = Field(gt=0, lt=1)
    oi_share: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _headline_listed(self) -> PortfolioConfig:
        if self.headline_top_k not in self.top_k:
            raise ValueError("portfolio.headline_top_k must be one of portfolio.top_k")
        if self.headline_min_quote_volume_usd not in self.min_quote_volume_usd:
            raise ValueError("headline volume filter must be listed")
        if self.headline_model not in self.models:
            raise ValueError("portfolio.headline_model must be one of portfolio.models")
        return self


class Episode(_Strict):
    name: str
    start: str
    end: str


class CounterpartyConfig(_Strict):
    venues: dict[str, float]  # annual failure hazard per venue
    hazard_scales: tuple[float, ...]
    loss_given_failure: tuple[float, ...]
    headline_loss_given_failure: float
    es_level: float = Field(gt=0, lt=1)
    max_share: float = Field(gt=0, le=1)


class StablecoinConfig(_Strict):
    shock: float = Field(gt=-1, lt=0)
    days: int = Field(ge=1)


class EvaluationConfig(_Strict):
    ewc_coefficient: float = Field(gt=0)
    hours_per_year: float = Field(gt=0)


class PlotConfig(_Strict):
    dpi: int = Field(ge=72)
    palette: dict[str, str]
    heatmap_symbols: int = Field(ge=5)


class Config(_Strict):
    sources: SourcesConfig
    download: DownloadConfig
    universe: UniverseConfig
    rebuild: RebuildConfig
    costs: CostConfig
    margin: MarginConfig
    carry: CarryConfig
    forecast: ForecastConfig
    portfolio: PortfolioConfig
    episodes: tuple[Episode, ...]
    counterparty: CounterpartyConfig
    stablecoin: StablecoinConfig
    evaluation: EvaluationConfig
    plot: PlotConfig

    @model_validator(mode="after")
    def _venues_known(self) -> Config:
        known = set(self.sources.venue_names)
        for group, keys in (
            ("sources.full_history_venues", set(self.sources.full_history_venues)),
            ("counterparty.venues", set(self.counterparty.venues)),
        ):
            if not keys <= known:
                raise ValueError(f"{group} names venues not in sources.venue_names: {keys - known}")
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


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(v) for v in value)
    if isinstance(value, dict):
        return {k: _tuples(v) for k, v in value.items()}
    return value


def load(*, path: Path, override: Path | None = None) -> Config:
    raw = _read_yaml(path)
    if override is not None:
        raw = _deep_merge(raw, _read_yaml(override))
    return Config.model_validate(_tuples(raw))


def write_resolved(config: Config, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"fingerprint": config.fingerprint(), "config": config.model_dump(mode="json")}
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

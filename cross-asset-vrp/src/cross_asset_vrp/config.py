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
    cboe_index: str
    cfe_futures: str
    cfe_archive: str
    tiingo: str
    tiingo_key_env: str
    fred: str
    vision_base: str
    vision_list: str
    deribit: str
    deribit_history: str
    official_hosts: tuple[str, ...]


class DownloadConfig(_Strict):
    retries: int = Field(ge=0)
    backoff_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    chunk_bytes: int = Field(gt=0)
    workers: int = Field(ge=1, le=32)
    rest_pause_seconds: float = Field(ge=0)


class AssetConfig(_Strict):
    key: str
    label: str
    implied: str
    instrument: str
    options_on: str
    reason: str


class TermStructureConfig(_Strict):
    short: str
    long: str


class RiskFreeConfig(_Strict):
    series: str


class WindowsConfig(_Strict):
    variance_days: int = Field(ge=1)
    days_per_year: int = Field(ge=360, le=366)
    first_date: str
    last_date: str | None


class RealizedConfig(_Strict):
    btc_sample_minutes: int = Field(ge=1)
    kernel: Literal["parzen"]
    estimators: tuple[str, ...]
    yang_zhang_window: int = Field(ge=2)


class HarConfig(_Strict):
    lags: tuple[int, int, int]
    btc_lags: tuple[int, int, int]
    min_train_days: int = Field(ge=100)
    refit: Literal["monthly"]


class PredictabilityConfig(_Strict):
    horizons_months: tuple[int, ...]
    btz_sample_end: str
    ewc_coefficient: float = Field(gt=0)


class FuturesConfig(_Strict):
    first_expiry: str
    new_files_from: str
    archive_until: str
    max_gap_days: int = Field(ge=1)
    rescale_before: str
    rescale_factor: float = Field(gt=1)
    rescale_check: tuple[float, float]
    target_days: int = Field(ge=1)
    cost_points_per_side: float = Field(ge=0)
    sizing_cap: float = Field(gt=1)
    min_signal_history_days: int = Field(ge=20)


class Episode(_Strict):
    name: str
    start: str
    end: str


class ReplicationConfig(_Strict):
    window_hours: int = Field(ge=1, le=168)
    page_size: int = Field(ge=10, le=10000)
    spread_buckets: tuple[float, ...]
    min_spread_trades: int = Field(ge=1)
    min_expiry_days: int = Field(ge=1)
    zero_bid_stop: int = Field(ge=1)


class HestonConfig(_Strict):
    kappa: float = Field(gt=0)
    theta: float = Field(gt=0)
    xi: float = Field(gt=0)
    rho: float = Field(ge=-1, le=1)
    v0: float = Field(gt=0)
    paths: int = Field(ge=1)
    days: int = Field(ge=5)
    steps_per_day: int = Field(ge=10)
    coarse_steps: tuple[int, ...]
    seed: int


class PlotConfig(_Strict):
    dpi: int = Field(ge=72)
    palette: dict[str, str]


ESTIMATORS = ("close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang")


class Config(_Strict):
    sources: SourcesConfig
    download: DownloadConfig
    assets: tuple[AssetConfig, ...]
    term_structure: TermStructureConfig
    risk_free: RiskFreeConfig
    windows: WindowsConfig
    realized: RealizedConfig
    har: HarConfig
    predictability: PredictabilityConfig
    futures: FuturesConfig
    episodes: tuple[Episode, ...]
    replication: ReplicationConfig
    heston: HestonConfig
    plot: PlotConfig

    @model_validator(mode="after")
    def _consistent(self) -> Config:
        keys = [a.key for a in self.assets]
        if len(set(keys)) != len(keys):
            raise ValueError("asset keys must be unique")
        unknown = set(self.realized.estimators) - set(ESTIMATORS)
        if unknown:
            raise ValueError(f"unknown estimators {sorted(unknown)}; known: {ESTIMATORS}")
        if "close_to_close" not in self.realized.estimators:
            raise ValueError("close_to_close is the unbiased target and must stay in the list")
        if not any(a.implied == "VIX" for a in self.assets):
            raise ValueError("the VIX futures test needs the asset whose implied index is VIX")
        if self.heston.steps_per_day % max(self.heston.coarse_steps, default=1):
            raise ValueError("coarse_steps must divide steps_per_day")
        return self

    def fingerprint(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def asset(self, key: str) -> AssetConfig:
        for a in self.assets:
            if a.key == key:
                return a
        raise KeyError(key)

    @property
    def crypto(self) -> AssetConfig:
        """The asset whose implied index is Deribit's DVOL and whose realized side is Binance."""
        for a in self.assets:
            if a.implied == "DVOL":
                return a
        raise KeyError("no DVOL asset in the configuration")

    @property
    def cboe_assets(self) -> tuple[AssetConfig, ...]:
        return tuple(a for a in self.assets if a.implied != "DVOL")


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

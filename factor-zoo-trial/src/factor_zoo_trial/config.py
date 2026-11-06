"""Research configuration: every choice lives in configs/*.yaml and is validated here.

The YAML carries a one-line reason next to each value. Models are strict and frozen,
so a typo or a wrong type fails on load rather than silently changing a result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SourcesConfig(_Strict):
    osap_resolve: str
    osap_api: str
    osap_revision: str
    jkp_base: str
    jkp_reference_base: str
    jkp_reference_commit: str
    french_base: str
    official_hosts: tuple[str, ...]


class DownloadConfig(_Strict):
    retries: int = Field(ge=0)
    backoff_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    chunk_bytes: int = Field(gt=0)


class OsapConfig(_Strict):
    signal_doc: str
    variants: dict[str, str]
    headline_variant: str
    liquidity_variant: str
    clear_label: str
    long_short_port: str

    @model_validator(mode="after")
    def _variants_known(self) -> OsapConfig:
        for name in (self.headline_variant, self.liquidity_variant):
            if name not in self.variants:
                raise ValueError(f"variant {name!r} is not listed under osap.variants")
        return self


class JkpConfig(_Strict):
    selector: str
    market_selector: str
    frequency: Literal["monthly"]
    weighting: Literal["vw_cap", "vw", "ew"]
    replication_regions: tuple[str, ...]
    min_stocks_per_leg: int = Field(ge=1)
    cluster_labels: str
    factor_details: str
    country_classification: str


class FrenchConfig(_Strict):
    five_factor_zip: str
    momentum_zip: str


class PeriodsConfig(_Strict):
    publication_month: int = Field(ge=1, le=12)
    publication_shifts: tuple[int, ...]
    headline_shift: int

    @model_validator(mode="after")
    def _shift_listed(self) -> PeriodsConfig:
        if self.headline_shift not in self.publication_shifts:
            raise ValueError("periods.headline_shift must be one of periods.publication_shifts")
        if any(s < 0 for s in self.publication_shifts):
            raise ValueError("publication shifts move dates back and must be >= 0")
        return self


class DecayConfig(_Strict):
    min_months_in_sample: int = Field(ge=12)
    return_bases: tuple[Literal["raw", "capm", "ff6"], ...]
    headline_basis: Literal["raw", "capm", "ff6"]


class MultipleTestingConfig(_Strict):
    level: float = Field(gt=0, lt=1)
    ewc_coefficient: float = Field(gt=0)
    hlz_threshold: float = Field(gt=0)
    romano_wolf_draws: int = Field(ge=50)
    romano_wolf_block_months: int = Field(ge=1)
    storey_lambda: float = Field(gt=0, lt=1)
    storey_lambdas: tuple[float, ...]
    storey_bootstrap_draws: int = Field(ge=50)
    seed: int

    @field_validator("storey_lambdas")
    @classmethod
    def _lambdas_in_unit(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not 0 < v < 1 for v in value):
            raise ValueError("storey_lambdas must lie strictly between 0 and 1")
        return tuple(sorted(value))


class EmpiricalBayesConfig(_Strict):
    min_months: int = Field(ge=24)
    annual_vol_target: float = Field(gt=0)
    fix_prior_mean_at_zero: bool
    covariance_shrinkage: Literal["schafer_strimmer", "none"]
    eigenvalue_floor: Literal["marchenko_pastur", "numerical"]
    minimum_eigenvalue: float = Field(gt=0, lt=1)
    min_pair_months: int = Field(ge=12)
    posterior_draws: int = Field(ge=100)
    interval: float = Field(gt=0, lt=1)
    require_original_significance: bool
    seed: int


class SpaceConfig(_Strict):
    source_region: str
    min_months: int = Field(ge=24)
    min_factors: int = Field(ge=2)


class SimulationConfig(_Strict):
    n_factors: int = Field(ge=20)
    n_months: int = Field(ge=60)
    latent_factors: int = Field(ge=1)
    null_shares: tuple[float, ...]
    replications: int = Field(ge=5)
    eb_clusters: int = Field(ge=2)
    seed: int


class McLeanPontiffBenchmark(_Strict):
    last_month: str
    max_publication_year: int
    in_sample_mean: float
    post_sample: float
    post_publication: float
    citation: str


class JkpBenchmark(_Strict):
    last_month: str
    replication_rate: float
    by_rate: float
    n_factors: int
    citation: str


class BenchmarksConfig(_Strict):
    mclean_pontiff: McLeanPontiffBenchmark
    jkp: JkpBenchmark


class PlotConfig(_Strict):
    dpi: int = Field(ge=72)
    palette: dict[str, str]


class Config(_Strict):
    sources: SourcesConfig
    download: DownloadConfig
    osap: OsapConfig
    jkp: JkpConfig
    french: FrenchConfig
    periods: PeriodsConfig
    decay: DecayConfig
    multiple_testing: MultipleTestingConfig
    empirical_bayes: EmpiricalBayesConfig
    space: SpaceConfig
    simulation: SimulationConfig
    benchmarks: BenchmarksConfig
    plot: PlotConfig

    def fingerprint(self) -> str:
        """Short SHA-256 of the resolved configuration, written next to every output."""
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
    """YAML gives lists; the frozen models hold tuples."""
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

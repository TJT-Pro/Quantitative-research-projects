"""Typed research configuration.

The YAML file is the single place research choices live. It is validated on load
so a typo fails immediately instead of silently changing a result, and the
resolved configuration is written next to every run's outputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    library_page: HttpUrl
    base_url: HttpUrl
    files: dict[str, str]
    portfolio_section: Literal["value_weighted", "equal_weighted"]
    missing_sentinels: list[float]
    percent_scale: float = Field(gt=0)
    max_download_bytes: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    download_attempts: int = Field(ge=1)
    retry_backoff_seconds: float = Field(ge=0)

    @field_validator("base_url")
    @classmethod
    def _https_only(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("data must be fetched over https")
        return value

    @field_validator("files")
    @classmethod
    def _safe_names(cls, value: dict[str, str]) -> dict[str, str]:
        for key, name in value.items():
            if "/" in name or "\\" in name or name.startswith(".") or not name.endswith(".zip"):
                raise ValueError(f"file name for {key!r} must be a bare .zip name, got {name!r}")
        return value


class UniverseConfig(_Strict):
    key: str
    file: str
    label: str


class FactorSet(_Strict):
    file: str | None = None
    files: list[str] | None = None
    columns: list[str]

    @model_validator(mode="after")
    def _one_source(self) -> FactorSet:
        if (self.file is None) == (self.files is None):
            raise ValueError("a factor set needs exactly one of `file` or `files`")
        return self

    @property
    def source_files(self) -> list[str]:
        return [self.file] if self.file is not None else list(self.files or [])


class FactorsConfig(_Strict):
    primary: FactorSet
    extended: FactorSet


class BacktestConfig(_Strict):
    rebalance: Literal["month_end"]
    windows: list[int]
    primary_window: int
    common_evaluation_sample: bool
    singular_tolerance: float = Field(gt=0, lt=1)
    annualization_days: int = Field(gt=0)

    @model_validator(mode="after")
    def _primary_in_windows(self) -> BacktestConfig:
        if any(w < 2 for w in self.windows):
            raise ValueError("every window needs at least two observations")
        if self.primary_window not in self.windows:
            raise ValueError("primary_window must be one of windows")
        if len(set(self.windows)) != len(self.windows):
            raise ValueError("windows must be unique")
        return self


class SolverConfig(_Strict):
    max_iterations: int = Field(gt=0)
    tolerance: float = Field(gt=0)


class PortfolioConfig(_Strict):
    long_only_solver: SolverConfig


ModelKind = Literal[
    "sample",
    "ewma",
    "linear_shrinkage",
    "nonlinear_shrinkage",
    "observable_factor",
    "statistical_factor",
    "dcc",
]


class ModelSpec(_Strict):
    name: str
    kind: ModelKind
    windows: Literal["all"] | list[int]
    label: str
    description: str
    params: dict[str, Any] = Field(default_factory=dict)

    def runs_on(self, window: int) -> bool:
        return self.windows == "all" or window in self.windows


class McsConfig(_Strict):
    alpha: float = Field(gt=0, lt=1)
    bootstrap_replications: int = Field(ge=100)
    mean_block_length: float = Field(ge=1)
    period: Literal["decade", "full"]
    min_months: int = Field(ge=12)


LossName = Literal["gmv_variance", "gmv_long_only_variance", "qlike", "frobenius"]


class EvaluationConfig(_Strict):
    mcs: McsConfig
    losses: list[LossName]
    seed: int


class ParametricTruth(_Strict):
    n_factors: int = Field(ge=1)
    factor_vols: list[float]
    loading_mean: list[float]
    loading_sd: list[float]
    idio_vol_range: tuple[float, float]

    @model_validator(mode="after")
    def _shapes(self) -> ParametricTruth:
        k = self.n_factors
        if not (len(self.factor_vols) == len(self.loading_mean) == len(self.loading_sd) == k):
            raise ValueError("factor_vols, loading_mean and loading_sd need n_factors entries")
        lo, hi = self.idio_vol_range
        if not 0 < lo < hi:
            raise ValueError("idio_vol_range must satisfy 0 < low < high")
        return self


class TruthConfig(_Strict):
    source: Literal["parametric", "data"]
    parametric: ParametricTruth


class SimulationConfig(_Strict):
    dimensions: list[int]
    sample_sizes: list[int]
    replications: int = Field(ge=1)
    degrees_of_freedom: float = Field(gt=2)
    truth: TruthConfig
    models: list[str]
    seed: int


class ReportConfig(_Strict):
    figure_dpi: int = Field(ge=72)
    headline_universe: str
    spectrum_universe: str
    headline_loss: LossName


class ResearchConfig(_Strict):
    data: DataConfig
    universes: list[UniverseConfig]
    factors: FactorsConfig
    backtest: BacktestConfig
    portfolio: PortfolioConfig
    models: list[ModelSpec]
    evaluation: EvaluationConfig
    simulation: SimulationConfig
    report: ReportConfig

    @model_validator(mode="after")
    def _cross_references(self) -> ResearchConfig:
        self._check_files()
        self._check_models()
        self._check_report()
        return self

    def _check_files(self) -> None:
        files = set(self.data.files)
        for universe in self.universes:
            if universe.file not in files:
                raise ValueError(
                    f"universe {universe.key!r} references unknown file {universe.file!r}"
                )
        for factor_set in (self.factors.primary, self.factors.extended):
            for name in factor_set.source_files:
                if name not in files:
                    raise ValueError(f"factor set references unknown file {name!r}")

    def _check_models(self) -> None:
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        for spec in self.models:
            if spec.windows != "all":
                unknown = set(spec.windows) - set(self.backtest.windows)
                if unknown:
                    raise ValueError(f"model {spec.name!r} uses windows not in backtest: {unknown}")
            if spec.kind == "observable_factor" and spec.params.get("factor_set") not in {
                "primary",
                "extended",
            }:
                raise ValueError(f"model {spec.name!r} needs params.factor_set primary|extended")
        unknown_sim = set(self.simulation.models) - set(names)
        if unknown_sim:
            raise ValueError(f"simulation lists unknown models: {unknown_sim}")
        if any(self.model(n).kind == "observable_factor" for n in self.simulation.models):
            raise ValueError("simulation draws no factor returns; drop factor models from it")

    def _check_report(self) -> None:
        universe_keys = {u.key for u in self.universes}
        for field_name in ("headline_universe", "spectrum_universe"):
            value = getattr(self.report, field_name)
            if value not in universe_keys:
                raise ValueError(f"report.{field_name} {value!r} is not a configured universe")
        if self.report.headline_loss not in self.evaluation.losses:
            raise ValueError("report.headline_loss must be one of evaluation.losses")

    def model(self, name: str) -> ModelSpec:
        for spec in self.models:
            if spec.name == name:
                return spec
        raise KeyError(name)

    def fingerprint(self) -> str:
        """Hash of every setting that can change a number.

        Presentation fields (labels, descriptions, the report section) are left
        out, so rewording the README never forces a rerun of the backtest.
        """
        payload = self.model_dump(mode="json")
        payload.pop("report")
        for model in payload["models"]:
            model.pop("label")
            model.pop("description")
        for universe in payload["universes"]:
            universe.pop("label")
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded


def load_config(path: Path, overrides: list[Path] | None = None) -> ResearchConfig:
    """Load the base config and apply override files in order (deep merge)."""
    raw = _read_yaml(path)
    for override in overrides or []:
        raw = _deep_merge(raw, _read_yaml(override))
    return ResearchConfig.model_validate(raw)

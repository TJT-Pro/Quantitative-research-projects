"""Typed research configuration, validated on load."""

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


class Source(_Strict):
    url: HttpUrl
    file: str

    @field_validator("url")
    @classmethod
    def _https(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("sources must be fetched over https")
        return value

    @field_validator("file")
    @classmethod
    def _bare_name(cls, value: str) -> str:
        if "/" in value or "\\" in value or value.startswith("."):
            raise ValueError(f"file must be a bare file name, got {value!r}")
        return value


class Sources(_Strict):
    gsw: Source
    acm: Source
    fedfunds: Source
    target: Source
    target_upper: Source

    def items(self) -> list[tuple[str, Source]]:
        return [(name, getattr(self, name)) for name in type(self).model_fields]


class DataConfig(_Strict):
    sources: Sources
    max_download_bytes: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    download_attempts: int = Field(ge=1)
    retry_backoff_seconds: float = Field(ge=0)


class MaturityRange(_Strict):
    first: int = Field(ge=1)
    last: int

    @model_validator(mode="after")
    def _ordered(self) -> MaturityRange:
        if self.last <= self.first:
            raise ValueError("last must exceed first")
        return self

    def months(self) -> list[int]:
        return list(range(self.first, self.last + 1))


class ShortRate(_Strict):
    method: Literal["fedfunds_splice", "gsw"]
    fit_start: str
    replace_before: str


class ModelConfig(_Strict):
    n_factors: int = Field(ge=1, le=10)
    pc_maturities: MaturityRange
    return_maturities: list[int]
    max_maturity: int
    convention: Literal["nyfed", "paper"]
    max_spectral_radius: float = Field(gt=0, lt=1)
    short_rate: ShortRate

    @model_validator(mode="after")
    def _consistent(self) -> ModelConfig:
        if sorted(set(self.return_maturities)) != self.return_maturities:
            raise ValueError("return_maturities must be strictly increasing")
        if self.return_maturities[0] < 2:
            raise ValueError("excess returns need maturities of at least two months")
        if max(self.return_maturities) > self.max_maturity:
            raise ValueError("return maturities cannot exceed max_maturity")
        if self.pc_maturities.last > self.max_maturity:
            raise ValueError("factor maturities cannot exceed max_maturity")
        if len(self.return_maturities) <= self.n_factors:
            raise ValueError("need more return maturities than factors to identify prices of risk")
        return self


class ValidationConfig(_Strict):
    published_years: list[int]
    headline_years: int

    @model_validator(mode="after")
    def _headline_published(self) -> ValidationConfig:
        if self.headline_years not in self.published_years:
            raise ValueError("headline_years must be one of published_years")
        return self


class UncertaintyConfig(_Strict):
    replications: int = Field(ge=50)
    block_months: int = Field(ge=1)
    band: float = Field(gt=0, lt=1)
    seed: int


class RealtimeConfig(_Strict):
    min_months: int = Field(ge=24)


class ForecastConfig(_Strict):
    horizon_months: int = Field(ge=1)
    maturities: list[int]
    cp_forward_years: list[int]
    cp_target_years: list[int]
    min_train_months: int = Field(ge=24)
    newey_west_lags: int = Field(ge=0)


class EventsConfig(_Strict):
    start: str
    maturity_years: int


class ReportConfig(_Strict):
    figure_dpi: int = Field(ge=72)


class ResearchConfig(_Strict):
    data: DataConfig
    model: ModelConfig
    validation: ValidationConfig
    uncertainty: UncertaintyConfig
    realtime: RealtimeConfig
    forecasting: ForecastConfig
    events: EventsConfig
    report: ReportConfig

    @model_validator(mode="after")
    def _cross(self) -> ResearchConfig:
        max_years = self.model.max_maturity // 12
        for years in [*self.validation.published_years, self.events.maturity_years]:
            if years * 12 > self.model.max_maturity:
                raise ValueError(f"{years}-year maturity exceeds model.max_maturity")
        for months in self.forecasting.maturities:
            if months > self.model.max_maturity or months <= self.forecasting.horizon_months:
                raise ValueError("forecast maturities must exceed the horizon and fit the model")
        if max(self.forecasting.cp_forward_years) > max_years:
            raise ValueError("CP forwards exceed the curve's maximum maturity")
        return self

    def fingerprint(self) -> str:
        """Hash of every setting that can change a number (the report section is excluded)."""
        payload = self.model_dump(mode="json")
        payload.pop("report")
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


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
    raw = _read_yaml(path)
    for override in overrides or []:
        raw = _deep_merge(raw, _read_yaml(override))
    return ResearchConfig.model_validate(raw)

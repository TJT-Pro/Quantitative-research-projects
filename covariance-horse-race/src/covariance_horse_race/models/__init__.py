"""Covariance forecasters and the factory that builds them from config."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from covariance_horse_race.config import ModelKind, ModelSpec
from covariance_horse_race.models.base import CovarianceModel
from covariance_horse_race.models.dcc import Dcc
from covariance_horse_race.models.factor import ObservableFactor, StatisticalFactor
from covariance_horse_race.models.shrinkage import LinearShrinkage, NonlinearShrinkage
from covariance_horse_race.models.unconditional import Ewma, SampleCovariance


def _dcc(params: dict[str, Any]) -> CovarianceModel:
    target = params["target"]
    if target not in ("sample", "nonlinear_shrinkage"):
        raise ValueError(f"unknown DCC target {target!r}")
    return Dcc(target=target, reestimate_every=int(params["reestimate_every_months"]))


_BUILDERS: dict[ModelKind, Callable[[dict[str, Any]], CovarianceModel]] = {
    "sample": lambda _: SampleCovariance(),
    "ewma": lambda p: Ewma(decay=float(p["decay"])),
    "linear_shrinkage": lambda _: LinearShrinkage(),
    "nonlinear_shrinkage": lambda _: NonlinearShrinkage(),
    "observable_factor": lambda _: ObservableFactor(),
    "statistical_factor": lambda p: StatisticalFactor(kmax=int(p["kmax"])),
    "dcc": _dcc,
}


def build_model(spec: ModelSpec) -> CovarianceModel:
    """A fresh, unfitted model for one spec. Missing parameters raise KeyError."""
    return _BUILDERS[spec.kind](spec.params)


__all__ = ["CovarianceModel", "build_model"]

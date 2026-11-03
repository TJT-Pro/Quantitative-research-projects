"""Known-covariance simulation study.

Returns are drawn from a multivariate Student-t with a fixed covariance Sigma, so
for any estimate H the true risk of its GMV portfolio, w(H)' Sigma w(H), is
known exactly and can be compared with the oracle w(Sigma)' Sigma w(Sigma).
Reported per estimator, dimension N and sample size T:

* PRIAL in the Frobenius norm relative to the sample covariance (Ledoit-Wolf),
  100 * (E||S - Sigma||^2 - E||H - Sigma||^2) / E||S - Sigma||^2;
* the excess true GMV variance over the oracle, in percent.

The truth is either a parametric factor structure from config or the full-sample
covariance of the matching real universe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from covariance_horse_race.config import ParametricTruth, ResearchConfig
from covariance_horse_race.evaluation.portfolio import SingularForecastError, gmv_weights
from covariance_horse_race.models import build_model
from covariance_horse_race.models.base import Matrix


def parametric_truth(
    n: int, spec: ParametricTruth, annualization: int, rng: np.random.Generator
) -> Matrix:
    """Daily covariance with a k-factor structure and heterogeneous idiosyncratic risk."""
    vols = np.asarray(spec.factor_vols) / np.sqrt(annualization)
    loadings = rng.normal(
        np.asarray(spec.loading_mean), np.asarray(spec.loading_sd), size=(n, spec.n_factors)
    )
    lo, hi = spec.idio_vol_range
    idio = rng.uniform(lo, hi, size=n) / np.sqrt(annualization)
    sigma = (loadings * vols**2) @ loadings.T + np.diag(idio**2)
    out: Matrix = 0.5 * (sigma + sigma.T)
    return out


def multivariate_t(sigma: Matrix, n_obs: int, dof: float, rng: np.random.Generator) -> np.ndarray:
    """Draws with covariance exactly `sigma` (the t is rescaled by (dof-2)/dof)."""
    chol = np.linalg.cholesky(sigma)
    z = rng.standard_normal((n_obs, sigma.shape[0])) @ chol.T
    w = rng.chisquare(dof, size=(n_obs, 1)) / dof
    out: np.ndarray = z / np.sqrt(w) * np.sqrt((dof - 2.0) / dof)
    return out


@dataclass(frozen=True)
class SimulationCell:
    n: int
    t: int
    model: str
    prial: float
    excess_gmv_variance_pct: float
    excess_gmv_variance_se: float
    singular_share: float


def run_simulation(
    config: ResearchConfig,
    truths: dict[int, Matrix] | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    sim = config.simulation
    rng = np.random.default_rng(sim.seed)
    tolerance = config.backtest.singular_tolerance
    specs = [config.model(name) for name in sim.models]
    cells: list[SimulationCell] = []
    for n in sim.dimensions:
        if truths is not None and n in truths:
            sigma = truths[n]
        else:
            sigma = parametric_truth(
                n, sim.truth.parametric, config.backtest.annualization_days, rng
            )
        oracle_w = gmv_weights(sigma, tolerance)
        oracle_var = float(oracle_w @ sigma @ oracle_w)
        for t in sim.sample_sizes:
            frob: dict[str, list[float]] = {s.name: [] for s in specs}
            excess: dict[str, list[float]] = {s.name: [] for s in specs}
            singular: dict[str, int] = {s.name: 0 for s in specs}
            sample_frob: list[float] = []
            for _ in range(sim.replications):
                x = multivariate_t(sigma, t, sim.degrees_of_freedom, rng)
                xd = x - x.mean(axis=0)
                sample = xd.T @ xd / (t - 1)
                sample_frob.append(float(np.sum((sample - sigma) ** 2)))
                for spec in specs:
                    model = build_model(spec)
                    try:
                        model.fit(x)
                        h = model.forecast(1)
                    except ValueError:
                        singular[spec.name] += 1
                        continue
                    frob[spec.name].append(float(np.sum((h - sigma) ** 2)))
                    try:
                        w = gmv_weights(h, tolerance)
                    except SingularForecastError:
                        singular[spec.name] += 1
                        continue
                    excess[spec.name].append(100.0 * (float(w @ sigma @ w) / oracle_var - 1.0))
            base = float(np.mean(sample_frob))
            for spec in specs:
                e = np.asarray(excess[spec.name])
                f = np.asarray(frob[spec.name])
                cells.append(
                    SimulationCell(
                        n=n,
                        t=t,
                        model=spec.name,
                        prial=100.0 * (base - float(f.mean())) / base if f.size else float("nan"),
                        excess_gmv_variance_pct=float(e.mean()) if e.size else float("nan"),
                        excess_gmv_variance_se=(
                            float(e.std(ddof=1) / np.sqrt(e.size)) if e.size > 1 else float("nan")
                        ),
                        singular_share=singular[spec.name] / sim.replications,
                    )
                )
            if progress is not None:
                progress(f"simulation N={n} T={t}")
    return pd.DataFrame([c.__dict__ for c in cells])

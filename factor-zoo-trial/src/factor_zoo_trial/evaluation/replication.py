"""Empirical Bayes replication of the JKP factors, one location at a time.

Each factor's return is made market neutral against the location's own market
(CAPM beta from the full sample), then scaled so the neutral return has the configured
annual volatility. The alpha estimate is the mean of the scaled neutral return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from factor_zoo_trial.config import EmpiricalBayesConfig
from factor_zoo_trial.models import covariance, empirical_bayes
from factor_zoo_trial.models import multiple_testing as mt
from factor_zoo_trial.models.empirical_bayes import Hyperparameters, Posterior, ReplicationRate
from factor_zoo_trial.models.inference import one_sided_p

Z95 = 1.959963984540054


@dataclass(frozen=True)
class LocationAlphas:
    location: str
    factors: list[str]
    clusters: list[str]
    significant: np.ndarray  # original-paper significance flag
    scaled: np.ndarray  # months x factors, scaled market-neutral returns
    alpha_hat: np.ndarray
    sigma: np.ndarray
    shrinkage: float
    eigenvalue_floor: float
    clipped: int
    months: int


def location_alphas(
    *,
    location: str,
    jkp: pd.DataFrame,
    market: pd.DataFrame,
    reference: pd.DataFrame,
    settings: EmpiricalBayesConfig,
    shrinkage: str,
    last_month: pd.Timestamp | None = None,
) -> LocationAlphas | None:
    rows = jkp.loc[jkp["location"] == location]
    mkt = market.loc[market["location"] == location].set_index("month")["mkt"]
    if last_month is not None:
        rows = rows.loc[rows["month"] <= last_month]
        mkt = mkt.loc[mkt.index <= last_month]
    wide = rows.pivot_table(index="month", columns="factor", values="ret", aggfunc="first")
    wide = wide.join(mkt.rename("__mkt__"), how="inner").sort_index()
    if wide.empty:
        return None
    m = wide.pop("__mkt__")
    keep = [c for c in wide.columns if wide[c].notna().sum() >= settings.min_months]
    ref = reference.set_index("factor")
    keep = [c for c in keep if c in ref.index]
    if len(keep) < 2:
        return None
    scaled = np.full((len(wide), len(keep)), np.nan)
    target = settings.annual_vol_target / np.sqrt(12.0)
    for j, factor in enumerate(keep):
        r = wide[factor]
        ok = r.notna() & m.notna()
        x = m[ok].to_numpy()
        y = r[ok].to_numpy()
        beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
        neutral = y - beta * x
        scaled[np.nonzero(ok.to_numpy())[0], j] = neutral * (target / np.std(neutral, ddof=1))
    used = np.isfinite(scaled).any(axis=1)
    scaled = scaled[used]
    cov = covariance.alpha_covariance(
        scaled,
        min_pair=settings.min_pair_months,
        shrinkage="none" if shrinkage == "none" else "schafer_strimmer",
        eigenvalue_floor=settings.eigenvalue_floor,
        minimum_floor=settings.minimum_eigenvalue,
    )
    return LocationAlphas(
        location=location,
        factors=keep,
        clusters=[str(ref.loc[f, "cluster"]) for f in keep],
        significant=ref.loc[keep, "significance"].to_numpy(dtype=bool),
        scaled=scaled,
        alpha_hat=np.nanmean(scaled, axis=0),
        sigma=cov.sigma,
        shrinkage=cov.intensity,
        eigenvalue_floor=cov.eigenvalue_floor,
        clipped=cov.clipped,
        months=int(used.sum()),
    )


@dataclass(frozen=True)
class RegionFit:
    alphas: LocationAlphas
    hyper: Hyperparameters
    post: Posterior
    omega: np.ndarray
    rate: ReplicationRate
    rate_all: ReplicationRate
    ols_rate: float
    by_rate: float

    def table(self) -> pd.DataFrame:
        a = self.alphas
        se = np.sqrt(np.diag(a.sigma))
        sd = self.post.sd
        return pd.DataFrame(
            {
                "location": a.location,
                "factor": a.factors,
                "cluster": a.clusters,
                "original_significance": a.significant,
                "alpha_ols": a.alpha_hat,
                "se_ols": se,
                "alpha_post": self.post.mean,
                "sd_post": sd,
                "low_post": self.post.mean - Z95 * sd,
                "high_post": self.post.mean + Z95 * sd,
            }
        )

    def summary(self) -> dict[str, Any]:
        h = self.hyper
        return {
            "location": self.alphas.location,
            "n_factors": len(self.alphas.factors),
            "months": self.alphas.months,
            "shrinkage_intensity": self.alphas.shrinkage,
            "eigenvalue_floor": self.alphas.eigenvalue_floor,
            "eigenvalues_clipped": self.alphas.clipped,
            "tau_c": h.tau_c,
            "tau_w": h.tau_w,
            "se_tau_c": h.se_tau_c,
            "se_tau_w": h.se_tau_w,
            "mu": h.mu,
            "converged": h.converged,
            "replication_rate": self.rate.rate,
            "replication_low": self.rate.lower,
            "replication_high": self.rate.upper,
            "replicated": self.rate.replicated,
            "n_counted": self.rate.n_factors,
            "replication_rate_all_factors": self.rate_all.rate,
            "ols_rate": self.ols_rate,
            "by_rate": self.by_rate,
        }


def fit_region(alphas: LocationAlphas, *, settings: EmpiricalBayesConfig) -> RegionFit:
    rng = np.random.default_rng(settings.seed)
    m, _ = empirical_bayes.membership(alphas.clusters)
    hyper = empirical_bayes.fit_hyperparameters(
        alpha_hat=alphas.alpha_hat,
        sigma=alphas.sigma,
        m=m,
        fix_mean=settings.fix_prior_mean_at_zero,
        rng=rng,
    )
    omega = empirical_bayes.prior_covariance(m, tau_c=hyper.tau_c, tau_w=hyper.tau_w)
    post = empirical_bayes.posterior(
        alpha_hat=alphas.alpha_hat, sigma=alphas.sigma, omega=omega, mu=hyper.mu
    )
    include = (
        alphas.significant
        if settings.require_original_significance
        else np.ones(len(alphas.factors), dtype=bool)
    )
    z = float(stats.norm.ppf(0.5 + settings.interval / 2.0))
    rate = empirical_bayes.replication_rate(
        post,
        include=include,
        z=z,
        draws=settings.posterior_draws,
        rng=rng,
        interval=settings.interval,
    )
    rate_all = empirical_bayes.replication_rate(
        post,
        include=np.ones(len(alphas.factors), dtype=bool),
        z=z,
        draws=settings.posterior_draws,
        rng=rng,
        interval=settings.interval,
    )
    se = np.sqrt(np.diag(alphas.sigma))
    t = alphas.alpha_hat / se
    n = np.isfinite(alphas.scaled).sum(axis=0).astype(np.int64)
    p = one_sided_p(t, n - 1)
    # Two-sided 95% intervals above zero are one-sided tests at 2.5%.
    level = (1.0 - settings.interval) / 2.0
    chosen = np.nonzero(include)[0]
    ols_rate = float(np.mean(p[chosen] <= level)) if chosen.size else float("nan")
    by_rate = (
        float(np.mean(mt.benjamini_yekutieli(p[chosen], level=level)))
        if chosen.size
        else float("nan")
    )
    return RegionFit(
        alphas=alphas,
        hyper=hyper,
        post=post,
        omega=omega,
        rate=rate,
        rate_all=rate_all,
        ols_rate=ols_rate,
        by_rate=by_rate,
    )

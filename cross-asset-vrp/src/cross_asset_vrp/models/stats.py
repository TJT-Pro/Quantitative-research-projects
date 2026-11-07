"""Sharpe ratios with autocorrelation-robust errors, and drawdowns.

Long-run variances use the equal-weighted cosine (EWC) estimator of Lazarus, Lewis, Stock
and Watson (2018) with nu = coefficient * T^(2/3) cosines and a t(nu) reference.
The Sharpe standard error is the delta method on the influence function
psi_t = z_t - SR/2 (z_t^2 - 1), z_t = (r_t - mu)/sigma, whose long-run variance carries
the autocorrelation of both the mean and the variance of returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def ewc_terms(n: int, *, coefficient: float) -> int:
    return int(min(max(1, round(coefficient * n ** (2.0 / 3.0))), n - 1))


def ewc_long_run_variance(x: FloatArray, *, coefficient: float) -> tuple[float, int]:
    """Long-run variance of a demeaned series and the degrees of freedom of its t reference."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 3:
        raise ValueError("need at least three observations")
    nu = ewc_terms(n, coefficient=coefficient)
    e = x - x.mean()
    j = np.arange(1, nu + 1)[:, None]
    t = np.arange(1, n + 1)[None, :]
    basis = np.sqrt(2.0 / n) * np.cos(np.pi * j * (t - 0.5) / n)
    return float(np.mean((basis @ e) ** 2)), nu


@dataclass(frozen=True)
class SharpeEstimate:
    sharpe: float  # annualized
    se: float  # annualized
    t: float
    p_value: float  # two-sided, t(nu)
    periods: int
    mean: float  # annualized return
    vol: float  # annualized volatility
    skew: float
    kurtosis: float  # not excess


def sharpe(returns: FloatArray, *, periods_per_year: float, coefficient: float) -> SharpeEstimate:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    sr = mu / sigma if sigma > 0 else 0.0
    z = (r - mu) / sigma if sigma > 0 else np.zeros_like(r)
    psi = z - 0.5 * sr * (z**2 - 1.0)
    lrv, nu = ewc_long_run_variance(psi, coefficient=coefficient)
    se = float(np.sqrt(lrv / r.size))
    scale = float(np.sqrt(periods_per_year))
    t_stat = sr / se if se > 0 else 0.0
    return SharpeEstimate(
        sharpe=sr * scale,
        se=se * scale,
        t=t_stat,
        p_value=float(2 * stats.t.sf(abs(t_stat), df=nu)),
        periods=int(r.size),
        mean=mu * periods_per_year,
        vol=sigma * scale,
        skew=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r, fisher=False)),
    )


@dataclass(frozen=True)
class Drawdown:
    depth: float  # worst peak-to-trough loss, as a fraction
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovered: pd.Timestamp | None
    duration_days: float  # peak to recovery, or to the end if not recovered


def worst_drawdown(equity: pd.Series) -> Drawdown:
    values = equity.to_numpy(dtype=np.float64)
    index = pd.DatetimeIndex(equity.index)
    dd = values / np.maximum.accumulate(values) - 1.0
    trough_at = int(np.argmin(dd))
    peak_at = int(np.argmax(values[: trough_at + 1]))
    back = np.nonzero(values[trough_at:] >= values[peak_at])[0]
    recovered = index[trough_at + int(back[0])] if back.size else None
    end = recovered if recovered is not None else index[-1]
    return Drawdown(
        depth=float(dd.min()),
        peak=index[peak_at],
        trough=index[trough_at],
        recovered=recovered,
        duration_days=float((end - index[peak_at]) / pd.Timedelta(days=1)),
    )

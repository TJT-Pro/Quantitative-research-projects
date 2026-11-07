"""Does the variance premium forecast returns? Bollerslev, Tauchen and Zhou (2009) regressions.

At each month end t, the annualized log excess return over the next h months is regressed on
the premium known at t:
    (12 / h) sum_{s=t+1}^{t+h} (r_s - rf_s) = a + b VRP_t + e
Overlapping h-month returns are serially correlated by construction, so the slope's standard
error uses the equal-weighted cosine (EWC) long-run variance of the score x_t e_t with
nu = coefficient * T^(2/3) terms and a t reference with nu degrees of freedom (Lazarus,
Lewis, Stock and Watson 2018).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from cross_asset_vrp.models.stats import ewc_long_run_variance


@dataclass(frozen=True)
class Regression:
    slope: float
    se: float
    t: float
    p_value: float
    ci_low: float
    ci_high: float
    r2: float
    observations: int
    dof: int
    first: str
    last: str


def ols_ewc(
    x: np.ndarray, y: np.ndarray, *, coefficient: float, index: pd.DatetimeIndex
) -> Regression | None:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    dates = index[ok]
    n = x.size
    if n < 24:
        return None
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        return None
    slope = float(xc @ (y - y.mean())) / sxx
    resid = y - y.mean() - slope * xc
    lrv, nu = ewc_long_run_variance(xc * resid, coefficient=coefficient)
    se = float(np.sqrt(n * lrv) / sxx)
    t = slope / se if se > 0 else float("nan")
    crit = float(sps.t.ppf(0.975, nu))
    total = float(((y - y.mean()) ** 2).sum())
    return Regression(
        slope=slope,
        se=se,
        t=t,
        p_value=float(2 * sps.t.sf(abs(t), nu)) if np.isfinite(t) else float("nan"),
        ci_low=slope - crit * se,
        ci_high=slope + crit * se,
        r2=1.0 - float(resid @ resid) / total if total > 0 else float("nan"),
        observations=n,
        dof=nu,
        first=str(dates[0].date()),
        last=str(dates[-1].date()),
    )


def forward_excess(monthly: pd.DataFrame, *, months: int) -> pd.Series:
    """Annualized log excess return over the next `months`, from columns ret (log) and rf."""
    excess = monthly["ret"] - monthly["rf"]
    ahead = excess[::-1].rolling(months, min_periods=months).sum()[::-1].shift(-1)
    return ahead * (12.0 / months)

"""Out-of-sample forecasts of annual bond excess returns.

Target: the log excess return from holding an n-month bond for h months,
rx_{t+h} = p_{t+h}^(n-h) - p_t^(n) - h y_t^(h) / 12, realized at t + h.

Forecasts made at month t use only information available at t: predictor
values dated t or earlier, and target values realized by t (so the last
training pair is (predictor_{t-h}, rx realized at t)). Three forecasters:

* historical mean of realized excess returns;
* the real-time ACM term premium of the same maturity, via a predictive regression;
* the Cochrane-Piazzesi (2005) factor: average excess return on bonds of
  `cp_target_years` regressed on forward rates of `cp_forward_years`, its fitted
  value used as the single predictor; both stages re-estimated each month.

Evaluation: out-of-sample R^2 against the historical mean (Campbell and Thompson,
2008) and the Clark and West (2007) MSPE-adjusted statistic, with Newey-West
standard errors because h-month returns overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from acm_term_premium.models.acm import log_prices


def excess_return(curve: pd.DataFrame, n: int, h: int) -> pd.Series:
    """rx of an n-month bond held h months, labelled by the date it is realized."""
    p = log_prices(curve)
    earned = p[n - h].shift(-h) - p[n] - h * curve[h] / 12.0
    return earned.shift(h).dropna().rename(f"rx{n}")


def forward_rates(curve: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """One-year forward rates ending at each year in `years`; the first is the 1-year yield."""
    p = log_prices(curve)
    out = {}
    for y in years:
        out[f"f{y}"] = -p[12] if y == 1 else p[12 * (y - 1)] - p[12 * y]
    return pd.DataFrame(out)


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    out: np.ndarray = coef
    return out


def newey_west_mean_se(series: np.ndarray, lags: int) -> float:
    """HAC standard error of the sample mean with Bartlett weights."""
    e = series - series.mean()
    t = e.size
    gamma0 = float(e @ e) / t
    total = gamma0
    for lag in range(1, min(lags, t - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        total += 2.0 * weight * float(e[lag:] @ e[:-lag]) / t
    return float(np.sqrt(max(total, 0.0) / t))


@dataclass(frozen=True)
class ForecastResult:
    maturity: int
    predictor: str
    months: int
    oos_r2: float
    clark_west: float
    p_value: float


def evaluate(
    curve: pd.DataFrame,
    realtime_tp: pd.DataFrame,
    *,
    maturities: list[int],
    horizon: int,
    cp_forward_years: list[int],
    cp_target_years: list[int],
    min_train: int,
    nw_lags: int,
) -> tuple[list[ForecastResult], pd.DataFrame]:
    """Recursive forecasts for every target maturity; returns statistics and the forecast panel."""
    dates = pd.DatetimeIndex(curve.index)
    position = {d: i for i, d in enumerate(dates)}
    fwd = forward_rates(curve, cp_forward_years)
    cp_target = pd.concat(
        [excess_return(curve, 12 * y, horizon) for y in cp_target_years], axis=1
    ).mean(axis=1)
    targets = {n: excess_return(curve, n, horizon) for n in maturities}

    records: list[dict[str, float | int | pd.Timestamp]] = []
    for t_date in dates:
        t = position[t_date]
        # Training pairs: predictor at s, return realized at s + h <= t.
        train_s = dates[: max(t - horizon + 1, 0)]
        if len(train_s) < min_train:
            continue
        realized = dates[[position[s] + horizon for s in train_s]]
        cp_y = cp_target.reindex(realized).to_numpy()
        cp_x = fwd.loc[train_s].to_numpy()
        ok = np.isfinite(cp_y)
        gamma = _ols(cp_x[ok], cp_y[ok])
        cp_factor_train = np.column_stack([np.ones(len(train_s)), cp_x]) @ gamma
        cp_now = float(np.concatenate([[1.0], fwd.loc[t_date].to_numpy()]) @ gamma)
        for n in maturities:
            y = targets[n].reindex(realized).to_numpy()
            if not np.isfinite(y).all():
                continue
            row: dict[str, float | int | pd.Timestamp] = {"date": t_date, "maturity": n}
            row["mean"] = float(y.mean())
            b_cp = _ols(cp_factor_train, y)
            row["cp"] = float(b_cp[0] + b_cp[1] * cp_now)
            if n in realtime_tp.columns and t_date in realtime_tp.index:
                tp_train = realtime_tp[n].reindex(train_s).to_numpy()
                use = np.isfinite(tp_train)
                if use.sum() >= min_train:
                    b_tp = _ols(tp_train[use], y[use])
                    row["acm"] = float(b_tp[0] + b_tp[1] * realtime_tp.loc[t_date, n])
            if t + horizon < len(dates):
                row["actual"] = float(targets[n].get(dates[t + horizon], np.nan))
            records.append(row)
    panel = pd.DataFrame(records)
    results: list[ForecastResult] = []
    for n in maturities:
        rows = panel[(panel["maturity"] == n)].dropna(subset=["actual"])
        # Both forecasters are scored on the months where both exist.
        if "acm" not in rows:
            continue
        common = rows.dropna(subset=["acm", "cp"])
        for predictor in ("acm", "cp"):
            r = common
            if len(r) < 24:
                continue
            e_mean = r["actual"] - r["mean"]
            e_model = r["actual"] - r[predictor]
            r2 = 1.0 - float((e_model**2).sum()) / float((e_mean**2).sum())
            adjusted = e_mean**2 - (e_model**2 - (r["mean"] - r[predictor]) ** 2)
            values = adjusted.to_numpy()
            cw = float(values.mean()) / newey_west_mean_se(values, nw_lags)
            results.append(
                ForecastResult(n, predictor, len(r), r2, cw, float(1.0 - stats.norm.cdf(cw)))
            )
    return results, panel

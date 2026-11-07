"""HAR forecasts of the variance over the next 30 calendar days (Corsi 2009), walk-forward.

The model is a direct projection: on each day t,
    y_t = sum of daily variance over (t, t + 30 calendar days]
    y_t = b0 + b_d x_t^(d) + b_w x_t^(w) + b_m x_t^(m) + e_t,
with x^(d), x^(w), x^(m) the mean of a daily variance proxy over the last 1, 5 and 22
trading days (1, 7 and 30 days for bitcoin). The target y uses the unbiased measure (squared
close-to-close returns for the ETFs, 5-minute realized variance for bitcoin); the regressors
can use a more efficient proxy. Coefficients are re-estimated at each month start on rows
whose 30-day target had fully settled by then, so a forecast never uses information from its
own future. A forecast below the smallest target seen in training is raised to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HarFit:
    coefficients: tuple[float, float, float, float]
    observations: int


def features(proxy: pd.Series, *, lags: tuple[int, int, int]) -> pd.DataFrame:
    """Rolling means of the daily proxy over each lag, indexed like `proxy`."""
    return pd.DataFrame(
        {f"x{lag}": proxy.rolling(lag, min_periods=lag).mean() for lag in lags}, index=proxy.index
    )


def forward_sum(daily: pd.Series, *, days: int) -> tuple[pd.Series, pd.Series]:
    """For each date t: the sum of `daily` over (t, t + days] and the date that window ends.
    `daily` is indexed by date (trading days, or every day for bitcoin)."""
    dates = daily.index.to_numpy()
    values = np.nan_to_num(daily.to_numpy(dtype=np.float64))
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    ends = dates + np.timedelta64(days, "D")
    last = np.searchsorted(dates, ends, side="right")  # first index past the window
    first = np.arange(1, dates.size + 1)  # first index inside the window (t excluded)
    total = cumulative[last] - cumulative[np.minimum(first, dates.size)]
    complete = ends <= dates[-1]
    out = pd.Series(np.where(complete, total, np.nan), index=daily.index)
    return out, pd.Series(pd.DatetimeIndex(ends), index=daily.index)


def fit(x: np.ndarray, y: np.ndarray) -> HarFit:
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return HarFit(
        coefficients=(float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])),
        observations=len(y),
    )


def walk_forward(
    proxy: pd.Series,
    target: pd.Series,
    target_end: pd.Series,
    *,
    lags: tuple[int, int, int],
    min_train_days: int,
) -> pd.DataFrame:
    """date, forecast, actual: out-of-sample HAR forecasts of `target`, refit monthly."""
    x = features(proxy, lags=lags)
    frame = x.assign(y=target, end=target_end).dropna(subset=list(x.columns))
    months = pd.DatetimeIndex(frame.index).to_period("M")
    out = []
    for month in months.unique():
        start = month.start_time
        train = frame.loc[(frame["end"] < start) & frame["y"].notna()]
        if len(train) < min_train_days:
            continue
        model = fit(train[list(x.columns)].to_numpy(), train["y"].to_numpy())
        live = frame.loc[months == month]
        b = np.array(model.coefficients)
        predicted = b[0] + live[list(x.columns)].to_numpy() @ b[1:]
        predicted = np.maximum(predicted, float(train["y"].min()))
        out.append(
            pd.DataFrame({"forecast": predicted, "actual": live["y"].to_numpy()}, index=live.index)
        )
    if not out:
        return pd.DataFrame(
            {"forecast": pd.Series(dtype=float), "actual": pd.Series(dtype=float)},
            index=pd.DatetimeIndex([]),
        )
    return pd.concat(out)


def oos_r2(forecast: np.ndarray, actual: np.ndarray, benchmark: np.ndarray) -> float:
    """1 - MSE(forecast) / MSE(benchmark) on rows where all three exist."""
    ok = np.isfinite(forecast) & np.isfinite(actual) & np.isfinite(benchmark)
    if ok.sum() < 2:
        return float("nan")
    return 1.0 - float(
        np.mean((actual[ok] - forecast[ok]) ** 2) / np.mean((actual[ok] - benchmark[ok]) ** 2)
    )

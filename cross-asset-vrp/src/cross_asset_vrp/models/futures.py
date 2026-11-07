"""Short VIX futures at a constant 30-day maturity, sized three ways.

On each trading day t the position holds the first two monthly contracts that have not
expired, weighted so their average time to expiry is 30 calendar days:
    w1 = (tau2 - 30) / (tau2 - tau1), w2 = 1 - w1, both held in [0, 1],
with tau the calendar days to expiry at the close of t. The weights set at the close of t-1
earn the settle-to-settle change of each contract on day t, scaled by the value of the
blend at t-1, so the series is the excess return of a fully collateralized long position
(the collateral's T-bill return is left out). The short position earns minus that.

Trading cost: a contract changes weight every day as the blend rolls; each unit of weight
traded pays `cost_points_per_side` index points, as a share of the blend's value.

Sizing, decided at the close of t-1 and applied to day t:
- unconditional: 1
- premium: the ex-ante VIX premium over its expanding mean, held in [0, cap]
- slope: 1 - VIX9D / VIX3M (positive in contango) over the expanding mean of its positive
  part, held in [0, cap]; zero when the curve is inverted.
Both normalizations use only past values and start after `min_history` days of signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def constant_maturity(vx: pd.DataFrame, *, target_days: int, cost_points: float) -> pd.DataFrame:
    """date, ret (long excess return net of roll costs), level (the blend at the close),
    tau1, tau2 for the contracts held."""
    settle = vx.pivot_table(
        index="trade_date", columns="expiry", values="settle", aggfunc="last"
    ).sort_index()
    expiries = np.array(sorted(settle.columns), dtype="datetime64[ns]")
    dates = settle.index.to_numpy(dtype="datetime64[ns]")
    values = settle.to_numpy(dtype=np.float64)
    rows = []
    previous: tuple[np.ndarray, float] | None = None  # weights on all contracts, blend value
    for i, day in enumerate(dates):
        alive = np.flatnonzero(expiries > day)
        weights = np.zeros(expiries.size)
        level = np.nan
        if alive.size >= 2:
            a, b = alive[0], alive[1]
            tau1 = (expiries[a] - day) / np.timedelta64(1, "D")
            tau2 = (expiries[b] - day) / np.timedelta64(1, "D")
            w1 = float(np.clip((tau2 - target_days) / (tau2 - tau1), 0.0, 1.0))
            weights[a], weights[b] = w1, 1.0 - w1
            level = float(np.nansum(weights * np.nan_to_num(values[i])))
            if not np.isfinite(values[i, a]) or not np.isfinite(values[i, b]):
                level = np.nan
        ret = np.nan
        if previous is not None and np.isfinite(previous[1]) and previous[1] > 0:
            held, value = previous
            change = values[i] - values[i - 1]
            used = held > 0
            if np.all(np.isfinite(change[used])):
                traded = float(np.abs(weights - held).sum()) if np.isfinite(level) else 0.0
                ret = (
                    float(np.sum(held[used] * change[used])) / value - traded * cost_points / value
                )
        rows.append({"date": pd.Timestamp(day), "ret": ret, "level": level})
        previous = (weights, level)
    return pd.DataFrame(rows).set_index("date")


def expanding_scale(
    signal: pd.Series, *, min_history: int, cap: float, positive_only: bool
) -> pd.Series:
    """signal / its expanding past mean (positive part if asked), lagged one day, in [0, cap]."""
    base = signal.clip(lower=0.0) if positive_only else signal
    mean = base.expanding(min_periods=min_history).mean()  # known at the close of t
    scaled = (signal / mean).where(mean > 0)
    return scaled.clip(lower=0.0, upper=cap).shift(1)


def strategy_returns(short: pd.Series, size: pd.Series) -> pd.Series:
    return (short * size.reindex(short.index)).dropna()


def limited_liability(returns: pd.Series) -> tuple[pd.Series, pd.Timestamp | None]:
    """Returns of an account that cannot lose more than its equity.

    A sized position can lose more than the account in one day (twice a 100% move at a
    size of 2). The first day the compounded equity reaches zero, the account is wiped
    out: that day's return becomes -100% and every later return is zero (no capital, no
    position). Returns the capped series and the ruin date, or None.
    """
    r = returns.astype(float).copy()
    equity = (1.0 + r).cumprod()
    hit = np.flatnonzero(equity.to_numpy(dtype=np.float64) <= 0.0)
    if hit.size == 0:
        return r, None
    at = int(hit[0])
    r.iloc[at] = -1.0
    r.iloc[at + 1 :] = 0.0
    return r, pd.Timestamp(str(r.index[at]))

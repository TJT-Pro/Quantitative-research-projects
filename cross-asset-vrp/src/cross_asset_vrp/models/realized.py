"""Realized measures from intraday returns and range estimators from daily bars.

All inputs are log prices or log returns; all outputs are variances over the period (not
annualized).

Range estimators, with o = ln(O / C_prev), u = ln(H / O), d = ln(L / O), c = ln(C / O):
- Parkinson (1980):        (u - d)^2 / (4 ln 2)
- Garman and Klass (1980): 0.5 (u - d)^2 - (2 ln 2 - 1) c^2
- Rogers and Satchell (1991): u (u - c) + d (d - c)
- Yang and Zhang (2000):   V_o + k V_c + (1 - k) V_RS over a window of n days, with
  k = 0.34 / (1.34 + (n + 1) / (n - 1)), V_o and V_c the sample variances of o and c and
  V_RS the mean Rogers-Satchell term; it alone includes the overnight move.
Parkinson, Garman-Klass and Rogers-Satchell measure the open-to-close part of the day.

Realized kernel (Barndorff-Nielsen, Hansen, Lunde and Shephard 2008, 2009), Parzen weights:
    RK = gamma_0 + 2 sum_{h=1}^{H} k(h / (H + 1)) gamma_h,  gamma_h = sum_j r_j r_{j-h}
with H = c* xi^(4/5) n^(3/5), c* = 3.5134 and xi^2 = omega^2 / IV, the noise variance
omega^2 estimated by RV_dense / (2 n) and IV by a sparse (20-minute) RV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LN2 = float(np.log(2.0))
PARZEN_C = 3.5134
SPARSE_MINUTES = 20


def _logs(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    o = np.log(bars["open"].to_numpy(dtype=np.float64))
    h = np.log(bars["high"].to_numpy(dtype=np.float64))
    lo = np.log(bars["low"].to_numpy(dtype=np.float64))
    c = np.log(bars["close"].to_numpy(dtype=np.float64))
    prev = np.concatenate([[np.nan], c[:-1]])
    return {"o": o - prev, "u": h - o, "d": lo - o, "c": c - o, "cc": c - prev}


def close_to_close(bars: pd.DataFrame) -> pd.Series:
    """Squared close-to-close log return: unbiased for the day's variance, overnight included."""
    x = _logs(bars)
    return pd.Series(x["cc"] ** 2, index=bars.index)


def parkinson(bars: pd.DataFrame) -> pd.Series:
    x = _logs(bars)
    return pd.Series((x["u"] - x["d"]) ** 2 / (4.0 * LN2), index=bars.index)


def garman_klass(bars: pd.DataFrame) -> pd.Series:
    x = _logs(bars)
    return pd.Series(
        0.5 * (x["u"] - x["d"]) ** 2 - (2.0 * LN2 - 1.0) * x["c"] ** 2, index=bars.index
    )


def rogers_satchell(bars: pd.DataFrame) -> pd.Series:
    x = _logs(bars)
    return pd.Series(x["u"] * (x["u"] - x["c"]) + x["d"] * (x["d"] - x["c"]), index=bars.index)


def overnight(bars: pd.DataFrame) -> pd.Series:
    """Squared close-to-open log return."""
    x = _logs(bars)
    return pd.Series(x["o"] ** 2, index=bars.index)


def yang_zhang(bars: pd.DataFrame, *, window: int) -> pd.Series:
    """Rolling Yang-Zhang daily variance over `window` days (it needs a window by design)."""
    x = _logs(bars)
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    o = pd.Series(x["o"], index=bars.index)
    c = pd.Series(x["c"], index=bars.index)
    rs = rogers_satchell(bars)
    v_o = o.rolling(window).var(ddof=1)
    v_c = c.rolling(window).var(ddof=1)
    v_rs = rs.rolling(window).mean()
    return v_o + k * v_c + (1.0 - k) * v_rs


DAILY_ESTIMATORS = {
    "close_to_close": close_to_close,
    "parkinson": parkinson,
    "garman_klass": garman_klass,
    "rogers_satchell": rogers_satchell,
}


def estimate(name: str, bars: pd.DataFrame, *, window: int) -> pd.Series:
    if name == "yang_zhang":
        return yang_zhang(bars, window=window)
    return DAILY_ESTIMATORS[name](bars)


def realized_variance(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    return float(np.sum(r * r))


def parzen(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    return np.where(a <= 0.5, 1 - 6 * a**2 + 6 * a**3, np.where(a <= 1.0, 2 * (1 - a) ** 3, 0.0))


def kernel_bandwidth(returns: np.ndarray, *, sparse_step: int) -> int:
    r = returns[np.isfinite(returns)]
    n = r.size
    if n < 4 * sparse_step:
        return 1
    omega2 = float(np.sum(r * r)) / (2.0 * n)
    prices = np.concatenate([[0.0], np.cumsum(r)])
    sparse = np.diff(prices[::sparse_step])
    iv = float(np.sum(sparse * sparse))
    if iv <= 0 or omega2 <= 0:
        return 1
    xi = np.sqrt(omega2 / iv)
    return max(1, int(np.ceil(PARZEN_C * xi**0.8 * n**0.6)))


def realized_kernel(
    returns: np.ndarray, *, bandwidth: int | None = None, sparse_step: int = SPARSE_MINUTES
) -> float:
    """Parzen realized kernel of one period's high-frequency returns."""
    r = returns[np.isfinite(returns)]
    if r.size < 2:
        return float("nan")
    h_max = bandwidth if bandwidth is not None else kernel_bandwidth(r, sparse_step=sparse_step)
    total = float(np.dot(r, r))
    for h in range(1, min(h_max, r.size - 1) + 1):
        weight = float(parzen(np.array([h / (h_max + 1)]))[0])
        total += 2.0 * weight * float(np.dot(r[h:], r[:-h]))
    return total

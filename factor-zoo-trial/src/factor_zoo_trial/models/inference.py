"""Sample means with autocorrelation-robust standard errors, and one-sided p-values.

Overlapping holding periods make long-short returns autocorrelated, so the standard error
needs a long-run variance. Newey-West with a Student-t reference over-rejects in the
far tail that Bonferroni and Holm use (at 12 lags and 360 months, about twice the
nominal rate at 0.1%). The equal-weighted cosine (EWC) estimator of Lazarus, Lewis,
Stock and Watson (2018) projects the demeaned series on nu cosines; its t-statistic is
exactly Student-t with nu degrees of freedom under Gaussian errors and has that fixed-b
limit in general, so the tail p-values stay valid. nu follows their rule 0.4 T^(2/3).
Newey-West is kept for comparison and for the property tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MeanTest:
    mean: float
    se: float
    t: float
    n: int
    df: int  # degrees of freedom of the Student-t reference


def _finite(values: FloatArray) -> FloatArray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("need at least three observations")
    return x


def _t(mean: float, se: float) -> float:
    if se > 0:
        return mean / se
    return float(np.copysign(np.inf, mean)) if mean != 0 else 0.0


def ewc_terms(n: int, *, coefficient: float) -> int:
    return int(min(max(1, round(coefficient * n ** (2.0 / 3.0))), n - 1))


def ewc_mean(values: FloatArray, *, coefficient: float) -> MeanTest:
    x = _finite(values)
    n = x.size
    nu = ewc_terms(n, coefficient=coefficient)
    e = x - x.mean()
    j = np.arange(1, nu + 1)[:, None]
    t_index = np.arange(1, n + 1)[None, :]
    basis = np.sqrt(2.0 / n) * np.cos(np.pi * j * (t_index - 0.5) / n)
    projections = basis @ e
    long_run = float(np.mean(projections**2))
    se = float(np.sqrt(long_run / n))
    mean = float(x.mean())
    return MeanTest(mean=mean, se=se, t=_t(mean, se), n=n, df=nu)


def newey_west_mean(values: FloatArray, *, lags: int) -> MeanTest:
    """Bartlett-kernel HAC standard error (lags=0 is the iid case), t reference on n - 1 df."""
    x = _finite(values)
    n = x.size
    mean = float(x.mean())
    e = x - mean
    long_run = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        long_run += 2.0 * weight * float(e[lag:] @ e[:-lag]) / n
    # Degrees-of-freedom correction so lags=0 matches the textbook s / sqrt(n).
    se = float(np.sqrt(max(long_run, 0.0) / (n - 1)))
    return MeanTest(mean=mean, se=se, t=_t(mean, se), n=n, df=n - 1)


def one_sided_p(t: FloatArray, df: NDArray[np.int64]) -> FloatArray:
    """P(T >= t) under H0: mean <= 0, Student-t reference with the given degrees of freedom."""
    return np.asarray(stats.t.sf(t, df=np.maximum(df, 1)), dtype=np.float64)

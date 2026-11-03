"""Sample covariance and the exponentially weighted moving average."""

from __future__ import annotations

import numpy as np

from covariance_horse_race.models.base import (
    CovarianceModel,
    Matrix,
    check_window,
    demean,
    symmetrize,
)


def sample_covariance(returns: Matrix) -> Matrix:
    """Unbiased sample covariance of a T x N window."""
    check_window(returns)
    x = demean(returns)
    return symmetrize(x.T @ x / (x.shape[0] - 1))


class SampleCovariance(CovarianceModel):
    def __init__(self) -> None:
        self._sigma: Matrix | None = None

    def fit(self, returns: Matrix, factors: Matrix | None = None) -> None:
        self._sigma = sample_covariance(returns)

    def forecast(self, horizon: int) -> Matrix:
        if self._sigma is None:
            raise RuntimeError("fit before forecast")
        return self._sigma.copy()


def ewma_covariance(returns: Matrix, decay: float) -> Matrix:
    """EWMA covariance with weights normalized to sum to one over the window.

    Normalizing over the finite window (rather than seeding with an initial
    matrix) means the estimate depends only on data inside the window.
    Returns are demeaned with the window mean, consistent with the other models.
    """
    check_window(returns)
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must lie in (0, 1)")
    t = returns.shape[0]
    weights = decay ** np.arange(t - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    x = demean(returns)
    return symmetrize((x * weights[:, None]).T @ x)


class Ewma(CovarianceModel):
    def __init__(self, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.decay = decay
        self._sigma: Matrix | None = None

    def fit(self, returns: Matrix, factors: Matrix | None = None) -> None:
        self._sigma = ewma_covariance(returns, self.decay)

    def forecast(self, horizon: int) -> Matrix:
        # EWMA is a random-walk forecaster: the h-step forecast equals the current estimate.
        if self._sigma is None:
            raise RuntimeError("fit before forecast")
        return self._sigma.copy()

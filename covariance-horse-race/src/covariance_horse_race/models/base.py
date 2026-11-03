"""Common interface for covariance forecasters.

A model is fitted on a T x N window of daily returns (and, for observable factor
models, the matching T x K factor returns) and produces the forecast of the
average daily covariance matrix over the next `horizon` trading days. Using the
average daily matrix keeps every model and every loss in the same units
regardless of how many trading days the next month has.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]


class CovarianceModel(ABC):
    requires_factors: bool = False

    @abstractmethod
    def fit(self, returns: Matrix, factors: Matrix | None = None) -> None:
        """Estimate on a T x N window. Must not read anything outside it."""

    @abstractmethod
    def forecast(self, horizon: int) -> Matrix:
        """Average daily covariance over the next `horizon` days (N x N, symmetric)."""


def demean(returns: Matrix) -> Matrix:
    out: Matrix = returns - returns.mean(axis=0, keepdims=True)
    return out


def symmetrize(matrix: Matrix) -> Matrix:
    out: Matrix = 0.5 * (matrix + matrix.T)
    return out


def check_window(returns: Matrix) -> None:
    if returns.ndim != 2:
        raise ValueError("returns must be a T x N matrix")
    if returns.shape[0] < 2:
        raise ValueError("need at least two observations")
    if not np.isfinite(returns).all():
        raise ValueError("returns window contains non-finite values")

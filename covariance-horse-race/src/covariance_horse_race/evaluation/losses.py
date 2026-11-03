"""Statistical losses for covariance forecasts against a realized proxy.

The realized proxy for month m is the average daily outer product of returns,
Sigma_m = (1/D) sum_d r_d r_d'. It is noisy and singular when D < N, so only
losses that rank forecasts consistently under a conditionally unbiased noisy
proxy are used (Laurent, Rombouts and Violante, 2013): the Frobenius distance
and the multivariate QLIKE

    L(Sigma, H) = log|H| + tr(H^-1 Sigma).
"""

from __future__ import annotations

import numpy as np

from covariance_horse_race.models.base import Matrix


def realized_proxy(period_returns: np.ndarray) -> Matrix:
    days = period_returns.shape[0]
    if days == 0:
        raise ValueError("empty holding period")
    out: Matrix = period_returns.T @ period_returns / days
    return out


def qlike(proxy: Matrix, forecast: Matrix) -> float:
    sign, logdet = np.linalg.slogdet(forecast)
    if sign <= 0:
        raise ValueError("forecast is not positive definite")
    return float(logdet + np.trace(np.linalg.solve(forecast, proxy)))


def frobenius(proxy: Matrix, forecast: Matrix) -> float:
    return float(np.sum((forecast - proxy) ** 2))


def portfolio_realized_variance(weights: np.ndarray, period_returns: np.ndarray) -> float:
    """Average squared daily portfolio return over the holding period."""
    returns = period_returns @ weights
    return float(np.mean(returns**2))

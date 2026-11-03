"""DCC-GARCH and DCC-NL.

Univariate step: GARCH(1,1) with variance targeting for each asset, Gaussian
quasi-maximum likelihood.

Correlation step: scalar DCC(1,1) (Engle, 2002) with correlation targeting,

    Q_t = (1 - a - b) C + a z_{t-1} z_{t-1}' + b Q_{t-1},   R_t = diag(Q_t)^-1/2 Q_t diag(Q_t)^-1/2,

estimated by the composite likelihood over contiguous pairs (Engle, Shephard and
Sheppard, 2008), which stays tractable for 100 assets where the full likelihood
does not. DCC uses the sample covariance of the devolatilized residuals as C;
DCC-NL (Engle, Ledoit and Wolf, 2019) replaces it with its nonlinear shrinkage
estimate rescaled to a correlation matrix.

Multi-step forecasts use exact GARCH variance mean reversion and the
Engle-Sheppard (2001) approximation for correlations, Q_{T+k} ~ C + (a+b)^(k-1)(Q_{T+1} - C).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np
from scipy.optimize import minimize

from covariance_horse_race.models.base import (
    CovarianceModel,
    Matrix,
    check_window,
    demean,
    symmetrize,
)
from covariance_horse_race.models.shrinkage import analytical_nonlinear_shrinkage

# Persistence above this makes the variance target meaningless (unit root).
_MAX_PERSISTENCE = 0.9999
# Ratios are kept this far inside (0, 1) when mapped to optimizer coordinates.
_EDGE = 1e-8
# A warm start with a parameter below this is on the boundary; search from scratch too.
_BOUNDARY = 1e-6
_LOG_2PI = float(np.log(2.0 * np.pi))


def _logistic(u: float) -> float:
    return float(1.0 / (1.0 + np.exp(-u)))


def _logit(p: float) -> float:
    q = min(max(p, _EDGE), 1.0 - _EDGE)
    return float(np.log(q / (1.0 - q)))


def _unpack(theta: np.ndarray) -> tuple[float, float]:
    """Map R^2 to (alpha, beta) with alpha, beta >= 0 and alpha + beta < _MAX_PERSISTENCE."""
    persistence = _MAX_PERSISTENCE * _logistic(float(theta[0]))
    alpha = persistence * _logistic(float(theta[1]))
    return alpha, persistence - alpha


def _pack(alpha: float, beta: float) -> np.ndarray:
    persistence = max(alpha + beta, _EDGE)
    return np.array([_logit(persistence / _MAX_PERSISTENCE), _logit(alpha / persistence)])


@numba.njit(cache=True)
def garch_filter(eps: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """Conditional variances h_1..h_{T+1}; h_{T+1} is the one-step-ahead forecast."""
    t = eps.shape[0]
    h = np.empty(t + 1)
    h[0] = h0
    for i in range(t):
        h[i + 1] = omega + alpha * eps[i] * eps[i] + beta * h[i]
    return h


@numba.njit(cache=True)
def garch_negloglik(eps: np.ndarray, alpha: float, beta: float, target: float) -> float:
    omega = target * (1.0 - alpha - beta)
    h = target
    total = 0.0
    for i in range(eps.shape[0]):
        if h <= 0.0:
            return 1e300
        total += np.log(h) + eps[i] * eps[i] / h
        h = omega + alpha * eps[i] * eps[i] + beta * h
    return 0.5 * total


@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    beta: float

    @property
    def unconditional(self) -> float:
        return self.omega / (1.0 - self.alpha - self.beta)


def fit_garch(eps: np.ndarray, start: GarchParams | None = None) -> GarchParams:
    """Variance-targeted GARCH(1,1) by QMLE on demeaned returns."""
    target = float(np.mean(eps**2))
    if target <= 0:
        raise ValueError("zero variance series")

    def objective(theta: np.ndarray) -> float:
        a, b = _unpack(theta)
        return float(garch_negloglik(eps, a, b, target))

    # A warm start from last month's estimate on an overlapping window is a single
    # local search; a cold fit uses several starts to avoid a poor local optimum.
    cold = [_pack(0.05, 0.90), _pack(0.10, 0.80), _pack(0.02, 0.97)]
    if start is None:
        starts = cold
    elif min(start.alpha, start.beta) < _BOUNDARY:
        starts = [_pack(start.alpha, start.beta), *cold]
    else:
        starts = [_pack(start.alpha, start.beta)]
    best = None
    for theta0 in starts:
        result = minimize(
            objective,
            theta0,
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 2000},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("GARCH optimizer produced no result")
    alpha, beta = _unpack(best.x)
    return GarchParams(omega=target * (1.0 - alpha - beta), alpha=alpha, beta=beta)


@numba.njit(cache=True)
def dcc_pair_negloglik(z: np.ndarray, c: np.ndarray, a: float, b: float) -> float:
    """Negative composite log-likelihood over contiguous pairs (i, i+1)."""
    t, n = z.shape
    w = 1.0 - a - b
    total = 0.0
    for i in range(n - 1):
        j = i + 1
        cii, cjj, cij = c[i, i], c[j, j], c[i, j]
        qii, qjj, qij = cii, cjj, cij
        for s in range(t):
            rho = qij / np.sqrt(qii * qjj)
            one_minus = 1.0 - rho * rho
            if one_minus <= 1e-12:
                return 1e300
            zi, zj = z[s, i], z[s, j]
            total += np.log(one_minus) + (zi * zi + zj * zj - 2.0 * rho * zi * zj) / one_minus
            qii = w * cii + a * zi * zi + b * qii
            qjj = w * cjj + a * zj * zj + b * qjj
            qij = w * cij + a * zi * zj + b * qij
    return 0.5 * total


@numba.njit(cache=True)
def dcc_last_q(z: np.ndarray, c: np.ndarray, a: float, b: float) -> np.ndarray:
    """Q_{T+1}, the one-step-ahead pseudo-correlation matrix."""
    t, n = z.shape
    w = 1.0 - a - b
    q = c.copy()
    for s in range(t):
        for i in range(n):
            for j in range(i, n):
                value = w * c[i, j] + a * z[s, i] * z[s, j] + b * q[i, j]
                q[i, j] = value
                q[j, i] = value
    return q


def _to_correlation(matrix: Matrix) -> Matrix:
    d = np.sqrt(np.diag(matrix))
    corr = matrix / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return symmetrize(corr)


def fit_dcc(
    z: np.ndarray, c: np.ndarray, start: tuple[float, float] | None = None
) -> tuple[float, float]:
    def objective(theta: np.ndarray) -> float:
        a, b = _unpack(theta)
        return float(dcc_pair_negloglik(z, c, a, b))

    cold = [_pack(0.01, 0.97), _pack(0.03, 0.90), _pack(0.005, 0.99)]
    if start is None:
        starts = cold
    elif min(start) < _BOUNDARY:
        starts = [_pack(*start), *cold]
    else:
        starts = [_pack(*start)]
    best = None
    for theta0 in starts:
        result = minimize(
            objective,
            theta0,
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 2000},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("DCC optimizer produced no result")
    return _unpack(best.x)


class Dcc(CovarianceModel):
    def __init__(
        self,
        target: Literal["sample", "nonlinear_shrinkage"] = "sample",
        reestimate_every: int = 1,
    ) -> None:
        if reestimate_every < 1:
            raise ValueError("reestimate_every must be at least 1")
        self.target = target
        self.reestimate_every = reestimate_every
        self._fits_since_estimate = 0
        self.garch: list[GarchParams] | None = None
        self.dcc_params: tuple[float, float] | None = None
        self._h_next: np.ndarray | None = None
        self._q_next: Matrix | None = None
        self._c: Matrix | None = None

    def _estimate_now(self, n_assets: int) -> bool:
        if self.garch is None or self.dcc_params is None or len(self.garch) != n_assets:
            return True
        return self._fits_since_estimate >= self.reestimate_every

    def fit(self, returns: Matrix, factors: Matrix | None = None) -> None:
        check_window(returns)
        eps = demean(returns)
        t, n = eps.shape
        if self._estimate_now(n):
            previous = self.garch if self.garch is not None and len(self.garch) == n else None
            self.garch = [
                fit_garch(np.ascontiguousarray(eps[:, i]), previous[i] if previous else None)
                for i in range(n)
            ]
            self._fits_since_estimate = 0
            reestimate_dcc = True
        else:
            reestimate_dcc = False
        if self.garch is None:
            raise RuntimeError("GARCH parameters missing after estimation")
        h = np.empty((t + 1, n))
        for i, g in enumerate(self.garch):
            # Each window starts from its own unconditional variance so the filter
            # depends only on data inside the window.
            h[:, i] = garch_filter(
                np.ascontiguousarray(eps[:, i]), g.omega, g.alpha, g.beta, g.unconditional
            )
        z = np.ascontiguousarray(eps / np.sqrt(h[:-1]))
        if self.target == "sample":
            c = _to_correlation(z.T @ z / t)
        else:
            c = _to_correlation(analytical_nonlinear_shrinkage(z, demeaned=True))
        if reestimate_dcc or self.dcc_params is None:
            self.dcc_params = fit_dcc(z, c, self.dcc_params)
        a, b = self.dcc_params
        self._h_next = h[-1]
        self._q_next = dcc_last_q(z, c, a, b)
        self._c = c
        self._fits_since_estimate += 1

    def forecast(self, horizon: int) -> Matrix:
        if self.garch is None or self.dcc_params is None or self._h_next is None:
            raise RuntimeError("fit before forecast")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if self._q_next is None or self._c is None:
            raise RuntimeError("fit before forecast")
        a, b = self.dcc_params
        omega = np.array([g.omega for g in self.garch])
        persistence = np.array([g.alpha + g.beta for g in self.garch])
        unconditional = omega / (1.0 - persistence)
        total = np.zeros((len(self.garch), len(self.garch)))
        for k in range(1, horizon + 1):
            h_k = unconditional + persistence ** (k - 1) * (self._h_next - unconditional)
            q_k = self._c + (a + b) ** (k - 1) * (self._q_next - self._c)
            r_k = _to_correlation(q_k)
            sd = np.sqrt(h_k)
            total += r_k * np.outer(sd, sd)
        return symmetrize(total / horizon)

"""Global minimum variance portfolios.

The unconstrained GMV portfolio, w = H^-1 1 / 1'H^-1 1, isolates the quality of
the covariance forecast because it ignores expected returns entirely. The
long-only version, min w'Hw subject to w >= 0 and 1'w = 1, is solved exactly by
a primal active-set method: each step solves the equality-constrained problem on
the current free set, and the method stops when the KKT conditions hold. It is
checked against SLSQP and against the KKT conditions in the test suite.
"""

from __future__ import annotations

import numpy as np

from covariance_horse_race.models.base import Matrix


class SingularForecastError(ValueError):
    """The forecast is numerically singular, so GMV weights are undefined."""


def is_well_conditioned(sigma: Matrix, tolerance: float) -> bool:
    eigenvalues = np.linalg.eigvalsh(sigma)
    return bool(eigenvalues[0] > tolerance * eigenvalues[-1] and eigenvalues[-1] > 0)


def gmv_weights(sigma: Matrix, tolerance: float) -> np.ndarray:
    if not is_well_conditioned(sigma, tolerance):
        raise SingularForecastError("forecast covariance is singular")
    ones = np.ones(sigma.shape[0])
    x = np.linalg.solve(sigma, ones)
    weights: np.ndarray = x / x.sum()
    return weights


def _equality_solution(h: Matrix, free: list[int]) -> np.ndarray:
    """argmin w_F' H_FF w_F subject to 1'w_F = 1, on the free set F."""
    x = np.linalg.solve(h[np.ix_(free, free)], np.ones(len(free)))
    out: np.ndarray = x / x.sum()
    return out


def gmv_long_only(
    sigma: Matrix, tolerance: float, max_iterations: int, solver_tolerance: float
) -> np.ndarray:
    """Long-only GMV weights by a primal active-set method.

    Starts from the single lowest-variance asset (a feasible vertex). At each
    step it solves the equality-constrained problem on the free set. If that
    solution is non-negative it checks the KKT condition on the excluded assets,
    gradient_i >= nu, where nu is the common gradient on the free set, and frees
    the most violating asset; otherwise it moves toward the solution until the
    first weight reaches zero and fixes that asset at zero. For a positive
    definite H the objective falls strictly at every move, so the method ends in
    finitely many steps at the exact optimum. `solver_tolerance` is relative to nu.
    """
    if not is_well_conditioned(sigma, tolerance):
        raise SingularForecastError("forecast covariance is singular")
    n = sigma.shape[0]
    # Rescaling leaves the minimizer unchanged and keeps gradients near unit size.
    h = sigma / (float(np.trace(sigma)) / n)
    start = int(np.argmin(np.diag(h)))
    free = [start]
    w = np.zeros(n)
    w[start] = 1.0
    for _ in range(max_iterations):
        v = np.zeros(n)
        v[free] = _equality_solution(h, free)
        if np.all(v[free] >= 0.0):
            w = v
            gradient = 2.0 * h @ w
            nu = float(gradient[free].mean())
            excluded = np.setdiff1d(np.arange(n), free)
            if excluded.size == 0:
                return w
            violation = gradient[excluded] - nu
            worst = int(np.argmin(violation))
            if violation[worst] >= -solver_tolerance * abs(nu):
                return w
            free.append(int(excluded[worst]))
            continue
        # Step from the feasible w toward v until the first free weight hits zero.
        direction = v - w
        shrinking = [i for i in free if direction[i] < 0.0]
        ratios = [w[i] / -direction[i] for i in shrinking]
        k = int(np.argmin(ratios))
        alpha = float(ratios[k])
        w = w + alpha * direction
        blocking = shrinking[k]
        w[blocking] = 0.0
        free.remove(blocking)
        w = np.maximum(w, 0.0)
        w /= w.sum()
    raise RuntimeError(f"long-only GMV did not converge in {max_iterations} active-set steps")


def drifted_weights(weights: np.ndarray, period_returns: np.ndarray) -> np.ndarray:
    """Weights at the end of a holding period after buy-and-hold drift.

    `period_returns` is D x N of daily simple returns. Wealth in each position
    grows by its compounded return; weights renormalize by total wealth.
    """
    growth = np.prod(1.0 + period_returns, axis=0)
    value = weights * growth
    total = value.sum()
    if total == 0 or not np.isfinite(total):
        return weights.copy()
    out: np.ndarray = value / total
    return out

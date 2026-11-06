"""Hierarchical empirical Bayes for factor alphas (Jensen, Kelly and Pedersen 2023).

    alpha_i = mu + c_{k(i)} + w_i,     c_k ~ N(0, tau_c^2),  w_i ~ N(0, tau_w^2)
    alpha_hat ~ N(alpha, Sigma)

so alpha ~ N(mu 1, Omega) with Omega = tau_c^2 M M' + tau_w^2 I, where M maps factors
to theme clusters. The hyperparameters maximize the marginal likelihood
alpha_hat ~ N(mu 1, Omega + Sigma). The posterior is computed in the form
Omega (Omega + Sigma)^-1 so that tau_w = 0 or tau_c = 0 needs no inverse of Omega.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Hyperparameters:
    mu: float
    tau_c: float
    tau_w: float
    se_tau_c: float
    se_tau_w: float
    log_likelihood: float
    converged: bool


@dataclass(frozen=True)
class Posterior:
    mean: FloatArray
    covariance: FloatArray

    @property
    def sd(self) -> FloatArray:
        return np.sqrt(np.maximum(np.diag(self.covariance), 0.0))


def membership(clusters: list[str]) -> tuple[FloatArray, list[str]]:
    names = sorted(set(clusters))
    index = {name: j for j, name in enumerate(names)}
    m = np.zeros((len(clusters), len(names)))
    for i, name in enumerate(clusters):
        m[i, index[name]] = 1.0
    return m, names


def prior_covariance(m: FloatArray, *, tau_c: float, tau_w: float) -> FloatArray:
    return np.asarray(tau_c**2 * (m @ m.T) + tau_w**2 * np.eye(m.shape[0]), dtype=np.float64)


def gaussian_log_density(x: FloatArray, mean: FloatArray, cov: FloatArray) -> float:
    factor = linalg.cho_factor(cov, lower=True, check_finite=False)
    resid = x - mean
    solved = linalg.cho_solve(factor, resid, check_finite=False)
    logdet = 2.0 * float(np.sum(np.log(np.diag(factor[0]))))
    return float(-0.5 * (resid @ solved + logdet + x.size * np.log(2.0 * np.pi)))


def _negative_log_likelihood(
    params: FloatArray, *, alpha_hat: FloatArray, sigma: FloatArray, mm: FloatArray, fix_mean: bool
) -> float:
    if fix_mean:
        mu, tau_c, tau_w = 0.0, params[0], params[1]
    else:
        mu, tau_c, tau_w = params[0], params[1], params[2]
    cov = sigma + tau_c**2 * mm + tau_w**2 * np.eye(sigma.shape[0])
    try:
        return -gaussian_log_density(alpha_hat, np.full(alpha_hat.size, mu), cov)
    except linalg.LinAlgError:
        return 1e12


Objective = Callable[[FloatArray], float]


def _hessian(f: Objective, x: FloatArray, *, step: float) -> FloatArray:
    n = x.size
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = step
            ej[j] = step
            value = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                4.0 * step**2
            )
            h[i, j] = h[j, i] = value
    return h


def fit_hyperparameters(
    *,
    alpha_hat: FloatArray,
    sigma: FloatArray,
    m: FloatArray,
    fix_mean: bool,
    rng: np.random.Generator,
    starts: int = 10,
) -> Hyperparameters:
    mm = m @ m.T
    cluster_means = (m.T @ alpha_hat) / np.maximum(m.sum(axis=0), 1.0)
    within = alpha_hat - m @ cluster_means
    base = [
        float(np.sqrt(max(np.mean(cluster_means**2) if fix_mean else np.var(cluster_means), 1e-6))),
        float(max(np.std(within), 1e-3)),
    ]
    if not fix_mean:
        base = [float(np.mean(alpha_hat)), *base]
    bounds = [(0.0, None), (0.0, None)] if fix_mean else [(None, None), (0.0, None), (0.0, None)]

    def objective(p: FloatArray) -> float:
        return _negative_log_likelihood(
            p, alpha_hat=alpha_hat, sigma=sigma, mm=mm, fix_mean=fix_mean
        )

    best: optimize.OptimizeResult | None = None
    for attempt in range(starts):
        jitter = 0.0 if attempt == 0 else 0.25
        start = np.array(base) * np.exp(rng.normal(0.0, jitter, size=len(base)))
        if not fix_mean:
            start[0] = base[0] + rng.normal(0.0, jitter * abs(base[0]) + 1e-3)
        result = optimize.minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("no optimizer run")
    x = np.asarray(best.x, dtype=np.float64)
    mu, tau_c, tau_w = (0.0, x[0], x[1]) if fix_mean else (x[0], x[1], x[2])
    se = _standard_errors(objective, x)
    se_c, se_w = (se[0], se[1]) if fix_mean else (se[1], se[2])
    return Hyperparameters(
        mu=float(mu),
        tau_c=float(abs(tau_c)),
        tau_w=float(abs(tau_w)),
        se_tau_c=float(se_c),
        se_tau_w=float(se_w),
        log_likelihood=-float(best.fun),
        converged=bool(best.success),
    )


def _standard_errors(objective: Objective, x: FloatArray) -> FloatArray:
    step = 1e-4 * np.maximum(np.abs(x), 1e-2)
    probe = np.maximum(x, 2.0 * step)  # stay off a zero bound for the finite differences
    hess = _hessian(objective, probe, step=float(step.min()))
    try:
        cov = np.linalg.inv(hess)
        return np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
    except np.linalg.LinAlgError:
        return np.full(x.size, np.nan)


def posterior(
    *, alpha_hat: FloatArray, sigma: FloatArray, omega: FloatArray, mu: float
) -> Posterior:
    prior_mean = np.full(alpha_hat.size, mu)
    total = omega + sigma
    factor = linalg.cho_factor(total, lower=True)
    gain = linalg.cho_solve(factor, omega).T  # Omega (Omega + Sigma)^-1
    mean = prior_mean + gain @ (alpha_hat - prior_mean)
    cov = omega - gain @ omega
    cov = (cov + cov.T) / 2.0
    return Posterior(mean=np.asarray(mean), covariance=np.asarray(cov))


@dataclass(frozen=True)
class ReplicationRate:
    rate: float
    lower: float
    upper: float
    n_factors: int
    replicated: int


def replication_rate(
    post: Posterior,
    *,
    include: NDArray[np.bool_],
    z: float,
    draws: int,
    rng: np.random.Generator,
    interval: float,
) -> ReplicationRate:
    """Share of included factors whose posterior interval lies above zero, with the JKP
    posterior interval for that share: draw alphas from the joint posterior and count the
    factors whose drawn alpha exceeds z posterior standard deviations."""
    sd = post.sd
    significant = (post.mean - z * sd) > 0
    chosen = np.nonzero(include)[0]
    if chosen.size == 0:
        return ReplicationRate(np.nan, np.nan, np.nan, 0, 0)
    mean = post.mean[chosen]
    cov = post.covariance[np.ix_(chosen, chosen)]
    eigval, eigvec = np.linalg.eigh(cov)
    root = eigvec * np.sqrt(np.maximum(eigval, 0.0))
    sims = mean[None, :] + rng.standard_normal((draws, chosen.size)) @ root.T
    shares = (sims / sd[chosen][None, :] > z).mean(axis=1)
    tail = (1.0 - interval) / 2.0
    return ReplicationRate(
        rate=float(significant[chosen].mean()),
        lower=float(np.quantile(shares, tail)),
        upper=float(np.quantile(shares, 1.0 - tail)),
        n_factors=int(chosen.size),
        replicated=int(significant[chosen].sum()),
    )

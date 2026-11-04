"""Adrian, Crump and Moench (2013), "Pricing the Term Structure with Linear Regressions".

Notation follows the paper. With K pricing factors X_t, monthly log zero prices
p_t^(n) = -n y_t^(n) / 12 and the one-month rate r_t = -p_t^(1):

1. Factors: the first K principal components of demeaned yields, scaled to unit
   variance and signed so their average loading is positive.
2. Dynamics: OLS VAR(1), X_{t+1} = mu + Phi X_t + v_{t+1}, Sigma = V'V / T.
3. Returns: for each bond n in the return set, the one-month log excess return
   rx_{t+1}^(n-1) = p_{t+1}^(n-1) - p_t^(n) - r_t is regressed on a constant,
   the innovation v_{t+1} and X_t:  rx = a i' + beta' V + c X_- + E,
   sigma^2 = tr(E E') / (N T).
4. Prices of risk from the cross-section of those coefficients,
   lambda_1 = (beta beta')^-1 beta c,
   lambda_0 = (beta beta')^-1 beta (a + (B* vec(Sigma) + sigma^2 i) / 2),
   with row n of B* equal to vec(beta_n beta_n')'.
5. Short rate r_t = delta_0 + delta_1' X_t by OLS, then the bond-pricing recursions
   A_n = A_{n-1} + B_{n-1}'(mu - lambda_0) + (B_{n-1}' Sigma B_{n-1} + sigma^2) / 2 - delta_0
   B_n' = B_{n-1}' (Phi - lambda_1) - delta_1',  A_0 = 0, B_0 = 0,
   and log prices p^(n) = A_n + B_n' X. Setting lambda_0 = lambda_1 = 0 gives the
   risk-neutral prices; the term premium is the fitted yield minus the
   risk-neutral yield.

Two conventions are implemented. "paper" is the estimator exactly as written
above. "nyfed" reproduces the series the New York Fed publishes: the VAR
intercept is folded into the innovations (v = X_{t+1} - Phi X_t), so the
recursions run with mu = 0; Sigma uses T - 1 degrees of freedom; and the
recursion starts from the one-month bond, A_1 = -delta_0, B_1 = -delta_1, with no
convexity term at one month. Fitted yields are identical under both conventions
because only mu - lambda_0 enters them; the risk-neutral yields, and therefore
the term premium, differ.

All yields here are continuously compounded annual decimals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Array = NDArray[np.float64]
Convention = Literal["paper", "nyfed"]


@dataclass(frozen=True)
class FactorMap:
    """Affine map from yields at `maturities` to factors: X = (Y - mean) @ loadings."""

    maturities: list[int]
    mean: Array
    loadings: Array  # n_maturities x K
    explained: Array

    def factors(self, curve: pd.DataFrame) -> Array:
        values = curve[self.maturities].to_numpy(dtype=float)
        out: Array = (values - self.mean) @ self.loadings
        return out


def principal_components(curve: pd.DataFrame, maturities: list[int], k: int) -> FactorMap:
    y = curve[maturities].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("factor yields contain missing values")
    mean = y.mean(axis=0)
    _, s, vt = np.linalg.svd(y - mean, full_matrices=False)
    loadings = vt[:k].T
    scores = (y - mean) @ loadings
    loadings = loadings / scores.std(axis=0, ddof=1)
    signs = np.sign(loadings.mean(axis=0))
    signs[signs == 0] = 1.0
    loadings = loadings * signs
    return FactorMap(list(maturities), mean, loadings, (s[:k] ** 2) / (s**2).sum())


def log_prices(curve: pd.DataFrame) -> pd.DataFrame:
    months = curve.columns.to_numpy(dtype=float)
    return -curve * months / 12.0


def excess_returns(curve: pd.DataFrame, maturities: list[int]) -> pd.DataFrame:
    """One-month log excess returns on n-month bonds, aligned to the end of the month earned."""
    p = log_prices(curve)
    r = -p[1]
    out = {n: p[n - 1].shift(-1) - p[n] - r for n in maturities}
    frame = pd.DataFrame(out).shift(1)  # label by the date the return is realized
    return frame.iloc[1:]


@dataclass(frozen=True)
class AcmModel:
    factor_map: FactorMap
    mu: Array
    phi: Array
    sigma: Array
    sigma2: float
    delta0: float
    delta1: Array
    lambda0: Array
    lambda1: Array
    a: Array  # A_n, n = 0..max
    b: Array  # B_n, (max+1) x K
    a_rn: Array
    b_rn: Array
    sample_start: pd.Timestamp
    sample_end: pd.Timestamp
    spectral_radius: float = float("nan")
    constrained: bool = False

    def decompose(self, curve: pd.DataFrame, months: list[int]) -> dict[str, pd.DataFrame]:
        """Fitted, risk-neutral and term-premium yields at `months` for every row of `curve`."""
        x = self.factor_map.factors(curve)
        n = np.asarray(months)
        fitted = -(self.a[n] + x @ self.b[n].T) / n * 12.0
        neutral = -(self.a_rn[n] + x @ self.b_rn[n].T) / n * 12.0
        frame = {"fitted": fitted, "risk_neutral": neutral, "term_premium": fitted - neutral}
        return {k: pd.DataFrame(v, index=curve.index, columns=months) for k, v in frame.items()}


def _recursions(
    params: Parameters,
    lambda0: Array,
    lambda1: Array,
    max_maturity: int,
    convention: Convention,
) -> tuple[Array, Array]:
    """Log-price coefficients A_n, B_n for n = 0..max_maturity (index n is maturity n)."""
    k = params.mu.size
    a = np.zeros(max_maturity + 1)
    b = np.zeros((max_maturity + 1, k))
    drift = params.mu - lambda0
    persistence = params.phi - lambda1
    first = 1
    if convention == "nyfed":
        a[1], b[1] = -params.delta0, -params.delta1
        first = 2
    for n in range(first, max_maturity + 1):
        prev = b[n - 1]
        convexity = 0.5 * (prev @ params.sigma @ prev + params.sigma2)
        a[n] = a[n - 1] + prev @ drift + convexity - params.delta0
        b[n] = prev @ persistence - params.delta1
    return a, b


@dataclass(frozen=True)
class Moments:
    """The regression inputs of one estimation sample, kept so the bootstrap can resample them."""

    x_now: Array  # X_t, t = 0..T-2
    x_next: Array  # X_{t+1}
    rx: Array  # rx_{t+1}, (T-1) x N
    x_all: Array  # X_t for the short-rate regression
    r_all: Array


def moments(curve: pd.DataFrame, factor_map: FactorMap, return_maturities: list[int]) -> Moments:
    x = factor_map.factors(curve)
    rx = excess_returns(curve, return_maturities).to_numpy(dtype=float)
    r = (curve[1] / 12.0).to_numpy(dtype=float)
    return Moments(x_now=x[:-1], x_next=x[1:], rx=rx, x_all=x, r_all=r)


@dataclass(frozen=True)
class Parameters:
    mu: Array
    phi: Array
    sigma: Array
    sigma2: float
    delta0: float
    delta1: Array
    lambda0: Array
    lambda1: Array
    spectral_radius: float  # of the unconstrained OLS Phi
    constrained: bool


def constrain_spectral_radius(phi: Array, bound: float) -> Array:
    """Pull every eigenvalue of Phi with modulus above `bound` down to `bound`.

    Eigenvectors and the phase of complex pairs are kept, so only the speed of
    mean reversion changes, and only along the offending directions.
    """
    values, vectors = np.linalg.eig(phi)
    moduli = np.abs(values)
    scale = np.where(moduli > bound, bound / moduli, 1.0)
    out: Array = np.real(vectors @ np.diag(values * scale) @ np.linalg.inv(vectors))
    return out


def estimate_parameters(
    m: Moments, convention: Convention, max_spectral_radius: float | None = None
) -> Parameters:
    """Steps 2-5. With `max_spectral_radius`, an explosive or near-unit-root VAR is
    constrained (see `constrain_spectral_radius`) and the intercept, innovations
    and everything downstream are re-estimated given the constrained Phi."""
    t, k = m.x_now.shape
    n_bonds = m.rx.shape[1]
    design = np.column_stack([np.ones(t), m.x_now])
    coef, *_ = np.linalg.lstsq(design, m.x_next, rcond=None)
    mu, phi = coef[0], coef[1:].T
    radius = float(np.abs(np.linalg.eigvals(phi)).max())
    constrained = False
    if max_spectral_radius is not None and radius > max_spectral_radius:
        constrained = True
        phi = constrain_spectral_radius(phi, max_spectral_radius)
        mu = (m.x_next - m.x_now @ phi.T).mean(axis=0)
    v = m.x_next - mu - m.x_now @ phi.T
    if convention == "nyfed":
        sigma = v.T @ v / (t - 1)
        v = v + mu  # intercept folded into the innovations
        mu = np.zeros_like(mu)
    else:
        sigma = v.T @ v / t

    z = np.column_stack([np.ones(t), v, m.x_now])
    gamma, *_ = np.linalg.lstsq(z, m.rx, rcond=None)
    e = m.rx - z @ gamma
    a = gamma[0]
    beta = gamma[1 : k + 1]  # K x N
    c = gamma[k + 1 :].T  # N x K
    sigma2 = float(np.trace(e.T @ e)) / (n_bonds * t)

    b_star = np.stack([np.outer(beta[:, i], beta[:, i]).ravel() for i in range(n_bonds)])
    gram = beta @ beta.T
    lambda1 = np.linalg.solve(gram, beta @ c)
    lambda0 = np.linalg.solve(gram, beta @ (a + 0.5 * (b_star @ sigma.ravel() + sigma2)))

    short = np.column_stack([np.ones(m.x_all.shape[0]), m.x_all])
    delta, *_ = np.linalg.lstsq(short, m.r_all, rcond=None)
    return Parameters(
        mu,
        phi,
        sigma,
        sigma2,
        float(delta[0]),
        delta[1:],
        lambda0,
        lambda1,
        radius,
        bool(constrained),
    )


def price(
    params: Parameters,
    factor_map: FactorMap,
    *,
    max_maturity: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    convention: Convention,
) -> AcmModel:
    a, b = _recursions(params, params.lambda0, params.lambda1, max_maturity, convention)
    a_rn, b_rn = _recursions(
        params,
        np.zeros_like(params.lambda0),
        np.zeros_like(params.lambda1),
        max_maturity,
        convention,
    )
    return AcmModel(
        factor_map,
        params.mu,
        params.phi,
        params.sigma,
        params.sigma2,
        params.delta0,
        params.delta1,
        params.lambda0,
        params.lambda1,
        a,
        b,
        a_rn,
        b_rn,
        start,
        end,
        params.spectral_radius,
        params.constrained,
    )


def fit(
    curve: pd.DataFrame,
    *,
    factor_maturities: list[int],
    n_factors: int,
    return_maturities: list[int],
    max_maturity: int,
    convention: Convention = "nyfed",
    max_spectral_radius: float | None = None,
) -> AcmModel:
    """Estimate the model on a monthly curve (rows: month ends, columns: maturities in months)."""
    if curve.shape[0] < 3 * (2 * n_factors + 1):
        raise ValueError("sample too short for the return regression")
    factor_map = principal_components(curve, factor_maturities, n_factors)
    params = estimate_parameters(
        moments(curve, factor_map, return_maturities), convention, max_spectral_radius
    )
    index = pd.DatetimeIndex(curve.index)
    return price(
        params,
        factor_map,
        max_maturity=max_maturity,
        start=index[0],
        end=index[-1],
        convention=convention,
    )


def bootstrap_term_premium(
    curve: pd.DataFrame,
    model: AcmModel,
    *,
    return_maturities: list[int],
    months: list[int],
    replications: int,
    block: int,
    rng: np.random.Generator,
    convention: Convention,
    max_spectral_radius: float | None = None,
) -> NDArray[np.float64]:
    """Moving-block bootstrap of steps 2-5 with the factor definition held fixed.

    The estimator is invariant to invertible affine transformations of the
    factors, so fixing the principal-component map loses nothing. Blocks of
    consecutive (X_t, X_{t+1}, rx_{t+1}, r_t) transitions keep the serial
    dependence of factors and returns together. Returns draws of the term
    premium, shape replications x T x len(months), evaluated on the actual factors.
    """
    m = moments(curve, model.factor_map, return_maturities)
    t = m.x_now.shape[0]
    n_blocks = int(np.ceil(t / block))
    x = model.factor_map.factors(curve)
    n = np.asarray(months)
    draws = np.empty((replications, x.shape[0], n.size))
    max_maturity = int(n.max())
    for i in range(replications):
        starts = rng.integers(0, t - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:t]
        sample = Moments(m.x_now[idx], m.x_next[idx], m.rx[idx], m.x_now[idx], m.r_all[:-1][idx])
        params = estimate_parameters(sample, convention, max_spectral_radius)
        a, b = _recursions(params, params.lambda0, params.lambda1, max_maturity, convention)
        a_rn, b_rn = _recursions(
            params,
            np.zeros_like(params.lambda0),
            np.zeros_like(params.lambda1),
            max_maturity,
            convention,
        )
        fitted = -(a[n] + x @ b[n].T) / n
        neutral = -(a_rn[n] + x @ b_rn[n].T) / n
        draws[i] = (fitted - neutral) * 12.0
    return draws

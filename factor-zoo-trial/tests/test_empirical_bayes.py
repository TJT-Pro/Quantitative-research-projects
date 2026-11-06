"""Empirical Bayes: posterior algebra, the OLS limit, and hyperparameter recovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from factor_zoo_trial.models import covariance, empirical_bayes, transfer


def _setup(seed: int, n: int = 40, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    m, _ = empirical_bayes.membership([f"c{i % k}" for i in range(n)])
    a = rng.standard_normal((n, n)) * 0.02
    sigma = a @ a.T + np.eye(n) * 0.01
    alpha_hat = rng.standard_normal(n) * 0.3
    return m, sigma, alpha_hat


def test_posterior_matches_the_precision_form() -> None:
    m, sigma, alpha_hat = _setup(0)
    omega = empirical_bayes.prior_covariance(m, tau_c=0.2, tau_w=0.1)
    post = empirical_bayes.posterior(alpha_hat=alpha_hat, sigma=sigma, omega=omega, mu=0.0)
    cov = np.linalg.inv(np.linalg.inv(omega) + np.linalg.inv(sigma))
    mean = cov @ np.linalg.inv(sigma) @ alpha_hat
    np.testing.assert_allclose(post.covariance, cov, atol=1e-9)
    np.testing.assert_allclose(post.mean, mean, atol=1e-9)


@settings(deadline=None, max_examples=25)
@given(seed=st.integers(0, 1000), scale=st.floats(min_value=1e2, max_value=1e4))
def test_posterior_converges_to_ols_as_prior_variance_grows(seed: int, scale: float) -> None:
    m, sigma, alpha_hat = _setup(seed)
    tight = empirical_bayes.posterior(
        alpha_hat=alpha_hat,
        sigma=sigma,
        omega=empirical_bayes.prior_covariance(m, tau_c=1.0, tau_w=1.0),
        mu=0.0,
    )
    loose = empirical_bayes.posterior(
        alpha_hat=alpha_hat,
        sigma=sigma,
        omega=empirical_bayes.prior_covariance(m, tau_c=scale, tau_w=scale),
        mu=0.0,
    )
    gap_tight = np.abs(tight.mean - alpha_hat).max()
    gap_loose = np.abs(loose.mean - alpha_hat).max()
    assert gap_loose <= gap_tight + 1e-12
    assert gap_loose < 1e-2


def test_zero_tau_w_needs_no_inverse_of_omega() -> None:
    m, sigma, alpha_hat = _setup(1)
    omega = empirical_bayes.prior_covariance(m, tau_c=0.2, tau_w=0.0)
    post = empirical_bayes.posterior(alpha_hat=alpha_hat, sigma=sigma, omega=omega, mu=0.0)
    assert np.all(np.isfinite(post.mean))


_FLOOR: dict[str, Any] = {"eigenvalue_floor": "marchenko_pastur", "minimum_floor": 0.01}


def test_hyperparameters_are_recovered() -> None:
    rng = np.random.default_rng(7)
    n, k = 150, 13
    m, _ = empirical_bayes.membership([f"c{i % k}" for i in range(n)])
    tau_c, tau_w = 0.2, 0.1
    estimates = []
    for _ in range(15):
        alpha = tau_c * (m @ rng.standard_normal(k)) + tau_w * rng.standard_normal(n)
        returns = alpha + rng.standard_normal((600, n)) * 2.9
        cov = covariance.alpha_covariance(
            returns, min_pair=24, shrinkage="schafer_strimmer", **_FLOOR
        )
        hyper = empirical_bayes.fit_hyperparameters(
            alpha_hat=returns.mean(axis=0), sigma=cov.sigma, m=m, fix_mean=True, rng=rng, starts=2
        )
        estimates.append((hyper.tau_c, hyper.tau_w))
    mean_c, mean_w = np.mean(estimates, axis=0)
    assert mean_c == pytest.approx(tau_c, rel=0.3)
    assert mean_w == pytest.approx(tau_w, rel=0.3)


def test_covariance_is_psd_when_factors_outnumber_months() -> None:
    rng = np.random.default_rng(2)
    returns = rng.standard_normal((40, 80))
    returns[:10, :20] = np.nan
    cov = covariance.alpha_covariance(returns, min_pair=24, shrinkage="schafer_strimmer", **_FLOOR)
    assert np.linalg.eigvalsh(cov.sigma).min() > 0
    assert 0.0 <= cov.intensity <= 1.0
    np.testing.assert_allclose(np.diag(cov.correlation), 1.0)


def test_transfer_density_matches_the_direct_gaussian() -> None:
    m, sigma, alpha_hat = _setup(3)
    omega = empirical_bayes.prior_covariance(m, tau_c=0.2, tau_w=0.1)
    case = transfer.CountryCase(
        name="x",
        alpha_hat=alpha_hat,
        sigma=sigma,
        us_mean=alpha_hat * 0.5,
        us_cov=omega * 0.3,
        us_prior=omega,
    )
    score = transfer.score_countries([case, case])[0]
    eta = score.eta
    direct = empirical_bayes.gaussian_log_density(
        alpha_hat, alpha_hat * 0.5, omega * 0.3 + sigma + eta**2 * np.eye(alpha_hat.size)
    )
    assert score.transfer == pytest.approx(direct, rel=1e-8)


def test_marchenko_pastur_floor() -> None:
    assert covariance.marchenko_pastur_floor(
        n_factors=100, n_months=400, minimum=0.01
    ) == pytest.approx(0.25)
    # At K = n the edge is zero, so the minimum applies.
    assert covariance.marchenko_pastur_floor(n_factors=50, n_months=50, minimum=0.01) == 0.01


def _staggered_panel(rng: np.random.Generator) -> np.ndarray:
    """Zero-alpha factors with a common component, starting on different dates, as in
    the JKP country files. Pairwise correlations from such a panel are indefinite."""
    months, k = 300, 60
    common = rng.standard_normal(months)
    returns = 0.7 * common[:, None] + rng.standard_normal((months, k))
    for j, start in enumerate(rng.integers(0, 220, size=k)):
        returns[:start, j] = np.nan
    return returns


def test_staggered_panel_alphas_are_not_treated_as_known() -> None:
    # True alphas are zero, so alpha_hat' Sigma^-1 alpha_hat / K should be near 1. The
    # numerical floor (1e-8) makes it explode: that produced per-factor log score gains
    # in the millions in the space test.
    rng = np.random.default_rng(11)
    medians = {}
    for floor in ("marchenko_pastur", "numerical"):
        quadratic = []
        for _ in range(20):
            returns = _staggered_panel(rng)
            cov = covariance.alpha_covariance(
                returns,
                min_pair=24,
                shrinkage="schafer_strimmer",
                eigenvalue_floor=floor,
                minimum_floor=0.01,
            )
            alpha_hat = np.nanmean(returns, axis=0)
            quadratic.append(float(alpha_hat @ np.linalg.solve(cov.sigma, alpha_hat)) / 60)
        medians[floor] = float(np.median(quadratic))
    assert medians["marchenko_pastur"] < 3.0
    assert medians["numerical"] > 100 * medians["marchenko_pastur"]

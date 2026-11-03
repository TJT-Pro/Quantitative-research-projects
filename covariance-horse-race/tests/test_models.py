"""Known-answer, property and recovery tests for every estimator."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from covariance_horse_race.config import ResearchConfig
from covariance_horse_race.evaluation.simulation import multivariate_t
from covariance_horse_race.models import build_model
from covariance_horse_race.models.dcc import (
    Dcc,
    GarchParams,
    dcc_last_q,
    fit_dcc,
    fit_garch,
    garch_filter,
)
from covariance_horse_race.models.factor import (
    bai_ng_ic2,
    observable_factor_covariance,
    pca_factor_covariance,
)
from covariance_horse_race.models.shrinkage import (
    analytical_nonlinear_shrinkage,
    constant_correlation_shrinkage,
)
from covariance_horse_race.models.unconditional import ewma_covariance, sample_covariance
from tests.conftest import factor_covariance


def _frob(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a - b) ** 2))


def _is_pd(m: np.ndarray) -> bool:
    return bool(np.allclose(m, m.T) and np.linalg.eigvalsh(m)[0] > 0)


windows = st.tuples(st.integers(30, 120), st.integers(2, 25), st.integers(0, 2**31 - 1))


@settings(max_examples=40, deadline=None)
@given(windows)
def test_all_unconditional_estimators_symmetric_and_psd(shape: tuple[int, int, int]) -> None:
    t, n, seed = shape
    x = np.random.default_rng(seed).standard_t(5, size=(t, n)) * 0.01
    for estimate in (
        sample_covariance(x),
        ewma_covariance(x, 0.97),
        constant_correlation_shrinkage(x).sigma,
        analytical_nonlinear_shrinkage(x),
        pca_factor_covariance(x, min(3, n - 1)),
    ):
        assert np.allclose(estimate, estimate.T)
        assert np.linalg.eigvalsh(estimate)[0] >= -1e-15


def test_sample_matches_numpy(rng: np.random.Generator) -> None:
    x = rng.normal(size=(200, 7))
    assert np.allclose(sample_covariance(x), np.cov(x, rowvar=False))


def test_ewma_weights_and_limit(rng: np.random.Generator) -> None:
    x = rng.normal(size=(300, 4))
    # As decay -> 1 the normalized weights become equal: the ML (1/T) covariance.
    t = x.shape[0]
    assert np.allclose(ewma_covariance(x, 1 - 1e-12), sample_covariance(x) * (t - 1) / t, atol=1e-9)
    # A spike in the latest observation dominates a short-memory estimate.
    y = x.copy()
    y[-1] = 25.0
    assert ewma_covariance(y, 0.5)[0, 0] > ewma_covariance(y, 0.99)[0, 0]
    with pytest.raises(ValueError, match="decay"):
        ewma_covariance(x, 1.0)


def test_linear_shrinkage_intensity_bounds_and_target(rng: np.random.Generator) -> None:
    x = rng.normal(size=(60, 20)) * 0.01
    result = constant_correlation_shrinkage(x)
    assert 0.0 <= result.intensity <= 1.0
    sd = np.sqrt(np.diag(result.target))
    corr = result.target / np.outer(sd, sd)
    off = corr[~np.eye(20, dtype=bool)]
    assert np.allclose(off, off[0]), "target has one common correlation"


def test_linear_shrinkage_beats_sample_when_truth_is_constant_correlation(
    rng: np.random.Generator,
) -> None:
    n, rho = 30, 0.4
    vols = rng.uniform(0.01, 0.03, n)
    sigma = rho * np.outer(vols, vols)
    np.fill_diagonal(sigma, vols**2)
    loss_s, loss_lw = [], []
    for _ in range(50):
        x = rng.multivariate_normal(np.zeros(n), sigma, size=60)
        loss_s.append(_frob(sample_covariance(x), sigma))
        loss_lw.append(_frob(constant_correlation_shrinkage(x).sigma, sigma))
    assert np.mean(loss_lw) < 0.7 * np.mean(loss_s)


def test_nonlinear_shrinkage_keeps_eigenvectors_and_improves(rng: np.random.Generator) -> None:
    n, t = 50, 100
    sigma = factor_covariance(n, rng)
    x = multivariate_t(sigma, t, 8.0, rng)
    shrunk = analytical_nonlinear_shrinkage(x)
    _, v_sample = np.linalg.eigh(sample_covariance(x))
    # Eigenvectors are preserved: V' H V is diagonal.
    rotated = v_sample.T @ shrunk @ v_sample
    assert np.allclose(rotated, np.diag(np.diag(rotated)), atol=1e-12 * np.abs(rotated).max())
    losses = [
        (_frob(sample_covariance(x), sigma), _frob(analytical_nonlinear_shrinkage(x), sigma))
        for x in (multivariate_t(sigma, t, 8.0, rng) for _ in range(30))
    ]
    s, nl = np.mean(losses, axis=0)
    assert nl < s


def test_nonlinear_shrinkage_defined_when_assets_exceed_observations(
    rng: np.random.Generator,
) -> None:
    x = rng.normal(size=(60, 100)) * 0.01
    estimate = analytical_nonlinear_shrinkage(x)
    assert _is_pd(estimate)
    assert np.linalg.eigvalsh(sample_covariance(x))[0] < 1e-12, "sample is singular here"


def test_nonlinear_shrinkage_rejects_tiny_window(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        analytical_nonlinear_shrinkage(rng.normal(size=(10, 3)))


def test_observable_factor_recovers_covariance(rng: np.random.Generator) -> None:
    n, k, t = 20, 3, 20000
    betas = rng.normal(1.0, 0.5, size=(n, k))
    sigma_f = np.diag([1e-4, 4e-5, 2e-5])
    idio = rng.uniform(1e-5, 5e-5, n)
    f = rng.multivariate_normal(np.zeros(k), sigma_f, size=t)
    x = f @ betas.T + rng.normal(0, np.sqrt(idio), size=(t, n))
    truth = betas @ sigma_f @ betas.T + np.diag(idio)
    estimate = observable_factor_covariance(x, f)
    assert np.max(np.abs(estimate - truth) / np.abs(truth).max()) < 0.03


def test_bai_ng_recovers_number_of_factors(rng: np.random.Generator) -> None:
    """IC_p2 is consistent for pervasive factors (loadings spread over all assets,
    factor variance comparable to idiosyncratic variance), which is the setting here."""
    n, t = 60, 500
    for k in (1, 3):
        loadings = rng.normal(0.0, 1.0, size=(n, k))
        factor_var = np.linspace(1e-4, 5e-5, k)
        idio = rng.uniform(2e-5, 6e-5, n)
        sigma = (loadings * factor_var) @ loadings.T + np.diag(idio)
        x = rng.multivariate_normal(np.zeros(n), sigma, size=t)
        assert bai_ng_ic2(x, 10) == k


def test_factory_builds_every_configured_model(
    config: ResearchConfig, rng: np.random.Generator
) -> None:
    x = rng.normal(size=(252, 10)) * 0.01
    f = rng.normal(size=(252, 6)) * 0.01
    for spec in config.models:
        model = build_model(spec)
        model.fit(
            x, f[:, : len(config.factors.extended.columns)] if model.requires_factors else None
        )
        h = model.forecast(21)
        assert h.shape == (10, 10)
        assert _is_pd(h)


def _simulate_garch(
    t: int, omega: float, alpha: float, beta: float, rng: np.random.Generator
) -> np.ndarray:
    h = omega / (1 - alpha - beta)
    out = np.empty(t)
    for i in range(t):
        out[i] = np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * out[i] ** 2 + beta * h
    return out


def test_garch_filter_matches_recursion() -> None:
    eps = np.array([0.01, -0.02, 0.015])
    h = garch_filter(eps, 1e-6, 0.1, 0.8, 2e-4)
    assert h[1] == pytest.approx(1e-6 + 0.1 * 1e-4 + 0.8 * 2e-4)
    assert h.shape == (4,)


@pytest.mark.slow
def test_garch_parameter_recovery() -> None:
    rng = np.random.default_rng(3)
    alpha, beta, uncond = 0.08, 0.90, 1e-4
    estimates = []
    for _ in range(8):
        eps = _simulate_garch(4000, uncond * (1 - alpha - beta), alpha, beta, rng)
        g = fit_garch(eps)
        estimates.append((g.alpha, g.beta))
    a_hat, b_hat = np.mean(estimates, axis=0)
    assert abs(a_hat - alpha) < 0.015
    assert abs(b_hat - beta) < 0.02


@pytest.mark.slow
def test_dcc_parameter_recovery() -> None:
    """Simulate a scalar DCC on standardized residuals and recover (a, b)."""
    rng = np.random.default_rng(5)
    n, t, a, b = 8, 4000, 0.04, 0.93
    base = 0.3 * np.ones((n, n)) + 0.7 * np.eye(n)
    q = base.copy()
    z = np.empty((t, n))
    for s in range(t):
        d = np.sqrt(np.diag(q))
        r = q / np.outer(d, d)
        z[s] = np.linalg.cholesky(r) @ rng.standard_normal(n)
        q = (1 - a - b) * base + a * np.outer(z[s], z[s]) + b * q
    c = z.T @ z / t
    d = np.sqrt(np.diag(c))
    a_hat, b_hat = fit_dcc(np.ascontiguousarray(z), c / np.outer(d, d))
    assert abs(a_hat - a) < 0.015
    assert abs(b_hat - b) < 0.03


def test_dcc_last_q_is_correlation_targeting_identity() -> None:
    z = np.zeros((5, 3))
    c = np.eye(3)
    assert np.allclose(
        dcc_last_q(z, c, 0.05, 0.9),
        (1 - 0.05 - 0.9) * c * (1 + 0.9 + 0.81 + 0.729 + 0.6561) + 0.9**5 * c,
    )


def test_dcc_forecast_converges_to_unconditional(rng: np.random.Generator) -> None:
    x = rng.standard_t(6, size=(500, 5)) * 0.01
    model = Dcc()
    model.fit(x)
    short, long = model.forecast(1), model.forecast(5000)
    assert model.garch is not None
    assert model.dcc_params is not None
    uncond = np.array([g.unconditional for g in model.garch])
    # Long-horizon average variance approaches the unconditional variance.
    assert np.allclose(np.diag(long), uncond, rtol=0.05)
    assert _is_pd(short)
    assert _is_pd(long)


def test_dcc_reestimation_cadence(rng: np.random.Generator) -> None:
    x = rng.standard_t(6, size=(300, 4)) * 0.01
    model = Dcc(reestimate_every=3)
    model.fit(x)
    params = model.dcc_params
    model.fit(x[1:])
    assert model.dcc_params == params, "parameters frozen between re-estimations"
    model.fit(x[2:])
    model.fit(x[3:])
    assert model._fits_since_estimate == 1


@pytest.mark.parametrize(("alpha", "beta"), [(0.2, 0.0), (0.0, 0.9), (0.0, 0.0), (0.5, 1e-14)])
def test_warm_start_on_parameter_boundary(
    alpha: float, beta: float, rng: np.random.Generator
) -> None:
    """Regression: a boundary estimate from last month must not break the next fit."""
    eps = rng.standard_t(6, size=400) * 0.01
    g = fit_garch(eps, GarchParams(omega=1e-6, alpha=alpha, beta=beta))
    assert g.alpha >= 0
    assert g.beta >= 0
    assert g.alpha + g.beta < 1
    z = rng.standard_normal((300, 4))
    a, b = fit_dcc(z, np.eye(4), (alpha, beta))
    assert a + b < 1

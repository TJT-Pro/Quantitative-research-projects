"""Estimator correctness: recovery of a known model, invariances, conventions, bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from acm_term_premium.models import acm
from tests.conftest import simulate_affine

RETURNS = [6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
FACTOR_MATS = list(range(3, 121))


def _true_term_premium(truth: acm.Parameters, x: np.ndarray, n: int) -> np.ndarray:
    a, b = acm._recursions(truth, truth.lambda0, truth.lambda1, n, "paper")
    a0, b0 = acm._recursions(truth, 0 * truth.lambda0, 0 * truth.lambda1, n, "paper")
    out: np.ndarray = (-(a[n] + x @ b[n]) + (a0[n] + x @ b0[n])) / n * 12.0
    return out


def test_recovers_known_term_premium(rng: np.random.Generator) -> None:
    curve, truth, x = simulate_affine(rng, months=3000)
    model = acm.fit(
        curve,
        factor_maturities=FACTOR_MATS,
        n_factors=3,
        return_maturities=RETURNS,
        max_maturity=120,
        convention="paper",
    )
    estimated = model.decompose(curve, [120])["term_premium"][120].to_numpy()
    true = _true_term_premium(truth, x, 120)
    # Movements are recovered almost exactly; the level carries the usual
    # small-sample bias of OLS VAR persistence, bounded here at a quarter.
    assert np.corrcoef(estimated, true)[0, 1] > 0.99
    assert abs(estimated.mean() - true.mean()) < 0.25 * abs(true.mean())


def test_fitted_yields_price_the_curve(rng: np.random.Generator) -> None:
    curve, _, _ = simulate_affine(rng)
    model = acm.fit(
        curve,
        factor_maturities=FACTOR_MATS,
        n_factors=3,
        return_maturities=RETURNS,
        max_maturity=120,
    )
    fitted = model.decompose(curve, [12, 60, 120])["fitted"]
    rmse_bp = np.sqrt(((fitted - curve[[12, 60, 120]]) ** 2).to_numpy().mean()) * 1e4
    assert rmse_bp < 1.0, "pricing error is at the level of the 0.5 bp measurement noise"


def test_conventions_agree_on_fitted_yields(rng: np.random.Generator) -> None:
    curve, _, _ = simulate_affine(rng)
    a = acm.fit(
        curve,
        factor_maturities=FACTOR_MATS,
        n_factors=3,
        return_maturities=RETURNS,
        max_maturity=120,
        convention="nyfed",
    ).decompose(curve, [24, 120])
    b = acm.fit(
        curve,
        factor_maturities=FACTOR_MATS,
        n_factors=3,
        return_maturities=RETURNS,
        max_maturity=120,
        convention="paper",
    ).decompose(curve, [24, 120])
    assert np.allclose(a["fitted"], b["fitted"], atol=1e-9)
    assert not np.allclose(a["term_premium"], b["term_premium"], atol=1e-9)


def test_invariant_to_affine_factor_rotation(rng: np.random.Generator) -> None:
    """Rotating and shifting the factors leaves every yield decomposition unchanged."""
    curve, _, _ = simulate_affine(rng)
    base = acm.principal_components(curve, FACTOR_MATS, 3)
    rotation = rng.normal(size=(3, 3)) + 3 * np.eye(3)
    rotated = acm.FactorMap(
        base.maturities, base.mean - 0.01, base.loadings @ rotation, base.explained
    )
    results = []
    for fmap in (base, rotated):
        params = acm.estimate_parameters(acm.moments(curve, fmap, RETURNS), "paper")
        model = acm.price(
            params,
            fmap,
            max_maturity=120,
            start=curve.index[0],
            end=curve.index[-1],
            convention="paper",
        )
        results.append(model.decompose(curve, [120])["term_premium"].to_numpy())
    assert np.allclose(results[0], results[1], atol=1e-10)


def test_excess_returns_alignment() -> None:
    dates = pd.date_range("2000-01-31", periods=3, freq="ME")
    curve = pd.DataFrame({1: [0.012, 0.024, 0.036], 2: [0.024, 0.036, 0.048]}, index=dates)
    rx = acm.excess_returns(curve, [2])
    # p1 at t+1 minus p2 at t minus r_t, labelled at t+1.
    expected = -0.024 / 12 - (-2 * 0.024 / 12) - 0.012 / 12
    assert rx.index[0] == dates[1]
    assert rx.iloc[0, 0] == pytest.approx(expected)


@settings(max_examples=50, deadline=None)
@given(st.floats(0.5, 0.999), st.integers(0, 2**31 - 1))
def test_spectral_bound_holds(bound: float, seed: int) -> None:
    phi = np.random.default_rng(seed).normal(0, 0.6, (5, 5))
    out = acm.constrain_spectral_radius(phi, bound)
    assert np.abs(np.linalg.eigvals(out)).max() <= bound + 1e-9
    if np.abs(np.linalg.eigvals(phi)).max() <= bound:
        assert np.allclose(out, phi)


def test_explosive_window_is_constrained(rng: np.random.Generator) -> None:
    curve, _, _ = simulate_affine(rng)
    fmap = acm.principal_components(curve, FACTOR_MATS, 3)
    m = acm.moments(curve, fmap, RETURNS)
    # Replace the factors with an explosive path (root 1.02 on every factor).
    x = np.zeros_like(m.x_all)
    for t in range(1, x.shape[0]):
        x[t] = 1.02 * x[t - 1] + rng.normal(0.0, 0.1, x.shape[1])
    explosive = acm.Moments(x[:-1], x[1:], m.rx, x, m.r_all)
    free = acm.estimate_parameters(explosive, "nyfed")
    bound = acm.estimate_parameters(explosive, "nyfed", 0.995)
    assert free.spectral_radius > 1.0
    assert bound.constrained
    assert np.abs(np.linalg.eigvals(bound.phi)).max() <= 0.995 + 1e-9


def test_bootstrap_shape_and_spread(rng: np.random.Generator) -> None:
    curve, _, _ = simulate_affine(rng)
    model = acm.fit(
        curve,
        factor_maturities=FACTOR_MATS,
        n_factors=3,
        return_maturities=RETURNS,
        max_maturity=120,
    )
    draws = acm.bootstrap_term_premium(
        curve,
        model,
        return_maturities=RETURNS,
        months=[60, 120],
        replications=40,
        block=24,
        rng=rng,
        convention="nyfed",
        max_spectral_radius=0.995,
    )
    assert draws.shape == (40, len(curve), 2)
    point = model.decompose(curve, [120])["term_premium"][120].to_numpy()
    lower, upper = np.quantile(draws[:, :, 1], [0.05, 0.95], axis=0)
    assert np.mean((point >= lower) & (point <= upper)) > 0.8


def test_rejects_short_sample(rng: np.random.Generator) -> None:
    curve, _, _ = simulate_affine(rng, months=12)
    with pytest.raises(ValueError, match="too short"):
        acm.fit(
            curve,
            factor_maturities=FACTOR_MATS,
            n_factors=3,
            return_maturities=RETURNS,
            max_maturity=120,
        )

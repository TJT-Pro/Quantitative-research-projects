"""Standard errors: EWC size, Newey-West at zero lags, and the fixed-effects sandwich."""

from __future__ import annotations

import numpy as np
import pytest

from factor_zoo_trial.models.inference import (
    ewc_mean,
    ewc_terms,
    newey_west_mean,
    one_sided_p,
)
from factor_zoo_trial.models.panel import fit_fixed_effects


def test_newey_west_zero_lags_is_the_textbook_standard_error() -> None:
    x = np.random.default_rng(0).standard_normal(250)
    test = newey_west_mean(x, lags=0)
    assert test.se == pytest.approx(x.std(ddof=1) / np.sqrt(x.size))


def test_ewc_terms_follow_the_rule() -> None:
    assert ewc_terms(360, coefficient=0.4) == round(0.4 * 360 ** (2 / 3))
    assert ewc_terms(5, coefficient=0.4) >= 1


def test_ewc_keeps_its_size_in_the_tail_where_newey_west_does_not() -> None:
    rng = np.random.default_rng(1)
    reps, n, level = 6000, 360, 0.005
    ewc, nw = [], []
    for _ in range(reps):
        x = rng.standard_normal(n)
        a = ewc_mean(x, coefficient=0.4)
        b = newey_west_mean(x, lags=12)
        ewc.append(one_sided_p(np.array([a.t]), np.array([a.df]))[0])
        nw.append(one_sided_p(np.array([b.t]), np.array([b.df]))[0])
    se = np.sqrt(level * (1 - level) / reps)
    assert np.mean(np.array(ewc) < level) <= level + 3 * se
    assert np.mean(np.array(nw) < level) > level + 3 * se


def _dense_fit(
    *, y: np.ndarray, groups: np.ndarray, clusters: np.ndarray, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    g = groups.max() + 1
    d = np.column_stack([np.eye(g)[groups], x])
    bread = np.linalg.inv(d.T @ d)
    theta = bread @ d.T @ y
    e = y - d @ theta
    _, codes = np.unique(clusters, return_inverse=True)
    scores = np.zeros((codes.max() + 1, d.shape[1]))
    np.add.at(scores, codes, d * e[:, None])
    n, p, c = y.size, d.shape[1], codes.max() + 1
    cov = (c / (c - 1)) * ((n - 1) / (n - p)) * bread @ scores.T @ scores @ bread
    return theta, cov


def test_fixed_effects_match_the_dense_dummy_regression() -> None:
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(6), 40)
    clusters = np.tile(np.arange(40), 6)
    x = rng.standard_normal((groups.size, 2))
    common = rng.standard_normal(40)[clusters]
    y = (
        np.arange(6)[groups] * 0.1
        + x @ np.array([0.5, -0.3])
        + common
        + rng.standard_normal(groups.size)
    )
    fit = fit_fixed_effects(y=y, groups=groups, clusters=clusters, regressors=x)
    theta, cov = _dense_fit(y=y, groups=groups, clusters=clusters, x=x)
    np.testing.assert_allclose(np.concatenate([fit.effects, fit.slopes]), theta, atol=1e-10)
    np.testing.assert_allclose(fit.covariance, cov, atol=1e-10)


def test_fixed_effects_recover_a_known_decay() -> None:
    rng = np.random.default_rng(4)
    n_groups, months = 40, 300
    groups = np.repeat(np.arange(n_groups), months)
    clusters = np.tile(np.arange(months), n_groups)
    post = (clusters > 200).astype(float)
    mu = rng.uniform(0.3, 0.9, n_groups)
    y = mu[groups] * (1 - 0.5 * post) + rng.standard_normal(groups.size)
    x = (post * mu.mean()).reshape(-1, 1)
    fit = fit_fixed_effects(y=y, groups=groups, clusters=clusters, regressors=x)
    assert fit.slopes[0] == pytest.approx(-0.5, abs=0.1)

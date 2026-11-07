"""The model against the New York Fed's code, against theory and against planted truths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize

from realtime_nowcast import config as config_mod
from realtime_nowcast.config import SeriesConfig
from realtime_nowcast.data import fixture, vintages
from realtime_nowcast.evaluation import market, realtime, score
from realtime_nowcast.models import dfm, kalman, news, simulate

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference" / "frbny"


def _cfg() -> config_mod.Config:
    return config_mod.load(path=ROOT / "configs" / "research.yaml")


# The New York Fed's code ---------------------------------------------------------------------


def _reference_panel() -> tuple[np.ndarray, pd.PeriodIndex, dfm.Structure]:
    spec = pd.read_csv(REFERENCE / "Spec_US_example.csv")
    spec = spec.loc[spec["Model"] == 1]
    spec = pd.concat([spec.loc[spec["Frequency"] == "m"], spec.loc[spec["Frequency"] == "q"]])
    specs = tuple(
        SeriesConfig.model_validate(
            {
                "id": str(r["SeriesID"]),
                "name": str(r["SeriesName"]),
                "frequency": str(r["Frequency"]),
                "transform": str(r["Transformation"]),
                "blocks": ("global",),
                "category": str(r["Category"]),
            }
        )
        for r in spec.to_dict("records")
    )
    data = pd.read_csv(REFERENCE / "2016-06-29.csv", parse_dates=["Date"]).set_index("Date")
    levels = data[[s.id for s in specs]].to_numpy(dtype=float)
    x = vintages.transform(levels, specs)[3:]  # load_data drops the first three months
    dates = data.index[3:]
    keep = dates >= pd.Timestamp("2000-01-01")  # example_DFM.m's sample_start
    blocks = spec[[c for c in spec.columns if c.startswith("Block")]].to_numpy(dtype=np.int64)
    structure = dfm.Structure(
        n_monthly=int((spec["Frequency"] == "m").sum()),
        n_quarterly=int((spec["Frequency"] == "q").sum()),
        blocks=blocks,
    )
    return x[keep], pd.PeriodIndex(dates[keep], freq="M"), structure


def test_em_matches_the_new_york_fed_code() -> None:
    x, periods, structure = _reference_panel()
    params = dfm.estimate(
        x,
        structure,
        periods=periods,
        settings=dfm.Settings(threshold=1e-4, max_iter=5000, measurement_variance=1e-4),
    )
    ref = sio.loadmat(REFERENCE / "ResDFM.mat", squeeze_me=True, struct_as_record=False)["Res"]
    for mine, theirs in (
        (params.c, ref.C),
        (params.a, ref.A),
        (params.q, ref.Q),
        (params.r, ref.R),
        (params.mx, ref.Mx),
        (params.wx, ref.Wx),
        (params.z0, ref.Z_0),
        (params.v0, ref.V_0),
    ):
        assert np.max(np.abs(mine - theirs)) < 1e-12


# The Kalman smoother against the Gaussian conditional expectation ----------------------------


def test_smoother_equals_the_conditional_expectation_with_missing_data() -> None:
    rng = np.random.default_rng(0)
    m, n, periods = 2, 3, 10
    a = np.array([[0.7, 0.1], [0.0, 0.5]])
    q = np.diag([0.6, 0.3])
    c = rng.normal(size=(n, m))
    r = np.diag([0.2, 0.1, 0.3])
    v0 = np.eye(m)
    y = rng.normal(size=(n, periods))
    y[rng.random(size=y.shape) < 0.3] = np.nan
    sm = kalman.smooth(y, a=a, c=c, q=q, r=r, z0=np.zeros(m), v0=v0)
    # Joint covariance of (z_1..z_T) and of the observed y, built directly.
    powers = [np.linalg.matrix_power(a, k) for k in range(periods + 1)]
    cov_z = np.zeros((periods * m, periods * m))
    for s in range(periods):
        for t in range(periods):
            total = powers[s + 1] @ v0 @ powers[t + 1].T
            for k in range(min(s, t) + 1):
                total += powers[s - k] @ q @ powers[t - k].T
            cov_z[s * m : (s + 1) * m, t * m : (t + 1) * m] = total
    big_c = np.kron(np.eye(periods), c)
    seen = ~np.isnan(y.T.reshape(-1))
    cov_y = big_c @ cov_z @ big_c.T + np.kron(np.eye(periods), r)
    gain = cov_z @ big_c.T[:, seen] @ np.linalg.inv(cov_y[np.ix_(seen, seen)])
    expected = (gain @ y.T.reshape(-1)[seen]).reshape(periods, m).T
    assert np.max(np.abs(sm.z[:, 1:] - expected)) < 1e-10


# Recovery ----------------------------------------------------------------------------------


def test_em_recovers_the_factor_from_a_ragged_panel() -> None:
    cfg = _cfg()
    structure = realtime.structure_of(cfg)
    rng = np.random.default_rng(11)
    truth = simulate.truth(structure, rng)
    periods = pd.period_range("2003-01", periods=180, freq="M")
    y, states = simulate.draw(structure, truth, periods=len(periods), rng=rng)
    y[:36, 1] = np.nan  # a series that starts late
    y[rng.random(size=y.shape) < 0.03] = np.nan  # scattered gaps
    for k, lag in enumerate(rng.integers(0, 3, size=structure.n_monthly)):
        if lag:
            y[-lag:, k] = np.nan  # the ragged edge
    params = dfm.estimate(
        y,
        structure,
        periods=periods,
        settings=dfm.Settings(threshold=1e-5, max_iter=2000, measurement_variance=1e-4),
    )
    fit = news.fit(y, params, periods)
    zs = fit.smoothed(1)
    assert abs(np.corrcoef(zs[0, 1:], states[:, 0])[0, 1]) > 0.9
    g = structure.n_monthly
    common = states[:, : 5 * structure.n_blocks] @ truth.c[g, : 5 * structure.n_blocks]
    estimated = zs[: 5 * structure.n_blocks, 1:].T @ params.c[g, : 5 * structure.n_blocks]
    seen = ~np.isnan(y[:, g])
    assert np.corrcoef(common[seen], estimated[seen])[0, 1] > 0.9


def test_pandemic_scales_are_recovered_and_maximize_the_likelihood() -> None:
    structure = dfm.Structure(n_monthly=8, n_quarterly=1, blocks=np.ones((9, 1), dtype=np.int64))
    rng = np.random.default_rng(5)
    truth = simulate.truth(structure, rng)
    periods = pd.period_range("2005-01", "2022-12", freq="M")
    free = tuple(pd.Period(p, freq="M") for p in ("2020-03", "2020-04", "2020-05"))
    planted = dfm.Pandemic(free=free, scales=(8.0, 5.0, 3.0), rho=0.6)
    y, _ = simulate.draw(
        structure, truth, periods=len(periods), rng=rng, scale=planted.scale(periods)
    )
    p = dfm.estimate(
        y,
        structure,
        periods=periods,
        settings=dfm.Settings(
            threshold=1e-9, max_iter=800, measurement_variance=1e-4, pandemic_free=free
        ),
    )
    got = np.array(p.pandemic.scales)
    assert np.all(np.abs(np.log(got / np.array(planted.scales))) < np.log(1.4))
    assert 0.3 < p.pandemic.rho < 0.8
    # The EM step for the scales lands on the maximum of the likelihood itself, holding the
    # other parameters at their estimates.
    xs = (y - p.mx) / p.wx

    def nll(theta: np.ndarray) -> float:
        pan = dfm.Pandemic(
            free=free, scales=tuple(np.exp(theta[:3])), rho=float(1 / (1 + np.exp(-theta[3])))
        )
        f = kalman.run_filter(
            xs.T, a=p.a, c=p.c, q=p.q, r=p.r, z0=p.z0, v0=p.v0, scale=pan.scale(periods)
        )
        return -f.loglik

    start = np.r_[np.log(got), np.log(p.pandemic.rho / (1 - p.pandemic.rho))]
    best = minimize(
        nll, start, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000}
    )
    assert nll(start) - best.fun < 1e-6


def test_accelerated_em_climbs_at_least_as_high_in_fewer_steps() -> None:
    structure = dfm.Structure(n_monthly=8, n_quarterly=1, blocks=np.ones((9, 1), dtype=np.int64))
    rng = np.random.default_rng(11)
    truth = simulate.truth(structure, rng)
    periods = pd.period_range("2008-01", "2019-12", freq="M")
    y, _ = simulate.draw(structure, truth, periods=len(periods), rng=rng)
    y[rng.random(size=y.shape) < 0.05] = np.nan
    plain, fast = (
        dfm.estimate(
            y,
            structure,
            periods=periods,
            settings=dfm.Settings(
                threshold=1e-6, max_iter=20000, measurement_variance=1e-4, accelerate=accelerate
            ),
        )
        for accelerate in (False, True)
    )
    assert plain.converged
    assert fast.converged
    # Plain EM creeps: its steps fall under the threshold while the likelihood still has room
    # to rise. The accelerated run passes the height plain EM stopped at in far fewer EM steps,
    # and stops higher.
    path = np.array(fast.loglik_path)
    reached = int(np.flatnonzero(path >= plain.loglik)[0]) + 1
    assert reached < plain.iterations / 3
    assert fast.loglik > plain.loglik
    # Every log likelihood the accelerated EM keeps is no lower than the one before it.
    kept = np.array(fast.loglik_path)[2::3]
    assert np.all(np.diff(kept) >= -1e-9 * np.abs(kept[1:]))


def test_a_free_month_with_no_data_keeps_its_scale_at_one() -> None:
    structure = dfm.Structure(n_monthly=8, n_quarterly=1, blocks=np.ones((9, 1), dtype=np.int64))
    rng = np.random.default_rng(7)
    truth = simulate.truth(structure, rng)
    periods = pd.period_range("2005-01", "2020-05", freq="M")
    free = tuple(pd.Period(p, freq="M") for p in ("2020-03", "2020-04", "2020-05"))
    planted = dfm.Pandemic(free=free, scales=(8.0, 5.0, 3.0), rho=0.6)
    y, _ = simulate.draw(
        structure, truth, periods=len(periods), rng=rng, scale=planted.scale(periods)
    )
    y[-1, 1:] = np.nan  # May 2020 as on 2020-06-01: one series out, the rest unpublished
    p = dfm.estimate(
        y,
        structure,
        periods=periods,
        settings=dfm.Settings(
            threshold=1e-5,
            max_iter=3000,
            measurement_variance=1e-4,
            pandemic_free=free,
            accelerate=True,
        ),
    )
    assert p.converged
    assert min(p.pandemic.scales) >= 1.0
    assert p.pandemic.scales[1] > 2.0  # the observed April shock still shows


def test_units_come_from_the_months_before_the_pandemic() -> None:
    structure = dfm.Structure(n_monthly=8, n_quarterly=1, blocks=np.ones((9, 1), dtype=np.int64))
    rng = np.random.default_rng(3)
    truth = simulate.truth(structure, rng)
    periods = pd.period_range("2010-01", "2021-12", freq="M")
    free = tuple(pd.Period(p, freq="M") for p in ("2020-03", "2020-04", "2020-05"))
    planted = dfm.Pandemic(free=free, scales=(8.0, 40.0, 3.0), rho=0.6)
    y, _ = simulate.draw(
        structure, truth, periods=len(periods), rng=rng, scale=planted.scale(periods)
    )
    p = dfm.estimate(
        y,
        structure,
        periods=periods,
        settings=dfm.Settings(
            threshold=1e-5,
            max_iter=3000,
            measurement_variance=1e-4,
            pandemic_free=free,
            accelerate=True,
        ),
    )
    before = np.asarray(periods < free[0])
    np.testing.assert_allclose(p.wx, np.nanstd(y[before], axis=0, ddof=1))
    np.testing.assert_allclose(p.mx, np.nanmean(y[before], axis=0))
    assert p.converged


def test_shock_moments_match_their_definition() -> None:
    rng = np.random.default_rng(0)
    m, periods = 7, 20
    z = rng.normal(size=(m, periods + 1))
    v = rng.normal(size=(periods + 1, m, m))
    v = v @ v.transpose(0, 2, 1)
    vv = rng.normal(size=(periods, m, m))
    a = rng.normal(size=(m, m))
    q = np.diag([1.0, 0.0, 2.0, 0.0, 0.5, 1.0, 0.0])
    shocked = np.flatnonzero(np.diag(q) > 0)
    expected = []
    for t in range(periods):
        e11 = np.outer(z[:, t + 1], z[:, t + 1]) + v[t + 1]
        e00 = np.outer(z[:, t], z[:, t]) + v[t]
        e10 = np.outer(z[:, t + 1], z[:, t]) + vv[t]
        mt = e11 - a @ e10.T - e10 @ a.T + a @ e00 @ a.T
        expected.append(sum(mt[i, i] / q[i, i] for i in shocked))
    got = dfm._shock_terms(dfm._Moments(z=z, v=v, vv=vv), a, q)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-10)


def test_without_pandemic_months_the_scales_stay_off() -> None:
    p = dfm.Pandemic(free=(), scales=(), rho=0.0)
    assert np.all(p.scale(pd.period_range("2019-01", "2021-01", freq="M")) == 1.0)


# News ---------------------------------------------------------------------------------------


def _known_model() -> tuple[dfm.Params, np.ndarray, pd.PeriodIndex, dfm.Structure]:
    cfg = _cfg()
    structure = realtime.structure_of(cfg)
    rng = np.random.default_rng(2)
    truth = simulate.truth(structure, rng)
    periods = pd.period_range("2010-01", periods=96, freq="M")
    y, _ = simulate.draw(structure, truth, periods=len(periods), rng=rng)
    m = truth.a.shape[0]
    params = dfm.Params(
        a=truth.a,
        c=truth.c,
        q=truth.q,
        r=np.eye(structure.n) * 1e-4,
        z0=np.zeros(m),
        v0=np.eye(m),
        mx=np.zeros(structure.n),
        wx=np.ones(structure.n),
        pandemic=dfm.Pandemic(free=(), scales=(), rho=0.0),
        loglik=0.0,
        iterations=0,
        converged=True,
    )
    return params, y, periods, structure


@pytest.mark.parametrize("backcast", [False, True])
def test_news_adds_up_and_weights_are_derivatives(*, backcast: bool) -> None:
    params, y, periods, structure = _known_model()
    g = structure.n_monthly
    last = y.shape[0] - 1
    row = last - 3 if backcast else last  # a quarter-end row: the earlier quarter or the last
    y[row:, g] = np.nan
    new = y.copy()
    new[-2:, :] = np.nan
    old = new.copy()
    old[-4:-2, :5] = np.nan  # five series' last two months are the releases
    new[-10, 3] += 0.4  # and one old value is revised
    rev = np.where(np.isnan(old), np.nan, new)
    before = news.value_at(news.fit(old, params, periods), series=g, row=row)
    ch = news.decompose(
        old=before,
        rev=news.fit(rev, params, periods),
        new=news.fit(new, params, periods),
        x_old=old,
        x_new=new,
        target=g,
        row=row,
    )
    assert abs(ch.residual) < 1e-10
    assert len(ch.releases) == 10
    assert ch.revisions != 0.0
    r0 = ch.releases[0]
    bumped = new.copy()
    bumped[r0.row, r0.series] += 1.0
    after = news.value_at(news.fit(bumped, params, periods), series=g, row=row)
    assert after - ch.new == pytest.approx(r0.weight, abs=1e-10)


# Transformations ------------------------------------------------------------------------------


@settings(deadline=None, max_examples=30)
@given(
    transform=st.sampled_from(["chg", "pch", "lin"]),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_fixture_levels_invert_the_transformation(*, transform: str, seed: int) -> None:
    spec = SeriesConfig.model_validate(
        {
            "id": "X",
            "name": "X",
            "frequency": "m",
            "transform": transform,
            "blocks": ("global",),
            "category": "c",
        }
    )
    x = np.random.default_rng(seed).normal(0.2, 0.5, size=40)
    x[0] = np.nan
    back = vintages.transform(fixture._levels(spec, x)[:, None], (spec,))[:, 0]
    if transform == "lin":
        np.testing.assert_allclose(back[1:], x[1:], rtol=1e-10)
    else:
        np.testing.assert_allclose(back[1:], x[1:], rtol=1e-8, atol=1e-10)


def test_quarterly_growth_is_annualized_on_third_months() -> None:
    spec = SeriesConfig(
        id="Q", name="Q", frequency="q", transform="pca", blocks=("global",), category="c"
    )
    levels = np.full(9, np.nan)
    levels[[2, 5, 8]] = [100.0, 101.0, 100.5]
    out = vintages.transform(levels[:, None], (spec,))[:, 0]
    assert np.isnan(out[[0, 1, 2, 3, 4, 6, 7]]).all()
    assert out[5] == pytest.approx(100 * (1.01**4 - 1))
    assert out[8] == pytest.approx(100 * ((100.5 / 101.0) ** 4 - 1))


# Scoring and the market test ------------------------------------------------------------------


def test_diebold_mariano() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=40) ** 2
    b = a + 1.0 + 0.1 * rng.normal(size=40)
    dm = score.diebold_mariano(a, b)
    assert dm.statistic < -10
    assert dm.p_value < 1e-6
    flipped = score.diebold_mariano(b, a)
    assert flipped.statistic == pytest.approx(-dm.statistic)
    assert np.isnan(score.diebold_mariano(a, a).statistic)


def test_hc3_regression_recovers_a_slope() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=500)
    y = 2.0 + 3.0 * x + rng.normal(size=500) * (1 + np.abs(x))
    fit = market.ols_hc3(x, y)
    assert fit["ci_low"] < 3.0 < fit["ci_high"]
    assert fit["t"] > 10


def test_binned_keeps_the_pair_closest_to_release() -> None:
    pairs = pd.DataFrame(
        {
            "quarter": ["2015Q1"] * 4,
            "date": pd.to_datetime(["2015-03-01", "2015-04-01", "2015-04-20", "2015-04-28"]),
            "horizon": [60, 29, 10, 2],
            "model": 1.0,
            "gdpnow": 1.0,
            "advance": 1.0,
            "model_error": 0.0,
            "gdpnow_error": 0.0,
        }
    )
    picked = score.binned(pairs, (0, 15, 30, 75))
    assert list(picked["horizon"]) == [2, 29, 60]

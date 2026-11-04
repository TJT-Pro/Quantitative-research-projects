"""Forecasts use only information available at the forecast date; events use the prior model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acm_term_premium.evaluation import events, forecast
from acm_term_premium.models import acm
from tests.conftest import simulate_affine

RETURNS = [6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]


def _evaluate(
    curve: pd.DataFrame, tp: pd.DataFrame
) -> tuple[list[forecast.ForecastResult], pd.DataFrame]:
    return forecast.evaluate(
        curve,
        tp,
        maturities=[24, 60, 120],
        horizon=12,
        cp_forward_years=[1, 2, 3, 4, 5],
        cp_target_years=[2, 3, 4, 5],
        min_train=60,
        nw_lags=12,
    )


def test_forecasts_ignore_the_future(rng: np.random.Generator) -> None:
    curve, _, _ = simulate_affine(rng, months=300)
    tp = pd.DataFrame(rng.normal(0.01, 0.005, (300, 3)), index=curve.index, columns=[24, 60, 120])
    _, full = _evaluate(curve, tp)
    cut = curve.index[200]
    # Scramble everything after the cut: forecasts made up to the cut must not move.
    scrambled = curve.copy()
    scrambled.loc[scrambled.index > cut] *= 3.0
    tp_scrambled = tp.copy()
    tp_scrambled.loc[tp_scrambled.index > cut] = -1.0
    _, other = _evaluate(scrambled, tp_scrambled)
    cols = ["mean", "cp", "acm"]
    a = full[full["date"] <= cut].set_index(["date", "maturity"])[cols]
    b = other[other["date"] <= cut].set_index(["date", "maturity"])[cols]
    assert len(a) > 100
    pd.testing.assert_frame_equal(a, b)


def test_excess_return_is_labelled_when_realized(rng: np.random.Generator) -> None:
    curve, _, _ = simulate_affine(rng, months=60)
    rx = forecast.excess_return(curve, 24, 12)
    p = acm.log_prices(curve)
    t, realized = curve.index[10], curve.index[22]
    expected = p.loc[realized, 12] - p.loc[t, 24] - curve.loc[t, 12]
    assert rx[realized] == pytest.approx(expected)
    assert rx.index[0] == curve.index[12]


def test_newey_west_without_lags_is_the_iid_standard_error(rng: np.random.Generator) -> None:
    x = rng.normal(size=500)
    expected = np.sqrt(((x - x.mean()) ** 2).mean() / x.size)
    assert forecast.newey_west_mean_se(x, 0) == pytest.approx(expected)
    # Positive autocorrelation widens the HAC error.
    ar = np.zeros(2_000)
    for i in range(1, ar.size):
        ar[i] = 0.8 * ar[i - 1] + rng.normal()
    assert forecast.newey_west_mean_se(ar, 12) > 2 * forecast.newey_west_mean_se(ar, 0)


def test_policy_day_uses_the_model_from_the_previous_month(rng: np.random.Generator) -> None:
    curve, _, _ = simulate_affine(rng, months=200)
    model = acm.fit(
        curve,
        factor_maturities=list(range(3, 121)),
        n_factors=3,
        return_maturities=RETURNS,
        max_maturity=120,
    )
    ends = list(curve.index[150:153])
    models = dict.fromkeys(ends, model)
    # A day inside the month after the second end uses that end, never its own month.
    day = ends[1] + pd.Timedelta(days=10)
    assert events._model_for(day, models, ends) is models[ends[1]]
    assert events._model_for(ends[1], models, ends) is models[ends[0]]
    assert events._model_for(ends[0], models, ends) is None


def test_event_decomposition_adds_up(rng: np.random.Generator) -> None:
    curve, _, _ = simulate_affine(rng, months=200)
    model = acm.fit(
        curve,
        factor_maturities=list(range(3, 121)),
        n_factors=3,
        return_maturities=RETURNS,
        max_maturity=120,
    )
    daily = curve.iloc[150:]
    days = pd.DatetimeIndex(daily.index[5:15])
    out = events.decompose_changes(daily, days, {curve.index[149]: model}, 120)
    assert len(out) == len(days)
    total = out["expectations_bp"] + out["term_premium_bp"] + out["fit_error_bp"]
    assert np.allclose(total, out["yield_bp"])
    # A change in pricing error differences two 0.5 bp measurement errors: sd about
    # 0.7 bp, so 3 bp is more than four standard deviations.
    assert out["fit_error_bp"].abs().max() < 3.0


def test_all_days_summary_matches_the_event_statistics(rng: np.random.Generator) -> None:
    # The all-days comparison uses the same formulas as the event summary, so on
    # the same frame both must agree, and the shares plus covariance equal the
    # variance of the sum when the fitting error is zero.
    rn = rng.normal(0, 2, 400)
    tp = rng.normal(0, 3, 400) - 0.3 * rn
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2000-01-03", periods=400),
            "yield_bp": rn + tp,
            "expectations_bp": rn,
            "term_premium_bp": tp,
            "fit_error_bp": 0.0,
        }
    )
    changes = pd.DataFrame({"date": frame["date"], "change_pp": 0.25})
    summary = events.summarize(frame, changes, frame)
    every = events.all_days_summary(frame)
    assert every["days"] == 400
    assert every["term_premium_variance_share"] == pytest.approx(
        summary.term_premium_variance_share
    )
    assert every["mean_abs_expectations_bp"] == pytest.approx(summary.mean_abs_expectations_bp)
    total = (
        every["term_premium_variance_share"]
        + every["expectations_variance_share"]
        + every["covariance_share"]
    )
    assert total == pytest.approx(1.0)

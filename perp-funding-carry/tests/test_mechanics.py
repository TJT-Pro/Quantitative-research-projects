"""Liquidation, funding and carry mechanics against cases with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from perp_funding_carry.models import funding
from perp_funding_carry.models.carry import CarrySettings, Position, simulate
from perp_funding_carry.models.carry import Path as MarginPath
from perp_funding_carry.models.costs import fit_spread_model
from perp_funding_carry.models.margin import liquidation_price

# Binance, "How to calculate liquidation price of USD-M futures contracts", cross mode examples.


def test_liquidation_price_matches_binance_eth_example() -> None:
    lp = liquidation_price(
        wallet=1535443.01,
        side=1,
        quantity=3683.979,
        entry=1456.84,
        maintenance_rate=0.10,
        maintenance_amount=135365.00,
        other_maintenance=71200.81144,
        other_unrealized=-56354.57,
    )
    assert lp == pytest.approx(1153.26, abs=0.01)


def test_liquidation_price_matches_binance_btc_example() -> None:
    lp = liquidation_price(
        wallet=1535443.01,
        side=1,
        quantity=109.488,
        entry=32481.98,
        maintenance_rate=0.025,
        maintenance_amount=16300.000,
        other_maintenance=356512.508,
        other_unrealized=-448192.89,
    )
    assert lp == pytest.approx(26316.89, abs=0.01)


@given(
    leverage=st.floats(min_value=1.0, max_value=20.0),
    mmr=st.floats(min_value=0.001, max_value=0.05),
    entry=st.floats(min_value=1.0, max_value=1e5),
)
def test_short_liquidation_sits_above_entry_and_falls_with_leverage(
    leverage: float, mmr: float, entry: float
) -> None:
    q = 1.0
    lp = liquidation_price(
        wallet=q * entry / leverage, side=-1, quantity=q, entry=entry, maintenance_rate=mmr
    )
    assert lp > entry
    higher = liquidation_price(
        wallet=q * entry / (leverage * 1.5), side=-1, quantity=q, entry=entry, maintenance_rate=mmr
    )
    assert higher < lp


def test_liquidation_rejects_a_bad_side() -> None:
    with pytest.raises(ValueError, match="side"):
        liquidation_price(wallet=1.0, side=0, quantity=1.0, entry=1.0, maintenance_rate=0.01)


def _bars(n: int, *, rate: float, price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="8h", tz="UTC")
    return pd.DataFrame(
        {
            "spot_close": price,
            "mark_close": price,
            "mark_high": price,
            "rate": rate,
            "spot_half": 0.0002,
            "perp_half": 0.0001,
        },
        index=index,
    )


def _settings(path: MarginPath, *, leverage: float = 2.0) -> CarrySettings:
    return CarrySettings(
        leverage=leverage,
        reserve_share=0.2,
        topup_multiple=1.5,
        maintenance_rate=0.005,
        spot_fee=0.001,
        perp_fee=0.0005,
        path=path,
    )


@pytest.mark.parametrize("path", ["ideal", "topup", "liquidate"])
def test_constant_funding_pays_the_analytic_carry_minus_costs(path: MarginPath) -> None:
    r, n, leverage = 0.0001, 300, 2.0
    s = _settings(path, leverage=leverage)
    res = simulate(_bars(n, rate=r), settings=s)
    notional = (1.0 - s.reserve_share) / (1.0 + 1.0 / leverage)
    entry_cost = notional * (s.spot_fee + 0.0002) + notional * (s.perp_fee + 0.0001)
    expected = 1.0 - entry_cost + (n - 1) * r * notional
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-12)
    assert res.funding.iloc[-1] == pytest.approx((n - 1) * r * notional, rel=1e-12)
    assert res.liquidations == 0


def test_a_spike_through_the_liquidation_price_loses_the_wallet() -> None:
    bars = _bars(5, rate=0.0)
    bars.loc[bars.index[2], "mark_high"] = 200.0
    res = simulate(bars, settings=_settings("liquidate"))
    assert res.liquidations == 1
    ideal = simulate(bars, settings=_settings("ideal"))
    assert ideal.liquidations == 0


def test_topup_restores_the_target_leverage() -> None:
    p = Position(settings=_settings("topup"))
    p.enter(capital=1.0, spot=100.0, mark=100.0, spot_half=0.0, perp_half=0.0)
    # A 20% rise takes effective leverage from 2 to 4, past the 1.5 x 2 = 3 trigger; the top-up
    # needed (0.16 of capital) fits in the reserve (0.2 less entry costs).
    p.step(spot=120.0, mark=120.0, mark_high=120.0, rate=0.0, spot_half=0.0, perp_half=0.0)
    margin = p.wallet - p.quantity * (120.0 - p.entry)
    assert p.topups == 1
    assert p.quantity * 120.0 / margin == pytest.approx(2.0)


def test_a_reserve_too_small_forces_a_close() -> None:
    p = Position(settings=_settings("topup"))
    p.enter(capital=1.0, spot=100.0, mark=100.0, spot_half=0.0, perp_half=0.0)
    cash = p.step(spot=125.0, mark=125.0, mark_high=125.0, rate=0.0, spot_half=0.0, perp_half=0.0)
    assert p.forced_closes == 1
    assert not p.open
    assert cash == pytest.approx(1.0 - p.costs)  # hedged: the close returns the capital less costs


def test_premium_weights_and_conventions() -> None:
    samples = np.array([1.0, 2.0, 3.0])
    assert funding.average_premium(samples, linear=True) == pytest.approx((1 + 4 + 9) / 6)
    assert funding.average_premium(samples, linear=False) == pytest.approx(2.0)
    for p in (-0.002, 0.0, 0.00005, 0.003):
        faq = funding.funding_rate(
            premium=p, interval_h=8, interest_8h=0.0001, clamp=0.0005, convention="faq"
        )
        per = funding.funding_rate(
            premium=p, interval_h=8, interest_8h=0.0001, clamp=0.0005, convention="per_interval"
        )
        assert faq == pytest.approx(per)
    # Binance FAQ example: an 8-hour premium of 0.0429% gives the 0.0100% interest rate.
    assert funding.funding_rate(
        premium=0.000429, interval_h=8, interest_8h=0.0001, clamp=0.0005, convention="faq"
    ) == pytest.approx(0.0001)


@given(p=st.floats(min_value=-0.01, max_value=0.01), hours=st.sampled_from([1, 4, 8]))
def test_per_interval_funding_stays_within_the_clamp_of_the_premium(p: float, hours: int) -> None:
    f = funding.funding_rate(
        premium=p, interval_h=hours, interest_8h=0.0001, clamp=0.0005, convention="per_interval"
    )
    assert abs(f - p) <= 0.0005 + 1e-15


def test_rebuild_reproduces_prints_built_from_the_premium() -> None:
    minutes = pd.date_range("2024-01-01", periods=60 * 24, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    premium = pd.DataFrame(
        {"symbol": "X", "open_time": minutes, "close": rng.normal(0.0002, 0.0004, minutes.size)}
    )
    rows = []
    for k in range(1, 4):
        t = minutes[0] + pd.Timedelta(hours=8 * k)
        window = premium.loc[
            (premium["open_time"] >= t - pd.Timedelta(hours=8)) & (premium["open_time"] < t),
            "close",
        ]
        p = funding.average_premium(window.to_numpy(), linear=True)
        rows.append(
            {
                "symbol": "X",
                "time": t,
                "interval_h": 8,
                "rate": funding.funding_rate(
                    premium=p,
                    interval_h=8,
                    interest_8h=0.0001,
                    clamp=0.0005,
                    convention="per_interval",
                ),
            }
        )
    out = funding.rebuild(
        pd.DataFrame(rows),
        premium,
        caps={},
        settings=funding.RebuildSettings(
            interest_8h=0.0001,
            clamp=0.0005,
            equal_weight_interval_hours=(1,),
            conventions=("faq", "per_interval"),
        ),
    )
    assert len(out) == 3
    np.testing.assert_allclose(out["per_interval"], out["reported"], atol=1e-15)


def _snapshots(
    *, slope: float, noise: float, n: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Books whose perpetual half-spread is 2 bp * (1e8 / V) ** -slope, spot 1.5 times that."""
    rng = np.random.default_rng(seed)
    volume = np.exp(rng.uniform(np.log(1e6), np.log(1e10), n))
    perp = 2e-4 * (volume / 1e8) ** slope * np.exp(rng.normal(0, noise, n))
    spot = 1.5 * 2e-4 * (volume / 1e8) ** slope * np.exp(rng.normal(0, noise, n))
    symbols = [f"C{i}USDT" for i in range(n)]
    rows = []
    for market, half in (("perp", perp), ("spot", spot)):
        for sym, h, v in zip(symbols, half, volume, strict=True):
            rows.append(
                {
                    "taken_at": "2026-11-04T00:00:00+00:00",
                    "market": market,
                    "symbol": sym,
                    "bid": 100 * (1 - h),
                    "ask": 100 * (1 + h),
                    "quote_volume": v,
                    "origin": "mirror",
                }
            )
    contracts = pd.DataFrame({"symbol": symbols, "spot": symbols})
    return pd.DataFrame(rows), contracts


def test_spread_fit_recovers_a_planted_power_law() -> None:
    snaps, contracts = _snapshots(slope=-0.4, noise=0.2, n=400, seed=3)
    model = fit_spread_model(snaps, contracts, clip_quantiles=(0.0, 1.0))
    assert model.perp.slope == pytest.approx(-0.4, abs=0.03)
    assert model.spot.slope == pytest.approx(-0.4, abs=0.03)
    at = np.array([1e8])
    assert model.half_spread("perp", at)[0] == pytest.approx(2e-4, rel=0.1)
    assert model.half_spread("spot", at)[0] == pytest.approx(3e-4, rel=0.1)


def test_smearing_makes_the_retransformed_mean_unbiased() -> None:
    """exp(fitted log) is the conditional median; with smearing it is the conditional mean."""
    snaps, contracts = _snapshots(slope=0.0, noise=0.5, n=4000, seed=5)
    model = fit_spread_model(snaps, contracts, clip_quantiles=(0.0, 1.0))
    measured = snaps.loc[snaps["market"] == "perp"]
    mean = float(((measured["ask"] - measured["bid"]) / (measured["ask"] + measured["bid"])).mean())
    assert model.half_spread("perp", np.array([1e8]))[0] == pytest.approx(mean, rel=0.03)


def test_spread_predictions_stay_inside_the_measured_range() -> None:
    snaps, contracts = _snapshots(slope=-0.4, noise=0.2, n=200, seed=9)
    model = fit_spread_model(snaps, contracts, clip_quantiles=(0.01, 0.99))
    tiny_huge_unknown = np.array([1.0, 1e15, np.nan, 0.0])
    out = model.half_spread("perp", tiny_huge_unknown)
    assert out[0] == model.perp.upper  # a dead market is held at the widest measured spread
    assert out[1] == model.perp.lower
    assert out[2] == model.perp.upper  # no volume history yet: the conservative bound
    assert out[3] == model.perp.upper

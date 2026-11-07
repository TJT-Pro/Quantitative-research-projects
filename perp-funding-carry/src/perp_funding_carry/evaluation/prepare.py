"""Bars ready for simulation: per-bar funding, trailing volume and modelled half-spreads.

Every derived column is known at the time it is used: a funding print settled at T is
credited to the bar ending at T; trailing volume at bar open uses complete UTC days before
that bar's day, and the half-spread is the spread model evaluated at that volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from perp_funding_carry.models.costs import SpreadModel


def interval_hours(text: str) -> float:
    unit = text[-1]
    value = float(text[:-1])
    return {"m": value / 60.0, "h": value, "d": value * 24.0}[unit]


def funding_per_bar(funding: pd.DataFrame, *, bar_hours: float) -> pd.DataFrame:
    """symbol, open_time, rate: prints settled in (open, open + bar] summed onto that bar."""
    width = pd.Timedelta(hours=bar_hours)
    open_time = (funding["time"] - pd.Timedelta(milliseconds=1)).dt.floor(width)
    summed: pd.DataFrame = (
        funding.assign(open_time=open_time)
        .groupby(["symbol", "open_time"], as_index=False)
        .agg(rate=("rate", "sum"))
    )
    return summed


def prepare_bars(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    bar_hours: float,
    spreads: SpreadModel,
    volume_window_days: int,
) -> pd.DataFrame:
    """Funding per bar, the trailing median daily quote volume known at each bar's open, and
    each leg's half-spread from the spread model at that volume."""
    out = bars.merge(
        funding_per_bar(funding, bar_hours=bar_hours), on=["symbol", "open_time"], how="left"
    ).fillna({"rate": 0.0})
    pieces = []
    for _, group in out.groupby("symbol", sort=False):
        g = group.sort_values("open_time").reset_index(drop=True)
        day = g["open_time"].dt.floor("1D")
        daily = g.groupby(day)["perp_quote_volume"].sum()
        trailing = (
            daily.shift(1)
            .rolling(volume_window_days, min_periods=max(1, volume_window_days // 2))
            .median()
        )
        g["volume_median"] = day.map(trailing).to_numpy()
        volume = g["volume_median"].to_numpy(dtype=np.float64)
        g["spot_half"] = spreads.half_spread("spot", volume)
        g["perp_half"] = spreads.half_spread("perp", volume)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def usd_per_usdt(stable: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """USDT marked to USD at each bar, carried forward from the stablecoin pair's closes."""
    series = stable.set_index("open_time")["usd_per_usdt"].sort_index()
    return series.reindex(series.index.union(index)).ffill().reindex(index).fillna(1.0)

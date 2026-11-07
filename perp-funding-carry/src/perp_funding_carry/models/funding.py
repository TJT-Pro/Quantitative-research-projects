"""Binance funding from first principles.

    P = weighted average of the premium index over the interval
        (weights 1, 2, ..., n for most contracts; equal weights for hourly contracts)
    faq:          F = [P + clamp(I8 - P, -c, c)] / (8 / N)
    per_interval: F = P + clamp(I8 N / 8 - P, -c, c)
    F is then held inside the contract's cap and floor when fundingInfo lists them.

N is the interval in hours, I8 the 8-hour interest rate, c the clamp. The two conventions
agree for 8-hour contracts and differ for shorter ones; the reproduction rate on the
exchange's own prints decides between them. The exchange samples the premium every 5
seconds; the published 1-minute premium index klines are the finest public record, so
each minute's close stands in for its twelve samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Convention = Literal["faq", "per_interval"]


def average_premium(samples: NDArray[np.float64], *, linear: bool) -> float:
    if samples.size == 0:
        raise ValueError("no premium samples")
    if not linear:
        return float(samples.mean())
    weights = np.arange(1, samples.size + 1, dtype=np.float64)
    return float(weights @ samples / weights.sum())


def funding_rate(
    *, premium: float, interval_h: float, interest_8h: float, clamp: float, convention: Convention
) -> float:
    if convention == "faq":
        raw = premium + float(np.clip(interest_8h - premium, -clamp, clamp))
        return raw / (8.0 / interval_h)
    interest = interest_8h * interval_h / 8.0
    return premium + float(np.clip(interest - premium, -clamp, clamp))


@dataclass(frozen=True)
class RebuildSettings:
    interest_8h: float
    clamp: float
    equal_weight_interval_hours: tuple[int, ...]
    conventions: tuple[Convention, ...]


def rebuild(
    prints: pd.DataFrame,
    premium: pd.DataFrame,
    *,
    caps: dict[str, tuple[float, float]],
    settings: RebuildSettings,
) -> pd.DataFrame:
    """One row per print with a complete minute window: reported rate and each convention's
    reproduction. `prints`: symbol, time, interval_h, rate; `premium`: symbol, open_time, close."""
    rows = []
    for symbol, group in prints.groupby("symbol"):
        series = (
            premium.loc[premium["symbol"] == symbol].set_index("open_time")["close"].sort_index()
        )
        if series.empty:
            continue
        index = series.index
        values = series.to_numpy(dtype=np.float64)
        for _, p in group.iterrows():
            minutes = round(float(p["interval_h"]) * 60)
            start = p["time"] - pd.Timedelta(minutes=minutes)
            lo = index.searchsorted(start, side="left")
            hi = index.searchsorted(p["time"], side="left")
            if hi - lo != minutes:
                continue  # an incomplete window cannot be rebuilt
            linear = int(p["interval_h"]) not in settings.equal_weight_interval_hours
            prem = average_premium(values[lo:hi], linear=linear)
            row = {
                "symbol": symbol,
                "time": p["time"],
                "interval_h": p["interval_h"],
                "reported": p["rate"],
                "premium": prem,
            }
            floor, cap = caps.get(str(symbol), (-np.inf, np.inf))
            for convention in settings.conventions:
                f = funding_rate(
                    premium=prem,
                    interval_h=float(p["interval_h"]),
                    interest_8h=settings.interest_8h,
                    clamp=settings.clamp,
                    convention=convention,
                )
                row[convention] = float(np.clip(f, floor, cap))
            rows.append(row)
    return pd.DataFrame(rows)

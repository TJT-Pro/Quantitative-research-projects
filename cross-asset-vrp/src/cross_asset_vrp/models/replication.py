"""Model-free 30-day variance swap rate from one Deribit option-chain snapshot.

Static replication (Carr and Madan 1998) in the discrete form of the Cboe VIX method. For one
expiry with T years to go, forward F and strikes K_i:

    sigma^2 = (2 / T) sum_i (dK_i / K_i^2) Q(K_i) - (1 / T) (F / K_0 - 1)^2

where K_0 is the first strike at or below F, Q(K_i) the forward value of the out-of-the-money
option at K_i (puts below K_0, calls above, their average at K_0), and dK_i half the distance
between the neighbouring strikes. Moving away from K_0, strikes stop after `zero_bid_stop`
consecutive options without a bid.

Deribit options are inverse: priced in BTC, settled in BTC. A premium of p BTC paid today
equals p BTC held to expiry, whose dollar value at expiry is locked at p F with a short
forward, so the forward dollar value is Q = p F with no interest rate needed. The forward of
each expiry is the chain's own underlying_price.

The 30-day rate interpolates the two expiries that bracket 30 days, the near one at least
`min_expiry_days` out, in total variance, as the VIX does:

    K30 = [T1 s1^2 (N2 - N30) / (N2 - N1) + T2 s2^2 (N30 - N1) / (N2 - N1)] * (N365 / N30)

Each side of the book gives its own rate: the bid strip is what a seller of variance
receives, the ask strip what a buyer pays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["bid", "mid", "ask"]
SECONDS_PER_YEAR = 365.0 * 86_400.0


@dataclass(frozen=True)
class ExpiryVariance:
    expiration: pd.Timestamp
    years: float
    forward: float
    variance: float  # annualized
    strikes: int


def _price(frame: pd.DataFrame, side: Side) -> pd.Series:
    if side == "mid":
        both = frame["bid"].notna() & frame["ask"].notna() & (frame["bid"] > 0)
        return ((frame["bid"] + frame["ask"]) / 2.0).where(both)
    return frame[side].where(frame[side] > 0)


def _walk(prices: pd.Series, *, stop: int) -> pd.Series:
    """Keep strikes moving outward until `stop` consecutive missing prices."""
    kept = {}
    misses = 0
    for strike, price in prices.items():
        if not np.isfinite(price):
            misses += 1
            if misses >= stop:
                break
            continue
        misses = 0
        kept[strike] = price
    return pd.Series(kept, dtype=float)


def expiry_variance(
    chain: pd.DataFrame, *, now: pd.Timestamp, side: Side, zero_bid_stop: int
) -> ExpiryVariance | None:
    """`chain`: one expiry's rows with type, strike, bid, ask (BTC) and forward."""
    expiration = pd.Timestamp(chain["expiration"].iloc[0])
    years = (expiration - now).total_seconds() / SECONDS_PER_YEAR
    forward = float(chain["forward"].median())
    if years <= 0 or not np.isfinite(forward) or forward <= 0:
        return None
    calls = chain.loc[chain["type"] == "call"].set_index("strike").sort_index()
    puts = chain.loc[chain["type"] == "put"].set_index("strike").sort_index()
    strikes = sorted(set(calls.index) & set(puts.index))
    below = [k for k in strikes if k <= forward]
    if not below:
        return None
    k0 = below[-1]
    put_prices = _price(puts, side)
    call_prices = _price(calls, side)
    lower = _walk(
        put_prices.loc[put_prices.index < k0].sort_index(ascending=False), stop=zero_bid_stop
    )
    upper = _walk(call_prices.loc[call_prices.index > k0].sort_index(), stop=zero_bid_stop)
    at_k0 = [
        p for p in (put_prices.get(k0), call_prices.get(k0)) if p is not None and np.isfinite(p)
    ]
    if not at_k0 or lower.empty or upper.empty:
        return None
    q = pd.concat([lower, pd.Series({k0: float(np.mean(at_k0))}), upper]).sort_index()
    k = q.index.to_numpy(dtype=np.float64)
    dk = np.empty_like(k)
    dk[1:-1] = (k[2:] - k[:-2]) / 2.0
    dk[0] = k[1] - k[0]
    dk[-1] = k[-1] - k[-2]
    forward_value = q.to_numpy() * forward  # BTC premium to forward dollars
    variance = (2.0 / years) * float(np.sum(dk / k**2 * forward_value)) - (
        forward / k0 - 1.0
    ) ** 2 / years
    return ExpiryVariance(
        expiration=expiration, years=years, forward=forward, variance=variance, strikes=len(k)
    )


def thirty_day_rate(
    chain: pd.DataFrame,
    *,
    now: pd.Timestamp,
    side: Side,
    target_days: int,
    min_expiry_days: int,
    zero_bid_stop: int,
) -> float | None:
    """Annualized 30-day variance swap rate from a snapshot's chain (all expiries)."""
    target = target_days / 365.0
    floor = min_expiry_days / 365.0
    terms = []
    for _, group in chain.groupby("expiration"):
        result = expiry_variance(group, now=now, side=side, zero_bid_stop=zero_bid_stop)
        if result is not None and result.years >= floor and result.variance > 0:
            terms.append(result)
    near = [t for t in terms if t.years <= target]
    far = [t for t in terms if t.years > target]
    if not near or not far:
        return None
    a = max(near, key=lambda t: t.years)
    b = min(far, key=lambda t: t.years)
    w = (b.years - target) / (b.years - a.years)
    total = a.years * a.variance * w + b.years * b.variance * (1.0 - w)
    return total / target

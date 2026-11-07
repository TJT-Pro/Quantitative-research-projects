"""A Deribit option chain at one moment, rebuilt from the trade history.

Deribit's history server keeps every trade, each with the option's mark price and the index
price at that instant, but no quote snapshots. For a snapshot time T and the trades in the
window before it:

1. Instruments: BTC-<expiry>-<strike>-<C|P>, expiring at 08:00 UTC.
2. Forward per expiry: a premium of p BTC is worth p F forward dollars (Deribit marks are
   the Black-76 dollar value over the forward), so parity, (C - P) F = F - K, gives
   F = K / (1 - (C - P)). Parity holds only for a call and a put priced at one moment, and
   the last trades of the two legs are hours apart, so each leg is carried to T (step 3 and
   4 on the current guess of the basis F / S) and parity at T updates the guess until it
   settles. The basis is the median over strikes within ATM_BAND of the index, and
   F_T = (F / S) S_T, with S_T the index at the last trade in the window.
3. Volatility per instrument: its last mark, worth mark F_t forward dollars at its own trade
   time t (F_t = (F / S) S_t), inverted through Black-76 for an implied volatility.
4. The chain at T: every instrument repriced by Black-76 at F_T with its own volatility and
   the time left from T, back in BTC as forward dollars over F_T.
5. Bid and ask: a taker buys above the mark and sells below it, so (price - mark) / mark,
   signed by the taker's side, is an effective half-spread. Its median per bucket of
   log(K / F) gives bid = mark (1 - h) and ask = mark (1 + h); a bucket with too few trades
   takes the median of all trades.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

SECONDS_PER_YEAR = 365.0 * 86_400.0
_NAME = re.compile(r"^BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:d\d+)?)-([CP])$")
_MONTHS = {
    m: i
    for i, m in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}
ATM_BAND = 0.15
BASIS_ITERATIONS = 20
BASIS_TOLERANCE = 1e-10


def black76(*, forward: float, strike: float, years: float, vol: float, call: bool) -> float:
    """Undiscounted Black-76 value in forward dollars."""
    if years <= 0 or vol <= 0:
        intrinsic = forward - strike if call else strike - forward
        return max(intrinsic, 0.0)
    s = vol * np.sqrt(years)
    d1 = (np.log(forward / strike) + 0.5 * s * s) / s
    d2 = d1 - s
    if call:
        return float(forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return float(strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def implied_vol(*, value: float, forward: float, strike: float, years: float, call: bool) -> float:
    """The Black-76 volatility that prices `value`; NaN when no volatility does."""
    intrinsic = max(forward - strike, 0.0) if call else max(strike - forward, 0.0)
    upper = forward if call else strike
    if not np.isfinite(value) or years <= 0 or value <= intrinsic or value >= upper:
        return float("nan")

    def gap(vol: float) -> float:
        return black76(forward=forward, strike=strike, years=years, vol=vol, call=call) - value

    try:
        return float(brentq(gap, 1e-4, 20.0, xtol=1e-8))
    except ValueError:
        return float("nan")


def parse_names(names: pd.Series) -> pd.DataFrame:
    """expiration (08:00 UTC), strike, type for each Deribit option name; NaN rows otherwise."""
    parts = names.str.extract(_NAME)
    day = parts[0].astype(float)
    month = parts[1].map(_MONTHS)
    year = 2000 + parts[2].astype(float)
    expiration = pd.to_datetime(
        pd.DataFrame({"year": year, "month": month, "day": day}), errors="coerce", utc=True
    ) + pd.Timedelta(hours=8)
    strike = pd.to_numeric(parts[3].str.replace("d", ".", regex=False), errors="coerce")
    kind = parts[4].map({"C": "call", "P": "put"})
    return pd.DataFrame(
        {"expiration": expiration, "strike": strike, "type": kind}, index=names.index
    )


@dataclass(frozen=True)
class Rebuilt:
    chain: pd.DataFrame  # symbol, type, strike, expiration, mark (BTC), forward, age_minutes, iv,
    # deribit_iv (Deribit's own volatility of the same mark, a check on the forward), bid, ask
    trades: int
    instruments: int
    median_age_minutes: float
    half_spreads: dict[str, float]


def _bucket_labels(edges: tuple[float, ...]) -> list[str]:
    bounds = [-np.inf, *edges, np.inf]
    return [f"{lo:g} to {hi:g}" for lo, hi in itertools.pairwise(bounds)]


def _carry(row: pd.Series, *, b: float, at: pd.Timestamp, spot_at: float) -> tuple[float, float]:
    """One option's last mark carried to `at` on basis `b`: its volatility, inverted at its own
    trade time on F_t = b S_t, and its value at `at` on F_T = b S_T, in BTC. NaNs when the mark
    has no volatility to invert."""
    call = row["type"] == "call"
    strike = float(row["strike"])
    expiry = pd.Timestamp(row["expiration"])
    f_trade = b * float(row["index_price"])
    vol = implied_vol(
        value=float(row["mark_price"]) * f_trade,
        forward=f_trade,
        strike=strike,
        years=(expiry - pd.Timestamp(row["time"])).total_seconds() / SECONDS_PER_YEAR,
        call=call,
    )
    if not np.isfinite(vol):
        return float("nan"), float("nan")
    f_at = b * spot_at
    years = (expiry - at).total_seconds() / SECONDS_PER_YEAR
    return vol, black76(forward=f_at, strike=strike, years=years, vol=vol, call=call) / f_at


def _expiry_basis(
    calls: pd.DataFrame, puts: pd.DataFrame, *, at: pd.Timestamp, spot_at: float
) -> float | None:
    """F / S for one expiry. Parity needs a call and a put at the same moment, and the last
    trades of the two legs are hours apart, so both are carried to `at` on the current guess
    and parity at `at` updates the guess until it settles."""
    strikes = calls.index.intersection(puts.index)
    strikes = strikes[np.abs(np.log(strikes.to_numpy(dtype=float) / spot_at)) <= ATM_BAND]
    if strikes.empty:
        return None
    gap = calls.loc[strikes, "mark_price"].to_numpy() - puts.loc[strikes, "mark_price"].to_numpy()
    raw = strikes.to_numpy(dtype=float) / (1.0 - gap) / spot_at
    raw = raw[(gap < 1.0) & np.isfinite(raw)]
    if raw.size == 0:
        return None
    b = float(np.median(raw))
    for _ in range(BASIS_ITERATIONS):
        ratios = []
        for k in strikes:
            _, c = _carry(calls.loc[k], b=b, at=at, spot_at=spot_at)
            _, q = _carry(puts.loc[k], b=b, at=at, spot_at=spot_at)
            if np.isfinite(c) and np.isfinite(q) and c - q < 1.0:
                ratios.append(float(k) / (1.0 - (c - q)) / spot_at)
        if not ratios:
            return None
        updated = float(np.median(ratios))
        if abs(updated - b) < BASIS_TOLERANCE:
            return updated
        b = updated
    return b


def _basis(last: pd.DataFrame, *, at: pd.Timestamp, spot_at: float) -> dict[pd.Timestamp, float]:
    basis: dict[pd.Timestamp, float] = {}
    for expiry, group in last.groupby("expiration"):
        calls = group.loc[group["type"] == "call"].set_index("strike", drop=False)
        puts = group.loc[group["type"] == "put"].set_index("strike", drop=False)
        b = _expiry_basis(calls, puts, at=at, spot_at=spot_at)
        if b is not None:
            basis[pd.Timestamp(str(expiry))] = b
    return basis


def _repriced(
    last: pd.DataFrame, *, basis: dict[pd.Timestamp, float], at: pd.Timestamp, spot_at: float
) -> pd.DataFrame:
    rows = []
    for name, row in last.iterrows():
        expiry = pd.Timestamp(row["expiration"])
        b = basis.get(expiry)
        if b is None:
            continue
        vol, mark = _carry(row, b=b, at=at, spot_at=spot_at)
        if not np.isfinite(vol):
            continue
        rows.append(
            {
                "symbol": name,
                "type": row["type"],
                "strike": float(row["strike"]),
                "expiration": expiry,
                "mark": mark,
                "forward": b * spot_at,
                "age_minutes": (at - pd.Timestamp(row["time"])).total_seconds() / 60.0,
                "iv": vol * 100.0,
                "deribit_iv": float(row["iv"]) if "iv" in row else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _half_spreads(
    t: pd.DataFrame,
    *,
    basis: dict[pd.Timestamp, float],
    edges: tuple[float, ...],
    min_trades: int,
) -> dict[str, float]:
    """Median signed (price - mark) / mark per log(K / F) bucket, floored at zero."""
    sign = np.where(t["direction"] == "buy", 1.0, -1.0)
    half = sign * (t["price"] - t["mark_price"]) / t["mark_price"]
    forward = t["expiration"].map(basis).astype(float).fillna(1.0) * t["index_price"]
    labels = _bucket_labels(edges)
    bucket = pd.cut(np.log(t["strike"] / forward), bins=[-np.inf, *edges, np.inf], labels=labels)
    overall = max(float(half.median()), 0.0)
    out: dict[str, float] = {}
    for label in labels:
        x = half.loc[bucket == label]
        out[label] = max(float(x.median()), 0.0) if len(x) >= min_trades else overall
    return out


def rebuild(
    trades: pd.DataFrame,
    *,
    at: pd.Timestamp,
    spread_buckets: tuple[float, ...],
    min_spread_trades: int,
) -> Rebuilt | None:
    """`trades`: instrument_name, timestamp (ms), price, mark_price, index_price, direction."""
    if trades.empty:
        return None
    t = trades.join(parse_names(trades["instrument_name"])).dropna(
        subset=["expiration", "strike", "type"]
    )
    t["time"] = pd.to_datetime(t["timestamp"].astype("int64"), unit="ms", utc=True)
    t = t.loc[(t["time"] <= at) & (t["expiration"] > at) & (t["mark_price"] > 0)].sort_values(
        "time"
    )
    if t.empty:
        return None
    last = t.groupby("instrument_name").tail(1).set_index("instrument_name")
    spot_at = float(t["index_price"].iloc[-1])
    basis = _basis(last, at=at, spot_at=spot_at)
    chain = _repriced(last, basis=basis, at=at, spot_at=spot_at)
    if chain.empty:
        return None
    halves = _half_spreads(t, basis=basis, edges=spread_buckets, min_trades=min_spread_trades)
    moneyness = pd.Series(np.log(chain["strike"] / chain["forward"]), index=chain.index)
    bucket = pd.Series(
        pd.cut(
            moneyness,
            bins=[-np.inf, *spread_buckets, np.inf],
            labels=_bucket_labels(spread_buckets),
        ),
        index=chain.index,
    )
    half = bucket.astype(str).map(halves).astype(float).to_numpy()
    chain["bid"] = chain["mark"] * (1.0 - half)
    chain["ask"] = chain["mark"] * (1.0 + half)
    return Rebuilt(
        chain=chain,
        trades=len(t),
        instruments=len(chain),
        median_age_minutes=float(chain["age_minutes"].median()),
        half_spreads=halves,
    )

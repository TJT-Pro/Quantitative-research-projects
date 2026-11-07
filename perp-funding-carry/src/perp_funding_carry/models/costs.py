"""Half-spreads from measured order books, carried back in time through trading volume.

Across contracts, the quoted half-spread falls with trading volume close to a power law.
For each leg (spot, perpetual) the model is

    log h = a + b log V + e,

fitted by OLS over every contract in every stored snapshot, with h the half-spread
(ask - bid) / (ask + bid) and V the perpetual's 24-hour quote volume. The perpetual's volume
is the regressor for both legs because it is the volume the history carries (the trailing
median daily volume of each contract), so the fitted law can be applied to any past bar.
The retransformed prediction uses Duan's (1983) smearing factor, mean(exp(e)), which removes
the downward bias of exp(fitted log). Predictions are held inside the 1st to 99th percentile
of measured half-spreads, so the law is never extrapolated beyond what the books showed. A bar
with no volume history yet (a new listing) gets the upper bound.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

LEGS = ("spot", "perp")


@dataclass(frozen=True)
class LegFit:
    intercept: float
    slope: float
    smearing: float
    r2: float
    observations: int
    lower: float  # clip bounds on the half-spread, as a fraction
    upper: float
    median_measured: float


@dataclass(frozen=True)
class SpreadModel:
    spot: LegFit
    perp: LegFit
    snapshots: int
    contracts: int
    first: str
    last: str

    def leg(self, name: str) -> LegFit:
        return self.spot if name == "spot" else self.perp

    def half_spread(self, name: str, volume: np.ndarray) -> np.ndarray:
        fit = self.leg(name)
        v = np.asarray(volume, dtype=np.float64)
        known = np.isfinite(v) & (v > 0)
        out = np.full(v.shape, fit.upper)
        logv = np.log(v[known])
        out[known] = np.exp(fit.intercept + fit.slope * logv) * fit.smearing
        clipped: np.ndarray = np.clip(out, fit.lower, fit.upper)
        return clipped

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def pair_snapshots(snapshots: pd.DataFrame, contracts: pd.DataFrame) -> pd.DataFrame:
    """One row per (snapshot, contract) with both legs' half-spreads and the perp volume.

    `contracts` needs symbol and spot columns."""
    s = snapshots.assign(
        half=(snapshots["ask"] - snapshots["bid"]) / (snapshots["ask"] + snapshots["bid"])
    )
    perp = s.loc[s["market"] == "perp", ["taken_at", "symbol", "half", "quote_volume"]].rename(
        columns={"half": "perp_half", "quote_volume": "volume"}
    )
    spot = s.loc[s["market"] == "spot", ["taken_at", "symbol", "half"]].rename(
        columns={"symbol": "spot", "half": "spot_half"}
    )
    pairs = (
        contracts[["symbol", "spot"]].merge(perp, on="symbol").merge(spot, on=["taken_at", "spot"])
    )
    keep = (pairs["volume"] > 0) & (pairs["perp_half"] > 0) & (pairs["spot_half"] > 0)
    return pairs.loc[keep].reset_index(drop=True)


def _fit_leg(
    log_volume: np.ndarray, half: np.ndarray, *, clip_quantiles: tuple[float, float]
) -> LegFit:
    y = np.log(half)
    x = np.column_stack([np.ones_like(log_volume), log_volume])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    total = float(((y - y.mean()) ** 2).sum())
    lo, hi = np.quantile(half, clip_quantiles)
    return LegFit(
        intercept=float(beta[0]),
        slope=float(beta[1]),
        smearing=float(np.mean(np.exp(resid))),
        r2=1.0 - float((resid**2).sum()) / total if total > 0 else 0.0,
        observations=int(y.size),
        lower=float(lo),
        upper=float(hi),
        median_measured=float(np.median(half)),
    )


def fit_spread_model(
    snapshots: pd.DataFrame, contracts: pd.DataFrame, *, clip_quantiles: tuple[float, float]
) -> SpreadModel:
    pairs = pair_snapshots(snapshots, contracts)
    if len(pairs) < 10:
        raise ValueError(
            f"only {len(pairs)} contract snapshots pair a spot and a perpetual book; "
            "the spread model needs at least 10"
        )
    logv = np.log(pairs["volume"].to_numpy(dtype=np.float64))
    fits = {
        leg: _fit_leg(
            logv, pairs[f"{leg}_half"].to_numpy(dtype=np.float64), clip_quantiles=clip_quantiles
        )
        for leg in LEGS
    }
    times = sorted(pairs["taken_at"].unique())
    return SpreadModel(
        spot=fits["spot"],
        perp=fits["perp"],
        snapshots=len(times),
        contracts=int(pairs["symbol"].nunique()),
        first=str(times[0]),
        last=str(times[-1]),
    )

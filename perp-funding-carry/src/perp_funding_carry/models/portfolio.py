"""Cross-sectional carry: at each rebalance, hold the top-k contracts by forecast funding.

Decisions at bar open t use information up to t: the score of the 8-hour window ending
at t and volume up to t. Hysteresis keeps turnover below the cost of chasing noise: a
held contract stays while it ranks inside exit_multiple * k, and the free slots go to
the best-ranked contracts not yet held. Entries and exits trade at the close of the bar ending at t.
Each position is an isolated long-spot, short-perpetual trade (models.carry.Position)
sized at total equity / k when it opens. A contract whose data stop (a delisting) is
closed at its last available prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from perp_funding_carry.models.carry import CarrySettings, Position


@dataclass(frozen=True)
class PortfolioSettings:
    top_k: int
    rebalance_hours: int
    exit_multiple: float
    carry: CarrySettings


@dataclass
class PortfolioResult:
    equity: pd.Series
    funding: pd.Series
    fees: pd.Series
    spread: pd.Series
    holdings: pd.Series  # number of positions held
    liquidations: int
    delisted_closes: int
    turnover: float  # positions opened per rebalance
    capacity_usd: pd.Series  # AUM the held set could absorb at each rebalance


@dataclass(frozen=True)
class Market:
    """Wide arrays, bars x symbols, NaN where a contract is not live."""

    times: pd.DatetimeIndex
    symbols: list[str]
    spot: np.ndarray
    mark: np.ndarray
    high: np.ndarray
    rate: np.ndarray
    spot_half: np.ndarray
    perp_half: np.ndarray
    daily_volume: np.ndarray  # trailing median daily quote volume known at bar open

    @classmethod
    def from_long(cls, frame: pd.DataFrame) -> Market:
        def wide(column: str) -> pd.DataFrame:
            return frame.pivot_table(
                index="open_time", columns="symbol", values=column, aggfunc="first"
            ).sort_index()

        spot = wide("spot_close")
        cols = list(spot.columns)
        get = {
            c: wide(c).reindex(index=spot.index, columns=cols)
            for c in ("mark_close", "mark_high", "rate", "spot_half", "perp_half", "volume_median")
        }
        return cls(
            times=pd.DatetimeIndex(spot.index),
            symbols=[str(c) for c in cols],
            spot=spot.to_numpy(dtype=np.float64),
            mark=get["mark_close"].to_numpy(dtype=np.float64),
            high=get["mark_high"].to_numpy(dtype=np.float64),
            rate=get["rate"].fillna(0.0).to_numpy(dtype=np.float64),
            spot_half=get["spot_half"].to_numpy(dtype=np.float64),
            perp_half=get["perp_half"].to_numpy(dtype=np.float64),
            daily_volume=get["volume_median"].to_numpy(dtype=np.float64),
        )


@dataclass
class _Book:
    """Cash, open positions and the running totals of positions already closed."""

    market: Market
    settings: PortfolioSettings
    maintenance: dict[str, float]
    default_maintenance: float
    cash: float = 1.0
    positions: dict[int, Position] = field(default_factory=dict)
    retired_funding: float = 0.0
    retired_fees: float = 0.0
    retired_spread: float = 0.0
    liquidations: int = 0
    delisted: int = 0
    opened: int = 0
    rebalances: int = 0

    def _prices(self, i: int, j: int) -> dict[str, float]:
        m = self.market
        return {
            "spot": m.spot[i, j],
            "mark": m.mark[i, j],
            "spot_half": m.spot_half[i, j],
            "perp_half": m.perp_half[i, j],
        }

    def retire(self, j: int) -> None:
        p = self.positions.pop(j)
        self.retired_funding += p.funding
        self.retired_fees += p.fees
        self.retired_spread += p.spread
        self.liquidations += p.liquidations

    def close(self, j: int, *, at: int) -> None:
        self.cash += self.positions[j].close(**self._prices(at, j))
        self.retire(j)

    def close_delisted(self, i: int) -> None:
        """Contracts whose data stopped are closed at their last prices."""
        m = self.market
        for j in [j for j in self.positions if np.isnan(m.mark[i, j]) or np.isnan(m.spot[i, j])]:
            self.close(j, at=i - 1)
            self.delisted += 1

    def equity(self, i: int) -> float:
        m = self.market
        return self.cash + sum(
            p.equity(spot=m.spot[i, j], mark=m.mark[i, j]) for j, p in self.positions.items()
        )

    def rebalance(self, i: int, *, score: np.ndarray, min_volume: float) -> list[int]:
        """Hysteresis ranking at bar open i; trades at the close of bar i - 1."""
        m, k, prev = self.market, self.settings.top_k, i - 1
        self.rebalances += 1
        live = (
            np.isfinite(score)
            & np.isfinite(m.mark[prev])
            & np.isfinite(m.mark[i])
            & np.isfinite(m.spot[i])
            & (np.nan_to_num(m.daily_volume[i]) >= min_volume)
        )
        ranked = [j for j in np.argsort(-np.where(live, score, -np.inf)) if live[j]]
        keep_until = int(np.ceil(self.settings.exit_multiple * k))
        kept = [j for j in ranked[:keep_until] if j in self.positions][:k]
        fresh = [j for j in ranked if j not in kept][: k - len(kept)]
        target = set(kept) | set(fresh)
        for j in [j for j in self.positions if j not in target]:
            self.close(j, at=prev)
        size = self.equity(prev) / k
        for j in sorted(target - set(self.positions), key=lambda j: -score[j]):
            amount = min(size, self.cash)
            if amount <= 1e-9:
                break
            mmr = self.maintenance.get(m.symbols[j], self.default_maintenance)
            position = Position(settings=replace(self.settings.carry, maintenance_rate=mmr))
            self.cash -= amount
            self.cash += position.enter(capital=amount, **self._prices(prev, j))
            self.positions[j] = position
            self.opened += 1
        return list(self.positions)

    def step(self, i: int) -> None:
        m = self.market
        for j, p in list(self.positions.items()):
            self.cash += p.step(mark_high=m.high[i, j], rate=m.rate[i, j], **self._prices(i, j))
            if not p.open:  # liquidated or force-closed: its cash came back through step()
                self.retire(j)


def simulate_portfolio(
    market: Market,
    scores: pd.DataFrame,
    *,
    settings: PortfolioSettings,
    min_volume: float,
    participation: float,
    maintenance: dict[str, float],
    default_maintenance: float,
) -> PortfolioResult:
    """`scores`: index = decision times (bar opens), columns = symbols, higher = better."""
    times = market.times
    score = scores.reindex(index=times, columns=market.symbols).to_numpy(dtype=np.float64)
    book = _Book(
        market=market,
        settings=settings,
        maintenance=maintenance,
        default_maintenance=default_maintenance,
    )
    n = times.size
    equity, funding, holdings = np.ones(n), np.zeros(n), np.zeros(n)
    fees, spread = np.zeros(n), np.zeros(n)
    capacity: dict[pd.Timestamp, float] = {}
    for i in range(1, n):
        book.close_delisted(i)
        if times[i].hour % settings.rebalance_hours == 0 and times[i].minute == 0:
            held = book.rebalance(i, score=score[i], min_volume=min_volume)
            if held:
                cap = min(participation * np.nan_to_num(market.daily_volume[i, j]) for j in held)
                capacity[pd.Timestamp(times[i])] = cap * settings.top_k
        book.step(i)
        equity[i] = book.equity(i)
        funding[i] = book.retired_funding + sum(p.funding for p in book.positions.values())
        fees[i] = book.retired_fees + sum(p.fees for p in book.positions.values())
        spread[i] = book.retired_spread + sum(p.spread for p in book.positions.values())
        holdings[i] = len(book.positions)
    return PortfolioResult(
        equity=pd.Series(equity, index=times),
        funding=pd.Series(funding, index=times),
        fees=pd.Series(fees, index=times),
        spread=pd.Series(spread, index=times),
        holdings=pd.Series(holdings, index=times),
        liquidations=book.liquidations + sum(p.liquidations for p in book.positions.values()),
        delisted_closes=book.delisted,
        turnover=book.opened / max(book.rebalances, 1),
        capacity_usd=pd.Series(capacity, dtype=float),
    )

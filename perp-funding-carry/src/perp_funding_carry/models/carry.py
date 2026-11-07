"""Long spot, short perpetual: one position, bar by bar.

Capital C buys q units of spot and posts N / L of isolated margin on a short of q units,
with N = (1 - reserve) C / (1 + 1/L), so every path deploys the same notional. Funding is
settled into the isolated wallet at the bar's closing mark. Three margin paths:

- ideal: margin is unlimited; the trade earns funding plus basis minus costs.
- topup: at each bar close, if effective leverage reaches `topup_multiple` times the
  target, margin is restored from the reserve; excess margin is swept back. If the
  reserve cannot cover it, the position is closed. A breach inside the bar still
  liquidates, because the top-up only happens at bar frequency.
- liquidate: no top-ups. When the bar's mark-price high reaches the liquidation price,
  the isolated wallet is lost (Binance passes the remainder to its insurance fund), the
  spot leg is sold at the bar close and the trade is re-entered at the next bar.

All amounts are in USDT. Costs are taker fees plus an estimated half-spread on each leg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from perp_funding_carry.models.margin import liquidation_price

Path = Literal["ideal", "topup", "liquidate"]


@dataclass(frozen=True)
class CarrySettings:
    leverage: float
    reserve_share: float
    topup_multiple: float
    maintenance_rate: float
    spot_fee: float
    perp_fee: float
    path: Path


@dataclass
class Position:
    settings: CarrySettings
    quantity: float = 0.0
    entry: float = 0.0
    wallet: float = 0.0
    reserve: float = 0.0
    funding: float = 0.0  # cumulative funding received
    fees: float = 0.0  # cumulative taker fees
    spread: float = 0.0  # cumulative half-spreads paid
    liquidations: int = 0
    topups: int = 0
    forced_closes: int = 0

    @property
    def open(self) -> bool:
        return self.quantity > 0

    @property
    def costs(self) -> float:
        return self.fees + self.spread

    def _pay(
        self, *, spot_notional: float, perp_notional: float, spot_half: float, perp_half: float
    ) -> float:
        """Charge a trade on each leg (either notional may be zero); returns the total."""
        s = self.settings
        fee = spot_notional * s.spot_fee + perp_notional * s.perp_fee
        spread = spot_notional * spot_half + perp_notional * perp_half
        self.fees += fee
        self.spread += spread
        return fee + spread

    def enter(
        self, *, capital: float, spot: float, mark: float, spot_half: float, perp_half: float
    ) -> float:
        """Open the trade with `capital`; returns the cash not deployed."""
        s = self.settings
        notional = (1.0 - s.reserve_share) * capital / (1.0 + 1.0 / s.leverage)
        self.quantity = notional / mark
        self.entry = mark
        self.wallet = notional / s.leverage
        cost = self._pay(
            spot_notional=self.quantity * spot,
            perp_notional=notional,
            spot_half=spot_half,
            perp_half=perp_half,
        )
        spent = self.quantity * spot + self.wallet + cost
        if s.path == "topup":
            self.reserve = capital - spent
            return 0.0
        return capital - spent

    def close(self, *, spot: float, mark: float, spot_half: float, perp_half: float) -> float:
        """Close both legs at these prices; returns the cash released."""
        cost = self._pay(
            spot_notional=self.quantity * spot,
            perp_notional=self.quantity * mark,
            spot_half=spot_half,
            perp_half=perp_half,
        )
        cash = self.quantity * spot + self.wallet - self.quantity * (mark - self.entry) - cost
        cash += self.reserve
        self.quantity = self.wallet = self.reserve = 0.0
        return cash

    def liquidation_level(self) -> float:
        return liquidation_price(
            wallet=self.wallet,
            side=-1,
            quantity=self.quantity,
            entry=self.entry,
            maintenance_rate=self.settings.maintenance_rate,
        )

    def step(
        self,
        *,
        spot: float,
        mark: float,
        mark_high: float,
        rate: float,
        spot_half: float,
        perp_half: float,
    ) -> float:
        """Advance one bar. Returns cash released by a liquidation or forced close, else 0."""
        if not self.open:
            return 0.0
        s = self.settings
        if s.path != "ideal" and mark_high >= self.liquidation_level():
            self.liquidations += 1
            self.wallet = 0.0
            cost = self._pay(
                spot_notional=self.quantity * spot,
                perp_notional=0.0,
                spot_half=spot_half,
                perp_half=perp_half,
            )
            cash = self.quantity * spot - cost + self.reserve
            self.quantity = self.reserve = 0.0
            return cash
        received = self.quantity * mark * rate
        self.wallet += received
        self.funding += received
        if s.path == "topup":
            return self._rebalance_margin(
                spot=spot, mark=mark, spot_half=spot_half, perp_half=perp_half
            )
        return 0.0

    def _rebalance_margin(
        self, *, spot: float, mark: float, spot_half: float, perp_half: float
    ) -> float:
        s = self.settings
        unrealized = -self.quantity * (mark - self.entry)
        margin = self.wallet + unrealized
        notional = self.quantity * mark
        target = notional / s.leverage
        if margin <= 0 or notional / margin >= s.leverage * s.topup_multiple:
            need = target - margin
            if need > self.reserve:
                self.forced_closes += 1
                return self.close(spot=spot, mark=mark, spot_half=spot_half, perp_half=perp_half)
            self.reserve -= need
            self.wallet += need
            self.topups += 1
        elif notional / margin <= s.leverage / s.topup_multiple:
            excess = margin - target
            self.wallet -= excess
            self.reserve += excess
        return 0.0

    def equity(self, *, spot: float, mark: float) -> float:
        if not self.open:
            return self.reserve
        return (
            self.quantity * spot + self.wallet - self.quantity * (mark - self.entry) + self.reserve
        )


@dataclass
class CarryResult:
    equity: pd.Series
    funding: pd.Series  # cumulative funding received, per unit of starting capital
    costs: pd.Series
    liquidations: int
    topups: int
    forced_closes: int
    reentries: int
    liquidation_times: list[pd.Timestamp] = field(default_factory=list)


def simulate(bars: pd.DataFrame, *, settings: CarrySettings, capital: float = 1.0) -> CarryResult:
    """One position held throughout, re-entered after a liquidation or forced close.

    `bars` is indexed by bar open time with spot_close, mark_close, mark_high, rate (sum of
    funding prints settled at this bar's close), spot_half and perp_half."""
    position = Position(settings=settings)
    spot = bars["spot_close"].to_numpy(dtype=np.float64)
    mark = bars["mark_close"].to_numpy(dtype=np.float64)
    high = bars["mark_high"].to_numpy(dtype=np.float64)
    rate = bars["rate"].to_numpy(dtype=np.float64)
    sh = bars["spot_half"].to_numpy(dtype=np.float64)
    ph = bars["perp_half"].to_numpy(dtype=np.float64)
    equity = np.empty(len(bars))
    funding = np.empty(len(bars))
    costs = np.empty(len(bars))
    cash = position.enter(
        capital=capital, spot=spot[0], mark=mark[0], spot_half=sh[0], perp_half=ph[0]
    )
    reentries = 0
    liquidation_times: list[pd.Timestamp] = []
    equity[0] = cash + position.equity(spot=spot[0], mark=mark[0])
    funding[0], costs[0] = position.funding, position.costs
    for i in range(1, len(bars)):
        if not position.open and cash > 0:
            cash = position.enter(
                capital=cash,
                spot=spot[i - 1],
                mark=mark[i - 1],
                spot_half=sh[i - 1],
                perp_half=ph[i - 1],
            )
            reentries += 1
        before = position.liquidations
        cash += position.step(
            spot=spot[i],
            mark=mark[i],
            mark_high=high[i],
            rate=rate[i],
            spot_half=sh[i],
            perp_half=ph[i],
        )
        if position.liquidations > before:
            liquidation_times.append(pd.Timestamp(bars.index[i]))
        equity[i] = cash + position.equity(spot=spot[i], mark=mark[i])
        funding[i], costs[i] = position.funding, position.costs
    index = bars.index
    return CarryResult(
        equity=pd.Series(equity / capital, index=index),
        funding=pd.Series(funding / capital, index=index),
        costs=pd.Series(costs / capital, index=index),
        liquidations=position.liquidations,
        topups=position.topups,
        forced_closes=position.forced_closes,
        reentries=reentries,
        liquidation_times=liquidation_times,
    )

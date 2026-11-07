"""Binance USD-M liquidation price.

Liquidation price, one-way mode (Binance, "How to calculate liquidation price"):

    LP = (WB - TMM1 + UPNL1 + cumB - Side1 * Position1 * EP1)
         / (Position1 * MMR_B - Side1 * Position1)

In isolated mode WB is the position's isolated wallet balance and TMM1 = UPNL1 = 0.
"""

from __future__ import annotations


def liquidation_price(
    *,
    wallet: float,
    side: int,
    quantity: float,
    entry: float,
    maintenance_rate: float,
    maintenance_amount: float = 0.0,
    other_maintenance: float = 0.0,
    other_unrealized: float = 0.0,
) -> float:
    if side not in (1, -1):
        raise ValueError("side is 1 (long) or -1 (short)")
    if quantity <= 0:
        raise ValueError("quantity is the absolute position size")
    numerator = (
        wallet - other_maintenance + other_unrealized + maintenance_amount - side * quantity * entry
    )
    denominator = quantity * maintenance_rate - side * quantity
    return numerator / denominator

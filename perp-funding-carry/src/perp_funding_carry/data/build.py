"""Build stage: parse data/raw into tidy Parquet tables in data/processed.

Prices of contracts quoted per 1,000 units are compared with spot times the multiplier,
so both legs are in the same unit. A contract counts as delisted when its data end more
than a week before the catalogue's last month ends.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from perp_funding_carry.config import Config
from perp_funding_carry.data import parsers, vision
from perp_funding_carry.data.fetch import CATALOG_PATH
from perp_funding_carry.data.vision import Catalog, Contract
from perp_funding_carry.paths import ProjectPaths

TABLE_NAMES = (
    "contracts",
    "funding",
    "bars",
    "core_bars",
    "premium",
    "oi",
    "stable",
    "venues",
    "funding_info",
)


@dataclass(frozen=True)
class Tables:
    contracts: pd.DataFrame  # symbol, spot, multiplier, first_bar, last_bar, delisted
    funding: pd.DataFrame  # symbol, time, interval_h, rate
    bars: pd.DataFrame  # symbol, open_time, spot/mark/perp prices, quote volume (universe bars)
    core_bars: pd.DataFrame  # same layout, core symbols at the core interval
    premium: pd.DataFrame  # symbol, open_time, close (1-minute premium index)
    oi: pd.DataFrame  # symbol, time, oi_usd
    stable: pd.DataFrame  # open_time, usd_per_usdt
    venues: pd.DataFrame  # venue, symbol, time, rate
    funding_info: pd.DataFrame  # symbol, cap, floor, interval_h


PRICE_COLUMNS = (
    "mark_high",
    "mark_low",
    "mark_close",
    "perp_close",
    "perp_high",
    "perp_low",
    "spot_close",
    "spot_high",
    "spot_low",
)


def _positive(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].to_numpy(dtype=np.float64)
    return pd.Series(np.isfinite(values).all(axis=1) & (values > 0).all(axis=1), index=frame.index)


def screen(tables: Tables) -> tuple[Tables, dict[str, int]]:
    """Drop records a price or size can never take: a zero, negative or non-finite price in a
    bar, a USDCUSDT close that is not positive, or open interest that is not positive. Binance
    files contain a few such rows; one of them turns a log return or a basis into infinity,
    which no NaN filter catches. Returns the cleaned tables and the rows dropped per table.
    Premium index values are left alone: they are signed by construction."""
    dropped: dict[str, int] = {}
    out = {}
    for name in ("bars", "core_bars"):
        frame = getattr(tables, name)
        columns = [c for c in PRICE_COLUMNS if c in frame.columns]
        keep = _positive(frame, columns) if columns and len(frame) else pd.Series(True, frame.index)
        dropped[name] = int((~keep).sum())
        out[name] = frame.loc[keep].reset_index(drop=True)
    keep = _positive(tables.stable, ["usd_per_usdt"])
    dropped["stable"] = int((~keep).sum())
    out["stable"] = tables.stable.loc[keep].reset_index(drop=True)
    if len(tables.oi):
        keep = _positive(tables.oi, ["oi_usd"])
        dropped["oi"] = int((~keep).sum())
        out["oi"] = tables.oi.loc[keep].reset_index(drop=True)
    else:
        dropped["oi"] = 0
    finite = np.isfinite(tables.funding["rate"].to_numpy(dtype=np.float64))
    dropped["funding"] = int((~finite).sum())
    out["funding"] = tables.funding.loc[finite].reset_index(drop=True)
    return replace(tables, **out), dropped


def _read_many(paths: list[Path], reader: Callable[[Path], pd.DataFrame]) -> pd.DataFrame:
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(reader, existing))
    return pd.concat(frames, ignore_index=True)


def _bars(raw: Path, *, contract: Contract, months: list[str], interval: str) -> pd.DataFrame:
    def load(folder: str, symbol: str) -> pd.DataFrame:
        return _read_many(
            [
                raw / vision.monthly_key(folder, symbol=symbol, interval=interval, month=m)
                for m in months
            ],
            parsers.read_klines,
        )

    mark = load(vision.MARK, contract.symbol)
    perp = load(vision.PERP, contract.symbol)
    spot = load(vision.SPOT, contract.spot)
    if mark.empty or perp.empty or spot.empty:
        return pd.DataFrame()
    k = float(contract.multiplier)
    merged = (
        mark[["open_time", "high", "low", "close"]]
        .rename(columns={"high": "mark_high", "low": "mark_low", "close": "mark_close"})
        .merge(
            perp[["open_time", "close", "high", "low", "quote_volume"]].rename(
                columns={
                    "close": "perp_close",
                    "high": "perp_high",
                    "low": "perp_low",
                    "quote_volume": "perp_quote_volume",
                }
            ),
            on="open_time",
        )
        .merge(
            spot[["open_time", "close", "high", "low"]].rename(
                columns={"close": "spot_close", "high": "spot_high", "low": "spot_low"}
            ),
            on="open_time",
        )
    )
    for column in ("spot_close", "spot_high", "spot_low"):
        merged[column] = merged[column] * k
    merged.insert(0, "symbol", contract.symbol)
    return merged


def _premium(config: Config, *, catalog: Catalog, raw: Path) -> pd.DataFrame:
    """The most recent complete months of 1-minute premium index for the rebuild symbols."""
    frames = []
    for symbol in config.rebuild.symbols:
        listed = catalog.files.get(f"premium/{symbol}/1m", {})
        keys = sorted(k for k, (m, _, _) in listed.items() if m <= catalog.last_month)
        frame = _read_many([raw / k for k in keys[-config.rebuild.months :]], parsers.read_klines)
        if not frame.empty:
            frames.append(frame[["open_time", "close"]].assign(symbol=symbol))
    if not frames:
        return pd.DataFrame(columns=["open_time", "close", "symbol"])
    return pd.concat(frames, ignore_index=True)


def _open_interest(config: Config, *, catalog: Catalog, raw: Path) -> pd.DataFrame:
    u = config.universe
    frames = []
    for symbol in u.core_symbols:
        days = sorted(catalog.files.get(f"metrics/{symbol}", {}).items(), key=lambda kv: kv[1][0])
        frame = _read_many([raw / k for k, _ in days[-u.oi_window_days :]], parsers.read_metrics)
        if not frame.empty:
            frames.append(frame.assign(symbol=symbol))
    if not frames:
        return pd.DataFrame(columns=["time", "oi_usd", "symbol"])
    return pd.concat(frames, ignore_index=True)


def _stablecoin(config: Config, *, catalog: Catalog, raw: Path) -> pd.DataFrame:
    """USD per USDT, through USDC: 1 / USDCUSDT."""
    u = config.universe
    listed = catalog.files.get(f"spot/{u.stablecoin_pair}/{u.bar_interval}", {})
    keys = sorted(k for k, (m, _, _) in listed.items() if u.first_month <= m <= catalog.last_month)
    bars = _read_many([raw / k for k in keys], parsers.read_klines)
    if bars.empty:
        raise RuntimeError(f"no {u.stablecoin_pair} bars to mark USDT to USD")
    return pd.DataFrame({"open_time": bars["open_time"], "usd_per_usdt": 1.0 / bars["close"]})


def build_tables(*, config: Config, paths: ProjectPaths) -> Tables:
    raw = paths.raw / "vision"
    catalog = Catalog.from_json(json.loads((paths.data / CATALOG_PATH).read_text(encoding="utf-8")))
    u = config.universe
    bar_frames, core_frames, funding_frames, rows = [], [], [], []
    for contract in catalog.contracts:
        months = vision.contract_months(config, catalog, contract)
        if len(months) < u.min_months and contract.symbol not in u.core_symbols:
            continue
        bars = _bars(raw, contract=contract, months=months, interval=u.bar_interval)
        funding = _read_many(
            [
                raw
                / vision.monthly_key(vision.FUNDING, symbol=contract.symbol, interval=None, month=m)
                for m in months
            ],
            parsers.read_funding,
        )
        if bars.empty or funding.empty:
            continue
        funding.insert(0, "symbol", contract.symbol)
        bar_frames.append(bars)
        funding_frames.append(funding)
        rows.append(
            {
                "symbol": contract.symbol,
                "spot": contract.spot,
                "multiplier": contract.multiplier,
                "first_bar": bars["open_time"].min(),
                "last_bar": bars["open_time"].max(),
            }
        )
        if contract.symbol in u.core_symbols:
            core = _bars(raw, contract=contract, months=months, interval=u.core_bar_interval)
            if not core.empty:
                core_frames.append(core)
    if not bar_frames:
        raise RuntimeError("no contract has bars and funding inside the window")
    contracts = pd.DataFrame(rows)
    end = pd.Timestamp(catalog.last_month + "-01", tz="UTC") + pd.offsets.MonthEnd(1)
    contracts["delisted"] = contracts["last_bar"] < end - pd.Timedelta(days=7)

    premium = _premium(config, catalog=catalog, raw=raw)
    oi = _open_interest(config, catalog=catalog, raw=raw)
    stable = _stablecoin(config, catalog=catalog, raw=raw)
    funding = pd.concat(funding_frames, ignore_index=True)
    venues = [
        funding.loc[funding["symbol"].isin(u.core_symbols), ["symbol", "time", "rate"]].assign(
            venue="binance"
        )
    ]
    for symbol in u.core_symbols:
        path = paths.raw / "rest" / f"bybit_{symbol}.json"
        if path.is_file():
            venues.append(parsers.read_bybit(path, symbol=symbol).assign(venue="bybit"))
    store = paths.collected / "okx_funding.csv"
    if store.is_file():
        venues.append(parsers.read_okx_store(store, quote=u.quote).assign(venue="okx"))
    return Tables(
        contracts=contracts,
        funding=funding,
        bars=pd.concat(bar_frames, ignore_index=True),
        core_bars=pd.concat(core_frames, ignore_index=True) if core_frames else pd.DataFrame(),
        premium=premium,
        oi=oi,
        stable=stable,
        venues=pd.concat(venues, ignore_index=True)[["venue", "symbol", "time", "rate"]],
        funding_info=parsers.read_funding_info(paths.raw / "rest" / "binance_fundingInfo.json"),
    )


def write_tables(tables: Tables, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        getattr(tables, name).to_parquet(directory / f"{name}.parquet", index=False)


def read_tables(directory: Path) -> Tables:
    return Tables(**{n: pd.read_parquet(directory / f"{n}.parquet") for n in TABLE_NAMES})

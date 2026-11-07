"""Build stage: parse data/raw into tidy Parquet tables in data/processed.

Tables:
- implied: date, symbol, value (every Cboe index and DVOL, in volatility points)
- etf: date, ticker, open, high, low, close (adjusted)
- rf: date, rate
- vx: trade_date, expiry, settle (pre-2007 contracts rescaled to index points)
- btc_daily: date (UTC day), open, high, low, close, minutes, rv1, rv5, rk
- btc_5m: time (end of the 5-minute interval), ret
- chains: snapshot, symbol, type, strike, expiration, bid, ask, mark, forward, age_minutes,
  window_trades (rebuilt from Deribit's trade history; see models.chain)
- dvol_snapshot: snapshot, dvol
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from cross_asset_vrp.config import Config
from cross_asset_vrp.data import cboe, crypto, prices
from cross_asset_vrp.data.fetch import DVOL_DAILY_KEY, DVOL_SNAPSHOT_KEY, TRADES_PREFIX
from cross_asset_vrp.data.manifest import Manifest
from cross_asset_vrp.models import chain
from cross_asset_vrp.models.realized import realized_kernel
from cross_asset_vrp.paths import ProjectPaths

Say = Callable[[str], object]

TABLE_NAMES = ("implied", "etf", "rf", "vx", "btc_daily", "btc_5m", "chains", "dvol_snapshot")
MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class Tables:
    implied: pd.DataFrame
    etf: pd.DataFrame
    rf: pd.DataFrame
    vx: pd.DataFrame
    btc_daily: pd.DataFrame
    btc_5m: pd.DataFrame
    chains: pd.DataFrame
    dvol_snapshot: pd.DataFrame


def _entries(manifest: Manifest, prefix: str) -> list[tuple[str, str]]:
    return sorted((k, e.path) for k, e in manifest.entries.items() if k.startswith(prefix))


def _implied(config: Config, data: Path, manifest: Manifest) -> pd.DataFrame:
    frames = []
    for key, rel in _entries(manifest, "cboe/"):
        symbol = key.split("/", 1)[1]
        frames.append(cboe.read_index(data / rel, symbol=symbol).assign(symbol=symbol))
    candles = json.loads((data / manifest.entries[DVOL_DAILY_KEY].path).read_text(encoding="utf-8"))
    dvol = crypto.dvol_frame(candles)
    frames.append(
        pd.DataFrame(
            {
                "date": dvol["time"].dt.tz_localize(None).dt.normalize(),
                "value": dvol["close"],
                "symbol": config.crypto.implied,
            }
        )
    )
    out = pd.concat(frames, ignore_index=True)[["date", "symbol", "value"]]
    return out.drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)


def _vx(config: Config, *, data: Path, manifest: Manifest, implied: pd.DataFrame) -> pd.DataFrame:
    f = config.futures
    frames, errors = [], []
    archived = {
        date.fromisoformat(key.split("/", 1)[1]): rel
        for key, rel in _entries(manifest, "cfe-archive/")
    }
    for key, rel in _entries(manifest, "cfe/"):
        expiry = date.fromisoformat(key.split("/", 1)[1])
        try:
            contract = cboe.read_futures(data / rel, expiry=expiry)
            if expiry in archived:
                contract = cboe.merge_sources(
                    contract, cboe.read_futures(data / archived[expiry], expiry=expiry)
                )
            frames.append(contract)
        except cboe.SchemaError as error:
            errors.append(str(error))
    if errors:  # every unreadable contract at once, not one per run
        raise cboe.SchemaError(f"{len(errors)} VX files could not be read:\n" + "\n".join(errors))
    vx = pd.concat(frames, ignore_index=True)
    old = vx["trade_date"] < pd.Timestamp(f.rescale_before)
    vx.loc[old, "settle"] = vx.loc[old, "settle"] / f.rescale_factor
    # Check the rescaling: the nearest contract's settle over VIX, year by year.
    vix = implied.loc[implied["symbol"] == "VIX"].set_index("date")["value"]
    front = vx.sort_values(["trade_date", "expiry"]).groupby("trade_date").first()
    ratio = (front["settle"] / vix.reindex(front.index)).dropna()
    by_year = ratio.groupby(pd.DatetimeIndex(ratio.index).year).median()
    lo, hi = f.rescale_check
    bad = by_year[(by_year < lo) | (by_year > hi)]
    if not bad.empty:
        raise ValueError(
            f"front VX settle over VIX is outside {lo}-{hi} in {list(bad.index)}: "
            "the 10x rescaling date or a file layout has changed"
        )
    return vx.sort_values(["expiry", "trade_date"]).reset_index(drop=True)


def _minute_month(path: Path, prev_close: float | None) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    m = crypto.read_minutes(path)
    logc = np.log(m["close"].to_numpy())
    r1 = np.diff(np.concatenate([[np.log(prev_close) if prev_close else np.nan], logc]))
    # A gap of missing minutes would fold into one long "minute"; such returns are dropped.
    # (The first row's return spans the month boundary and is kept when the previous month
    # ended on the minute before.)
    step = m["time"].diff().to_numpy().copy()
    step[0] = np.timedelta64(60, "s")
    r1[step != np.timedelta64(60, "s")] = np.nan
    m["r1"] = r1
    m["day"] = m["time"].dt.tz_localize(None).dt.normalize()
    grid = m.set_index("time")["close"].resample("5min", label="right", closed="left").last()
    r5 = np.log(grid).diff()
    if prev_close:
        r5.iloc[0] = np.log(grid.iloc[0]) - np.log(prev_close)
    five = pd.DataFrame({"time": grid.index, "ret": r5.to_numpy()}).dropna()
    daily = m.groupby("day").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        minutes=("close", "size"),
    )
    daily["rv1"] = m.groupby("day")["r1"].apply(lambda x: float(np.nansum(x.to_numpy() ** 2)))
    five["day"] = (five["time"] - pd.Timedelta(minutes=5)).dt.tz_localize(None).dt.normalize()
    daily["rv5"] = five.groupby("day")["ret"].apply(lambda x: float(np.sum(x.to_numpy() ** 2)))
    daily["rk"] = m.groupby("day")["r1"].apply(lambda x: realized_kernel(x.to_numpy()))
    return (
        daily.reset_index().rename(columns={"day": "date"}),
        five[["time", "ret"]],
        float(m["close"].iloc[-1]),
    )


def _btc(data: Path, manifest: Manifest, *, say: Say) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries = [(k, rel) for k, rel in _entries(manifest, "vision/") if rel.endswith(".zip")]
    days, fives = [], []
    prev: float | None = None
    for i, (_, rel) in enumerate(entries, start=1):
        daily, five, prev = _minute_month(data / rel, prev)
        days.append(daily)
        fives.append(five)
        if i % 12 == 0 or i == len(entries):
            say(f"  bitcoin minute files: {i} / {len(entries)}")
    daily = (
        pd.concat(days, ignore_index=True)
        .groupby("date", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            minutes=("minutes", "sum"),
            rv1=("rv1", "sum"),
            rv5=("rv5", "sum"),
            rk=("rk", "sum"),
        )
    )
    return daily, pd.concat(fives, ignore_index=True).drop_duplicates("time")


def _chains(
    config: Config, data: Path, manifest: Manifest, *, say: Say
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The chain at each month start, rebuilt from that month's trade window."""
    r = config.replication
    frames = []
    entries = list(_entries(manifest, TRADES_PREFIX))
    for n, (key, rel) in enumerate(entries, start=1):
        snapshot = pd.Timestamp(key.removeprefix(TRADES_PREFIX), tz="UTC")
        trades = crypto.trades_frame(json.loads((data / rel).read_text(encoding="utf-8")))
        rebuilt = chain.rebuild(
            trades,
            at=snapshot,
            spread_buckets=r.spread_buckets,
            min_spread_trades=r.min_spread_trades,
        )
        if rebuilt is not None:
            frames.append(rebuilt.chain.assign(snapshot=snapshot, window_trades=rebuilt.trades))
        if n % max(1, len(entries) // 5) == 0 or n == len(entries):
            say(f"  chains rebuilt: {n:,} / {len(entries):,}")
    chains = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    snaps = json.loads(
        (data / manifest.entries[DVOL_SNAPSHOT_KEY].path).read_text(encoding="utf-8")
    )
    snapshot_rows = []
    for day, candles in snaps.items():
        frame = crypto.dvol_frame(candles)
        if frame.empty:
            continue
        snapshot_rows.append(
            {"snapshot": pd.Timestamp(day, tz="UTC"), "dvol": float(frame["close"].iloc[-1])}
        )
    return chains, pd.DataFrame(snapshot_rows, columns=["snapshot", "dvol"])


def build_tables(*, config: Config, paths: ProjectPaths, say: Say = print) -> Tables:
    manifest = Manifest.read(paths.manifest)
    if not manifest.entries:
        raise FileNotFoundError("data/manifest.json is empty: run `vrp fetch` first")
    data = paths.data
    implied = _implied(config, data, manifest)
    etf = pd.concat(
        [
            prices.read_tiingo(data / rel, ticker=key.split("/", 1)[1])
            for key, rel in _entries(manifest, "tiingo/")
        ],
        ignore_index=True,
    )
    fred_key = f"fred/{config.risk_free.series}"
    rf = prices.read_fred(data / manifest.entries[fred_key].path, series=config.risk_free.series)
    vx = _vx(config, data=data, manifest=manifest, implied=implied)
    btc_daily, btc_5m = _btc(data, manifest, say=say)
    say("Rebuilding the bitcoin option chains from the trade windows")
    chains, dvol_snapshot = _chains(config, data, manifest, say=say)
    return Tables(
        implied=implied,
        etf=etf,
        rf=rf,
        vx=vx,
        btc_daily=btc_daily,
        btc_5m=btc_5m,
        chains=chains,
        dvol_snapshot=dvol_snapshot,
    )


def write_tables(tables: Tables, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        getattr(tables, name).to_parquet(directory / f"{name}.parquet", index=False)


def read_tables(directory: Path) -> Tables:
    return Tables(**{n: pd.read_parquet(directory / f"{n}.parquet") for n in TABLE_NAMES})

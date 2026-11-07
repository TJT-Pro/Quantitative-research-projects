"""Readers for each source format. Every reader checks its layout and fails loudly.

Times are UTC pandas Timestamps; rates are decimals per funding interval.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


class SchemaError(ValueError):
    """A source file does not have the layout this pipeline was written against."""


def _csv_bytes(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise SchemaError(f"{path.name}: expected one CSV inside, found {members}")
        return archive.read(members[0])


def _has_header(data: bytes) -> bool:
    first = data.split(b"\n", 1)[0].split(b",", 1)[0].strip()
    return bool(first) and not first.lstrip(b"-").replace(b".", b"").isdigit()


def epoch_to_time(values: pd.Series) -> pd.Series:
    """Milliseconds, or microseconds (spot files from 2025), to UTC timestamps."""
    numbers = pd.to_numeric(values, errors="raise").astype("int64")
    micro = numbers > 10**14
    millis = np.where(micro, numbers // 1000, numbers)
    return pd.Series(pd.to_datetime(millis, unit="ms", utc=True), index=values.index)


def read_klines(path: Path) -> pd.DataFrame:
    data = _csv_bytes(path)
    header = 0 if _has_header(data) else None
    frame = pd.read_csv(io.BytesIO(data), header=header)
    if frame.shape[1] != len(KLINE_COLUMNS):
        raise SchemaError(f"{path.name}: {frame.shape[1]} columns, expected {len(KLINE_COLUMNS)}")
    frame.columns = KLINE_COLUMNS
    out = pd.DataFrame({"open_time": epoch_to_time(frame["open_time"])})
    for column in ("open", "high", "low", "close", "quote_volume"):
        out[column] = pd.to_numeric(frame[column], errors="coerce")
    return out.dropna().drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def read_funding(path: Path) -> pd.DataFrame:
    data = _csv_bytes(path)
    if _has_header(data):
        frame = pd.read_csv(io.BytesIO(data))
        missing = [c for c in FUNDING_COLUMNS if c not in frame.columns]
        if missing:
            raise SchemaError(f"{path.name}: missing {missing}; found {list(frame.columns)}")
    else:
        frame = pd.read_csv(io.BytesIO(data), header=None, names=FUNDING_COLUMNS)
    out = pd.DataFrame(
        {
            "time": epoch_to_time(frame["calc_time"]).dt.round("1min"),
            "interval_h": pd.to_numeric(frame["funding_interval_hours"], errors="coerce"),
            "rate": pd.to_numeric(frame["last_funding_rate"], errors="coerce"),
        }
    )
    return out.dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)


def read_metrics(path: Path) -> pd.DataFrame:
    data = _csv_bytes(path)
    frame = pd.read_csv(io.BytesIO(data))
    needed = ["create_time", "sum_open_interest_value"]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise SchemaError(f"{path.name}: missing {missing}; found {list(frame.columns)}")
    times = pd.to_datetime(frame["create_time"], utc=True, format="mixed")
    return pd.DataFrame(
        {"time": times, "oi_usd": pd.to_numeric(frame["sum_open_interest_value"], errors="coerce")}
    ).dropna()


def read_funding_info(path: Path) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SchemaError(f"{path.name}: expected a JSON list")
    return pd.DataFrame(
        {
            "symbol": [r["symbol"] for r in rows],
            "cap": [float(r["adjustedFundingRateCap"]) for r in rows],
            "floor": [float(r["adjustedFundingRateFloor"]) for r in rows],
            "interval_h": [int(r["fundingIntervalHours"]) for r in rows],
        }
    )


def read_bybit(path: Path, *, symbol: str) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [int(r["fundingRateTimestamp"]) for r in rows], unit="ms", utc=True
            ),
            "rate": [float(r["fundingRate"]) for r in rows],
        }
    )
    frame.insert(0, "symbol", symbol)
    return frame.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def read_okx_store(path: Path, *, quote: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    symbol = frame["instrument"].str.split("-").str[0] + quote
    return (
        pd.DataFrame(
            {
                "symbol": symbol,
                "time": pd.to_datetime(frame["time_ms"], unit="ms", utc=True).dt.round("1min"),
                "rate": frame["rate"].astype(float),
            }
        )
        .drop_duplicates(["symbol", "time"])
        .sort_values(["symbol", "time"])
        .reset_index(drop=True)
    )

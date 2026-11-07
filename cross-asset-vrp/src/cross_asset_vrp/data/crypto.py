"""Bitcoin sources: Binance Vision minute bars, Deribit DVOL and Deribit's option trade history.

Binance Vision: data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-<YYYY-MM>.zip, one CSV of
twelve columns per month (a header only in newer months; open times in microseconds from
2025). The bucket listing gives each file's MD5.

Deribit: public/get_volatility_index_data?currency=BTC&start_timestamp=&end_timestamp=
&resolution=, answering result.data = [[ms, open, high, low, close], ...] and
result.continuation, the end_timestamp of the next (older) page, or null.

Deribit history (history.deribit.com): public/get_last_trades_by_currency_and_time keeps every
trade since 2016, expired options included, with the option's mark price, the index price,
the taker's side and Deribit's own implied volatility of the mark. Pages hold up to 10,000
trades in time order with result.has_more; the next page starts at the last timestamp
received, and trades are de-duplicated by trade_id.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from cross_asset_vrp.config import Config

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
MINUTE_FOLDER = "data/spot/monthly/klines/{symbol}/1m/"
_MONTH = re.compile(r"-(\d{4}-\d{2})\.zip$")


class SchemaError(ValueError):
    """A source file does not have the layout this pipeline was written against."""


# Binance ------------------------------------------------------------------------------------


def minute_folder(symbol: str) -> str:
    return MINUTE_FOLDER.format(symbol=symbol)


def month_of(key: str) -> str | None:
    match = _MONTH.search(key)
    return match.group(1) if match else None


def _csv_bytes(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise SchemaError(f"{path.name}: expected one CSV inside, found {members}")
        return archive.read(members[0])


def read_minutes(path: Path) -> pd.DataFrame:
    """time (UTC open of the minute), open, high, low, close."""
    data = _csv_bytes(path)
    first = data.split(b"\n", 1)[0].split(b",", 1)[0].strip()
    header = 0 if first and not first.isdigit() else None
    frame = pd.read_csv(io.BytesIO(data), header=header)
    if frame.shape[1] != len(KLINE_COLUMNS):
        raise SchemaError(f"{path.name}: {frame.shape[1]} columns, expected {len(KLINE_COLUMNS)}")
    frame.columns = KLINE_COLUMNS
    stamp = pd.to_numeric(frame["open_time"], errors="raise").astype("int64").to_numpy()
    millis = np.where(stamp > 10**14, stamp // 1000, stamp)
    out = pd.DataFrame({"time": pd.to_datetime(millis, unit="ms", utc=True)})
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy()
    out = out.dropna()
    out = out.loc[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


# Deribit ------------------------------------------------------------------------------------


def dvol_url(config: Config, *, start_ms: int, end_ms: int, resolution: str) -> str:
    return (
        f"{config.sources.deribit}/public/get_volatility_index_data?currency=BTC"
        f"&start_timestamp={start_ms}&end_timestamp={end_ms}&resolution={resolution}"
    )


def parse_dvol_page(text: str) -> tuple[list[list[float]], int | None]:
    payload = json.loads(text)
    if "result" not in payload:
        raise SchemaError(f"Deribit answered without a result: {payload.get('error', payload)}")
    result = payload["result"]
    data = result.get("data")
    if not isinstance(data, list):
        raise SchemaError("Deribit result has no data list")
    continuation = result.get("continuation")
    return data, int(continuation) if continuation is not None else None


def dvol_frame(candles: Iterable[list[float]]) -> pd.DataFrame:
    """time, close (DVOL in volatility points)."""
    rows = list(candles)
    if rows and len(rows[0]) != 5:
        raise SchemaError(f"DVOL candles have {len(rows[0])} fields, expected 5")
    frame = pd.DataFrame(rows, columns=["ms", "open", "high", "low", "close"])
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(frame["ms"].astype("int64"), unit="ms", utc=True),
            "close": frame["close"].astype(float),
        }
    )
    return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


# Deribit trade history --------------------------------------------------------------------

TRADE_FIELDS = (
    "trade_id",
    "instrument_name",
    "timestamp",
    "price",
    "mark_price",
    "index_price",
    "direction",
    "iv",
)


def trades_url(config: Config, *, start_ms: int, end_ms: int) -> str:
    return (
        f"{config.sources.deribit_history}/public/get_last_trades_by_currency_and_time"
        f"?currency=BTC&kind=option&start_timestamp={start_ms}&end_timestamp={end_ms}"
        f"&count={config.replication.page_size}&include_old=true&sorting=asc"
    )


def parse_trades_page(text: str) -> tuple[list[dict[str, object]], bool]:
    """The trades on one page and whether more follow."""
    payload = json.loads(text)
    if "result" not in payload:
        raise SchemaError(f"Deribit answered without a result: {payload.get('error', payload)}")
    result = payload["result"]
    trades = result.get("trades")
    if not isinstance(trades, list):
        raise SchemaError("Deribit trade history has no trades list")
    for trade in trades[:1]:
        missing = [f for f in TRADE_FIELDS if f not in trade]
        if missing:
            raise SchemaError(f"Deribit trades are missing {missing}; found {sorted(trade)}")
    return trades, bool(result.get("has_more"))


def trades_frame(trades: Iterable[dict[str, object]]) -> pd.DataFrame:
    rows = list(trades)
    frame = (
        pd.DataFrame(rows, columns=list(TRADE_FIELDS))
        if rows
        else pd.DataFrame(columns=list(TRADE_FIELDS))
    )
    frame = frame[list(TRADE_FIELDS)].drop_duplicates("trade_id")
    for column in ("timestamp", "price", "mark_price", "index_price", "iv"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().reset_index(drop=True)

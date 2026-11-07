"""Daily ETF prices from Tiingo and the 3-month Treasury bill from FRED.

Tiingo: GET tiingo/daily/<ticker>/prices?startDate=...&format=json with the key in an
`Authorization: Token <key>` header, so the key never appears in a URL, a file or the
manifest. Each row carries raw and adjusted OHLC; the adjusted fields are used, because the
overnight return across an ex-dividend date is otherwise a fake price drop.

FRED: fredgraph.csv?id=DTB3 with columns observation_date (DATE in older files) and DTB3, in
percent on a discount basis, "." for a missing day.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from cross_asset_vrp.config import Config

TIINGO_FIELDS = ("date", "adjOpen", "adjHigh", "adjLow", "adjClose")


class SchemaError(ValueError):
    """A source file does not have the layout this pipeline was written against."""


def tiingo_url(config: Config, ticker: str) -> str:
    return (
        f"{config.sources.tiingo}/{ticker.lower()}/prices"
        f"?startDate={config.windows.first_date}&format=json&resampleFreq=daily"
    )


def tiingo_headers(config: Config, *, required: bool) -> dict[str, str]:
    key = os.environ.get(config.sources.tiingo_key_env, "").strip()
    if not key:
        if required:
            raise RuntimeError(
                f"set {config.sources.tiingo_key_env} to your Tiingo API key (free at "
                "tiingo.com): in PowerShell, $env:"
                f'{config.sources.tiingo_key_env} = "<key>"'
            )
        return {}
    return {"Authorization": f"Token {key}", "Content-Type": "application/json"}


def read_tiingo(path: Path, *, ticker: str) -> pd.DataFrame:
    """date, ticker, open, high, low, close (dividend- and split-adjusted)."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        raise SchemaError(f"{path.name}: Tiingo returned an error: {rows.get('detail', rows)}")
    if not rows:
        raise SchemaError(f"{path.name}: no rows")
    missing = [f for f in TIINGO_FIELDS if f not in rows[0]]
    if missing:
        raise SchemaError(f"{path.name}: missing {missing}; found {sorted(rows[0])}")
    frame = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None).dt.normalize(),
            "ticker": ticker,
            "open": pd.to_numeric(frame["adjOpen"], errors="coerce"),
            "high": pd.to_numeric(frame["adjHigh"], errors="coerce"),
            "low": pd.to_numeric(frame["adjLow"], errors="coerce"),
            "close": pd.to_numeric(frame["adjClose"], errors="coerce"),
        }
    )
    out = out.dropna()
    ok = (out[["open", "high", "low", "close"]] > 0).all(axis=1) & (out["high"] >= out["low"])
    return out.loc[ok].drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fred_url(config: Config) -> str:
    return f"{config.sources.fred}?id={config.risk_free.series}"


def read_fred(path: Path, *, series: str) -> pd.DataFrame:
    """date, rate (a decimal annual rate)."""
    frame = pd.read_csv(path)
    date_column = next((c for c in ("observation_date", "DATE") if c in frame.columns), None)
    if date_column is None or series not in frame.columns:
        raise SchemaError(f"{path.name}: expected a date column and {series}; found {list(frame)}")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], format="%Y-%m-%d", errors="coerce"),
            "rate": pd.to_numeric(frame[series], errors="coerce") / 100.0,
        }
    )
    return out.dropna().sort_values("date").reset_index(drop=True)

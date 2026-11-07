"""Order-book snapshots: the best bid and ask of every spot and USD-M pair, with 24-hour volume.

Spreads cannot be read from historical bars. On a crypto pair the quoted spread is a few
basis points while the price moves several percent in four hours, so a range-based estimator
measures volatility, not spread. Binance serves the live best bid and ask of every symbol
in one public request per market. Each run appends one snapshot to a local store under
data/collected, so the sample of measured spreads grows the way the OKX store does, and the
cost model fits half-spreads to volume across contracts and snapshots
(models.costs.fit_spread_model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from perp_funding_carry.config import Config
from perp_funding_carry.data.download import Client
from perp_funding_carry.data.manifest import Entry, sha256_of

STORE = "collected/book_snapshots.csv"
COLUMNS = ["taken_at", "market", "symbol", "bid", "ask", "quote_volume", "origin"]


def endpoints(config: Config) -> dict[str, tuple[str, str]]:
    """Market -> (bookTicker URL, 24-hour ticker URL)."""
    spot, fapi = config.sources.binance_spot_api, config.sources.binance_fapi
    return {
        "spot": (f"{spot}/api/v3/ticker/bookTicker", f"{spot}/api/v3/ticker/24hr"),
        "perp": (f"{fapi}/fapi/v1/ticker/bookTicker", f"{fapi}/fapi/v1/ticker/24hr"),
    }


def parse_snapshot(
    *, books: str, tickers: str, market: str, quote: str, taken_at: str, origin: str
) -> pd.DataFrame:
    """One market's snapshot: quote-currency symbols with a positive, uncrossed book."""
    book_rows = json.loads(books)
    ticker_rows = json.loads(tickers)
    if not isinstance(book_rows, list) or not isinstance(ticker_rows, list):
        raise ValueError(f"{market} snapshot: expected JSON lists")
    volume = {r["symbol"]: float(r["quoteVolume"]) for r in ticker_rows}
    frame = pd.DataFrame(
        {
            "symbol": [r["symbol"] for r in book_rows],
            "bid": [float(r["bidPrice"]) for r in book_rows],
            "ask": [float(r["askPrice"]) for r in book_rows],
        }
    )
    frame["quote_volume"] = frame["symbol"].map(volume)
    keep = (
        frame["symbol"].str.endswith(quote)
        & (frame["bid"] > 0)
        & (frame["ask"] > frame["bid"])
        & (frame["quote_volume"] > 0)
    )
    frame = frame.loc[keep].reset_index(drop=True)
    frame.insert(0, "market", market)
    frame.insert(0, "taken_at", taken_at)
    frame["origin"] = origin
    return frame[COLUMNS]


@dataclass(frozen=True)
class Snapshot:
    rows_added: int
    snapshots: int


def snapshot(config: Config, client: Client, *, data_dir: Path) -> Snapshot:
    """Take one snapshot of both markets and append it to the store."""
    taken_at = pd.Timestamp.now(tz="UTC").floor("min").isoformat()
    frames = []
    for market, (book_url, ticker_url) in endpoints(config).items():
        frames.append(
            parse_snapshot(
                books=client.get_text(book_url),
                tickers=client.get_text(ticker_url),
                market=market,
                quote=config.universe.quote,
                taken_at=taken_at,
                origin=client.origin,
            )
        )
    fresh = pd.concat(frames, ignore_index=True)
    store = data_dir / STORE
    old = pd.read_csv(store) if store.is_file() else fresh.iloc[0:0]
    merged = (
        pd.concat([old, fresh], ignore_index=True)
        .drop_duplicates(["taken_at", "market", "symbol"], keep="first")
        .sort_values(["taken_at", "market", "symbol"])
        .reset_index(drop=True)
    )
    store.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(store, index=False)
    origins = set(merged["origin"])
    client.manifest.put(
        "collected/books",
        Entry(
            url=endpoints(config)["perp"][0],
            path=STORE,
            sha256=sha256_of(store),
            bytes=store.stat().st_size,
            retrieved_at=taken_at,
            origin="official" if origins == {"official"} else "mirror",
        ),
    )
    return Snapshot(rows_added=len(merged) - len(old), snapshots=int(merged["taken_at"].nunique()))


def read_store(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: run `pfc snapshot` once to measure bid-ask spreads"
        )
    frame = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing {missing}")
    return frame

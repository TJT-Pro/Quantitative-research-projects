"""Public REST sources: Binance funding info, Bybit and OKX funding history.

Bybit pages backward: a request with only endTime returns the 200 prints up to it, so
the next page ends one millisecond before the earliest print received. OKX keeps about
three months; each run's prints are appended to a local store under data/collected, so
the cross-venue sample grows with every run instead of being capped at 93 days.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from perp_funding_carry.config import Config
from perp_funding_carry.data.download import Client, FetchResult
from perp_funding_carry.data.manifest import Entry, sha256_of


def funding_info_url(config: Config) -> str:
    return f"{config.sources.binance_fapi}/fapi/v1/fundingInfo"


def bybit_page_url(config: Config, *, symbol: str, end: int | None) -> str:
    base = f"{config.sources.bybit_api}/v5/market/funding/history"
    url = f"{base}?category=linear&symbol={symbol}&limit=200"
    return url + (f"&endTime={end}" if end is not None else "")


def okx_instrument(config: Config, symbol: str) -> str:
    quote = config.universe.quote
    return f"{symbol[: -len(quote)]}-{quote}-SWAP"


def okx_page_url(config: Config, *, instrument: str, after: int | None) -> str:
    url = (
        f"{config.sources.okx_api}/api/v5/public/funding-rate-history?instId={instrument}&limit=100"
    )
    return url + (f"&after={after}" if after is not None else "")


@dataclass(frozen=True)
class Collected:
    results: list[FetchResult]
    okx_rows_added: int


def _bybit(
    config: Config, client: Client, *, symbol: str, pause: Callable[[], None]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    end: int | None = None
    while True:
        payload = json.loads(client.get_text(bybit_page_url(config, symbol=symbol, end=end)))
        if payload.get("retCode") != 0:
            raise RuntimeError(f"Bybit {symbol}: {payload.get('retMsg')}")
        page = payload["result"]["list"]
        if not page:
            return rows
        rows += page
        earliest = min(int(r["fundingRateTimestamp"]) for r in page)
        if end is not None and earliest >= end:
            return rows
        end = earliest - 1
        pause()


def _okx(
    config: Config, client: Client, *, instrument: str, pause: Callable[[], None]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    after: int | None = None
    while True:
        payload = json.loads(
            client.get_text(okx_page_url(config, instrument=instrument, after=after))
        )
        if str(payload.get("code")) != "0":
            raise RuntimeError(f"OKX {instrument}: {payload.get('msg')}")
        page = payload["data"]
        if not page:
            return rows
        rows += page
        earliest = min(int(r["fundingTime"]) for r in page)
        if after is not None and earliest >= after:
            return rows
        after = earliest
        pause()


def _append_okx_store(store: Path, rows: list[dict[str, str]], *, origin: str) -> int:
    fresh = pd.DataFrame(
        {
            "instrument": [r["instId"] for r in rows],
            "time_ms": [int(r["fundingTime"]) for r in rows],
            "rate": [float(r.get("realizedRate") or r["fundingRate"]) for r in rows],
            "origin": origin,
        }
    )
    if store.is_file():
        old = pd.read_csv(store)
        merged = pd.concat([old, fresh], ignore_index=True)
    else:
        old = fresh.iloc[0:0]
        merged = fresh
    merged = merged.drop_duplicates(["instrument", "time_ms"], keep="first")
    merged = merged.sort_values(["instrument", "time_ms"]).reset_index(drop=True)
    store.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(store, index=False)
    return int(len(merged) - len(old))


def collect(config: Config, client: Client, *, data_dir: Path) -> Collected:
    def pause() -> None:
        if client.mirror is None:
            time.sleep(config.download.rest_pause_seconds)

    results = []
    url = funding_info_url(config)
    results.append(
        client.save_text(
            key="rest/binance/fundingInfo",
            url=url,
            relpath="raw/rest/binance_fundingInfo.json",
            text=client.get_text(url),
        )
    )
    okx_rows: list[dict[str, str]] = []
    for symbol in config.universe.core_symbols:
        bybit = _bybit(config, client, symbol=symbol, pause=pause)
        results.append(
            client.save_text(
                key=f"rest/bybit/{symbol}",
                url=bybit_page_url(config, symbol=symbol, end=None),
                relpath=f"raw/rest/bybit_{symbol}.json",
                text=json.dumps(bybit),
            )
        )
        instrument = okx_instrument(config, symbol)
        page = _okx(config, client, instrument=instrument, pause=pause)
        okx_rows += page
        results.append(
            client.save_text(
                key=f"rest/okx/{instrument}",
                url=okx_page_url(config, instrument=instrument, after=None),
                relpath=f"raw/rest/okx_{instrument}.json",
                text=json.dumps(page),
            )
        )
    store = data_dir / "collected" / "okx_funding.csv"
    added = _append_okx_store(store, okx_rows, origin=client.origin) if okx_rows else 0
    if store.is_file():
        origins = set(pd.read_csv(store)["origin"])
        entry = Entry(
            url=okx_page_url(config, instrument="*", after=None),
            path="collected/okx_funding.csv",
            sha256=sha256_of(store),
            bytes=store.stat().st_size,
            retrieved_at=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            origin="official" if origins == {"official"} else "mirror",
        )
        client.manifest.put("collected/okx", entry)
    return Collected(results=results, okx_rows_added=added)

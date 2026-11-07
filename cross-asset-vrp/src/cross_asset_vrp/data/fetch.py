"""Fetch stage: every source, in order, with progress lines and a SHA-256 manifest entry each.

A file that has not changed is not downloaded again: bucket files are skipped on a matching
MD5, plain URLs are revalidated with a conditional GET, and option trade windows of past
months, which never change, are kept once their file matches the manifest.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from cross_asset_vrp.config import Config
from cross_asset_vrp.data import cboe, crypto, prices
from cross_asset_vrp.data.download import Client, FetchResult, MissingError, Target
from cross_asset_vrp.data.listing import list_prefix
from cross_asset_vrp.data.manifest import Manifest, official, sha256_of
from cross_asset_vrp.paths import ProjectPaths

TRADES_PREFIX = "deribit/trades/"
DVOL_SNAPSHOT_KEY = "deribit/dvol_snapshots"
DVOL_DAILY_KEY = "deribit/dvol_1D"
BTC_LISTING_KEY = "vision/listing"


@dataclass
class FetchSummary:
    files: int = 0
    requested: int = 0
    changed: int = 0
    missing_contracts: list[str] = field(default_factory=list)
    btc_months: int = 0
    snapshots: int = 0


Say = Callable[[str], None]


def client_for(
    *, config: Config, paths: ProjectPaths, manifest: Manifest, mirror: Path | None
) -> Client:
    return Client(data_dir=paths.data, manifest=manifest, settings=config.download, mirror=mirror)


def _record(summary: FetchSummary, result: FetchResult) -> None:
    summary.files += 1
    summary.requested += int(result.requested)
    summary.changed += int(result.changed)


def _fetch_group(
    client: Client, targets: list[Target], *, summary: FetchSummary, say: Say, label: str
) -> list[tuple[Target, Exception]]:
    lock = threading.Lock()
    done = {"n": 0}
    step = max(1, len(targets) // 10)

    def on_done(_: Target, result: FetchResult) -> None:
        with lock:
            _record(summary, result)
            done["n"] += 1
            if done["n"] % step == 0 or done["n"] == len(targets):
                say(f"  {label}: {done['n']:,} / {len(targets):,}")

    return client.fetch_many(targets, on_done=on_done)


def _cboe(config: Config, client: Client, *, summary: FetchSummary, say: Say, today: date) -> None:
    symbols = [a.implied for a in config.cboe_assets]
    symbols += [config.term_structure.short, config.term_structure.long]
    targets = [
        Target(key=f"cboe/{s}", url=cboe.index_url(config, s), relpath=f"raw/cboe/{s}_History.csv")
        for s in symbols
    ]
    say(f"Cboe volatility indices: {len(targets)} files")
    failures = _fetch_group(client, targets, summary=summary, say=say, label="indices")
    if failures:
        raise RuntimeError(
            "Cboe index downloads failed: " + "; ".join(f"{t.key}: {e}" for t, e in failures)
        )
    horizon = (pd.Timestamp(today) + pd.DateOffset(months=9)).date()
    contracts = cboe.expiries(config, last=horizon)
    futures = [
        Target(
            key=f"cfe/{e.isoformat()}",
            url=cboe.futures_url(config, e),
            relpath=f"raw/cfe/VX_{e.isoformat()}.csv",
            # Cboe's CDN answers 403 for a contract file that does not exist.
            missing_statuses=(403, 404),
        )
        for e in contracts
    ]
    # Expiries held in both places also get their archive file (merged in the build).
    archives = [
        Target(
            key=f"cfe-archive/{e.isoformat()}",
            url=cboe.archive_url(config, e),
            relpath=f"raw/cfe/archive/{cboe.archive_name(e)}",
            missing_statuses=(403, 404),
        )
        for e in contracts
        if cboe.needs_archive_too(config, e)
    ]
    say(
        f"VX futures: {len(futures)} monthly contracts from {contracts[0]} to {contracts[-1]}, "
        f"plus {len(archives)} archive files for expiries held in both places"
    )
    archive_failures = _fetch_group(client, archives, summary=summary, say=say, label="archive")
    hard = [(t, e) for t, e in archive_failures if not isinstance(e, MissingError)]
    if hard:
        raise RuntimeError(
            "VX archive downloads failed: " + "; ".join(f"{t.key}: {e}" for t, e in hard[:5])
        )
    for t, _ in archive_failures:
        client.manifest.entries.pop(t.key, None)
    failures = _fetch_group(client, futures, summary=summary, say=say, label="contracts")
    hard = [(t, e) for t, e in failures if not isinstance(e, MissingError)]
    if hard:
        raise RuntimeError(
            "VX downloads failed: " + "; ".join(f"{t.key}: {e}" for t, e in hard[:5])
        )
    # A month with no file is a contract that was never listed (the 2004-2006 cycle was not
    # monthly) or is not listed yet; those keys are dropped from the manifest.
    summary.missing_contracts = sorted(t.key.split("/", 1)[1] for t, _ in failures)
    for t, _ in failures:
        client.manifest.entries.pop(t.key, None)
    missing = {date.fromisoformat(k) for k in summary.missing_contracts}
    if len(missing) == len(contracts):
        raise RuntimeError("no VX contract could be downloaded: Cboe refused every file")
    gaps = cboe.gaps_inside_listing(contracts, set(contracts) - missing)
    if gaps:
        raise RuntimeError(
            f"{len(gaps)} VX contracts inside the monthly cycle could not be downloaded "
            f"(first {gaps[0]}); run `vrp fetch` again"
        )


def _tiingo(config: Config, client: Client, *, summary: FetchSummary, say: Say) -> None:
    headers = tuple(prices.tiingo_headers(config, required=client.mirror is None).items())
    targets = [
        Target(
            key=f"tiingo/{a.instrument}",
            url=prices.tiingo_url(config, a.instrument),
            relpath=f"raw/tiingo/{a.instrument}.json",
            headers=headers,
        )
        for a in config.cboe_assets
    ]
    say(f"Tiingo daily prices: {len(targets)} ETFs")
    failures = _fetch_group(client, targets, summary=summary, say=say, label="ETFs")
    if failures:
        raise RuntimeError(
            "Tiingo downloads failed: " + "; ".join(f"{t.key}: {e}" for t, e in failures)
        )
    fred = Target(
        key=f"fred/{config.risk_free.series}",
        url=prices.fred_url(config),
        relpath=f"raw/fred/{config.risk_free.series}.csv",
    )
    _record(summary, client.fetch(fred))
    say(f"FRED {config.risk_free.series}: done")


def _btc_minutes(config: Config, client: Client, *, summary: FetchSummary, say: Say) -> list[str]:
    symbol = config.crypto.instrument
    folder = crypto.minute_folder(symbol)
    say(f"Binance Vision: listing {folder}")
    listing = list_prefix(client, config.sources.vision_list, folder)
    months = {
        o.key: (crypto.month_of(o.key), o.md5) for o in listing.objects if o.key.endswith(".zip")
    }
    first = pd.Timestamp(config.windows.first_date).strftime("%Y-%m")
    keep = {k: v for k, v in months.items() if v[0] is not None and v[0] >= first}
    if not keep:
        raise RuntimeError(f"the bucket listing has no {symbol} minute files")
    client.save_text(
        key=BTC_LISTING_KEY,
        url=config.sources.vision_list + "?prefix=" + folder,
        relpath="raw/vision/listing.json",
        text=json.dumps({k: list(v) for k, v in sorted(keep.items())}),
    )
    targets = [
        Target(
            key=f"vision/{k}",
            url=f"{config.sources.vision_base}/{k}",
            relpath=f"raw/vision/{k}",
            md5=md5,
        )
        for k, (_, md5) in sorted(keep.items())
    ]
    say(f"Binance {symbol} minute bars: {len(targets)} months")
    failures = _fetch_group(client, targets, summary=summary, say=say, label="months")
    if failures:
        raise RuntimeError(
            "Binance downloads failed: " + "; ".join(f"{t.key}: {e}" for t, e in failures[:5])
        )
    summary.btc_months = len(targets)
    return sorted(m for m, _ in keep.values() if m is not None)


def _month_end_ms(month: str) -> int:
    end = pd.Period(month, freq="M").end_time.tz_localize("UTC").ceil("D")
    return int(end.value // 1_000_000)


def _dvol(config: Config, client: Client, *, months: list[str], say: Say) -> None:
    """Daily DVOL candles, paged backward through `continuation`, saved as one file."""
    start = int(pd.Timestamp(config.windows.first_date, tz="UTC").value // 1_000_000)
    end = _month_end_ms(months[-1])
    candles: list[list[float]] = []
    first_url = crypto.dvol_url(config, start_ms=start, end_ms=end, resolution="1D")
    pages = 0
    while True:
        url = crypto.dvol_url(config, start_ms=start, end_ms=end, resolution="1D")
        data, continuation = crypto.parse_dvol_page(client.get_text(url))
        candles += data
        pages += 1
        if continuation is None or continuation >= end or not data:
            break
        end = continuation
    client.save_text(
        key=DVOL_DAILY_KEY,
        url=first_url,
        relpath="raw/deribit/dvol_1D.json",
        text=json.dumps(candles),
    )
    say(f"Deribit DVOL: {len(candles):,} daily candles in {pages} pages")


def snapshot_days(months: list[str]) -> list[pd.Timestamp]:
    """00:00 UTC on the first day of each bitcoin month (the realized side starts there)."""
    first = pd.Timestamp(months[0] + "-01", tz="UTC")
    last = pd.Timestamp(months[-1] + "-01", tz="UTC")
    return list(pd.date_range(first, last, freq="MS"))


def _ms(moment: pd.Timestamp) -> int:
    return int(moment.value // 1_000_000)


def trades_window(
    config: Config, client: Client, day: pd.Timestamp, *, pause: Callable[[], None]
) -> FetchResult:
    """Every option trade in the window before `day`, paged forward in time, kept as the fields
    the rebuild reads. The history does not change, so a day already in the manifest with a
    matching file is not asked for again."""
    key = f"{TRADES_PREFIX}{day:%Y-%m-%d}"
    relpath = f"raw/deribit/trades_{day:%Y-%m-%d}.json"
    dest = client.data_dir / relpath
    previous = client.manifest.entries.get(key)
    if previous is not None and dest.is_file() and sha256_of(dest) == previous.sha256:
        return FetchResult(entry=previous, changed=False, requested=False)
    start = _ms(day - pd.Timedelta(hours=config.replication.window_hours))
    end = _ms(day)
    first_url = crypto.trades_url(config, start_ms=start, end_ms=end)
    seen: dict[str, dict[str, object]] = {}
    while True:
        trades, more = crypto.parse_trades_page(
            client.get_text(crypto.trades_url(config, start_ms=start, end_ms=end))
        )
        for trade in trades:
            seen[str(trade["trade_id"])] = {f: trade[f] for f in crypto.TRADE_FIELDS}
        if not more or not trades:
            break
        last = int(str(trades[-1]["timestamp"]))
        if last <= start:
            raise RuntimeError(f"{day:%Y-%m-%d}: more than one page of trades in one millisecond")
        start = last
        pause()
    # Pages arrive in time order and a repeat keeps its first position: the server's order.
    return client.save_text(
        key=key, url=first_url, relpath=relpath, text=json.dumps(list(seen.values()))
    )


def _trades(
    config: Config, client: Client, *, months: list[str], summary: FetchSummary, say: Say
) -> list[pd.Timestamp]:
    days = snapshot_days(months)
    hours = config.replication.window_hours
    say(f"Deribit option trades: the {hours} hours before each of {len(days)} month starts")

    def pause() -> None:
        if client.mirror is None:
            time.sleep(config.download.rest_pause_seconds)

    errors: list[str] = []
    step = max(1, len(days) // 10)
    for n, day in enumerate(days, start=1):
        try:
            _record(summary, trades_window(config, client, day, pause=pause))
        except (RuntimeError, MissingError, ValueError) as error:
            errors.append(f"{day:%Y-%m-%d}: {error}")
        if n % step == 0 or n == len(days):
            say(f"  months: {n:,} / {len(days):,}")
        pause()
    if errors:
        raise RuntimeError(
            f"{len(errors)} months of trades failed; run `vrp fetch` again. First: {errors[0]}"
        )
    summary.snapshots = len(days)
    return days


def _dvol_at_snapshots(
    config: Config, client: Client, *, days: list[pd.Timestamp], say: Say
) -> None:
    """One-minute DVOL around each snapshot time, so the replicated rate is compared at the
    same minute."""
    out: dict[str, list[list[float]]] = {}
    for day in days:
        start = _ms(day - pd.Timedelta(minutes=5))
        end = _ms(day + pd.Timedelta(minutes=1))
        url = crypto.dvol_url(config, start_ms=start, end_ms=end, resolution="60")
        data, _ = crypto.parse_dvol_page(client.get_text(url))
        out[f"{day:%Y-%m-%d}"] = data
    client.save_text(
        key=DVOL_SNAPSHOT_KEY,
        url=crypto.dvol_url(config, start_ms=0, end_ms=0, resolution="60"),
        relpath="raw/deribit/dvol_snapshots.json",
        text=json.dumps(out),
    )
    say(f"Deribit DVOL at the snapshot minute: {len(out)} months")


def drop_retired(manifest: Manifest, *, data_dir: Path, hosts: tuple[str, ...], say: Say) -> int:
    """Remove entries, and their files, from hosts that are no longer configured sources, so a
    source taken out of the configuration leaves nothing behind for the provenance check."""
    retired = [k for k, e in manifest.entries.items() if not official(e.url, hosts)]
    for key in retired:
        (data_dir / manifest.entries[key].path).unlink(missing_ok=True)
        del manifest.entries[key]
    if retired:
        say(f"Removed {len(retired):,} files from sources no longer in the configuration")
    return len(retired)


def fetch_all(
    *, config: Config, paths: ProjectPaths, mirror: Path | None = None, say: Say = print
) -> FetchSummary:
    manifest = Manifest.read(paths.manifest)
    drop_retired(manifest, data_dir=paths.data, hosts=config.sources.official_hosts, say=say)
    client = client_for(config=config, paths=paths, manifest=manifest, mirror=mirror)
    summary = FetchSummary()
    today = datetime.now(UTC).date()
    try:
        _cboe(config, client, summary=summary, say=say, today=today)
        _tiingo(config, client, summary=summary, say=say)
        months = _btc_minutes(config, client, summary=summary, say=say)
        _dvol(config, client, months=months, say=say)
        days = _trades(config, client, months=months, summary=summary, say=say)
        _dvol_at_snapshots(config, client, days=days, say=say)
    finally:
        manifest.write(paths.manifest)
    return summary

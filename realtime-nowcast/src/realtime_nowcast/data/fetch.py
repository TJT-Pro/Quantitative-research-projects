"""Download every source into data/raw and record it in data/manifest.json.

- ALFRED: every vintage of each panel series, walked page by page, saved per series.
- FRED: the daily 2- and 10-year Treasury yields (never revised, so today's vintage).
- Atlanta Fed: the GDPNow workbook, revalidated with a conditional GET.

ALFRED and FRED need the key in the environment variable named in the configuration; it
is sent as a query parameter only at request time and never written anywhere.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from realtime_nowcast.config import Config
from realtime_nowcast.data import fred
from realtime_nowcast.data.download import Client, MissingError, Target
from realtime_nowcast.data.manifest import Manifest, official
from realtime_nowcast.paths import ProjectPaths

Say = Callable[[str], object]
ALFRED_PREFIX = "alfred/"
YIELDS_PREFIX = "fred/"
GDPNOW_KEY = "gdpnow/workbook"


@dataclass
class FetchSummary:
    files: int = 0
    changed: int = 0
    rows: dict[str, int] = field(default_factory=dict)


def client_for(
    *, config: Config, paths: ProjectPaths, manifest: Manifest, mirror: Path | None
) -> Client:
    return Client(data_dir=paths.data, manifest=manifest, settings=config.download, mirror=mirror)


def _secret(config: Config, client: Client) -> dict[str, str]:
    if client.mirror is not None:
        return {}
    key = os.environ.get(config.sources.fred_key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"set the {config.sources.fred_key_env} environment variable to your FRED API key "
            "(free at https://fredaccount.stlouisfed.org/apikeys)"
        )
    return {"api_key": key}


def _pages(
    client: Client,
    url_for: Callable[[int], str],
    *,
    secret: dict[str, str],
    pause: Callable[[], None],
) -> tuple[list[dict[str, str]], str]:
    rows: list[dict[str, str]] = []
    offset = 0
    first = url_for(0)
    while True:
        page, count = fred.parse_page(client.get_text(url_for(offset), secret=secret))
        rows.extend(page)
        offset += len(page)
        if offset >= count or not page:
            break
        pause()
    if len(rows) != count:
        raise fred.SchemaError(f"{first}: expected {count} rows, received {len(rows)}")
    return rows, first


def _save(
    client: Client,
    *,
    key: str,
    url: str,
    relpath: str,
    series: str,
    rows: list[dict[str, str]],
    summary: FetchSummary,
) -> None:
    text = json.dumps({"series_id": series, "count": len(rows), "observations": rows})
    result = client.save_text(key=key, url=url, relpath=relpath, text=text)
    summary.files += 1
    summary.changed += int(result.changed)
    summary.rows[series] = len(rows)


def _alfred(
    config: Config,
    client: Client,
    *,
    secret: dict[str, str],
    summary: FetchSummary,
    say: Say,
    pause: Callable[[], None],
) -> None:
    specs = config.panel.series
    say(f"ALFRED: every vintage of {len(specs)} series")
    for n, s in enumerate(specs, start=1):
        rows, first = _pages(
            client,
            fred.vintages_pages(config, s.id),
            secret=secret,
            pause=pause,
        )
        _save(
            client,
            key=f"{ALFRED_PREFIX}{s.id}",
            url=first,
            relpath=f"raw/alfred/{s.id}.json",
            series=s.id,
            rows=rows,
            summary=summary,
        )
        say(f"  {n:>2} / {len(specs)}  {s.id:<20} {len(rows):>8,} rows")
        pause()


def _yields(
    config: Config,
    client: Client,
    *,
    secret: dict[str, str],
    summary: FetchSummary,
    say: Say,
    pause: Callable[[], None],
) -> None:
    start = (
        pd.Period(config.evaluation.first_quarter, freq="Q").start_time - pd.DateOffset(years=1)
    ).strftime("%Y-%m-%d")
    for s in config.market.yields:
        rows, first = _pages(
            client,
            fred.current_pages(config, s, start=start),
            secret=secret,
            pause=pause,
        )
        _save(
            client,
            key=f"{YIELDS_PREFIX}{s}",
            url=first,
            relpath=f"raw/fred/{s}.json",
            series=s,
            rows=rows,
            summary=summary,
        )
        say(f"FRED {s}: {len(rows):,} days")
        pause()


def _gdpnow(config: Config, client: Client, *, summary: FetchSummary, say: Say) -> None:
    """The workbook from the first configured URL that has it. A URL already in the manifest
    is tried first, so a later run revalidates the same file."""
    urls = list(config.sources.gdpnow)
    previous = client.manifest.entries.get(GDPNOW_KEY)
    if previous is not None and previous.url in urls:
        urls.remove(previous.url)
        urls.insert(0, previous.url)
    relpath = "raw/gdpnow/GDPTrackingModelDataAndForecasts.xlsx"
    refused: list[str] = []
    for url in urls:
        target = Target(key=GDPNOW_KEY, url=url, relpath=relpath, missing_statuses=(403, 404))
        try:
            result = client.fetch(target)
        except MissingError as error:
            refused.append(str(error))
            continue
        summary.files += 1
        summary.changed += int(result.changed)
        state = "downloaded" if result.changed else "unchanged"
        say(f"GDPNow workbook: {state} ({result.entry.bytes / 1e6:.1f} MB)")
        return
    raise RuntimeError(
        "no configured URL serves the GDPNow workbook; open "
        "https://www.atlantafed.org/research-and-data/data/gdpnow, copy the link of "
        "'GDPNow Model Data and Historical Forecasts' into sources.gdpnow in "
        f"configs/research.yaml and run again ({'; '.join(refused)})"
    )


def drop_retired(manifest: Manifest, *, data_dir: Path, hosts: tuple[str, ...], say: Say) -> int:
    """Remove entries, and their files, from hosts that are no longer configured sources."""
    retired = [k for k, e in manifest.entries.items() if not official(e.url, hosts)]
    for key in retired:
        (data_dir / manifest.entries[key].path).unlink(missing_ok=True)
        del manifest.entries[key]
    if retired:
        say(f"Removed {len(retired):,} files from sources no longer in the configuration")
    return len(retired)


def drop_unlisted(manifest: Manifest, *, data_dir: Path, config: Config, say: Say) -> int:
    """Remove series that left the configuration, so the build reads exactly the panel."""
    wanted = (
        {f"{ALFRED_PREFIX}{s.id}" for s in config.panel.series}
        | {f"{YIELDS_PREFIX}{s}" for s in config.market.yields}
        | {GDPNOW_KEY}
    )
    stale = [k for k in manifest.entries if k not in wanted]
    for key in stale:
        (data_dir / manifest.entries[key].path).unlink(missing_ok=True)
        del manifest.entries[key]
    if stale:
        say(f"Removed {len(stale):,} files no longer in the configuration")
    return len(stale)


def fetch_all(
    *, config: Config, paths: ProjectPaths, mirror: Path | None = None, say: Say = print
) -> FetchSummary:
    manifest = Manifest.read(paths.manifest)
    drop_retired(manifest, data_dir=paths.data, hosts=config.sources.official_hosts, say=say)
    drop_unlisted(manifest, data_dir=paths.data, config=config, say=say)
    client = client_for(config=config, paths=paths, manifest=manifest, mirror=mirror)
    secret = _secret(config, client)
    summary = FetchSummary()

    def pause() -> None:
        if client.mirror is None:
            time.sleep(config.download.pause_seconds)

    try:
        _alfred(config, client, secret=secret, summary=summary, say=say, pause=pause)
        _yields(config, client, secret=secret, summary=summary, say=say, pause=pause)
        _gdpnow(config, client, summary=summary, say=say)
    finally:
        manifest.write(paths.manifest)
    return summary

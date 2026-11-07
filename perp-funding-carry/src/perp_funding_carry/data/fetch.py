"""Fetch stage: catalogue the bucket, download what changed, collect the REST sources."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

from perp_funding_carry.config import Config
from perp_funding_carry.data import books, rest, vision
from perp_funding_carry.data.download import Client, FetchResult, MissingError, Target
from perp_funding_carry.data.manifest import Manifest
from perp_funding_carry.paths import ProjectPaths

CATALOG_KEY = "catalog"
CATALOG_PATH = "raw/catalog.json"


@dataclass(frozen=True)
class FetchSummary:
    contracts: int
    files: int
    requested: int
    changed: int
    last_month: str
    okx_rows_added: int


def client_for(
    *, config: Config, paths: ProjectPaths, manifest: Manifest, mirror: Path | None
) -> Client:
    return Client(data_dir=paths.data, manifest=manifest, settings=config.download, mirror=mirror)


def fetch_all(
    *, config: Config, paths: ProjectPaths, mirror: Path | None = None, log: bool = True
) -> FetchSummary:
    manifest = Manifest.read(paths.manifest)
    client = client_for(config=config, paths=paths, manifest=manifest, mirror=mirror)
    say = (lambda line: print(line, flush=True)) if log else None
    catalog = vision.build_catalog(config, client, report=say)
    client.save_text(
        key=CATALOG_KEY,
        url=config.sources.vision_list,
        relpath=CATALOG_PATH,
        text=json.dumps(catalog.as_json()),
    )
    targets = vision.plan_targets(config, catalog)
    if log:
        print(
            f"catalogue: {len(catalog.contracts)} contracts with a spot pair, "
            f"{len(targets)} files up to {catalog.last_month}",
            flush=True,
        )

    counter = {"done": 0, "requested": 0, "changed": 0}
    lock = threading.Lock()

    def on_done(_: Target, result: FetchResult) -> None:
        with lock:
            counter["done"] += 1
            counter["requested"] += int(result.requested)
            counter["changed"] += int(result.changed)
            if counter["done"] % 500 == 0:
                manifest.write(paths.manifest)  # keep progress if the run is interrupted
                if log:
                    print(f"  {counter['done']:,} / {len(targets):,} files", flush=True)

    failures = client.fetch_many(targets, on_done=on_done)
    manifest.write(paths.manifest)
    if failures:
        missing = [t.key for t, e in failures if isinstance(e, MissingError)]
        other = [(t.key, str(e)) for t, e in failures if not isinstance(e, MissingError)]
        detail = "; ".join(f"{k}: {e}" for k, e in other[:5]) or ", ".join(missing[:5])
        raise RuntimeError(
            f"{len(failures)} files failed ({len(missing)} missing upstream). "
            f"Run `pfc fetch` again to retry them. First failures: {detail}"
        )
    if log:
        print("funding history from the Binance, Bybit and OKX APIs", flush=True)
    collected = rest.collect(config, client, data_dir=paths.data)
    if log:
        print("best bid and ask of every spot and USD-M pair", flush=True)
    books.snapshot(config, client, data_dir=paths.data)
    wanted = {t.key for t in targets} | {r.entry.path for r in collected.results}
    wanted |= {CATALOG_KEY, "collected/okx", "collected/books"} | {
        k for k in manifest.entries if k.startswith("rest/")
    }
    for stale in [k for k in manifest.entries if k not in wanted]:
        del manifest.entries[stale]
    manifest.write(paths.manifest)
    return FetchSummary(
        contracts=len(catalog.contracts),
        files=len(targets),
        requested=counter["requested"],
        changed=counter["changed"],
        last_month=catalog.last_month,
        okx_rows_added=collected.okx_rows_added,
    )

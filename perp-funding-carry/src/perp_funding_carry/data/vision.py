"""Binance Vision paths and the catalogue of what exists, built from the bucket listing.

No symbol list is typed anywhere. Perpetuals are the prefixes under the monthly
fundingRate folder (delisted contracts keep their folders), spot pairs the prefixes under
spot klines, and each file's month comes from its own listed key.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

from perp_funding_carry.config import Config
from perp_funding_carry.data.download import Client, Target
from perp_funding_carry.data.listing import ListedObject, list_prefix

FUNDING = "data/futures/um/monthly/fundingRate"
MARK = "data/futures/um/monthly/markPriceKlines"
PERP = "data/futures/um/monthly/klines"
PREMIUM = "data/futures/um/monthly/premiumIndexKlines"
SPOT = "data/spot/monthly/klines"
METRICS = "data/futures/um/daily/metrics"

_MONTH = re.compile(r"-(\d{4}-\d{2})\.zip$")
_DAY = re.compile(r"-(\d{4}-\d{2}-\d{2})\.zip$")


def monthly_key(folder: str, *, symbol: str, interval: str | None, month: str) -> str:
    if interval is None:
        name = folder.rsplit("/", 1)[1]
        return f"{folder}/{symbol}/{symbol}-{name}-{month}.zip"
    return f"{folder}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def metrics_key(symbol: str, day: str) -> str:
    return f"{METRICS}/{symbol}/{symbol}-metrics-{day}.zip"


@dataclass(frozen=True)
class Contract:
    symbol: str
    spot: str
    multiplier: int


@dataclass
class Catalog:
    """Listed files per (kind, symbol, interval): key -> (month or day, size, md5)."""

    last_month: str
    contracts: list[Contract]
    files: dict[str, dict[str, tuple[str, int, str | None]]] = field(default_factory=dict)

    def months(self, group: str) -> set[str]:
        return {period for period, _, _ in self.files.get(group, {}).values()}

    def as_json(self) -> dict[str, Any]:
        return {
            "last_month": self.last_month,
            "contracts": [asdict(c) for c in self.contracts],
            "files": {g: {k: list(v) for k, v in f.items()} for g, f in self.files.items()},
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Catalog:
        return cls(
            last_month=payload["last_month"],
            contracts=[Contract(**c) for c in payload["contracts"]],
            files={
                g: {k: (v[0], int(v[1]), v[2]) for k, v in f.items()}
                for g, f in payload["files"].items()
            },
        )


def _symbols(client: Client, base: str, folder: str) -> set[str]:
    listing = list_prefix(client, base, folder + "/")
    return {p.rstrip("/").rsplit("/", 1)[1] for p in listing.prefixes}


def map_contracts(config: Config, perps: set[str], spots: set[str]) -> list[Contract]:
    """Each USDT perpetual with a USDT spot pair. A perpetual quoted per 1,000 units maps to
    the spot pair without the prefix, unless the spot market lists the prefixed name itself."""
    quote = config.universe.quote
    prefixes = sorted(config.universe.multiplier_prefixes.items(), key=lambda kv: -len(kv[0]))
    out = []
    for symbol in sorted(perps):
        if not symbol.endswith(quote) or symbol == quote:
            continue
        if symbol in spots:
            out.append(Contract(symbol=symbol, spot=symbol, multiplier=1))
            continue
        base = symbol[: -len(quote)]
        for prefix, multiple in prefixes:
            stripped = base[len(prefix) :]
            if base.startswith(prefix) and stripped and f"{stripped}{quote}" in spots:
                out.append(Contract(symbol=symbol, spot=f"{stripped}{quote}", multiplier=multiple))
                break
    return out


def _periods(
    objects: list[ListedObject], pattern: re.Pattern[str]
) -> dict[str, tuple[str, int, str | None]]:
    out = {}
    for obj in objects:
        match = pattern.search(obj.key)
        if match and obj.key.endswith(".zip"):
            out[obj.key] = (match.group(1), obj.size, obj.md5)
    return out


def groups_for(config: Config, contract: Contract) -> list[tuple[str, str, re.Pattern[str]]]:
    """(group name, folder to list, period pattern) for every file family a contract needs."""
    u = config.universe
    s, p = contract.symbol, contract.spot
    out = [
        (f"funding/{s}", f"{FUNDING}/{s}/", _MONTH),
        (f"mark/{s}/{u.bar_interval}", f"{MARK}/{s}/{u.bar_interval}/", _MONTH),
        (f"perp/{s}/{u.bar_interval}", f"{PERP}/{s}/{u.bar_interval}/", _MONTH),
        (f"spot/{p}/{u.bar_interval}", f"{SPOT}/{p}/{u.bar_interval}/", _MONTH),
    ]
    if s in u.core_symbols:
        iv = u.core_bar_interval
        out += [
            (f"mark/{s}/{iv}", f"{MARK}/{s}/{iv}/", _MONTH),
            (f"perp/{s}/{iv}", f"{PERP}/{s}/{iv}/", _MONTH),
            (f"spot/{p}/{iv}", f"{SPOT}/{p}/{iv}/", _MONTH),
            (f"metrics/{s}", f"{METRICS}/{s}/", _DAY),
        ]
    if s in config.rebuild.symbols:
        out.append((f"premium/{s}/1m", f"{PREMIUM}/{s}/1m/", _MONTH))
    return out


def build_catalog(
    config: Config, client: Client, *, report: Callable[[str], None] | None = None
) -> Catalog:
    """List the bucket. `report` receives progress lines: the listing takes thousands of
    requests (one or more per contract and file family) and would otherwise run silently."""
    say = report or (lambda _: None)
    base = config.sources.vision_list
    say("listing the Binance Vision bucket: perpetual and spot symbols")
    perps = _symbols(client, base, FUNDING)
    spots = _symbols(client, base, SPOT)
    contracts = map_contracts(config, perps, spots)
    jobs: list[tuple[str, str, re.Pattern[str]]] = []
    for contract in contracts:
        jobs += groups_for(config, contract)
    stable = config.universe.stablecoin_pair
    jobs.append(
        (
            f"spot/{stable}/{config.universe.bar_interval}",
            f"{SPOT}/{stable}/{config.universe.bar_interval}/",
            _MONTH,
        )
    )
    say(
        f"{len(perps):,} perpetuals listed, {len(contracts):,} with a {config.universe.quote} "
        f"spot pair; listing {len(jobs):,} folders"
    )

    def run(
        job: tuple[str, str, re.Pattern[str]],
    ) -> tuple[str, dict[str, tuple[str, int, str | None]]]:
        group, folder, pattern = job
        return group, _periods(list_prefix(client, base, folder).objects, pattern)

    files: dict[str, dict[str, tuple[str, int, str | None]]] = {}
    step = max(1, len(jobs) // 20)
    with ThreadPoolExecutor(max_workers=config.download.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            group, periods = future.result()
            files[group] = periods
            if done % step == 0 or done == len(jobs):
                say(f"  {done:,} / {len(jobs):,} folders listed")
    reference = config.universe.core_symbols[0]
    listed = sorted({m for m, _, _ in files.get(f"funding/{reference}", {}).values()})
    if not listed:
        raise RuntimeError(f"the listing has no funding files for {reference}")
    last = config.universe.last_month or listed[-1]
    return Catalog(last_month=last, contracts=contracts, files=files)


def _target(config: Config, key: str, md5: str | None) -> Target:
    return Target(
        key=f"vision/{key}",
        url=f"{config.sources.vision_base}/{key}",
        relpath=f"raw/vision/{key}",
        md5=md5,
    )


def contract_months(config: Config, catalog: Catalog, contract: Contract) -> list[str]:
    """Months inside the window where the funding, mark, perpetual and spot files all exist."""
    u = config.universe
    s, p = contract.symbol, contract.spot
    common = (
        catalog.months(f"funding/{s}")
        & catalog.months(f"mark/{s}/{u.bar_interval}")
        & catalog.months(f"perp/{s}/{u.bar_interval}")
        & catalog.months(f"spot/{p}/{u.bar_interval}")
    )
    return sorted(m for m in common if u.first_month <= m <= catalog.last_month)


def _contract_targets(config: Config, catalog: Catalog, contract: Contract) -> list[Target]:
    u = config.universe
    s, p = contract.symbol, contract.spot
    window = set(contract_months(config, catalog, contract))
    if len(window) < u.min_months and s not in u.core_symbols:
        return []
    groups = [
        f"funding/{s}",
        f"mark/{s}/{u.bar_interval}",
        f"perp/{s}/{u.bar_interval}",
        f"spot/{p}/{u.bar_interval}",
    ]
    if s in u.core_symbols:
        iv = u.core_bar_interval
        groups += [f"mark/{s}/{iv}", f"perp/{s}/{iv}", f"spot/{p}/{iv}"]
    return [
        _target(config, key, md5)
        for group in groups
        for key, (month, _, md5) in catalog.files.get(group, {}).items()
        if month in window
    ]


def _latest(
    listed: dict[str, tuple[str, int, str | None]], *, count: int, last: str
) -> list[tuple[str, str | None]]:
    """The `count` most recent listed files up to `last`, as (key, md5)."""
    kept = sorted((kv for kv in listed.items() if kv[1][0] <= last), key=lambda kv: kv[1][0])
    return [(key, md5) for key, (_, _, md5) in kept[-count:]]


def plan_targets(config: Config, catalog: Catalog) -> list[Target]:
    """Every file to fetch: each contract's months inside the window, the trailing
    open-interest window for the core symbols, recent 1-minute premiums for the rebuild
    symbols and the stablecoin pair."""
    u = config.universe
    last = catalog.last_month
    targets = [t for c in catalog.contracts for t in _contract_targets(config, catalog, c)]
    for s in u.core_symbols:
        days = catalog.files.get(f"metrics/{s}", {})
        targets += [
            _target(config, k, md5) for k, md5 in _latest(days, count=u.oi_window_days, last="9999")
        ]
    listed_symbols = {c.symbol for c in catalog.contracts}
    for s in (s for s in config.rebuild.symbols if s in listed_symbols):
        premium = catalog.files.get(f"premium/{s}/1m", {})
        targets += [
            _target(config, k, md5)
            for k, md5 in _latest(premium, count=config.rebuild.months, last=last)
        ]
    stable = catalog.files.get(f"spot/{u.stablecoin_pair}/{u.bar_interval}", {})
    targets += [
        _target(config, k, md5) for k, (m, _, md5) in stable.items() if u.first_month <= m <= last
    ]
    return sorted(set(targets), key=lambda t: t.key)

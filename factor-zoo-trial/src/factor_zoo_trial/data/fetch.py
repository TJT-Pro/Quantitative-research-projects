"""Fetch stage: download (or copy from a mirror) every source file into data/raw."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from factor_zoo_trial.config import Config
from factor_zoo_trial.data import sources
from factor_zoo_trial.data.download import Fetcher, Target
from factor_zoo_trial.data.manifest import Manifest
from factor_zoo_trial.data.parsers import read_country_classification
from factor_zoo_trial.paths import ProjectPaths


@dataclass(frozen=True)
class FetchSummary:
    files: int
    changed: tuple[str, ...]
    countries: int
    regions: int
    skipped: dict[str, str]


def _country_file(config: Config, paths: ProjectPaths) -> Path:
    return paths.raw / "jkp" / "reference" / Path(config.jkp.country_classification).name


def plan_path(paths: ProjectPaths) -> Path:
    return paths.raw / "jkp" / "plan.json"


def fetch_all(
    *, config: Config, paths: ProjectPaths, mirror: Path | None = None, log: bool = True
) -> FetchSummary:
    manifest = Manifest.read(paths.manifest)
    fetcher = Fetcher(raw_dir=paths.raw, manifest=manifest, settings=config.download, mirror=mirror)
    wanted: list[str] = []
    changed: list[str] = []

    def get(target: Target) -> None:
        result = fetcher.fetch(target)
        wanted.append(target.key)
        if result.changed:
            changed.append(target.key)
        manifest.write(paths.manifest)  # after every file, so an interruption keeps progress
        if log:
            state = "new or changed" if result.changed else "unchanged"
            print(f"{state:>15}  {target.key}  ({result.entry.bytes:,} bytes)")

    for target in sources.osap_targets(config):
        get(target)
    for target in sources.jkp_reference_targets(config):
        get(target)
    availability = sources.jkp_availability_target(config)
    get(availability)
    for target in sources.french_targets(config):
        get(target)

    codes = set(read_country_classification(_country_file(config, paths))["location"])
    plan = sources.plan_locations(
        config, availability_path=paths.raw / availability.relpath, country_codes=codes
    )
    for location in plan.locations:
        for selector in (config.jkp.selector, config.jkp.market_selector):
            get(sources.jkp_location_target(config, location=location, selector=selector))

    plan_path(paths).write_text(json.dumps(asdict(plan), indent=2) + "\n", encoding="utf-8")
    for stale in set(manifest.entries) - set(wanted):
        del manifest.entries[stale]
    manifest.write(paths.manifest)
    return FetchSummary(
        files=len(wanted),
        changed=tuple(changed),
        countries=len(plan.countries),
        regions=len(plan.regions),
        skipped=plan.skipped,
    )

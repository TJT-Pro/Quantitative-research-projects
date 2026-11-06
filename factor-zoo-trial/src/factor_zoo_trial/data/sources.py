"""The list of files to fetch, built from the config plus catalogues read at runtime.

The set of JKP locations is not typed anywhere: it is the intersection of the
locations in the bucket's availability.json with the JKP country classification,
plus the configured regional aggregates. A location missing either the factor
file or the market file is skipped and the reason is kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from factor_zoo_trial.config import Config
from factor_zoo_trial.data.download import Target


def osap_target(config: Config, path: str) -> Target:
    revision = config.sources.osap_revision
    url = f"{config.sources.osap_resolve}/{quote(revision, safe='')}/{quote(path)}"
    return Target(key=f"osap/{path}", url=url, relpath=f"osap/{revision}/{path}")


def osap_targets(config: Config) -> list[Target]:
    paths = [config.osap.signal_doc, *config.osap.variants.values()]
    return [osap_target(config, p) for p in dict.fromkeys(paths)]


def jkp_reference_targets(config: Config) -> list[Target]:
    base = f"{config.sources.jkp_reference_base}/{config.sources.jkp_reference_commit}"
    files = (
        config.jkp.cluster_labels,
        config.jkp.factor_details,
        config.jkp.country_classification,
    )
    return [
        Target(
            key=f"jkp/reference/{Path(f).name}",
            url=f"{base}/{quote(f)}",
            relpath=f"jkp/reference/{Path(f).name}",
        )
        for f in files
    ]


def jkp_availability_target(config: Config) -> Target:
    return Target(
        key="jkp/availability.json",
        url=f"{config.sources.jkp_base}/availability.json",
        relpath="jkp/availability.json",
    )


def jkp_file_name(config: Config, *, location: str, selector: str) -> str:
    jkp = config.jkp
    return f"[{location}]_[{selector}]_[{jkp.frequency}]_[{jkp.weighting}].zip"


def jkp_location_target(config: Config, *, location: str, selector: str) -> Target:
    name = jkp_file_name(config, location=location, selector=selector)
    return Target(
        key=f"jkp/{selector}/{location}",
        url=f"{config.sources.jkp_base}/{quote(name)}",
        relpath=f"jkp/{selector}/{name}",
    )


def french_targets(config: Config) -> list[Target]:
    return [
        Target(key=f"french/{z}", url=f"{config.sources.french_base}/{z}", relpath=f"french/{z}")
        for z in (config.french.five_factor_zip, config.french.momentum_zip)
    ]


@dataclass(frozen=True)
class LocationPlan:
    countries: tuple[str, ...]
    regions: tuple[str, ...]
    skipped: dict[str, str]

    @property
    def locations(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.regions, *self.countries)))


def plan_locations(
    config: Config, *, availability_path: Path, country_codes: set[str]
) -> LocationPlan:
    payload = json.loads(availability_path.read_text(encoding="utf-8"))
    offered: dict[str, list[str]] = payload["factors"]
    needed = {config.jkp.selector, config.jkp.market_selector}
    skipped: dict[str, str] = {}

    def usable(location: str) -> bool:
        missing = needed - set(offered.get(location, []))
        if missing:
            skipped[location] = f"availability.json lists no {', '.join(sorted(missing))}"
            return False
        return True

    unknown = [r for r in config.jkp.replication_regions if r not in offered]
    if unknown:
        raise ValueError(
            f"replication regions {unknown} are not in availability.json; "
            f"offered keys include {sorted(offered)[:20]}"
        )
    regions = tuple(r for r in config.jkp.replication_regions if usable(r))
    countries = tuple(
        sorted(
            loc for loc in offered if loc in country_codes and loc not in regions and usable(loc)
        )
    )
    for location in sorted(offered):
        if location not in country_codes and location not in regions and location not in skipped:
            skipped[location] = "aggregate not used (not a country, not a configured region)"
    return LocationPlan(countries=countries, regions=regions, skipped=skipped)

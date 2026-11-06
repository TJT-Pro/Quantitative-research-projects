"""Build stage: parse data/raw into tidy Parquet tables in data/processed.

Raw files are never modified. Every processed table is rebuilt from raw on each run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from factor_zoo_trial.config import Config
from factor_zoo_trial.data import parsers, sources
from factor_zoo_trial.data.fetch import plan_path
from factor_zoo_trial.paths import ProjectPaths


@dataclass(frozen=True)
class Tables:
    signals: pd.DataFrame  # one row per OSAP predictor
    osap: pd.DataFrame  # variant, signal, month, ret
    jkp: pd.DataFrame  # location, factor, month, ret, n_stocks_min, n_countries, aggregate
    market: pd.DataFrame  # location, month, mkt
    jkp_reference: pd.DataFrame  # factor, cluster, significance, direction, cite
    locations: pd.DataFrame  # location, kind, development, geo_region, note (set when unused)
    french: pd.DataFrame  # month, mkt_rf, smb, hml, rmw, cma, rf, mom


TABLE_NAMES = ("signals", "osap", "jkp", "market", "jkp_reference", "locations", "french")


def build_tables(*, config: Config, paths: ProjectPaths) -> Tables:
    raw = paths.raw
    signal_doc = raw / sources.osap_target(config, config.osap.signal_doc).relpath
    signals = parsers.read_signal_doc(signal_doc)
    signals = signals.loc[signals["kind"] == "Predictor"].reset_index(drop=True)
    signals["clear"] = signals["predictability"] == config.osap.clear_label

    frames = []
    for variant, path in config.osap.variants.items():
        ls = parsers.read_long_short(
            raw / sources.osap_target(config, path).relpath, port=config.osap.long_short_port
        )
        ls.insert(0, "variant", variant)
        frames.append(ls)
    osap = pd.concat(frames, ignore_index=True)
    osap = osap.loc[osap["signal"].isin(signals["signal"])].reset_index(drop=True)

    reference_dir = raw / "jkp" / "reference"
    clusters = parsers.read_cluster_labels(reference_dir / Path(config.jkp.cluster_labels).name)
    details = parsers.read_factor_details(reference_dir / Path(config.jkp.factor_details).name)
    jkp_reference = clusters.merge(details, on="factor", how="left")
    jkp_reference["significance"] = jkp_reference["significance"].fillna(value=False).astype(bool)

    plan = json.loads(plan_path(paths).read_text(encoding="utf-8"))
    classification = parsers.read_country_classification(
        reference_dir / Path(config.jkp.country_classification).name
    )
    rows = [{"location": r, "kind": "region"} for r in plan["regions"]]
    rows += [{"location": c, "kind": "country"} for c in plan["countries"]]
    locations = pd.DataFrame(rows).merge(classification, on="location", how="left")

    factor_frames, market_frames = [], []
    empty: dict[str, str] = {}
    for location in locations["location"]:
        factor_target = sources.jkp_location_target(
            config, location=location, selector=config.jkp.selector
        )
        market_target = sources.jkp_location_target(
            config, location=location, selector=config.jkp.market_selector
        )
        factors = parsers.read_jkp_factors(raw / factor_target.relpath)
        market = parsers.read_jkp_market(raw / market_target.relpath)
        if factors.empty or market.empty:
            empty[location] = "factor or market file has no returns"
            continue
        factors.insert(0, "location", location)
        factor_frames.append(factors)
        market.insert(0, "location", location)
        market_frames.append(market)
    # Locations whose files hold no returns stay listed, with the reason, and are not used.
    locations["note"] = locations["location"].map(empty).astype("string")
    jkp = pd.concat(factor_frames, ignore_index=True)
    # Country rows need enough stocks in each leg; regional rows were filtered by JKP already.
    enough = jkp["n_stocks_min"].fillna(0) >= config.jkp.min_stocks_per_leg
    jkp = jkp.loc[jkp["aggregate"].astype(bool) | enough]
    jkp = jkp.loc[jkp["factor"].isin(jkp_reference["factor"])].reset_index(drop=True)
    market = pd.concat(market_frames, ignore_index=True)

    five = parsers.read_french_monthly(raw / "french" / config.french.five_factor_zip)
    momentum = parsers.read_french_monthly(raw / "french" / config.french.momentum_zip)
    five = five.rename(columns=lambda c: c.lower().replace("-", "_"))
    momentum = momentum.rename(columns={momentum.columns[1]: "mom"})[["month", "mom"]]
    french = five.merge(momentum, on="month", how="inner")

    return Tables(
        signals=signals,
        osap=osap,
        jkp=jkp,
        market=market,
        jkp_reference=jkp_reference,
        locations=locations,
        french=french,
    )


def write_tables(tables: Tables, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        frame: pd.DataFrame = getattr(tables, name)
        frame.to_parquet(directory / f"{name}.parquet", index=False)


def read_tables(directory: Path) -> Tables:
    loaded = {name: pd.read_parquet(directory / f"{name}.parquet") for name in TABLE_NAMES}
    return Tables(**loaded)

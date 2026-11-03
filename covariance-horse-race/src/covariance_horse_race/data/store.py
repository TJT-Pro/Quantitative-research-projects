"""Processed datasets: raw zips in, typed Parquet out, plus a coverage audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from covariance_horse_race.config import FactorSet, ResearchConfig
from covariance_horse_race.data import french
from covariance_horse_race.data.source import (
    SourceError,
    load_manifest,
    read_single_csv,
    verify_raw,
)


def file_roles(config: ResearchConfig) -> dict[str, str]:
    """Which configured files are needed, and whether each is a portfolio or factor file."""
    roles = {u.file: "portfolio" for u in config.universes}
    for factor_set in (config.factors.primary, config.factors.extended):
        for key in factor_set.source_files:
            if roles.get(key) == "portfolio":
                raise SourceError(f"{key} is used both as a universe and as a factor file")
            roles[key] = "factor"
    return roles


def build_processed(
    config: ResearchConfig, raw_dir: Path, processed_dir: Path, manifest: Path
) -> None:
    """Parse every needed raw file into Parquet, after verifying it against the manifest.

    Portfolio files carry several weighting sections and the configured one is
    selected; factor files must contain exactly one block.
    """
    verify_raw(raw_dir, manifest)
    processed_dir.mkdir(parents=True, exist_ok=True)
    entries = load_manifest(manifest)
    for key, role in file_roles(config).items():
        if key not in entries:
            raise SourceError(f"{key} is configured but was never fetched; run `chr fetch`")
        entry = entries[key]
        _, payload = read_single_csv(raw_dir / entry.file)
        section = config.data.portfolio_section if role == "portfolio" else None
        frame = french.parse(
            payload, section, config.data.missing_sentinels, config.data.percent_scale
        )
        frame.attrs["source_sha256"] = entry.csv_sha256
        frame.to_parquet(processed_dir / f"{key}.parquet")


def load(processed_dir: Path, key: str) -> pd.DataFrame:
    path = processed_dir / f"{key}.parquet"
    if not path.is_file():
        raise SourceError(f"{path.name} not built; run `chr build`")
    return pd.read_parquet(path)


def load_factor_set(processed_dir: Path, factor_set: FactorSet) -> pd.DataFrame:
    """Join the files of a factor set on common dates and keep the requested columns.

    The risk-free rate is dropped: factor models here explain return comovement,
    and RF has no cross-sectional loading.
    """
    frames = [load(processed_dir, key) for key in factor_set.source_files]
    joined = frames[0]
    for frame in frames[1:]:
        overlap = set(joined.columns) & set(frame.columns)
        joined = joined.join(frame.drop(columns=list(overlap)), how="inner")
    missing = [c for c in factor_set.columns if c not in joined.columns]
    if missing:
        raise SourceError(f"factor columns {missing} not found; available {list(joined.columns)}")
    selected = joined[factor_set.columns]
    if selected.isna().any().any():
        selected = selected.dropna()
    return selected


@dataclass(frozen=True)
class Coverage:
    key: str
    first_date: str
    last_date: str
    n_days: int
    n_columns: int
    missing_share: float
    columns_complete_from: str


def coverage(processed_dir: Path, keys: list[str]) -> list[Coverage]:
    rows: list[Coverage] = []
    for key in keys:
        frame = load(processed_dir, key)
        complete = frame.dropna()
        rows.append(
            Coverage(
                key=key,
                first_date=frame.index[0].date().isoformat(),
                last_date=frame.index[-1].date().isoformat(),
                n_days=len(frame),
                n_columns=frame.shape[1],
                missing_share=float(np.isnan(frame.to_numpy()).mean()),
                columns_complete_from=(
                    complete.index[0].date().isoformat() if len(complete) else "never"
                ),
            )
        )
    return rows

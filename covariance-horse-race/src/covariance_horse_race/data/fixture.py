"""Synthetic mirror of the library's ftp folder for offline tests and CI.

Writes zip archives in the same layout as the real files: free-text preamble,
"Average Value Weighted Returns -- Daily" and "Average Equal Weighted Returns --
Daily" blocks for portfolio files, a single block for factor files, percent
units, the -99.99 sentinel for portfolios that do not exist yet, and a trailing
copyright line. Returns follow a factor model with GARCH volatility so every
estimator, including DCC, has structure to find. The numbers are synthetic and
are never used for any reported result.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from covariance_horse_race.config import ResearchConfig


class FixtureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: str
    n_days: int = Field(gt=100)
    seed: int
    columns: dict[str, int]
    # Share of each portfolio file's columns that start late (filled with the sentinel).
    late_start_share: float = Field(ge=0, lt=1)
    late_start_max_days: int = Field(ge=0)


def load_fixture_spec(path: Path) -> FixtureSpec:
    with path.open(encoding="utf-8") as handle:
        return FixtureSpec.model_validate(yaml.safe_load(handle))


def _garch_path(n: int, rng: np.random.Generator, daily_vol: float) -> np.ndarray:
    alpha, beta = 0.08, 0.90
    omega = daily_vol**2 * (1 - alpha - beta)
    h = daily_vol**2
    out = np.empty(n)
    for i in range(n):
        out[i] = np.sqrt(h) * rng.standard_normal()
        h = omega + alpha * out[i] ** 2 + beta * h
    return out


def _format(dates: pd.DatetimeIndex, frame: np.ndarray, names: list[str]) -> list[str]:
    lines = ["         ," + ",".join(f"{n:<6}" for n in names)]
    for date, row in zip(dates, frame, strict=True):
        cells = ",".join(f"{v:7.2f}" for v in row)
        lines.append(f"{date:%Y%m%d},{cells}")
    return lines


def _write_zip(path: Path, member: str, text: str) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    path.write_bytes(buffer.getvalue())


def write_fixture(config: ResearchConfig, spec: FixtureSpec, target: Path) -> None:
    rng = np.random.default_rng(spec.seed)
    target.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(spec.start, periods=spec.n_days)

    factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    factor_vols = [0.010, 0.005, 0.005, 0.003, 0.003, 0.006]
    factors = np.column_stack([_garch_path(len(dates), rng, v) for v in factor_vols])
    rf = np.full(len(dates), 0.0001)

    for key, name in config.data.files.items():
        member = name.replace("_CSV.zip", ".csv").replace(".zip", ".csv")
        preamble = [
            "This file was created by a synthetic fixture generator for testing.",
            "It mirrors the layout of the Kenneth R. French Data Library daily files.",
            "Missing data are indicated by -99.99 or -999.",
            "",
        ]
        if key in spec.columns:
            n = spec.columns[key]
            loadings = rng.normal([1.0, 0.0, 0.0], [0.25, 0.6, 0.6], size=(n, 3))
            idio = np.column_stack(
                [_garch_path(len(dates), rng, v) for v in rng.uniform(0.004, 0.012, n)]
            )
            returns = factors[:, :3] @ loadings.T + idio
            equal = returns + rng.normal(0, 0.002, returns.shape)
            percent_vw = np.round(returns * 100, 2)
            percent_ew = np.round(equal * 100, 2)
            # A fixed count (at least one) guarantees every file exercises the sentinel path.
            n_late = max(1, round(spec.late_start_share * n)) if spec.late_start_share > 0 else 0
            for column in rng.choice(n, size=n_late, replace=False):
                cut = int(rng.integers(1, max(spec.late_start_max_days, 2)))
                percent_vw[:cut, column] = -99.99
                percent_ew[:cut, column] = -99.99
            names = [f"P{i + 1:03d}" for i in range(n)]
            body = [
                "  Average Value Weighted Returns -- Daily",
                *_format(dates, percent_vw, names),
                "",
                "  Average Equal Weighted Returns -- Daily",
                *_format(dates, percent_ew, names),
            ]
        elif key == "ff3":
            data = np.column_stack([factors[:, :3], rf]) * 100
            body = _format(dates, np.round(data, 2), ["Mkt-RF", "SMB", "HML", "RF"])
        elif key == "ff5":
            # The five-factor file starts later than the others, as the real one does.
            offset = len(dates) // 3
            data = np.column_stack([factors[offset:, :5], rf[offset:]]) * 100
            body = _format(dates[offset:], np.round(data, 2), [*factor_names[:5], "RF"])
        elif key == "mom":
            data = factors[:, 5:6] * 100
            body = ["Mom", *_format(dates, np.round(data, 2), ["Mom"])]
        else:
            raise ValueError(f"fixture has no generator for file key {key!r}")
        text = "\n".join([*preamble, *body, "", "Copyright 2026 synthetic fixture", ""])
        _write_zip(target / name, member, text.replace("\n", "\r\n"))

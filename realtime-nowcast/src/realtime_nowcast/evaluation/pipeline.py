"""Estimate stage: every analysis in order, written to data/runs/estimates.json and
reports/tables/*.csv. Each stage prints a progress line with the elapsed time."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realtime_nowcast.config import Config
from realtime_nowcast.data import build, vintages
from realtime_nowcast.data.build import Tables
from realtime_nowcast.evaluation import market, realtime, score

Say = Callable[[str], object]
HISTORY_TABLES = ("nowcasts", "changes", "releases", "estimations")


def _example_quarter(config: Config, pairs: pd.DataFrame) -> str:
    wanted = config.evaluation.example_quarter
    quarters = sorted(pairs["quarter"].unique())
    if wanted == "latest":
        return str(quarters[-1])
    if wanted not in quarters:
        raise ValueError(f"example_quarter {wanted} is not among the scored quarters")
    return wanted


def _waterfall(config: Config, history: realtime.History, quarter: str) -> pd.DataFrame:
    """step, value: the entry nowcast, the news by release category, revisions,
    re-estimation and the final nowcast of one quarter."""
    categories = {s.id: s.category for s in config.panel.series}
    ch = history.changes.loc[history.changes["quarter"] == quarter]
    rel = history.releases.loc[history.releases["quarter"] == quarter]
    entry = float(ch.loc[ch["kind"] == "entry", "after"].iloc[0])
    sums: dict[str, float] = {}
    for series, impact in zip(
        rel["series"].astype(str), rel["impact"].to_numpy(dtype=float), strict=True
    ):
        sums[categories[series]] = sums.get(categories[series], 0.0) + float(impact)
    rows: list[dict[str, object]] = [{"step": "First nowcast", "value": entry, "kind": "level"}]
    for name in sorted(sums, key=lambda c: -abs(sums[c])):
        rows.append({"step": name, "value": sums[name], "kind": "change"})
    rows.append({"step": "Revisions", "value": float(ch["revisions"].sum()), "kind": "change"})
    rows.append(
        {"step": "Re-estimation", "value": float(ch["reestimation"].sum()), "kind": "change"}
    )
    rows.append({"step": "Last nowcast", "value": float(ch["after"].iloc[-1]), "kind": "level"})
    return pd.DataFrame(rows)


def _advance_check(tables: Tables) -> dict[str, Any]:
    """Our advance estimates (from ALFRED) against the Atlanta Fed's TrackRecord sheet."""
    merged = tables.advance.merge(tables.track_record, on="quarter", suffixes=("", "_atl"))
    diff = (merged["advance"] - merged["advance_atl"]).abs()
    same_day = pd.to_datetime(merged["release"]) == pd.to_datetime(merged["release_atl"])
    worst = (
        str(merged["quarter"].to_numpy()[int(np.argmax(diff.to_numpy()))]) if len(merged) else None
    )
    return {
        "quarters": len(merged),
        "within_005": int((diff <= 0.05).sum()),
        "max_abs_diff": float(diff.max()) if len(merged) else None,
        "worst_quarter": worst,
        "release_dates_agree": int(same_day.sum()),
        "release_dates_differ": [str(q) for q in merged.loc[~same_day, "quarter"]],
    }


def _entries(config: Config, est: pd.DataFrame) -> list[dict[str, str | None]]:
    """Series not in the first estimation, and the first estimation they entered."""
    first = set(est["series"].iloc[0])
    out = []
    for s in config.panel.ordered:
        if s.id in first:
            continue
        dates = [d for d, ids in zip(est["date"], est["series"], strict=True) if s.id in ids]
        out.append(
            {
                "id": s.id,
                "name": s.name,
                "entered": str(pd.Timestamp(dates[0]).date()) if dates else None,
            }
        )
    return out


def _coverage(config: Config, series: tuple[vintages.SeriesVintages, ...]) -> list[dict[str, Any]]:
    out = []
    for spec, v in zip(config.panel.ordered, series, strict=True):
        out.append(
            {
                "id": spec.id,
                "name": spec.name,
                "frequency": spec.frequency,
                "first_vintage": str(v.start.min()) if v.start.size else None,
                "vintages": int(np.unique(v.start).size),
                "published": int(v.value.size),
            }
        )
    return out


def _units(
    config: Config, series: tuple[vintages.SeriesVintages, ...], *, day: np.datetime64
) -> dict[str, Any] | None:
    """Why the units come from the months before the pandemic, on the last estimation's
    vintage: the series whose standard deviation the pandemic months inflate most, by how
    much, and how many pre-pandemic standard deviations its largest pandemic month is."""
    free = [pd.Period(p, freq="M") for p in config.model.pandemic_free]
    if not free:
        return None
    last = vintages.month_index(pd.Timestamp(day).to_period("M"))
    x, periods = realtime.panel(series, config, day=day, last=last)
    before = np.asarray(periods < free[0])
    inside = np.asarray(periods.isin(free))
    if not before.any() or not inside.any():
        return None
    full = np.nanstd(x, axis=0, ddof=1)
    base_sd = np.nanstd(x[before], axis=0, ddof=1)
    base_mean = np.nanmean(x[before], axis=0)
    ratio = full / base_sd
    i = int(np.nanargmax(ratio))
    spec = config.panel.ordered[i]
    z = np.abs(x[inside, i] - base_mean[i]) / base_sd[i]
    return {
        "series": spec.id,
        "name": spec.name,
        "sd_ratio": float(ratio[i]),
        "largest_z": float(np.nanmax(z)),
        "largest_month": str(periods[inside][int(np.nanargmax(z))]),
    }


def run(
    *, config: Config, tables: Tables, tables_dir: Path, runs_dir: Path, say: Say
) -> dict[str, Any]:
    started = time.perf_counter()

    def stage(line: str) -> None:
        say(f"  [{time.perf_counter() - started:5.0f} s] {line}")

    series = build.series_vintages(config, tables.vintages)
    stage("real-time run: every day on which a nowcast could change")
    history = realtime.run(config, series, advance=tables.advance, gdpnow=tables.gdpnow, say=say)
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in history.names().items():
        frame.to_parquet(runs_dir / f"{name}.parquet", index=False)
    stage("scoring against GDPNow")
    pairs = score.paired(history.nowcasts, tables.gdpnow, tables.advance)
    evaluation = score.evaluate(config, pairs)
    tables_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(tables_dir / "pairs.csv", index=False)
    score.binned(pairs, config.evaluation.horizon_edges).to_csv(
        tables_dir / "binned.csv", index=False
    )
    stage("market test")
    market_test = market.test(config, history.changes, tables.yields)
    example = _example_quarter(config, pairs)
    _waterfall(config, history, example).to_csv(tables_dir / "waterfall.csv", index=False)
    path = history.nowcasts.loc[history.nowcasts["quarter"] == example]
    path.to_csv(tables_dir / "path_model.csv", index=False)
    tables.gdpnow.loc[tables.gdpnow["quarter"] == example].to_csv(
        tables_dir / "path_gdpnow.csv", index=False
    )
    est = history.estimations
    last = est.iloc[-1]
    changes = history.changes
    identity = (
        changes["after"]
        - changes["before"]
        - changes["revisions"]
        - changes["news"]
        - changes["reestimation"]
    ).abs()
    advance = dict(
        zip(
            tables.advance["quarter"].astype(str),
            tables.advance["advance"].to_numpy(dtype=float),
            strict=True,
        )
    )
    released = dict(
        zip(
            tables.advance["quarter"].astype(str),
            pd.to_datetime(tables.advance["release"]),
            strict=True,
        )
    )
    stage("done")
    return {
        "evaluation": evaluation,
        "market": market_test,
        "example_quarter": example,
        "example_advance": float(advance[example]),
        "example_release": str(released[example].date()),
        "model": {
            "estimations": len(est),
            "all_converged": bool(est["converged"].all()),
            "converged_count": int(est["converged"].sum()),
            "max_iterations": int(est["iterations"].max()),
            "median_iterations": float(est["iterations"].median()),
            "first_iterations": int(est["iterations"].iloc[0]),
            "median_seconds": float(est["seconds"].median()),
            "last_date": str(pd.Timestamp(last["date"]).date()),
            "pandemic_scales": [float(v) for v in last["scales"]],
            "pandemic_rho": float(last["rho"]),
            "states": int(last["states"]),
            "series_last": len(last["series"]),
            "entered_later": _entries(config, est),
            "units": _units(
                config, series, day=np.datetime64(pd.Timestamp(last["date"]).date(), "D")
            ),
        },
        "history": {
            "days": int(history.nowcasts["date"].nunique()),
            "releases": len(history.releases),
            "release_days": int(history.releases["date"].nunique()),
            "identity_max": float(identity.loc[changes["kind"] == "update"].max()),
        },
        "data": {
            "coverage": _coverage(config, series),
            "advance_check": _advance_check(tables),
            "advance_quarters": len(tables.advance),
            "gdpnow_nowcasts": len(tables.gdpnow),
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write(estimates: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimates, indent=2, default=_json_default) + "\n", encoding="utf-8")

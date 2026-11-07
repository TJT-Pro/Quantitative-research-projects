"""Score the model against GDPNow on the days GDPNow published, against the advance estimate.

For each quarter, every GDPNow nowcast is paired with the model's nowcast as of the same day
(its latest value on or before that day; the model can only have used data published by
then). The pairs are grouped by days before the advance estimate; within a group each
quarter contributes its pair closest to the release, so the quarters are the independent
observations of the Diebold and Mariano (1995) test, with the Harvey, Leybourne and
Newbold (1997) small-sample correction. Squared and absolute errors, on every quarter and
without the excluded years, are all reported, so no single metric decides the answer.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from realtime_nowcast.config import Config


def paired(history: pd.DataFrame, gdpnow: pd.DataFrame, advance: pd.DataFrame) -> pd.DataFrame:
    """quarter, date, horizon, model, gdpnow, advance, model_error, gdpnow_error."""
    releases = dict(
        zip(advance["quarter"].astype(str), pd.to_datetime(advance["release"]), strict=True)
    )
    values = dict(
        zip(advance["quarter"].astype(str), advance["advance"].to_numpy(dtype=float), strict=True)
    )
    rows = []
    for key, published in gdpnow.groupby("quarter"):
        quarter = str(key)
        if quarter not in releases:
            continue
        release = releases[quarter]
        bench = published.loc[published["date"] < release].sort_values("date")
        model = history.loc[history["quarter"] == quarter].sort_values("date")
        if bench.empty or model.empty:
            continue
        bench = bench.assign(date=bench["date"].astype("datetime64[ns]"))
        model = model.assign(date=model["date"].astype("datetime64[ns]"))
        merged = pd.merge_asof(
            bench,
            model[["date", "nowcast"]].rename(columns={"nowcast": "model"}),
            on="date",
            direction="backward",
        )
        merged = merged.dropna(subset=["model"])
        merged["horizon"] = (release - merged["date"]).dt.days
        merged["advance"] = values[quarter]
        rows.append(merged.rename(columns={"nowcast": "gdpnow"}))
    out = pd.concat(rows, ignore_index=True)
    out["model_error"] = out["model"] - out["advance"]
    out["gdpnow_error"] = out["gdpnow"] - out["advance"]
    return out[
        ["quarter", "date", "horizon", "model", "gdpnow", "advance", "model_error", "gdpnow_error"]
    ]


def binned(pairs: pd.DataFrame, edges: tuple[int, ...]) -> pd.DataFrame:
    """One pair per quarter and horizon group: the one closest to the release."""
    labels = [f"{lo}-{hi - 1}" for lo, hi in itertools.pairwise(edges)]
    group = pd.cut(pairs["horizon"], bins=list(edges), right=False, labels=labels)
    frame = pairs.assign(group=group).dropna(subset=["group"])
    keep = frame.sort_values("horizon").groupby(["quarter", "group"], observed=True).head(1)
    return keep.sort_values(["group", "quarter"]).reset_index(drop=True)


@dataclass(frozen=True)
class DM:
    statistic: float
    p_value: float
    n: int


def diebold_mariano(loss_model: np.ndarray, loss_benchmark: np.ndarray) -> DM:
    """One-step DM test with the HLN correction; negative statistics favour the model."""
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_benchmark, dtype=float)
    n = d.size
    if n < 3 or np.allclose(d, d[0]):
        return DM(statistic=float("nan"), p_value=float("nan"), n=n)
    dm = d.mean() / np.sqrt(d.var(ddof=0) / n)
    hln = dm * np.sqrt((n - 1) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return DM(statistic=float(hln), p_value=float(p), n=n)


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    em, eg = frame["model_error"].to_numpy(), frame["gdpnow_error"].to_numpy()
    sq = diebold_mariano(em**2, eg**2)
    ab = diebold_mariano(np.abs(em), np.abs(eg))
    return {
        "n": int(frame.shape[0]),
        "model_rmse": float(np.sqrt(np.mean(em**2))),
        "gdpnow_rmse": float(np.sqrt(np.mean(eg**2))),
        "model_mae": float(np.mean(np.abs(em))),
        "gdpnow_mae": float(np.mean(np.abs(eg))),
        "dm_squared": sq.statistic,
        "dm_squared_p": sq.p_value,
        "dm_absolute": ab.statistic,
        "dm_absolute_p": ab.p_value,
    }


def evaluate(config: Config, pairs: pd.DataFrame) -> dict[str, Any]:
    e = config.evaluation
    picks = binned(pairs, e.horizon_edges)
    years = pd.PeriodIndex(picks["quarter"], freq="Q").year
    ex = ~np.isin(years, list(e.excluded_years))
    groups = []
    for label, g in picks.groupby("group", observed=True):
        mask = ex[picks.index.get_indexer(g.index)]
        groups.append(
            {"group": str(label), "all": _metrics(g), "ex_excluded": _metrics(g.loc[mask])}
        )
    nearest = picks.sort_values("horizon").groupby("quarter").head(1)
    near_years = pd.PeriodIndex(nearest["quarter"], freq="Q").year
    near_ex = ~np.isin(near_years, list(e.excluded_years))
    return {
        "quarters": int(pairs["quarter"].nunique()),
        "first_quarter": str(pairs["quarter"].min()),
        "last_quarter": str(pairs["quarter"].max()),
        "pairs": int(pairs.shape[0]),
        "groups": groups,
        "final": {"all": _metrics(nearest), "ex_excluded": _metrics(nearest.loc[near_ex])},
        "excluded_years": list(e.excluded_years),
    }

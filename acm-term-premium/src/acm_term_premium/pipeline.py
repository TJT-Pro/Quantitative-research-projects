"""Pipeline stages. Outputs go to data/runs/<config fingerprint>/ with the
resolved configuration, so any stage can be rerun and every number traced."""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from acm_term_premium.config import ResearchConfig
from acm_term_premium.data import panel
from acm_term_premium.data.source import FetchReport, fetch_all
from acm_term_premium.evaluation import events, forecast, realtime
from acm_term_premium.models.acm import AcmModel, Convention, FactorMap, bootstrap_term_premium
from acm_term_premium.models.configured import fit_model
from acm_term_premium.paths import ProjectPaths

Log = Callable[[str], None]


def run_dir(paths: ProjectPaths, config: ResearchConfig) -> Path:
    directory = paths.runs / config.fingerprint()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.resolved.json").write_text(
        json.dumps(config.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    return directory


def fetch(
    config: ResearchConfig, paths: ProjectPaths, mirror: Path | None, log: Log
) -> FetchReport:
    paths.ensure()
    return fetch_all(config.data, paths.raw, paths.manifest, mirror=mirror, log=log)


def build(config: ResearchConfig, paths: ProjectPaths, log: Log) -> None:
    paths.ensure()
    report = panel.build(config, paths.raw, paths.processed, paths.manifest)
    if report is not None:
        (paths.processed / "splice.json").write_text(
            json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8"
        )
        log(
            f"short-rate splice: {report.replaced_months} months replaced, "
            f"R^2 {report.r_squared:.3f}"
        )
    log(f"built processed inputs in {paths.processed.relative_to(paths.root)}")


def published_months(config: ResearchConfig) -> list[int]:
    return [12 * y for y in config.validation.published_years]


def _save_decomposition(
    model: AcmModel, curve: pd.DataFrame, months: list[int], path: Path
) -> None:
    dec = model.decompose(curve, months)
    long = pd.concat({key: frame.stack() for key, frame in dec.items()}, axis=1).reset_index()
    long.columns = ["date", "maturity", *dec.keys()]
    long.to_parquet(path)


def save_models(models: dict[pd.Timestamp, AcmModel], path: Path) -> None:
    """Store the per-month real-time models as plain arrays (no pickle)."""
    dates = sorted(models)
    first = models[dates[0]]
    np.savez_compressed(
        path,
        dates=np.array([d.value for d in dates], dtype=np.int64),
        maturities=np.array(first.factor_map.maturities),
        mean=np.stack([models[d].factor_map.mean for d in dates]),
        loadings=np.stack([models[d].factor_map.loadings for d in dates]),
        a=np.stack([models[d].a for d in dates]),
        b=np.stack([models[d].b for d in dates]),
        a_rn=np.stack([models[d].a_rn for d in dates]),
        b_rn=np.stack([models[d].b_rn for d in dates]),
    )


def load_models(path: Path) -> dict[pd.Timestamp, AcmModel]:
    """Rebuild the pricing part of each model; parameters not needed for pricing are zeroed."""
    data = np.load(path, allow_pickle=False)
    maturities = [int(m) for m in data["maturities"]]
    k = data["loadings"].shape[2]
    zero_k, zero_kk = np.zeros(k), np.zeros((k, k))
    out: dict[pd.Timestamp, AcmModel] = {}
    for i, stamp in enumerate(data["dates"]):
        date = pd.Timestamp(int(stamp))
        fmap = FactorMap(maturities, data["mean"][i], data["loadings"][i], np.zeros(k))
        out[date] = AcmModel(
            fmap,
            zero_k,
            zero_kk,
            zero_kk,
            0.0,
            0.0,
            zero_k,
            zero_k,
            zero_kk,
            data["a"][i],
            data["b"][i],
            data["a_rn"][i],
            data["b_rn"][i],
            date,
            date,
        )
    return out


def _record_timing(directory: Path, stage: str, seconds: float) -> None:
    path = directory / "timing.json"
    timing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    timing[stage] = {
        "seconds": round(seconds, 1),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")


def estimate(config: ResearchConfig, paths: ProjectPaths, log: Log) -> None:
    """Full-sample fits, bootstrap bands, real-time estimates, forecasts, policy days."""
    started = time.perf_counter()
    out = run_dir(paths, config)
    months = published_months(config)
    curve = panel.load_curve(paths.processed, "curve_monthly")
    raw_curve = panel.load_curve(paths.processed, "curve_monthly_unspliced")
    daily = panel.load_curve(paths.processed, "curve_daily")

    main = fit_model(curve, config.model)
    other: Convention = "paper" if config.model.convention == "nyfed" else "nyfed"
    _save_decomposition(main, curve, months, out / "fullsample.parquet")
    other_model = fit_model(curve, config.model, convention=other)
    _save_decomposition(other_model, curve, months, out / "fullsample_other.parquet")
    _save_decomposition(
        fit_model(raw_curve, config.model),
        raw_curve,
        months,
        out / "fullsample_unspliced.parquet",
    )
    _save_decomposition(main, daily, months, out / "daily.parquet")
    log("full-sample estimates written")

    u = config.uncertainty
    draws = bootstrap_term_premium(
        curve,
        main,
        return_maturities=config.model.return_maturities,
        months=[12 * config.validation.headline_years],
        replications=u.replications,
        block=u.block_months,
        rng=np.random.default_rng(u.seed),
        convention=config.model.convention,
        max_spectral_radius=config.model.max_spectral_radius,
    )[:, :, 0]
    tail = (1.0 - u.band) / 2.0
    bands = pd.DataFrame(
        {"lower": np.quantile(draws, tail, axis=0), "upper": np.quantile(draws, 1 - tail, axis=0)},
        index=curve.index,
    )
    bands.to_parquet(out / "bands.parquet")
    log(f"bootstrap: {u.replications} replications")

    rt_months = sorted(set(months) | set(config.forecasting.maturities))
    rt = realtime.run(config, curve, rt_months, progress=log)
    for key in ("term_premium", "risk_neutral", "fitted", "unconstrained_term_premium"):
        frame = getattr(rt, key).copy()
        frame.columns = frame.columns.astype(str)
        frame.to_parquet(out / f"realtime_{key}.parquet")
    rt.diagnostics.to_parquet(out / "realtime_diagnostics.parquet")
    save_models(rt.models, out / "realtime_models.npz")
    log(f"real-time estimates: {len(rt.models)} months")

    f = config.forecasting
    results, forecasts = forecast.evaluate(
        curve,
        rt.term_premium.rename(columns=int),
        maturities=f.maturities,
        horizon=f.horizon_months,
        cp_forward_years=f.cp_forward_years,
        cp_target_years=f.cp_target_years,
        min_train=f.min_train_months,
        nw_lags=f.newey_west_lags,
    )
    forecasts.to_parquet(out / "forecasts.parquet")
    pd.DataFrame([asdict(r) for r in results]).to_parquet(out / "forecast_stats.parquet")
    log("out-of-sample forecasts evaluated")

    changes = panel.load_table(paths.processed, "policy_changes")
    changes = changes[changes["date"] >= pd.Timestamp(config.events.start)]
    maturity = 12 * config.events.maturity_years
    event_days = events.decompose_changes(
        daily, pd.DatetimeIndex(changes["date"]), rt.models, maturity
    )
    after = pd.DatetimeIndex(daily.index[daily.index >= pd.Timestamp(config.events.start)])
    all_days = events.decompose_changes(daily, after, rt.models, maturity)
    event_days.to_parquet(out / "events.parquet")
    all_days.to_parquet(out / "all_days.parquet")
    if len(event_days) >= 3:
        summary = events.summarize(event_days, changes, all_days)
        (out / "events_summary.json").write_text(
            json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8"
        )
    log(f"policy-rate change days decomposed: {len(event_days)}")
    _record_timing(out, "estimate", time.perf_counter() - started)

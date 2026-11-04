"""Assemble results.json, tables and figures from a run. The README and paper read
every number from results.json; nothing in the prose is typed by hand."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from acm_term_premium.config import ResearchConfig
from acm_term_premium.data import panel
from acm_term_premium.data.source import load_manifest
from acm_term_premium.evaluation import events
from acm_term_premium.paths import ProjectPaths
from acm_term_premium.plotting import figures

SCHEMA_VERSION = 1


def provenance(paths: ProjectPaths) -> str:
    entries = load_manifest(paths.manifest)
    if entries and all(e.url.startswith("https://") for e in entries.values()):
        return "official"
    return "synthetic"


def _md(frame: pd.DataFrame, fmt: dict[str, str]) -> str:
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---:" if c in fmt else "---" for c in frame.columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = row[c]
            if c in fmt and isinstance(v, (int, float, np.floating)):
                cells.append("" if pd.isna(v) else format(v, fmt[c]))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table(paths: ProjectPaths, frame: pd.DataFrame, stem: str, fmt: dict[str, str]) -> str:
    paths.tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.tables / f"{stem}.csv", index=False)
    text = _md(frame, fmt)
    (paths.tables / f"{stem}.md").write_text(text + "\n", encoding="utf-8")
    return text


def _wide(long: pd.DataFrame, value: str) -> pd.DataFrame:
    out = long.pivot(index="date", columns="maturity", values=value)
    out.columns = [int(c) // 12 for c in out.columns]
    return out


def _rel(paths: ProjectPaths, written: list[Path]) -> list[str]:
    return [p.relative_to(paths.root).as_posix() for p in written]


def _validation(
    paths: ProjectPaths, run: Path, years: int
) -> tuple[dict[str, Any], dict[str, str]]:
    mine = pd.read_parquet(run / "fullsample.parquet")
    other = pd.read_parquet(run / "fullsample_other.parquet")
    unspliced = pd.read_parquet(run / "fullsample_unspliced.parquet")
    published = panel.load_table(paths.processed, "published_monthly")
    rows = []
    for key, label in (
        ("fitted", "Fitted yield"),
        ("risk_neutral", "Risk-neutral yield"),
        ("term_premium", "Term premium"),
    ):
        ours = _wide(mine, key)
        pub = published.pivot(index="date", columns="maturity_years", values=key)
        common = ours.index.intersection(pub.index)
        diff = (ours.loc[common] - pub.loc[common, ours.columns]) * 1e4
        for maturity in ours.columns:
            d = diff[maturity]
            rows.append(
                {
                    "Series": label,
                    "Maturity (years)": int(maturity),
                    "RMSE (bp)": float(np.sqrt((d**2).mean())),
                    "Max abs (bp)": float(d.abs().max()),
                    "Correlation": float(
                        np.corrcoef(ours.loc[common, maturity], pub.loc[common, maturity])[0, 1]
                    ),
                    "Months": len(common),
                }
            )
    table = pd.DataFrame(rows)
    headline = table[(table["Series"] == "Term premium") & (table["Maturity (years)"] == years)]
    tp = _wide(mine, "term_premium")[years]
    tp_other = _wide(other, "term_premium")[years]
    tp_raw = _wide(unspliced, "term_premium")[years]
    pub_tp = _published_tp(paths, years)
    tp_dates = pd.DatetimeIndex(tp.index)
    by_decade = ((tp - pub_tp.reindex(tp_dates)) * 1e4).groupby(tp_dates.year // 10 * 10)
    decade = pd.DataFrame(
        {
            "Decade": [f"{d}s" for d in by_decade.groups],
            "RMSE (bp)": [float(np.sqrt((g**2).mean())) for _, g in by_decade],
            "Max abs (bp)": [float(g.abs().max()) for _, g in by_decade],
        }
    )
    daily_ours = _wide(pd.read_parquet(run / "daily.parquet"), "term_premium")[years]
    daily_pub = panel.load_table(paths.processed, "published_daily").pivot(
        index="date", columns="maturity_years", values="term_premium"
    )[years]
    common_days = daily_ours.index.intersection(daily_pub.index)
    daily_diff = (daily_ours.loc[common_days] - daily_pub.loc[common_days]) * 1e4
    convention_gap = (tp_other - tp) * 1e4
    splice_gap = (tp_raw - tp.reindex(tp_raw.index)) * 1e4
    summary = {
        "headline_years": years,
        "tp_rmse_bp": float(headline["RMSE (bp)"].iloc[0]),
        "tp_max_abs_bp": float(headline["Max abs (bp)"].iloc[0]),
        "max_abs_all_series_bp": float(table["Max abs (bp)"].max()),
        "months": int(headline["Months"].iloc[0]),
        "start": _iso(tp.index[0]),
        "end": _iso(tp.index[-1]),
        "daily_days": len(common_days),
        "daily_max_abs_bp": float(daily_diff.abs().max()),
        "convention_gap_mean_bp": float(convention_gap.mean()),
        "convention_gap_max_abs_bp": float(convention_gap.abs().max()),
        "splice_gap_mean_abs_bp": float(splice_gap.abs().mean()),
        "splice_gap_max_abs_bp": float(splice_gap.abs().max()),
        "latest_tp_pct": float(tp.iloc[-1] * 100),
        "latest_date": _iso(tp.index[-1]),
    }
    tables = {
        "validation": _table(
            paths,
            table,
            "validation",
            {"RMSE (bp)": ".4f", "Max abs (bp)": ".4f", "Correlation": ".6f"},
        ),
        "validation_decades": _table(
            paths, decade, "validation_decades", {"RMSE (bp)": ".4f", "Max abs (bp)": ".4f"}
        ),
    }
    return summary, tables


def _iso(value: object) -> str:
    return pd.Timestamp(str(value)).date().isoformat()


def _published_tp(paths: ProjectPaths, years: int) -> pd.Series:
    published = panel.load_table(paths.processed, "published_monthly")
    wide = published.pivot(index="date", columns="maturity_years", values="term_premium")
    return wide[years]


def _settings(config: ResearchConfig) -> dict[str, Any]:
    return {
        "n_factors": config.model.n_factors,
        "published_years": config.validation.published_years,
        "convention": config.model.convention,
        "max_spectral_radius": config.model.max_spectral_radius,
        "return_maturities": config.model.return_maturities,
        "short_rate": config.model.short_rate.model_dump(),
        "bootstrap": config.uncertainty.model_dump(),
        "realtime_min_months": config.realtime.min_months,
        "forecasting": config.forecasting.model_dump(),
        "events_start": config.events.start,
    }


def _full_sample(
    paths: ProjectPaths, run: Path, years: int, dpi: int, results: dict[str, Any]
) -> pd.Series:
    """Replication and decomposition figures and the bootstrap summary."""
    mine = pd.read_parquet(run / "fullsample.parquet")
    tp = _wide(mine, "term_premium")[years]
    results["figures"]["replication"] = _rel(
        paths,
        figures.replication(tp, _published_tp(paths, years), years, out=paths.figures, dpi=dpi),
    )
    bands = pd.read_parquet(run / "bands.parquet")
    results["figures"]["decomposition"] = _rel(
        paths,
        figures.decomposition(
            _wide(mine, "fitted")[years],
            _wide(mine, "risk_neutral")[years],
            tp,
            years=years,
            lower=bands["lower"],
            upper=bands["upper"],
            out=paths.figures,
            dpi=dpi,
        ),
    )
    width = (bands["upper"] - bands["lower"]) * 1e4
    results["uncertainty"] = {
        "median_band_width_bp": float(width.median()),
        "latest_lower_pct": float(bands["lower"].iloc[-1] * 100),
        "latest_upper_pct": float(bands["upper"].iloc[-1] * 100),
    }
    return tp


def _realtime(
    paths: ProjectPaths,
    run: Path,
    tp: pd.Series,
    *,
    years: int,
    dpi: int,
    results: dict[str, Any],
) -> None:
    n = str(12 * years)
    rt_tp = pd.read_parquet(run / "realtime_term_premium.parquet")[n]
    rt_raw = pd.read_parquet(run / "realtime_unconstrained_term_premium.parquet")[n]
    diag = pd.read_parquet(run / "realtime_diagnostics.parquet")
    dates = pd.DatetimeIndex(rt_tp.index)
    revision = (rt_tp - tp.reindex(dates)) * 1e4
    raw_revision = (rt_raw - tp.reindex(rt_raw.index)) * 1e4
    by_decade = revision.abs().groupby(dates.year // 10 * 10).mean()
    worst = dates[int(np.argmax(revision.abs().to_numpy()))]
    constrained = pd.DatetimeIndex(diag.index[diag["constrained"].to_numpy(dtype=bool)])
    results["realtime"] = {
        "months": len(rt_tp),
        "start": _iso(dates[0]),
        "mean_abs_revision_bp": float(revision.abs().mean()),
        "max_abs_revision_bp": float(revision.abs().max()),
        "max_revision_date": _iso(worst),
        "unconstrained_max_abs_revision_bp": float(raw_revision.abs().max()),
        "correlation_with_full_sample": float(np.corrcoef(rt_tp, tp.reindex(dates))[0, 1]),
        "constrained_months": len(constrained),
        "constrained_first": _iso(constrained.min()) if len(constrained) else None,
        "constrained_last": _iso(constrained.max()) if len(constrained) else None,
        "largest_ols_spectral_radius": float(diag["spectral_radius"].max()),
    }
    decade_table = pd.DataFrame(
        {
            "Decade": [f"{d}s" for d in by_decade.index],
            "Mean abs revision (bp)": by_decade.to_numpy(),
        }
    )
    results["tables"]["revisions"] = _table(
        paths, decade_table, "revisions", {"Mean abs revision (bp)": ".1f"}
    )
    results["figures"]["revisions"] = _rel(
        paths,
        figures.revisions(
            rt_tp, rt_raw, tp, diag["constrained"], years=years, out=paths.figures, dpi=dpi
        ),
    )


def _forecasts(paths: ProjectPaths, run: Path, dpi: int, results: dict[str, Any]) -> None:
    stats = pd.read_parquet(run / "forecast_stats.parquet")
    if stats.empty:
        return
    names = {"acm": "ACM term premium (real time)", "cp": "Cochrane-Piazzesi factor"}
    table = pd.DataFrame(
        {
            "Bond": [f"{m // 12}-year" for m in stats["maturity"]],
            "Predictor": stats["predictor"].map(names),
            "Months": stats["months"],
            "OOS R2 (%)": stats["oos_r2"] * 100,
            "Clark-West": stats["clark_west"],
            "p-value": stats["p_value"],
        }
    )
    results["tables"]["forecasts"] = _table(
        paths, table, "forecasts", {"OOS R2 (%)": ".2f", "Clark-West": ".2f", "p-value": ".3f"}
    )
    results["forecasts"] = stats.to_dict(orient="records")
    results["figures"]["forecasts"] = _rel(
        paths, figures.forecast_r2(stats, out=paths.figures, dpi=dpi)
    )
    scored = pd.read_parquet(run / "forecasts.parquet").dropna(subset=["actual", "acm", "cp"])
    results["forecast_sample"] = {
        "start": _iso(scored["date"].min()),
        "end": _iso(scored["date"].max()),
    }


def _events(
    config: ResearchConfig, paths: ProjectPaths, run: Path, dpi: int, results: dict[str, Any]
) -> None:
    summary_path = run / "events_summary.json"
    if not summary_path.is_file():
        return
    results["events"] = json.loads(summary_path.read_text(encoding="utf-8"))
    event_days = pd.read_parquet(run / "events.parquet")
    every_day = events.all_days_summary(pd.read_parquet(run / "all_days.parquet"))
    results["events"]["all_days"] = {**every_day, "days": int(every_day["days"])}
    results["figures"]["policy_days"] = _rel(
        paths,
        figures.policy_days(
            results["events"],
            years=config.events.maturity_years,
            out=paths.figures,
            dpi=dpi,
        ),
    )
    results["events"]["first"] = _iso(event_days["date"].min())
    results["events"]["last"] = _iso(event_days["date"].max())


def build_results(config: ResearchConfig, paths: ProjectPaths) -> dict[str, Any]:
    run = paths.runs / config.fingerprint()
    if not (run / "fullsample.parquet").is_file():
        raise FileNotFoundError(
            f"no estimates for configuration {config.fingerprint()}: "
            "run `acm estimate` with the same config first"
        )
    years = config.validation.headline_years
    dpi = config.report.figure_dpi
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_fingerprint": config.fingerprint(),
        "provenance": provenance(paths),
        "settings": _settings(config),
        "tables": {},
        "figures": {},
    }
    validation, tables = _validation(paths, run, years)
    results["validation"] = validation
    results["tables"].update(tables)
    splice_path = paths.processed / "splice.json"
    if splice_path.is_file():
        results["splice"] = json.loads(splice_path.read_text(encoding="utf-8"))
    tp = _full_sample(paths, run, years, dpi, results)
    _realtime(paths, run, tp, years=years, dpi=dpi, results=results)
    _forecasts(paths, run, dpi, results)
    _events(config, paths, run, dpi, results)
    timing_path = run / "timing.json"
    if timing_path.is_file():
        results["runtime"] = json.loads(timing_path.read_text(encoding="utf-8"))
    return results


def write(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")


def read(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded

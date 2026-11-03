"""Turn backtest and simulation output into tables, figures and results.json.

results.json is the only place the README and the paper read numbers from;
nothing quoted in prose is typed by hand. Each section below builds one part
of it and writes the matching table and figure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from covariance_horse_race.config import ResearchConfig
from covariance_horse_race.data import store
from covariance_horse_race.data.source import load_manifest
from covariance_horse_race.evaluation import summary
from covariance_horse_race.models.shrinkage import analytical_nonlinear_shrinkage
from covariance_horse_race.models.unconditional import sample_covariance
from covariance_horse_race.paths import ProjectPaths
from covariance_horse_race.plotting import figures
from covariance_horse_race.plotting.style import ModelStyle, model_styles

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _Context:
    config: ResearchConfig
    paths: ProjectPaths
    labels: dict[str, str]
    universe_labels: dict[str, str]
    styles: dict[str, ModelStyle]

    @property
    def dpi(self) -> int:
        return self.config.report.figure_dpi

    def rel(self, written: list[Path]) -> list[str]:
        return [p.relative_to(self.paths.root).as_posix() for p in written]


def provenance(paths: ProjectPaths) -> str:
    """'library' when every raw file came over https from the source, else 'synthetic'."""
    entries = load_manifest(paths.manifest)
    if entries and all(e.url.startswith("https://") for e in entries.values()):
        return "library"
    return "synthetic"


def _pairs(columns: pd.Index) -> list[tuple[Any, Any]]:
    """Two-level column labels of a pivot table as plain tuples."""
    return [(key[0], key[1]) for key in columns.to_list()]


def _md_table(frame: pd.DataFrame, floatfmt: dict[str, str]) -> str:
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---:" if c in floatfmt else "---" for c in frame.columns) + " |"
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if column in floatfmt and isinstance(value, (int, float, np.floating)):
                cells.append("" if pd.isna(value) else format(value, floatfmt[column]))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_table(ctx: _Context, frame: pd.DataFrame, stem: str, fmt: dict[str, str]) -> str:
    ctx.paths.tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(ctx.paths.tables / f"{stem}.csv", index=False)
    markdown = _md_table(frame, fmt)
    (ctx.paths.tables / f"{stem}.md").write_text(markdown + "\n", encoding="utf-8")
    return markdown


def _data_section(ctx: _Context, monthly: pd.DataFrame) -> tuple[dict[str, Any], str]:
    keys = sorted(store.file_roles(ctx.config))
    cov = pd.DataFrame([c.__dict__ for c in store.coverage(ctx.paths.processed, keys)])
    manifest = load_manifest(ctx.paths.manifest)
    cov["retrieved_at"] = [manifest[k].retrieved_at[:10] for k in cov["key"]]
    columns = [
        "key",
        "first_date",
        "last_date",
        "n_days",
        "n_columns",
        "missing_share",
        "columns_complete_from",
        "retrieved_at",
    ]
    readable = cov[columns].rename(
        columns={
            "key": "File",
            "first_date": "First day",
            "last_date": "Last day",
            "n_days": "Days",
            "n_columns": "Series",
            "missing_share": "Missing share",
            "columns_complete_from": "All series from",
            "retrieved_at": "Retrieved",
        }
    )
    table = _write_table(ctx, readable, "coverage", {"Missing share": ".3f"})
    # Every model-month that could not be evaluated, and why.
    status = (
        monthly.assign(status=monthly["status"].str.split(":").str[0])
        .groupby(["universe", "window", "model", "status"])
        .size()
        .rename("months")
        .reset_index()
    )
    data = {
        "coverage": cov.to_dict(orient="records"),
        "status_counts": status.to_dict(orient="records"),
    }
    return data, table


def _performance_section(
    ctx: _Context, monthly: pd.DataFrame, daily: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    performance: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    annualization = ctx.config.backtest.annualization_days
    for comparison in summary.comparisons(monthly, "full"):
        table = summary.performance_table(monthly, daily, comparison, annualization)
        if table.empty:
            continue
        best_vol = float(table["vol_gmv"].min())
        for model, row in table.iterrows():
            record: dict[str, Any] = {
                "universe": comparison.universe,
                "window": comparison.window,
                "model": str(model),
                "label": ctx.labels[str(model)],
            }
            record.update({str(k): float(v) for k, v in row.items()})
            record["excess_vol_pct"] = 100.0 * (float(row["vol_gmv"]) / best_vol - 1.0)
            record["start"] = comparison.months[0].date().isoformat()
            record["end"] = comparison.months[-1].date().isoformat()
            performance.append(record)
        excluded.extend(
            {
                "universe": comparison.universe,
                "universe_label": ctx.universe_labels[comparison.universe],
                "window": comparison.window,
                "model": model,
                "label": ctx.labels[model],
                "reason": reason,
            }
            for model, reason in sorted(comparison.excluded.items())
        )
    return performance, excluded


def _performance_tables(ctx: _Context, perf: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    bt = ctx.config.backtest
    head = perf[
        (perf["universe"] == ctx.config.report.headline_universe)
        & (perf["window"] == bt.primary_window)
    ].sort_values("vol_gmv")
    head_table = pd.DataFrame(
        {
            "Model": head["label"],
            "GMV vol (%)": head["vol_gmv"] * 100,
            "Long-only vol (%)": head["vol_gmv_long_only"] * 100,
            "Turnover (%/month)": head["turnover"] * 100,
            "QLIKE": head["qlike"],
        }
    )
    tables = {
        "headline": _write_table(
            ctx,
            head_table,
            "headline_performance",
            {
                "GMV vol (%)": ".2f",
                "Long-only vol (%)": ".2f",
                "Turnover (%/month)": ".1f",
                "QLIKE": ".3f",
            },
        )
    }
    sweep = perf.pivot_table(index="label", columns=["universe", "window"], values="vol_gmv")
    sweep = sweep.mul(100)
    sweep.columns = pd.Index(
        [f"{ctx.universe_labels[str(u)]} / {w}d" for u, w in _pairs(sweep.columns)]
    )
    tables["sweep"] = _write_table(
        ctx,
        sweep.reset_index().rename(columns={"label": "Model"}),
        "sweep_vol",
        dict.fromkeys([str(c) for c in sweep.columns], ".2f"),
    )
    return head, tables


def _mcs_section(
    ctx: _Context, monthly: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = ctx.config
    by_period = [
        {
            "universe": rec.comparison.universe,
            "period": rec.comparison.period,
            "loss": rec.loss,
            "months": len(rec.comparison.months),
            "included": rec.result.included,
            "included_labels": [ctx.labels[m] for m in rec.result.included],
            "p_values": rec.result.p_values,
            "elimination_order": rec.result.elimination_order,
            "excluded": rec.comparison.excluded,
        }
        for rec in summary.run_mcs(config, monthly, config.evaluation.losses)
    ]
    full_settings = config.evaluation.mcs.model_copy(update={"period": "full"})
    full_config = config.model_copy(
        update={"evaluation": config.evaluation.model_copy(update={"mcs": full_settings})}
    )
    full = [
        {
            "universe": rec.comparison.universe,
            "loss": rec.loss,
            "months": len(rec.comparison.months),
            "included": rec.result.included,
            "included_labels": [ctx.labels[m] for m in rec.result.included],
            "p_values": rec.result.p_values,
            "models": rec.comparison.models,
        }
        for rec in summary.run_mcs(full_config, monthly, [config.report.headline_loss])
    ]
    return by_period, full


def _mcs_figure(ctx: _Context, by_period: list[dict[str, Any]]) -> tuple[list[str], str] | None:
    config = ctx.config
    head = config.report.headline_universe
    rows = [
        r for r in by_period if r["universe"] == head and r["loss"] == config.report.headline_loss
    ]
    if not rows:
        return None
    window = config.backtest.primary_window
    order = [m.name for m in config.models if m.runs_on(window)]
    grid = pd.DataFrame(index=order, columns=[r["period"] for r in rows], dtype=float)
    for r in rows:
        for model, p in r["p_values"].items():
            grid.loc[model, r["period"]] = p
    grid = grid.dropna(how="all")
    written = figures.mcs_heatmap(
        grid,
        styles=ctx.styles,
        alpha=config.evaluation.mcs.alpha,
        title=f"Model Confidence Set by period: {ctx.universe_labels[head]}, {window}-day window",
        out=ctx.paths.figures,
        stem="mcs_heatmap",
        dpi=ctx.dpi,
    )
    table = grid.reset_index(names="model")
    table.insert(1, "Model", [ctx.labels[str(m)] for m in table["model"]])
    markdown = _write_table(
        ctx,
        table.drop(columns="model"),
        "mcs_heatmap",
        dict.fromkeys([str(c) for c in grid.columns], ".2f"),
    )
    return ctx.rel(written), markdown


def _spectrum(ctx: _Context) -> tuple[np.ndarray, np.ndarray, str, int] | None:
    frame = store.load(ctx.paths.processed, ctx.config.report.spectrum_universe)
    window = frame.iloc[-ctx.config.backtest.primary_window :]
    window = window.loc[:, window.notna().all()]
    if window.shape[1] < 2:
        return None
    x = window.to_numpy(dtype=float)
    sample_ev = np.linalg.eigvalsh(sample_covariance(x))
    shrunk_ev = np.linalg.eigvalsh(analytical_nonlinear_shrinkage(x))
    # A singular sample covariance has zero eigenvalues that a log axis cannot show.
    sample_ev = np.clip(sample_ev, np.finfo(float).tiny, None)
    end = pd.Timestamp(window.index[-1]).date().isoformat()
    return sample_ev, shrunk_ev, end, int(x.shape[1])


def _spectrum_section(ctx: _Context) -> tuple[list[str], dict[str, Any]] | None:
    spectrum = _spectrum(ctx)
    if spectrum is None:
        return None
    sample_ev, shrunk_ev, end, n = spectrum
    universe = ctx.config.report.spectrum_universe
    window = ctx.config.backtest.primary_window
    written = figures.eigen_spectrum(
        sample_ev,
        shrunk_ev,
        title=f"Eigenvalues, {ctx.universe_labels[universe]}: {window} days to {end}",
        out=ctx.paths.figures,
        stem="eigen_spectrum",
        dpi=ctx.dpi,
    )
    info = {
        "universe": universe,
        "end": end,
        "n_assets": n,
        "window": window,
        "sample_condition_number": float(sample_ev.max() / sample_ev.min()),
        "shrunk_condition_number": float(shrunk_ev.max() / shrunk_ev.min()),
    }
    return ctx.rel(written), info


def _simulation_section(ctx: _Context, simulation: pd.DataFrame) -> tuple[list[str], str]:
    written = figures.simulation_panels(
        simulation, styles=ctx.styles, out=ctx.paths.figures, stem="simulation", dpi=ctx.dpi
    )
    table = simulation.assign(label=simulation["model"].map(ctx.labels)).pivot_table(
        index="label", columns=["n", "t"], values="excess_gmv_variance_pct"
    )
    table.columns = pd.Index([f"N={n}, T={t}" for n, t in _pairs(table.columns)])
    markdown = _write_table(
        ctx,
        table.reset_index().rename(columns={"label": "Model"}),
        "simulation_excess_variance",
        dict.fromkeys([str(c) for c in table.columns], ".2f"),
    )
    return ctx.rel(written), markdown


def _figures(ctx: _Context, perf: pd.DataFrame, head: pd.DataFrame) -> dict[str, list[str]]:
    """The two figures drawn straight from the performance table."""
    bt = ctx.config.backtest
    out = {
        "dimension_sweep": ctx.rel(
            figures.dimension_sweep(
                perf[["universe", "window", "model", "excess_vol_pct"]],
                styles=ctx.styles,
                universe_labels=ctx.universe_labels,
                out=ctx.paths.figures,
                stem="dimension_sweep",
                dpi=ctx.dpi,
            )
        )
    }
    if not head.empty:
        universe = ctx.universe_labels[ctx.config.report.headline_universe]
        out["turnover_vs_vol"] = ctx.rel(
            figures.turnover_vs_vol(
                head.set_index("model")[["vol_gmv", "turnover"]],
                styles=ctx.styles,
                title=f"Risk against trading: {universe}, {bt.primary_window}-day window",
                out=ctx.paths.figures,
                stem="turnover_vs_vol",
                dpi=ctx.dpi,
            )
        )
    return out


def _settings(config: ResearchConfig) -> dict[str, Any]:
    bt = config.backtest
    return {
        "primary_window": bt.primary_window,
        "windows": bt.windows,
        "mcs_alpha": config.evaluation.mcs.alpha,
        "mcs_bootstrap_replications": config.evaluation.mcs.bootstrap_replications,
        "mcs_mean_block_length": config.evaluation.mcs.mean_block_length,
        "universes": [{"key": u.key, "label": u.label} for u in config.universes],
        "mcs_period": config.evaluation.mcs.period,
        "models": [
            {
                "name": m.name,
                "label": m.label,
                "kind": m.kind,
                "description": m.description,
                "windows": m.windows,
            }
            for m in config.models
        ],
        "simulation": config.simulation.model_dump(mode="json"),
    }


def build_results(
    config: ResearchConfig,
    paths: ProjectPaths,
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    simulation: pd.DataFrame,
) -> dict[str, Any]:
    ctx = _Context(
        config=config,
        paths=paths,
        labels={m.name: m.label for m in config.models},
        universe_labels={u.key: u.label for u in config.universes},
        styles=model_styles([(m.name, m.kind, m.label) for m in config.models]),
    )
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_fingerprint": config.fingerprint(),
        "provenance": provenance(paths),
        "settings": _settings(config),
        "tables": {},
    }
    results["data"], results["tables"]["coverage"] = _data_section(ctx, monthly)
    extended = store.load_factor_set(paths.processed, config.factors.extended)
    results["data"]["extended_factor_start"] = pd.Timestamp(extended.index[0]).date().isoformat()
    results["performance"], results["excluded"] = _performance_section(ctx, monthly, daily)
    perf = pd.DataFrame(results["performance"])
    head, perf_tables = _performance_tables(ctx, perf)
    results["tables"].update(perf_tables)
    results["mcs"], results["mcs_full"] = _mcs_section(ctx, monthly)
    results["figures"] = _figures(ctx, perf, head)

    mcs_figure = _mcs_figure(ctx, results["mcs"])
    if mcs_figure is not None:
        results["figures"]["mcs_heatmap"], results["tables"]["mcs_heatmap"] = mcs_figure
    spectrum = _spectrum_section(ctx)
    if spectrum is not None:
        results["figures"]["eigen_spectrum"], results["spectrum"] = spectrum
    if not simulation.empty:
        results["simulation"] = simulation.to_dict(orient="records")
        sim_fig, sim_table = _simulation_section(ctx, simulation)
        results["figures"]["simulation"], results["tables"]["simulation"] = sim_fig, sim_table
    results["headline"] = _headline(ctx, results, head)
    return results


def _headline(ctx: _Context, results: dict[str, Any], head: pd.DataFrame) -> dict[str, Any]:
    if head.empty:
        return {}
    config = ctx.config
    best = head.iloc[0]
    worst = head.iloc[-1]
    universe = config.report.headline_universe
    out: dict[str, Any] = {
        "universe": universe,
        "universe_label": ctx.universe_labels[universe],
        "window": config.backtest.primary_window,
        "start": best["start"],
        "end": best["end"],
        "start_year": str(best["start"])[:4],
        "end_year": str(best["end"])[:4],
        "months": int(best["months"]),
        "best_model": best["model"],
        "best_label": best["label"],
        "best_vol_pct": float(best["vol_gmv"]) * 100,
        "worst_label": worst["label"],
        "worst_vol_pct": float(worst["vol_gmv"]) * 100,
        "n_models": len(head),
        "mcs_alpha": config.evaluation.mcs.alpha,
        "mcs_confidence_pct": round(100 * (1 - config.evaluation.mcs.alpha)),
    }
    perf = pd.DataFrame(results["performance"])
    widest = perf.iloc[int(perf["assets_per_observation"].to_numpy().argmax())]
    out["max_assets_per_observation"] = float(widest["assets_per_observation"])
    out["max_ratio_universe_label"] = ctx.universe_labels[str(widest["universe"])]
    out["max_ratio_window"] = int(widest["window"])
    sample_row = head[head["model"] == "sample"]
    if not sample_row.empty:
        sample_vol = float(sample_row["vol_gmv"].iloc[0])
        out["sample_vol_pct"] = sample_vol * 100
        out["reduction_vs_sample_pct"] = 100 * (1 - float(best["vol_gmv"]) / sample_vol)
    full = next((r for r in results["mcs_full"] if r["universe"] == universe), None)
    if full is not None:
        out["mcs_full_included"] = full["included_labels"]
        out["mcs_full_size"] = len(full["included"])
        out["mcs_full_models"] = len(full["models"])
    return out


def write_results(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")


def read_results(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded

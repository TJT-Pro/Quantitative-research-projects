"""Estimate stage: every analysis, from processed tables to estimates.json and tables/*.csv."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_zoo_trial.config import Config
from factor_zoo_trial.data.build import Tables
from factor_zoo_trial.evaluation import decay as decay_mod
from factor_zoo_trial.evaluation import osap_panel, replication, simulation, space
from factor_zoo_trial.evaluation.discoveries import Discoveries, run_procedures

SUBSETS = ("all", "clear")


class Workspace:
    """Caches period-labelled panels so each (variant, basis, shift) is built once."""

    def __init__(self, config: Config, tables: Tables) -> None:
        self.config = config
        self.tables = tables
        self._adjusted: dict[tuple[str, str], pd.DataFrame] = {}
        self._panels: dict[tuple[str, str, int], pd.DataFrame] = {}

    def signals(self, subset: str) -> pd.DataFrame:
        s = self.tables.signals
        return s.loc[s["clear"]] if subset == "clear" else s

    def adjusted(self, variant: str, basis: str) -> pd.DataFrame:
        key = (variant, basis)
        if key not in self._adjusted:
            raw = self.tables.osap.loc[self.tables.osap["variant"] == variant]
            self._adjusted[key] = osap_panel.adjust_returns(
                raw[["signal", "month", "ret"]], self.tables.french, basis=basis
            )
        return self._adjusted[key]

    def panel(self, *, variant: str, basis: str, shift: int, subset: str = "all") -> pd.DataFrame:
        key = (variant, basis, shift)
        if key not in self._panels:
            self._panels[key] = osap_panel.label_periods(
                self.adjusted(variant, basis),
                self.tables.signals,
                publication_month=self.config.periods.publication_month,
                shift=shift,
            )
        frame = self._panels[key]
        if subset == "clear":
            frame = frame.loc[frame["signal"].isin(self.signals("clear")["signal"])]
        return self._eligible(frame)

    def _eligible(self, frame: pd.DataFrame) -> pd.DataFrame:
        n_in = frame.loc[frame["period"] == "in_sample"].groupby("signal").size()
        ok = n_in.index[n_in >= self.config.decay.min_months_in_sample]
        return frame.loc[frame["signal"].isin(ok)].reset_index(drop=True)


def _decay_grid(ws: Workspace) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = ws.config
    rows: list[dict[str, Any]] = []
    stored: dict[str, Any] = {}
    for subset in SUBSETS:
        for variant in cfg.osap.variants:
            for basis in cfg.decay.return_bases:
                for shift in cfg.periods.publication_shifts:
                    est = decay_mod.estimate_decay(
                        ws.panel(variant=variant, basis=basis, shift=shift, subset=subset)
                    )
                    d = est.as_dict()
                    stored[f"{subset}|{variant}|{basis}|{shift}"] = d
                    rows.append(
                        {
                            "subset": subset,
                            "variant": variant,
                            "basis": basis,
                            "shift": shift,
                            "n_signals": est.n_signals,
                            "n_obs": est.n_obs,
                            "in_sample": est.in_sample.value,
                            "post_sample": est.post_sample.value,
                            "post_publication": est.post_publication.value,
                            "beta_post_sample": est.beta_post_sample.value,
                            "beta_post_sample_se": est.beta_post_sample.se,
                            "beta_post_publication": est.beta_post_publication.value,
                            "beta_post_publication_se": est.beta_post_publication.se,
                            "share_post_sample": est.share_post_sample.value,
                            "share_post_publication": est.share_post_publication.value,
                        }
                    )
    return pd.DataFrame(rows), stored


def _tests(ws: Workspace, panel: pd.DataFrame) -> pd.DataFrame:
    return osap_panel.in_sample_tests(
        panel,
        coefficient=ws.config.multiple_testing.ewc_coefficient,
        min_months=ws.config.decay.min_months_in_sample,
    )


def _discover(ws: Workspace, panel: pd.DataFrame) -> Discoveries:
    tests = _tests(ws, panel)
    matrix = osap_panel.in_sample_matrix(panel, list(tests["signal"]))
    return run_procedures(tests, matrix=matrix, settings=ws.config.multiple_testing)


def _interactions(ws: Workspace) -> dict[str, Any]:
    cfg = ws.config
    shift = cfg.periods.headline_shift
    panel = ws.panel(variant=cfg.osap.headline_variant, basis=cfg.decay.headline_basis, shift=shift)
    screened = ws.panel(
        variant=cfg.osap.liquidity_variant, basis=cfg.decay.headline_basis, shift=shift
    )
    tests = _tests(ws, panel).set_index("signal")
    liquidity = osap_panel.liquidity_share(_tests(ws, panel), _tests(ws, screened))
    covariates = pd.DataFrame({"t_in_sample": tests["t"]}).join(liquidity, how="inner").dropna()
    covariates["t_in_sample"] = covariates["t_in_sample"] / covariates["t_in_sample"].std()
    return decay_mod.estimate_interactions(panel, covariates)


def _mclean_pontiff(ws: Workspace) -> dict[str, Any]:
    cfg = ws.config
    bench = cfg.benchmarks.mclean_pontiff
    panel = ws.panel(
        variant=cfg.osap.headline_variant, basis="raw", shift=cfg.periods.headline_shift
    )
    last = pd.Timestamp(bench.last_month)
    early = ws.tables.signals.loc[ws.tables.signals["year"] <= bench.max_publication_year, "signal"]
    window = panel.loc[(panel["month"] <= last) & panel["signal"].isin(early)]
    est = decay_mod.estimate_decay(window.reset_index(drop=True))
    published_ps = bench.post_sample / bench.in_sample_mean
    published_pp = bench.post_publication / bench.in_sample_mean
    return {
        "estimate": est.as_dict(),
        "published": {
            "in_sample_mean": bench.in_sample_mean,
            "post_sample": bench.post_sample,
            "post_publication": bench.post_publication,
            "share_post_sample": published_ps,
            "share_post_publication": published_pp,
            "citation": bench.citation,
        },
        "difference": {
            "share_post_sample": est.share_post_sample.value - published_ps,
            "share_post_publication": est.share_post_publication.value - published_pp,
            "in_sample_mean": est.in_sample.value - bench.in_sample_mean,
        },
        "last_month": bench.last_month,
    }


def _jkp(ws: Workspace, tables_dir: Path) -> dict[str, Any]:
    cfg = ws.config
    t = ws.tables
    eb = cfg.empirical_bayes
    out: dict[str, Any] = {"regions": {}, "skipped": {}}
    unused = t.locations.loc[t.locations["note"].notna()]
    out["skipped"].update(dict(zip(unused["location"], unused["note"], strict=True)))
    usable = t.locations.loc[t.locations["note"].isna()]
    fits: dict[str, replication.RegionFit] = {}
    for location in usable.loc[usable["kind"] == "region", "location"]:
        alphas = replication.location_alphas(
            location=location,
            jkp=t.jkp,
            market=t.market,
            reference=t.jkp_reference,
            settings=eb,
            shrinkage=eb.covariance_shrinkage,
        )
        if alphas is None:
            out["skipped"][location] = "too few factors with enough months"
            continue
        fit = replication.fit_region(alphas, settings=eb)
        fits[location] = fit
        out["regions"][location] = fit.summary()
        fit.table().to_csv(tables_dir / f"posterior_{location}.csv", index=False)

    source = cfg.space.source_region
    if source not in fits:
        raise ValueError(f"source region {source!r} has no fit; it must be a replication region")
    us = fits[source]
    clusters = us.table().groupby("cluster")
    out["clusters"] = {
        str(name): {
            "n": len(g),
            "mean_post": float(g["alpha_post"].mean()),
            "share_significant": float((g["low_post"] > 0).mean()),
        }
        for name, g in clusters
    }

    bench = cfg.benchmarks.jkp
    alphas_b = replication.location_alphas(
        location=source,
        jkp=t.jkp,
        market=t.market,
        reference=t.jkp_reference,
        settings=eb,
        shrinkage="none",
        last_month=pd.Timestamp(bench.last_month),
    )
    if alphas_b is not None:
        bfit = replication.fit_region(alphas_b, settings=eb)
        out["benchmark"] = {
            "estimate": bfit.summary(),
            "published": {
                "replication_rate": bench.replication_rate / 100.0,
                "by_rate": bench.by_rate / 100.0,
                "n_factors": bench.n_factors,
                "citation": bench.citation,
            },
            "difference": {
                "replication_rate": bfit.rate.rate - bench.replication_rate / 100.0,
                "by_rate": bfit.by_rate - bench.by_rate / 100.0,
                "n_factors": bfit.rate.n_factors - bench.n_factors,
            },
            "last_month": bench.last_month,
        }

    countries = []
    for location in usable.loc[usable["kind"] == "country", "location"]:
        if location == source:
            continue
        a = replication.location_alphas(
            location=location,
            jkp=t.jkp,
            market=t.market,
            reference=t.jkp_reference,
            settings=eb.model_copy(update={"min_months": cfg.space.min_months}),
            shrinkage=eb.covariance_shrinkage,
        )
        if a is None:
            out["skipped"][location] = "too few factors with enough months"
            continue
        countries.append(a)
    scores, skipped = space.run_space(us, countries, min_factors=cfg.space.min_factors)
    out["skipped"].update(skipped)
    frame = space.scores_frame(scores, t.locations)
    frame.to_csv(tables_dir / "space_scores.csv", index=False)
    out["space"] = space.summarize(scores, t.locations)
    out["n_locations"] = len(t.locations)
    # Countries are the classified locations, including a region (the US) that is also a country.
    out["n_countries_downloaded"] = int(usable["development"].notna().sum())
    out["n_factors"] = int(t.jkp["factor"].nunique())
    out["n_clusters"] = int(t.jkp_reference["cluster"].nunique())
    out["us_fit"] = {"tau_c": us.hyper.tau_c, "tau_w": us.hyper.tau_w}
    return out


def _simulation(ws: Workspace, us_tau: dict[str, float], tables_dir: Path) -> dict[str, Any]:
    cfg = ws.config
    panel = ws.panel(
        variant=cfg.osap.headline_variant, basis="raw", shift=cfg.periods.headline_shift
    )
    tests = _tests(ws, panel)
    matrix = osap_panel.in_sample_matrix(panel, list(tests["signal"]))
    rng = np.random.default_rng(cfg.simulation.seed)
    structure = simulation.fit_structure(
        matrix,
        k=cfg.simulation.latent_factors,
        n=cfg.simulation.n_factors,
        rng=rng,
        min_pair=cfg.empirical_bayes.min_pair_months,
    )
    positive = tests.loc[tests["mean"] > 0, "mean"]
    effect = float(positive.median()) if len(positive) else 0.5
    testing = simulation.multiple_testing_study(
        structure, effect_mean=effect, sim=cfg.simulation, testing=cfg.multiple_testing
    )
    testing.to_csv(tables_dir / "simulation_testing.csv", index=False)
    eb_structure = simulation.fit_structure(
        matrix,
        k=cfg.simulation.latent_factors,
        n=cfg.simulation.n_factors,
        rng=rng,
        min_pair=cfg.empirical_bayes.min_pair_months,
    )
    return {
        "effect_mean": effect,
        "testing": simulation.summarize_testing(testing, level=cfg.multiple_testing.level),
        "empirical_bayes": simulation.eb_study(
            eb_structure,
            tau_c=us_tau["tau_c"],
            tau_w=us_tau["tau_w"],
            sim=cfg.simulation,
            eb=cfg.empirical_bayes,
        ),
    }


def run(*, config: Config, tables: Tables, tables_dir: Path) -> dict[str, Any]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ws = Workspace(config, tables)
    grid, stored = _decay_grid(ws)
    grid.to_csv(tables_dir / "decay_grid.csv", index=False)

    testing: dict[str, Any] = {}
    shift = config.periods.headline_shift
    for subset in SUBSETS:
        for variant in config.osap.variants:
            found = _discover(
                ws, ws.panel(variant=variant, basis="raw", shift=shift, subset=subset)
            )
            testing[f"{subset}|{variant}|raw"] = found.summary()
            if subset == "all" and variant == config.osap.headline_variant:
                found.flags.to_csv(tables_dir / "discoveries.csv", index=False)
    for basis in config.decay.return_bases:
        if basis == "raw":
            continue
        variant = config.osap.headline_variant
        found = _discover(ws, ws.panel(variant=variant, basis=basis, shift=shift))
        testing[f"all|{variant}|{basis}"] = found.summary()

    signals = tables.signals
    osap = {
        "n_predictors": len(signals),
        "n_clear": int(signals["clear"].sum()),
        "n_in_panel": int(stored[f"all|{config.osap.headline_variant}|raw|{shift}"]["n_signals"]),
        "first_month": str(tables.osap["month"].min().date()),
        "last_month": str(tables.osap["month"].max().date()),
        "publication_years": [int(signals["year"].min()), int(signals["year"].max())],
        "decay": stored,
        "interactions": _interactions(ws),
        "mclean_pontiff": _mclean_pontiff(ws),
        "testing": testing,
    }
    jkp = _jkp(ws, tables_dir)
    sim = _simulation(ws, jkp["us_fit"], tables_dir)
    return {
        "osap": osap,
        "jkp": jkp,
        "simulation": sim,
        "runtime_seconds": time.perf_counter() - started,
    }


def write(estimates: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimates, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")

"""Score the US posterior as a forecast of each other country's realized alphas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_zoo_trial.evaluation.replication import LocationAlphas, RegionFit
from factor_zoo_trial.models.transfer import CountryCase, CountryScore, score_countries


def build_cases(
    source: RegionFit, countries: list[LocationAlphas], *, min_factors: int
) -> tuple[list[CountryCase], dict[str, str]]:
    index = {f: i for i, f in enumerate(source.alphas.factors)}
    cases, skipped = [], {}
    for country in countries:
        shared = [j for j, f in enumerate(country.factors) if f in index]
        if len(shared) < min_factors:
            skipped[country.location] = f"{len(shared)} factors shared with the source region"
            continue
        us = np.array([index[country.factors[j]] for j in shared])
        cases.append(
            CountryCase(
                name=country.location,
                alpha_hat=country.alpha_hat[shared],
                sigma=country.sigma[np.ix_(shared, shared)],
                us_mean=source.post.mean[us],
                us_cov=source.post.covariance[np.ix_(us, us)],
                us_prior=source.omega[np.ix_(us, us)],
            )
        )
    return cases, skipped


def summarize(scores: list[CountryScore], locations: pd.DataFrame) -> dict[str, Any]:
    if not scores:
        return {"n_countries": 0}
    frame = scores_frame(scores, locations)
    total = {
        model: float(frame[model].sum() - frame["no_skill"].sum())
        for model in ("transfer", "strict", "dispersion")
    }
    return {
        "n_countries": len(scores),
        "median_factors": float(frame["n_factors"].median()),
        "eta_median": float(frame["eta"].median()),
        "share_transfer_beats_no_skill": float((frame["gain_vs_no_skill"] > 0).mean()),
        "share_transfer_beats_dispersion": float((frame["gain_vs_dispersion"] > 0).mean()),
        "share_strict_beats_no_skill": float((frame["strict"] > frame["no_skill"]).mean()),
        "median_gain_vs_no_skill": float(frame["gain_vs_no_skill"].median()),
        "median_gain_vs_dispersion": float(frame["gain_vs_dispersion"].median()),
        "total_log_score_vs_no_skill": total,
        "by_development": {
            str(k): {
                "n": len(g),
                "share_beats_no_skill": float((g["gain_vs_no_skill"] > 0).mean()),
                "median_gain_vs_no_skill": float(g["gain_vs_no_skill"].median()),
            }
            for k, g in frame.groupby("development")
        },
    }


def scores_frame(scores: list[CountryScore], locations: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "location": [s.name for s in scores],
            "n_factors": [s.n_factors for s in scores],
            "eta": [s.eta for s in scores],
            "transfer": [s.transfer for s in scores],
            "strict": [s.strict for s in scores],
            "dispersion": [s.dispersion for s in scores],
            "no_skill": [s.no_skill for s in scores],
            "gain_vs_no_skill": [s.per_factor("transfer", "no_skill") for s in scores],
            "gain_vs_dispersion": [s.per_factor("transfer", "dispersion") for s in scores],
        }
    )
    meta = locations[["location", "development", "geo_region"]]
    return frame.merge(meta, on="location", how="left").fillna({"development": "unclassified"})


def run_space(
    source: RegionFit, countries: list[LocationAlphas], *, min_factors: int
) -> tuple[list[CountryScore], dict[str, str]]:
    cases, skipped = build_cases(source, countries, min_factors=min_factors)
    return score_countries(cases), skipped

"""McLean-Pontiff decay regression and the quantities the README reports from it."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from factor_zoo_trial.models.panel import FixedEffectsFit, fit_fixed_effects

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Estimate:
    value: float
    se: float

    @property
    def low(self) -> float:
        return self.value - Z95 * self.se

    @property
    def high(self) -> float:
        return self.value + Z95 * self.se

    def as_dict(self) -> dict[str, float]:
        return {
            "value": self.value,
            "se": self.se,
            "low": self.low,
            "high": self.high,
            "t": self.value / self.se if self.se > 0 else float("nan"),
        }


@dataclass(frozen=True)
class Decay:
    n_signals: int
    n_obs: int
    n_months: int
    in_sample: Estimate
    post_sample: Estimate
    post_publication: Estimate
    beta_post_sample: Estimate
    beta_post_publication: Estimate
    share_post_sample: Estimate  # beta / average in-sample effect, negative = decay
    share_post_publication: Estimate
    obs_by_period: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            attr = getattr(self, key)
            out[key] = attr.as_dict() if isinstance(attr, Estimate) else value
        return out


def _design(
    panel: pd.DataFrame, extra: dict[str, np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    signals = panel["signal"].astype("category")
    groups = signals.cat.codes.to_numpy(dtype=np.int64)
    clusters = panel["month"].to_numpy().astype("datetime64[M]").astype(np.int64)
    ps = (panel["period"] == "post_sample").to_numpy(dtype=np.float64)
    pp = (panel["period"] == "post_publication").to_numpy(dtype=np.float64)
    columns = [ps, pp]
    names = ["post_sample", "post_publication"]
    for name, values in (extra or {}).items():
        columns += [ps * values, pp * values]
        names += [f"post_sample_x_{name}", f"post_publication_x_{name}"]
    return (
        panel["ret"].to_numpy(dtype=np.float64),
        groups,
        clusters,
        np.column_stack(columns),
        names,
    )


def _ratio(fit: FixedEffectsFit, slope: int) -> Estimate:
    g = fit.n_groups
    theta = np.concatenate([fit.effects, fit.slopes])
    base = float(fit.effects.mean())
    beta = float(fit.slopes[slope])
    grad = np.zeros(theta.size)
    grad[:g] = -beta / base**2 / g
    grad[g + slope] = 1.0 / base
    var = float(grad @ fit.covariance @ grad)
    return Estimate(beta / base, float(np.sqrt(max(var, 0.0))))


def estimate_decay(panel: pd.DataFrame) -> Decay:
    """Average in-sample return and its change after the sample and after publication."""
    keep = panel.groupby("signal")["period"].transform(lambda p: (p == "in_sample").any())
    panel = panel.loc[keep.astype(bool)].reset_index(drop=True)
    y, groups, clusters, x, _ = _design(panel)
    fit = fit_fixed_effects(y=y, groups=groups, clusters=clusters, regressors=x)
    g = fit.n_groups

    def combo(slope: int | None) -> Estimate:
        w = np.zeros(g + fit.slopes.size)
        w[:g] = 1.0 / g
        if slope is not None:
            w[g + slope] = 1.0
        value, se = fit.linear(w)
        return Estimate(value, se)

    def slope(i: int) -> Estimate:
        return Estimate(float(fit.slopes[i]), float(np.sqrt(fit.covariance[g + i, g + i])))

    counts = panel["period"].value_counts()
    return Decay(
        n_signals=g,
        n_obs=fit.n_obs,
        n_months=fit.n_clusters,
        in_sample=combo(None),
        post_sample=combo(0),
        post_publication=combo(1),
        beta_post_sample=slope(0),
        beta_post_publication=slope(1),
        share_post_sample=_ratio(fit, 0),
        share_post_publication=_ratio(fit, 1),
        obs_by_period={str(k): int(v) for k, v in counts.items()},
    )


def estimate_interactions(panel: pd.DataFrame, covariates: pd.DataFrame) -> dict[str, Any]:
    """Decay interacted with signal-level covariates (one column each, indexed by signal).

    Covariates are centred across signals so the post dummies keep their meaning at the
    average signal."""
    centred = covariates - covariates.mean()
    panel = panel.loc[panel["signal"].isin(centred.index)].reset_index(drop=True)
    keep = panel.groupby("signal")["period"].transform(lambda p: (p == "in_sample").any())
    panel = panel.loc[keep.astype(bool)].reset_index(drop=True)
    extra = {
        str(c): centred[c].reindex(panel["signal"]).to_numpy(dtype=np.float64)
        for c in centred.columns
    }
    y, groups, clusters, x, names = _design(panel, extra)
    fit = fit_fixed_effects(y=y, groups=groups, clusters=clusters, regressors=x)
    g = fit.n_groups
    return {
        "n_signals": g,
        "n_obs": fit.n_obs,
        "coefficients": {
            name: Estimate(
                float(fit.slopes[i]), float(np.sqrt(fit.covariance[g + i, g + i]))
            ).as_dict()
            for i, name in enumerate(names)
        },
        "covariate_sd": {c: float(covariates[c].std()) for c in covariates},
    }

"""Run every multiple-testing procedure on one set of in-sample tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factor_zoo_trial.config import MultipleTestingConfig
from factor_zoo_trial.models import multiple_testing as mt

PROCEDURES = ("unadjusted", "bonferroni", "holm", "bhy", "hlz", "romano_wolf")
LABELS = {
    "unadjusted": "Unadjusted",
    "bonferroni": "Bonferroni",
    "holm": "Holm",
    "bh": "Benjamini-Hochberg",
    "bhy": "Benjamini-Hochberg-Yekutieli",
    "hlz": "Harvey-Liu-Zhu t > 3",
    "romano_wolf": "Romano-Wolf bootstrap",
}


@dataclass(frozen=True)
class Discoveries:
    flags: pd.DataFrame  # signal, t, p, and one boolean column per procedure
    counts: dict[str, int]
    storey_pi0: float
    storey_lambda: float
    storey_pi0_bootstrap: float
    storey_lambda_bootstrap: float
    n_tests: int
    median_rho1: float  # median first-order autocorrelation of the tested returns

    def summary(self) -> dict[str, Any]:
        return {
            "n_tests": self.n_tests,
            "counts": self.counts,
            "shares": {k: v / self.n_tests for k, v in self.counts.items()},
            "storey_pi0": self.storey_pi0,
            "storey_lambda": self.storey_lambda,
            "storey_pi0_bootstrap": self.storey_pi0_bootstrap,
            "storey_lambda_bootstrap": self.storey_lambda_bootstrap,
            "storey_true_effects": (1.0 - self.storey_pi0) * self.n_tests,
            "median_rho1": self.median_rho1,
        }


def run_procedures(
    tests: pd.DataFrame, *, matrix: np.ndarray, settings: MultipleTestingConfig
) -> Discoveries:
    """`tests` holds signal, t, p (EWC long-run variance) and optionally rho1; `matrix` the
    in-sample returns, months x signals, in the same signal order, for the bootstrap."""
    rng = np.random.default_rng(settings.seed)
    p = tests["p"].to_numpy(dtype=np.float64)
    t = tests["t"].to_numpy(dtype=np.float64)
    level = settings.level
    rw = mt.romano_wolf(
        matrix,
        level=level,
        draws=settings.romano_wolf_draws,
        block=settings.romano_wolf_block_months,
        rng=rng,
    )
    flags = tests[["signal", "t", "p"]].copy()
    flags["unadjusted"] = mt.unadjusted(p, level=level)
    flags["bonferroni"] = mt.bonferroni(p, level=level)
    flags["holm"] = mt.holm(p, level=level)
    flags["bh"] = mt.benjamini_hochberg(p, level=level)
    flags["bhy"] = mt.benjamini_yekutieli(p, level=level)
    flags["hlz"] = mt.harvey_liu_zhu(t, threshold=settings.hlz_threshold)
    flags["romano_wolf"] = rw.reject
    flags["romano_wolf_t"] = rw.t
    flags["romano_wolf_p"] = rw.adjusted_p
    storey = mt.storey_pi0(
        p,
        lam=settings.storey_lambda,
        lambdas=settings.storey_lambdas,
        draws=settings.storey_bootstrap_draws,
        rng=rng,
    )
    counts = {name: int(flags[name].sum()) for name in (*PROCEDURES, "bh")}
    return Discoveries(
        flags=flags,
        counts=counts,
        storey_pi0=storey.pi0,
        storey_lambda=storey.fixed_lambda,
        storey_pi0_bootstrap=storey.pi0_bootstrap,
        storey_lambda_bootstrap=storey.chosen_lambda,
        n_tests=int(p.size),
        median_rho1=float(tests["rho1"].median()) if "rho1" in tests else float("nan"),
    )

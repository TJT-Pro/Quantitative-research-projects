"""Simulation with known truth, from a factor structure fitted to the real predictor panel.

Two studies:
- multiple testing: a share of true alphas is exactly zero, the rest positive; each
  procedure's realized false discovery rate, family-wise error rate and power;
- empirical Bayes: alphas drawn from the hierarchical prior with known tau_c and tau_w;
  coverage of the 95% posterior intervals and recovery of the hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from factor_zoo_trial.config import EmpiricalBayesConfig, MultipleTestingConfig, SimulationConfig
from factor_zoo_trial.evaluation.discoveries import run_procedures
from factor_zoo_trial.models import covariance, empirical_bayes
from factor_zoo_trial.models.inference import ewc_mean, one_sided_p

FWER_PROCEDURES = ("bonferroni", "holm", "romano_wolf")
FDR_PROCEDURES = ("bh", "bhy")


@dataclass(frozen=True)
class Structure:
    loadings: np.ndarray  # n x k
    idio_sd: np.ndarray  # n

    def draw(self, *, months: int, rng: np.random.Generator) -> np.ndarray:
        f = rng.standard_normal((months, self.loadings.shape[1]))
        e = rng.standard_normal((months, self.idio_sd.size)) * self.idio_sd
        return np.asarray(f @ self.loadings.T + e)


def fit_structure(
    returns: np.ndarray, *, k: int, n: int, rng: np.random.Generator, min_pair: int
) -> Structure:
    """Top-k principal components of the pairwise covariance of the real panel; n rows are
    drawn (without replacement when the panel is large enough) to reach the target size."""
    corr, _ = covariance.pairwise_correlation(returns, min_pair=min_pair)
    corr = covariance.nearest_correlation(corr)
    sd = np.nanstd(returns, axis=0, ddof=1)
    cov = corr * sd[:, None] * sd[None, :]
    eigval, eigvec = np.linalg.eigh(cov)
    top = np.argsort(eigval)[::-1][:k]
    loadings = eigvec[:, top] * np.sqrt(np.maximum(eigval[top], 0.0))
    idio = np.diag(cov) - np.sum(loadings**2, axis=1)
    idio = np.sqrt(np.maximum(idio, 0.05 * np.diag(cov)))
    rows = rng.choice(loadings.shape[0], size=n, replace=n > loadings.shape[0])
    return Structure(loadings=loadings[rows], idio_sd=idio[rows])


def multiple_testing_study(
    structure: Structure,
    *,
    effect_mean: float,
    sim: SimulationConfig,
    testing: MultipleTestingConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(sim.seed)
    n = structure.idio_sd.size
    rows: list[dict[str, Any]] = []
    for null_share in sim.null_shares:
        n_null = round(null_share * n)
        for rep in range(sim.replications):
            alpha = np.zeros(n)
            alpha[n_null:] = rng.exponential(effect_mean, size=n - n_null)
            returns = alpha + structure.draw(months=sim.n_months, rng=rng)
            tests = []
            for j in range(n):
                test = ewc_mean(returns[:, j], coefficient=testing.ewc_coefficient)
                tests.append({"signal": f"s{j}", "t": test.t, "df": test.df})
            frame = pd.DataFrame(tests)
            frame["p"] = one_sided_p(frame["t"].to_numpy(), frame["df"].to_numpy(dtype=np.int64))
            seeded = testing.model_copy(update={"seed": testing.seed + rep})
            found = run_procedures(frame, matrix=returns, settings=seeded)
            truth = alpha > 0
            for procedure in (
                "unadjusted",
                "bonferroni",
                "holm",
                "bh",
                "bhy",
                "hlz",
                "romano_wolf",
            ):
                reject = found.flags[procedure].to_numpy(dtype=bool)
                false = int((reject & ~truth).sum())
                total = int(reject.sum())
                rows.append(
                    {
                        "null_share": null_share,
                        "replication": rep,
                        "procedure": procedure,
                        "fdp": false / max(total, 1),
                        "any_false": false > 0,
                        "power": float((reject & truth).sum()) / max(int(truth.sum()), 1),
                    }
                )
            rows.append(
                {
                    "null_share": null_share,
                    "replication": rep,
                    "procedure": "storey_pi0",
                    "estimate": found.storey_pi0,
                    "estimate_bootstrap": found.storey_pi0_bootstrap,
                }
            )
    return pd.DataFrame(rows)


def summarize_testing(frame: pd.DataFrame, *, level: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for share_key, group in frame.groupby("null_share"):
        share = float(str(share_key))
        block: dict[str, Any] = {}
        for procedure, g in group.groupby("procedure"):
            if procedure == "storey_pi0":
                block[str(procedure)] = {
                    "mean": float(g["estimate"].mean()),
                    "mean_bootstrap": float(g["estimate_bootstrap"].mean()),
                    "truth": share,
                }
                continue
            fdr = float(g["fdp"].mean())
            fwer = float(g["any_false"].mean())
            target = (
                "fwer"
                if procedure in FWER_PROCEDURES
                else "fdr"
                if procedure in FDR_PROCEDURES
                else None
            )
            realized = fwer if target == "fwer" else fdr
            se = float(np.sqrt(max(realized * (1 - realized), 1e-12) / len(g)))
            block[str(procedure)] = {
                "fdr": fdr,
                "fwer": fwer,
                "power": float(g["power"].mean()),
                "target": target,
                "realized": realized,
                "within_target": None if target is None else bool(realized <= level + 2 * se),
            }
        out[f"{share:g}"] = block
    return out


def eb_study(
    structure: Structure,
    *,
    tau_c: float,
    tau_w: float,
    sim: SimulationConfig,
    eb: EmpiricalBayesConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(sim.seed + 1)
    n = structure.idio_sd.size
    clusters = [f"c{i % sim.eb_clusters}" for i in range(n)]
    m, _ = empirical_bayes.membership(clusters)
    target_sd = eb.annual_vol_target / np.sqrt(12.0)
    z = float(stats.norm.ppf(0.5 + eb.interval / 2.0))
    coverage, est_c, est_w, rmse_post, rmse_ols = [], [], [], [], []
    for _ in range(sim.replications):
        alpha = tau_c * (m @ rng.standard_normal(m.shape[1])) + tau_w * rng.standard_normal(n)
        noise = structure.draw(months=sim.n_months, rng=rng)
        noise = noise / noise.std(axis=0, ddof=1) * target_sd
        returns = alpha + noise
        cov = covariance.alpha_covariance(
            returns,
            min_pair=eb.min_pair_months,
            shrinkage=eb.covariance_shrinkage,
            eigenvalue_floor=eb.eigenvalue_floor,
            minimum_floor=eb.minimum_eigenvalue,
        )
        alpha_hat = returns.mean(axis=0)
        hyper = empirical_bayes.fit_hyperparameters(
            alpha_hat=alpha_hat, sigma=cov.sigma, m=m, fix_mean=True, rng=rng, starts=3
        )
        omega = empirical_bayes.prior_covariance(m, tau_c=hyper.tau_c, tau_w=hyper.tau_w)
        post = empirical_bayes.posterior(alpha_hat=alpha_hat, sigma=cov.sigma, omega=omega, mu=0.0)
        coverage.append(float(np.mean(np.abs(alpha - post.mean) <= z * post.sd)))
        est_c.append(hyper.tau_c)
        est_w.append(hyper.tau_w)
        rmse_post.append(float(np.sqrt(np.mean((post.mean - alpha) ** 2))))
        rmse_ols.append(float(np.sqrt(np.mean((alpha_hat - alpha) ** 2))))
    return {
        "replications": sim.replications,
        "n_factors": n,
        "clusters": sim.eb_clusters,
        "nominal": eb.interval,
        "coverage": float(np.mean(coverage)),
        "coverage_se": float(np.std(coverage, ddof=1) / np.sqrt(len(coverage))),
        "tau_c_true": tau_c,
        "tau_w_true": tau_w,
        "tau_c_mean": float(np.mean(est_c)),
        "tau_w_mean": float(np.mean(est_w)),
        "tau_c_se": float(np.std(est_c, ddof=1) / np.sqrt(len(est_c))),
        "tau_w_se": float(np.std(est_w, ddof=1) / np.sqrt(len(est_w))),
        "rmse_posterior": float(np.mean(rmse_post)),
        "rmse_ols": float(np.mean(rmse_ols)),
    }

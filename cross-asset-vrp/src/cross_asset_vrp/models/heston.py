"""Heston paths with known integrated variance, to check the range estimators before use.

    dS / S = sqrt(v) dW1,   dv = kappa (theta - v) dt + xi sqrt(v) dW2,   corr(dW1, dW2) = rho

simulated by full-truncation Euler on a fine grid with zero drift. Each simulated day gives
an OHLC bar sampled at every step and the true integrated variance sum(v dt). Sampling the
same path at coarser steps shows how a discretely observed range understates the true one:
by Broadie, Glasserman and Kou (1997), the discrete maximum of a Brownian path sits about
beta sigma sqrt(dt) below the continuous one, with beta = -zeta(1/2) / sqrt(2 pi) = 0.5826,
and the minimum as far above it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BGK_BETA = 0.5826


@dataclass(frozen=True)
class Simulated:
    bars: dict[int, pd.DataFrame]  # sampling step -> daily OHLC (columns open, high, low, close)
    integrated: np.ndarray  # true daily integrated variance, paths x days, flattened row-wise
    sigma_step: np.ndarray  # mean per-step volatility sqrt(v dt), same shape


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    paths: int
    days: int
    steps_per_day: int
    seed: int


def simulate(params: HestonParams, *, samplings: tuple[int, ...]) -> Simulated:
    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0
    paths, days, steps_per_day, seed = params.paths, params.days, params.steps_per_day, params.seed
    rng = np.random.default_rng(seed)
    dt = 1.0 / (365.0 * steps_per_day)
    n = days * steps_per_day
    log_s = np.zeros((paths, n + 1))
    v = np.full(paths, v0)
    var_path = np.empty((paths, n))
    for i in range(n):
        z1 = rng.standard_normal(paths)
        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal(paths)
        vp = np.maximum(v, 0.0)
        var_path[:, i] = vp
        log_s[:, i + 1] = log_s[:, i] - 0.5 * vp * dt + np.sqrt(vp * dt) * z1
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
    integrated = (var_path * dt).reshape(paths, days, steps_per_day).sum(axis=2).ravel()
    sigma_step = np.sqrt(var_path * dt).reshape(paths, days, steps_per_day).mean(axis=2).ravel()
    bars = {}
    for step in samplings:
        frames = []
        for p in range(paths):
            prices = np.exp(log_s[p])
            for d in range(days):
                start, stop = d * steps_per_day, (d + 1) * steps_per_day
                window = prices[start : stop + 1 : step]
                frames.append((window[0], window.max(), window.min(), window[-1]))
        bars[step] = pd.DataFrame(frames, columns=["open", "high", "low", "close"])
    return Simulated(bars=bars, integrated=integrated, sigma_step=sigma_step)


def range_deficit(sim: Simulated, *, fine: int, coarse: int) -> tuple[float, float]:
    """Measured mean shrinkage of log(high / low) from the fine to the coarse sampling, and
    the Broadie-Glasserman-Kou prediction 2 beta sigma (sqrt(dt_coarse) - sqrt(dt_fine))."""

    def log_range(step: int) -> np.ndarray:
        b = sim.bars[step]
        out: np.ndarray = np.log(b["high"].to_numpy() / b["low"].to_numpy())
        return out

    measured = float(np.mean(log_range(fine) - log_range(coarse)))
    predicted = float(np.mean(2.0 * BGK_BETA * sim.sigma_step * (np.sqrt(coarse) - np.sqrt(fine))))
    return measured, predicted

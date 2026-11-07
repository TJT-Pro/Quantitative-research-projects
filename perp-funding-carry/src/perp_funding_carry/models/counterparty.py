"""Venue failure cannot be backtested from prices, so it is priced as a scenario.

Each venue fails within a year with probability h_v, independently. A failure loses
LGD times the capital held there. With V venues there are 2^V failure states, so the loss
distribution is enumerated exactly rather than simulated. The collateral split w that
minimizes expected shortfall at level beta is the Rockafellar-Uryasev linear program

    min  a + 1 / (1 - beta) * sum_s p_s u_s
    s.t. u_s >= L_s(w) - a,  u_s >= 0,  sum w = 1,  0 <= w <= max_share,

with L_s(w) = LGD * sum_v w_v * 1{v fails in s}. Margin needs are met by construction:
positions are placed on each venue in proportion to the collateral held there.

When the chance that any venue fails is smaller than the tail 1 - beta, the tail also holds
no-failure states, and spreading collateral raises the chance of some loss more than it cuts
the size of the worst one; expected shortfall then favours concentration. The level is set
so that the tail lies inside the failure states.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True)
class Split:
    venues: list[str]
    weights: dict[str, float]
    expected_loss: float  # per year, as a share of capital
    expected_shortfall: float
    single_venue_shortfall: float  # everything on the lowest-hazard venue
    equal_split_shortfall: float  # the same share on every venue
    any_failure_probability: float  # chance that at least one venue fails in the year


def failure_states(hazards: list[float]) -> tuple[np.ndarray, np.ndarray]:
    states = np.array(list(itertools.product([0.0, 1.0], repeat=len(hazards))))
    h = np.asarray(hazards)
    probs = np.prod(np.where(states == 1.0, h, 1.0 - h), axis=1)
    return states, probs


def expected_shortfall(losses: np.ndarray, probs: np.ndarray, *, level: float) -> float:
    """Mean loss in the worst (1 - level) tail of a discrete distribution."""
    order = np.argsort(-losses)
    tail = 1.0 - level
    taken = 0.0
    total = 0.0
    for k in order:
        take = min(probs[k], tail - taken)
        if take <= 0:
            break
        total += take * losses[k]
        taken += take
    return total / tail


def optimal_split(
    hazards: dict[str, float], *, lgd: float, level: float, max_share: float
) -> Split:
    venues = list(hazards)
    h = [hazards[v] for v in venues]
    states, probs = failure_states(h)
    n_v, n_s = len(venues), len(probs)
    # Variables: w (n_v), a (1), u (n_s).
    c = np.concatenate([np.zeros(n_v), [1.0], probs / (1.0 - level)])
    a_ub = np.zeros((n_s, n_v + 1 + n_s))
    a_ub[:, :n_v] = lgd * states
    a_ub[:, n_v] = -1.0
    a_ub[np.arange(n_s), n_v + 1 + np.arange(n_s)] = -1.0
    a_eq = np.concatenate([np.ones(n_v), [0.0], np.zeros(n_s)])[None, :]
    bounds = [(0.0, max_share)] * n_v + [(None, None)] + [(0.0, None)] * n_s
    result = optimize.linprog(
        c, A_ub=a_ub, b_ub=np.zeros(n_s), A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs"
    )
    if not result.success:
        raise RuntimeError(f"collateral split did not solve: {result.message}")
    # The solver can return -1e-17 for a share at its bound; clip and renormalize.
    w = np.clip(np.asarray(result.x, dtype=np.float64)[:n_v], 0.0, None)
    w /= w.sum()
    losses = lgd * states @ w
    safest = int(np.argmin(h))
    single = lgd * states[:, safest]
    equal = lgd * states @ np.full(n_v, 1.0 / n_v)
    return Split(
        venues=venues,
        weights={v: float(x) for v, x in zip(venues, w, strict=True)},
        expected_loss=float(probs @ losses),
        expected_shortfall=expected_shortfall(losses, probs, level=level),
        single_venue_shortfall=expected_shortfall(single, probs, level=level),
        equal_split_shortfall=expected_shortfall(equal, probs, level=level),
        any_failure_probability=float(1.0 - probs[states.sum(axis=1) == 0].sum()),
    )

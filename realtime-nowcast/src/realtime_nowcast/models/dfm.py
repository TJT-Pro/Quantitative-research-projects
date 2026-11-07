"""Mixed-frequency dynamic factor model estimated by EM (Banbura and Modugno, 2014).

The structure and every step follow dfm.m in the New York Fed's public nowcasting code
(Bok et al., 2018), so on the same data the estimates match theirs:

- one AR(1) factor per block, every series loading on the global block and, where the
  specification assigns one, on its own block;
- monthly series = loadings x factor + AR(1) idiosyncratic component;
- quarterly series sit on the third month of the quarter and load on the factor and its
  four lags with the Mariano and Murasawa (2003) weights 1, 2, 3, 2, 1, imposed as linear
  restrictions on the loadings; their idiosyncratic part aggregates the same way;
- measurement error variance fixed at 1e-4 (Doz, Giannone and Reichlin, 2012);
- starting values from principal components on gap-filled data.

Added here, following Lenza and Primiceri (2022): the variance of every state shock is scaled by
s_t^2, with s_t free in the first pandemic months and decaying to 1 at rate rho afterwards,
so 2020 is down-weighted by the data instead of being dropped by hand. The scales are
estimated inside EM as one more conditional maximization step; with no pandemic months in
the sample the model is exactly the reference one.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import block_diag, solve_discrete_lyapunov
from scipy.optimize import minimize

from realtime_nowcast.models import fill, kalman

Array = NDArray[np.float64]

LAGS = 5  # the factor and four lags carry a quarterly aggregate
TENT = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
# Rows say: the lag-k loading is (k+1) times the lag-0 one (k = 1, 2) and back down.
TENT_CONSTRAINTS = np.array(
    [
        [2.0, -1.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, -1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, -1.0],
    ]
)
QUARTERLY_IDIO_RHO0 = 0.1  # starting AR coefficient of the quarterly idiosyncratic parts
QUARTERLY_IDIO_SHARE = 1.0 / 19.0  # their starting variance share, as in the reference


@dataclass(frozen=True)
class Structure:
    """Which series load on which block; monthly series come first."""

    n_monthly: int
    n_quarterly: int
    blocks: NDArray[np.int64]  # (N, n_blocks), first column all ones

    @property
    def n(self) -> int:
        return self.n_monthly + self.n_quarterly

    @property
    def n_blocks(self) -> int:
        return int(self.blocks.shape[1])

    @property
    def n_states(self) -> int:
        return LAGS * self.n_blocks + self.n_monthly + LAGS * self.n_quarterly

    def factor_state(self, block: int) -> int:
        return LAGS * block

    def monthly_idio_state(self, i: int) -> int:
        return LAGS * self.n_blocks + i

    def quarterly_idio_states(self, j: int) -> slice:
        start = LAGS * self.n_blocks + self.n_monthly + LAGS * j
        return slice(start, start + LAGS)


@dataclass(frozen=True)
class Pandemic:
    """Shock scales s_t: free in `free` periods, then 1 + (s_last - 1) rho^(t - last)."""

    free: tuple[pd.Period, ...]
    scales: tuple[float, ...]
    rho: float

    def scale(self, periods: pd.PeriodIndex) -> Array:
        out = np.ones(len(periods))
        if not self.free:
            return out
        for p, v in zip(self.free, self.scales, strict=True):
            out[periods == p] = v
        last = self.free[-1]
        after = np.asarray(periods > last)
        if after.any():
            steps = np.array([(p - last).n for p in periods[after]], dtype=np.float64)
            out[after] = 1.0 + (self.scales[-1] - 1.0) * self.rho**steps
        return out


@dataclass(frozen=True)
class Params:
    a: Array
    c: Array
    q: Array
    r: Array
    z0: Array
    v0: Array
    mx: Array  # means and standard deviations that standardized the estimation vintage
    wx: Array
    pandemic: Pandemic
    loglik: float
    iterations: int
    converged: bool
    loglik_path: tuple[float, ...] = field(default=())


@dataclass(frozen=True)
class Settings:
    threshold: float
    max_iter: int
    measurement_variance: float
    pandemic_free: tuple[pd.Period, ...] = ()
    rho_start: float = 0.5
    accelerate: bool = False  # SQUAREM (Varadhan and Roland 2008); False runs plain EM


# Starting values ----------------------------------------------------------------------------


def _ols(x: Array, y: Array) -> Array:
    return np.asarray(np.linalg.solve(x.T @ x, x.T @ y), dtype=np.float64)


def _restricted(coef: Array, inv_xx: Array, r: Array) -> Array:
    """Least squares coefficients moved onto r b = 0."""
    return np.asarray(
        coef - inv_xx @ r.T @ np.linalg.solve(r @ inv_xx @ r.T, r @ coef), dtype=np.float64
    )


def _edge_count(missing: NDArray[np.bool_]) -> int:
    """Length of the leading run of missing values."""
    run = np.cumsum(missing) == np.arange(1, missing.size + 1)
    return int(run.sum())


@dataclass(frozen=True)
class _BlockStart:
    c: Array  # (N, LAGS) loadings on the block's factor and its lags
    a: Array
    q: Array
    v0: Array
    residual: Array  # the panel with this block's fit taken out


def _block_start(res: Array, res_nan: Array, *, structure: Structure, block: int) -> _BlockStart:
    """One block's starting values: its first principal component on the monthly series,
    restricted least squares tent loadings for the quarterly ones, an AR(1) for the factor."""
    t_len, n = res.shape
    nm = structure.n_monthly
    loaded = np.flatnonzero(structure.blocks[:, block])
    monthly, quarterly = loaded[loaded < nm], loaded[loaded >= nm]
    c_b = np.zeros((n, LAGS))
    eigval, eigvec = np.linalg.eigh(np.atleast_2d(np.cov(res[:, monthly], rowvar=False)))
    v = eigvec[:, np.argmax(np.abs(eigval))]
    v = -v if v.sum() < 0 else v
    c_b[monthly, 0] = v
    f = res[:, monthly] @ v
    lagged = np.column_stack([f[LAGS - 1 - k : t_len - k] for k in range(LAGS)])
    for j in quarterly:
        xx = res_nan[LAGS - 1 :, j]
        if np.sum(~np.isnan(xx)) < lagged.shape[1] + 2:
            xx = res[LAGS - 1 :, j]
        keep = ~np.isnan(xx)
        inv_xx = np.linalg.inv(lagged[keep].T @ lagged[keep])
        c_b[j, :] = _restricted(inv_xx @ lagged[keep].T @ xx[keep], inv_xx, TENT_CONSTRAINTS)
    residual = res - np.vstack([np.zeros((LAGS - 1, LAGS)), lagged]) @ c_b.T
    z, zlag = lagged[:, :1], lagged[:, 1:2]
    a_coef = float(_ols(zlag, z)[0, 0])
    a_b = np.zeros((LAGS, LAGS))
    a_b[0, 0] = a_coef
    a_b[1:, :-1] = np.eye(LAGS - 1)
    q_b = np.zeros((LAGS, LAGS))
    q_b[0, 0] = float(np.var(z[:, 0] - zlag[:, 0] * a_coef, ddof=1))
    return _BlockStart(c=c_b, a=a_b, q=q_b, v0=solve_discrete_lyapunov(a_b, q_b), residual=residual)


def _monthly_idio_start(res: Array, res_nan: Array, n_monthly: int) -> tuple[Array, Array]:
    """AR(1) coefficient and shock variance of each monthly residual, over its observed span."""
    t_len = res.shape[0]
    bm, sm = np.zeros(n_monthly), np.zeros(n_monthly)
    for i in range(n_monthly):
        lead = _edge_count(np.isnan(res_nan[:, i]))
        end = _edge_count(np.isnan(res_nan[::-1, i]))
        x = res[lead : t_len - end, i]
        bm[i] = float(x[:-1] @ x[1:] / (x[:-1] @ x[:-1]))
        sm[i] = float(np.var(x[1:] - x[:-1] * bm[i], ddof=1))
    return bm, sm


def _quarterly_idio_start(sig_e: Array) -> tuple[Array, Array]:
    one = np.zeros((LAGS, LAGS))
    one[0, 0] = QUARTERLY_IDIO_RHO0
    one[1:, :-1] = np.eye(LAGS - 1)
    bq = np.kron(np.eye(sig_e.size), one)
    sq = np.kron(np.diag((1 - QUARTERLY_IDIO_RHO0**2) * sig_e), np.diag([1.0, 0, 0, 0, 0]))
    return np.asarray(bq, dtype=np.float64), np.asarray(sq, dtype=np.float64)


def initial(xs: Array, structure: Structure, *, measurement_variance: float) -> Params:
    """Starting values as in InitCond: principal components block by block on the filled
    panel, the quarterly tent loadings by restricted least squares, AR(1) fits for the
    factors and the monthly idiosyncratic parts."""
    filled = fill.fill_for_start(xs)
    res = filled.filled.copy()
    missing = filled.missing.copy()
    res_nan = np.where(missing, np.nan, res)
    n, nm, nq = res.shape[1], structure.n_monthly, structure.n_quarterly
    missing[: LAGS - 1, :] = True
    starts = []
    for b in range(structure.n_blocks):
        start = _block_start(res, res_nan, structure=structure, block=b)
        res = start.residual
        res_nan = np.where(missing, np.nan, res)
        starts.append(start)
    tent = np.vstack([np.zeros((nm, LAGS * nq)), np.kron(np.eye(nq), TENT)])
    c = np.hstack([*(s.c for s in starts), np.eye(n)[:, :nm], tent])
    bm, sm = _monthly_idio_start(res, res_nan, nm)
    bq, sq = _quarterly_idio_start(np.nanvar(res_nan, axis=0, ddof=1)[nm:] * QUARTERLY_IDIO_SHARE)
    a = block_diag(*(s.a for s in starts), np.diag(bm), bq)
    return Params(
        a=a,
        c=c,
        q=block_diag(*(s.q for s in starts), np.diag(sm), sq),
        r=np.eye(n) * measurement_variance,
        z0=np.zeros(a.shape[0]),
        v0=block_diag(
            *(s.v0 for s in starts), np.diag(sm / (1 - bm**2)), solve_discrete_lyapunov(bq, sq)
        ),
        mx=np.zeros(n),
        wx=np.ones(n),
        pandemic=Pandemic(free=(), scales=(), rho=0.0),
        loglik=-np.inf,
        iterations=0,
        converged=False,
    )


# EM ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Moments:
    """Smoothed first and second moments, data periods 1..T."""

    z: Array  # (m, T + 1)
    v: Array  # (T + 1, m, m)
    vv: Array  # (T, m, m)

    def ezz(self, idx: Array | slice, w: Array) -> Array:
        z = self.z[idx, 1:]
        return np.asarray((z * w) @ z.T + np.einsum("t,tij->ij", w, self.v[1:][:, idx][:, :, idx]))

    def ezz_lag(self, idx: Array | slice, w: Array) -> Array:
        z = self.z[idx, :-1]
        return np.asarray((z * w) @ z.T + np.einsum("t,tij->ij", w, self.v[:-1][:, idx][:, :, idx]))

    def ezz_cross(self, idx: Array | slice, w: Array) -> Array:
        z1, z0 = self.z[idx, 1:], self.z[idx, :-1]
        return np.asarray((z1 * w) @ z0.T + np.einsum("t,tij->ij", w, self.vv[:, idx][:, :, idx]))


def _transition(
    mom: _Moments, *, structure: Structure, params: Params, w: Array
) -> tuple[Array, Array]:
    """Factor AR coefficients and shock variances, then the idiosyncratic ones (eq. 6, 8)."""
    t_len = mom.z.shape[1] - 1
    a, q = params.a.copy(), params.q.copy()
    for b in range(structure.n_blocks):
        k = np.array([structure.factor_state(b)])
        fb = mom.ezz_cross(k, w)[0, 0]
        bb = mom.ezz_lag(k, w)[0, 0]
        ff = mom.ezz(k, w)[0, 0]
        a[k[0], k[0]] = fb / bb
        q[k[0], k[0]] = (ff - fb * fb / bb) / t_len
    idio = [structure.monthly_idio_state(i) for i in range(structure.n_monthly)] + [
        structure.quarterly_idio_states(j).start for j in range(structure.n_quarterly)
    ]
    for s in idio:
        k = np.array([s])
        fb = mom.ezz_cross(k, w)[0, 0]
        bb = mom.ezz_lag(k, w)[0, 0]
        ff = mom.ezz(k, w)[0, 0]
        a[s, s] = fb / bb
        q[s, s] = (ff - (fb / bb) * fb) / t_len
    return a, q


def _loadings(y: Array, mom: _Moments, *, structure: Structure, params: Params) -> Array:
    """Monthly loadings by block pattern (BGR eq. 13) and restricted quarterly loadings."""
    seen = ~np.isnan(y)
    y0 = np.where(seen, y, 0.0)
    w = seen.astype(np.float64)
    c = params.c.copy()
    nm, nb = structure.n_monthly, structure.n_blocks
    patterns = np.unique(structure.blocks, axis=0)
    zs, vs = mom.z[:, 1:], mom.v[1:]
    constraints = block_diag(*([TENT_CONSTRAINTS] * nb))
    for pattern in patterns:
        loaded = [int(b) for b in np.flatnonzero(pattern)]
        fac = np.array([structure.factor_state(b) for b in loaded])
        rows = np.flatnonzero((structure.blocks == pattern).all(axis=1))
        monthly = rows[rows < nm]
        if monthly.size:
            idio = np.array([structure.monthly_idio_state(i) for i in monthly])
            f = zs[fac]  # (rs, T)
            eff = np.einsum("at,bt->tab", f, f) + vs[:, fac][:, :, fac]  # (T, rs, rs)
            rs, ni = fac.size, monthly.size
            d = np.einsum("tab,it->abi", eff, w[monthly])
            denom = np.zeros((rs * ni, rs * ni))
            for ia in range(rs):
                for ib in range(rs):
                    denom[ia * ni : (ia + 1) * ni, ib * ni : (ib + 1) * ni] = np.diag(d[ia, ib])
            e_idio_f = np.einsum("it,at->tia", zs[idio], f) + vs[:, idio][:, :, fac]
            cross = np.einsum("it,tia->ia", w[monthly], e_idio_f)
            nom = y0[monthly] @ f.T - cross
            vec = np.linalg.solve(denom, nom.reshape(-1, order="F"))
            c[np.ix_(monthly, fac)] = vec.reshape(ni, rs, order="F")
        quarterly = rows[rows >= nm]
        if quarterly.size:
            fq = np.concatenate(
                [
                    np.arange(structure.factor_state(b), structure.factor_state(b) + LAGS)
                    for b in loaded
                ]
            )
            cols = np.concatenate([np.arange(LAGS * b, LAGS * b + LAGS) for b in loaded])
            r_con = constraints[:, cols]
            r_con = r_con[np.any(r_con != 0, axis=1)]
            fz = zs[fq]
            eff = np.einsum("at,bt->tab", fz, fz) + vs[:, fq][:, :, fq]
            for j in quarterly:
                q_idio = structure.quarterly_idio_states(j - nm)
                wj = w[j]
                denom = np.einsum("t,tab->ab", wj, eff)
                e_idio_f = np.einsum("it,at->tia", zs[q_idio], fz) + vs[:, q_idio][:, :, fq]
                nom = y0[j] @ fz.T - np.einsum("t,i,tia->a", wj, TENT, e_idio_f)
                inv_d = np.linalg.inv(denom)
                coef = inv_d @ nom
                c[j, fq] = _restricted(coef, inv_d, r_con)
    return c


def _shocked(q: Array) -> NDArray[np.int64]:
    return np.flatnonzero(np.diag(q) > 0).astype(np.int64)


def _shock_terms(mom: _Moments, a: Array, q: Array) -> Array:
    """d_t = sum over shocked states k of E[(z_t - A z_{t-1})_k^2] / Q_kk, data periods 1..T.

    E[(z1 - A z0)_k^2] = (E z1 - A E z0)_k^2 + V1_kk - 2 (A Cov(z0, z1))_kk + (A V0 A')_kk,
    with Cov(z0, z1)_ik = vv[t, k, i] since vv[t] = Cov(z_{t+1}, z_t)."""
    k = _shocked(q)
    rows = a[k]  # (|k|, m)
    z1, z0 = mom.z[:, 1:], mom.z[:, :-1]
    e = z1[k] - rows @ z0  # (|k|, T)
    v1 = mom.v[1:][:, k, k]  # (T, |k|)
    a_v0_a = np.sum(np.matmul(rows, mom.v[:-1]) * rows, axis=2)  # (T, |k|)
    a_cross = np.sum(mom.vv[:, k, :] * rows, axis=2)  # (T, |k|)
    second = e.T**2 + v1 - 2.0 * a_cross + a_v0_a
    return np.asarray(second @ (1.0 / np.diag(q)[k]), dtype=np.float64)


def _fit_scales(
    d: Array, *, n_shocks: int, positions: tuple[int, ...], start: Pandemic
) -> Pandemic:
    """Maximize sum_t [-k log s_t - d_t / (2 s_t^2)] over the free scales and rho, with every
    scale at least 1, the support of Lenza and Primiceri's Pareto prior. Without the bound the
    likelihood has no maximum once a free month is mostly unpublished: its expected squared
    shocks shrink with its own scale, so EM drives that scale toward zero without end (on
    the vintage of 2020-06-01, when little of May 2020 was out)."""
    t_len = d.size
    last = positions[-1]
    after = np.arange(last + 1, t_len)

    def scales(theta: Array) -> Array:
        free = np.exp(theta[:-1])
        rho = 1.0 / (1.0 + np.exp(-theta[-1]))
        s = np.ones(t_len)
        s[list(positions)] = free
        s[after] = 1.0 + (free[-1] - 1.0) * rho ** (after - last)
        return s

    def objective(theta: Array) -> float:
        s = scales(theta)
        return float(np.sum(n_shocks * np.log(s) + d / (2.0 * s**2)))

    values = start.scales if len(start.scales) == len(positions) else (1.0,) * len(positions)
    rho0 = min(max(start.rho, 1e-3), 1 - 1e-3) if start.free else 0.5
    theta0 = np.array([*np.log(np.maximum(values, 1.0)), np.log(rho0 / (1 - rho0))])
    bounds = [(0.0, None)] * len(positions) + [(None, None)]
    best = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)
    theta = best.x
    return Pandemic(
        free=start.free,
        scales=tuple(float(v) for v in np.exp(theta[:-1])),
        rho=float(1.0 / (1.0 + np.exp(-theta[-1]))),
    )


def converged(loglik: float, previous: float, threshold: float) -> bool:
    """The reference rule: relative change in the log likelihood below `threshold`."""
    # A log likelihood near zero would make the relative change meaningless, so the average
    # is kept at one or more; the reference values are in the thousands, where this is idle.
    average = max((abs(loglik) + abs(previous)) / 2, 1.0)
    return bool(abs(loglik - previous) / average < threshold)


Progress = Callable[[int, float], object]


@dataclass(frozen=True)
class _State:
    """What one EM step changes."""

    a: Array
    c: Array
    q: Array
    z0: Array
    pandemic: Pandemic


@dataclass(frozen=True)
class _Problem:
    y: Array  # (N, T) standardized
    structure: Structure
    periods: pd.PeriodIndex
    positions: tuple[int, ...]
    r: Array
    v0: Array


def _em_step(problem: _Problem, state: _State) -> tuple[float, _State]:
    """The log likelihood at `state` and the state one EM step later."""
    scale = state.pandemic.scale(problem.periods)
    sm = kalman.smooth(
        problem.y,
        a=state.a,
        c=state.c,
        q=state.q,
        r=problem.r,
        z0=state.z0,
        v0=problem.v0,
        scale=scale,
    )
    mom = _Moments(z=sm.z, v=sm.v, vv=sm.vv)
    w = 1.0 / scale**2
    current = Params(
        a=state.a,
        c=state.c,
        q=state.q,
        r=problem.r,
        z0=state.z0,
        v0=problem.v0,
        mx=np.zeros(0),
        wx=np.zeros(0),
        pandemic=state.pandemic,
        loglik=sm.loglik,
        iterations=0,
        converged=False,
    )
    a_new, q_new = _transition(mom, structure=problem.structure, params=current, w=w)
    c_new = _loadings(problem.y, mom, structure=problem.structure, params=current)
    pandemic = state.pandemic
    if problem.positions:
        d = _shock_terms(mom, a_new, q_new)
        pandemic = _fit_scales(
            d, n_shocks=_shocked(q_new).size, positions=problem.positions, start=pandemic
        )
    return float(sm.loglik), _State(a=a_new, c=c_new, q=q_new, z0=sm.z[:, 0], pandemic=pandemic)


def _pack(state: _State) -> Array:
    """The state as one vector; the scales in logs and rho in logits, where a straight line
    between two states stays inside their ranges."""
    p = state.pandemic
    theta = np.array([*np.log(p.scales), np.log(p.rho / (1.0 - p.rho))]) if p.free else np.zeros(0)
    return np.concatenate([state.a.ravel(), state.c.ravel(), state.q.ravel(), state.z0, theta])


def _unpack(vector: Array, like: _State) -> _State:
    sizes = [like.a.size, like.c.size, like.q.size, like.z0.size]
    a, c, q, z0, theta = np.split(vector, np.cumsum(sizes))
    p = like.pandemic
    pandemic = (
        Pandemic(
            free=p.free,
            scales=tuple(float(v) for v in np.exp(theta[:-1])),
            rho=float(1.0 / (1.0 + np.exp(-theta[-1]))),
        )
        if p.free
        else p
    )
    return _State(
        a=a.reshape(like.a.shape),
        c=c.reshape(like.c.shape),
        q=q.reshape(like.q.shape),
        z0=z0,
        pandemic=pandemic,
    )


def _admissible(state: _State, shocked: NDArray[np.int64]) -> bool:
    """Shock variances positive where the model has shocks, scales at least 1, everything
    finite."""
    finite = all(np.isfinite(m).all() for m in (state.a, state.c, state.q, state.z0))
    scales = all(v >= 1.0 for v in state.pandemic.scales)
    return finite and scales and bool(np.all(np.diag(state.q)[shocked] > 0))


def _squarem(
    problem: _Problem,
    start: _State,
    *,
    settings: Settings,
    progress: Progress | None,
) -> tuple[_State, list[float], int, bool]:
    """EM accelerated by SQUAREM (Varadhan and Roland 2008, scheme S3).

    From s0, two EM steps give s1 and s2; with r = s1 - s0 and v = s2 - 2 s1 + s0 the
    extrapolated point is s0 - 2 alpha r + alpha^2 v with alpha = -|r| / |v| (at most -1;
    alpha = -1 is s2 itself). One more EM step from there is taken, and the point is kept
    only if its log likelihood is no lower than at s0 and its shock variances are positive;
    otherwise alpha is moved halfway to -1 until it is, which at worst returns plain EM's
    two steps. Every likelihood kept is therefore no lower than the one before, as in EM.
    Convergence uses the reference rule on the log likelihood from one cycle to the next."""
    shocked = _shocked(start.q)
    state, path, evaluations, done = start, [], 0, False
    previous = -np.inf
    while evaluations < settings.max_iter and not done:
        ll0, s1 = _em_step(problem, state)
        ll1, s2 = _em_step(problem, s1)
        evaluations += 2
        x0, x1, x2 = _pack(state), _pack(s1), _pack(s2)
        r, v = x1 - x0, x2 - 2.0 * x1 + x0
        norm_v = float(np.linalg.norm(v))
        alpha = min(-float(np.linalg.norm(r)) / norm_v, -1.0) if norm_v > 0 else -1.0
        while True:
            candidate = s2 if alpha == -1.0 else _unpack(x0 - 2.0 * alpha * r + alpha**2 * v, state)
            if alpha == -1.0 or _admissible(candidate, shocked):
                ll_c, after = _em_step(problem, candidate)
                evaluations += 1
                if alpha == -1.0 or (np.isfinite(ll_c) and ll_c >= ll0):
                    break
            alpha = (alpha - 1.0) / 2.0
            if alpha > -1.0 + 1e-6:
                continue
            alpha = -1.0
        path.extend([ll0, ll1, ll_c])
        if progress is not None:
            progress(evaluations, ll_c)
        done = converged(ll_c, previous, settings.threshold) if np.isfinite(previous) else False
        previous = ll_c
        state = after
    return state, path, evaluations, done


def _standardization(
    x: Array, periods: pd.PeriodIndex, *, free: tuple[pd.Period, ...]
) -> tuple[Array, Array]:
    """Means and standard deviations that put every series in common units.

    As in the reference, over the whole vintage, unless the vintage reaches the months whose
    shocks are scaled: then over the months before the first of them. Payrolls fell by 20.5
    million in April 2020, over a hundred standard deviations of the months before; left in,
    that month sets the units, squeezes every other month toward zero, and the idiosyncratic
    shock variances of the series it hit shrink toward zero with them, a boundary EM only
    creeps toward. The scaled months are exactly the ones the model treats as unlike the rest
    (results.json reports the figures for the last vintage)."""
    rows = np.asarray(periods < free[0]) if free else np.ones(len(periods), dtype=bool)
    base = x[rows]
    return np.nanmean(base, axis=0), np.nanstd(base, axis=0, ddof=1)


def estimate(
    x: Array,
    structure: Structure,
    *,
    periods: pd.PeriodIndex,
    settings: Settings,
    start: Params | None = None,
    progress: Progress | None = None,
) -> Params:
    """`x` (T, N): the transformed panel of one vintage, monthly series first. With `start`,
    EM begins from those parameters (last month's estimates) instead of principal
    components; it still runs to the same convergence rule."""
    any_data = ~np.isnan(x).all(axis=1)
    if not any_data[0]:
        raise ValueError("the panel starts with a period that has no data")
    last = int(np.flatnonzero(any_data)[-1]) + 1
    x, periods = x[:last], periods[:last]  # closing periods with no data carry no information
    free = tuple(p for p in settings.pandemic_free if p in periods)
    mx, wx = _standardization(x, periods, free=free)
    xs = (x - mx) / wx
    p0 = (
        initial(xs, structure, measurement_variance=settings.measurement_variance)
        if start is None
        else start
    )
    positions = tuple(int(np.flatnonzero(periods == p)[0]) for p in free)
    pandemic = (
        start.pandemic
        if start is not None and start.pandemic.free == free
        else Pandemic(free=free, scales=(1.0,) * len(free), rho=settings.rho_start)
    )
    problem = _Problem(
        y=xs.T, structure=structure, periods=periods, positions=positions, r=p0.r, v0=p0.v0
    )
    state = _State(a=p0.a, c=p0.c, q=p0.q, z0=p0.z0, pandemic=pandemic)
    if settings.accelerate:
        state, path, iteration, done = _squarem(
            problem, state, settings=settings, progress=progress
        )
        previous = path[-1] if path else -np.inf
    else:
        previous, path, iteration, done = -np.inf, [], 0, False
        while iteration < settings.max_iter and not done:
            loglik, new = _em_step(problem, state)
            if iteration > 2:
                done = converged(loglik, previous, settings.threshold)
            state = new
            path.append(loglik)
            previous = loglik
            iteration += 1
            if progress is not None:
                progress(iteration, loglik)
    return Params(
        a=state.a,
        c=state.c,
        q=state.q,
        r=p0.r,
        z0=state.z0,
        v0=p0.v0,
        mx=mx,
        wx=wx,
        pandemic=state.pandemic,
        loglik=previous,
        iterations=iteration,
        converged=done,
        loglik_path=tuple(path),
    )

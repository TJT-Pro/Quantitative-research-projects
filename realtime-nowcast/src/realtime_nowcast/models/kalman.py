"""Kalman filter and fixed-interval smoother with arbitrary missing data.

State space, for data periods t = 1..T:

    y_t = C z_t + e_t,                e_t ~ N(0, R)
    z_t = A z_{t-1} + u_t,            u_t ~ N(0, s_t^2 Q)

Missing entries of y_t are dropped from that period's update (the rows of C and R go
with them). The scale s_t is 1 except where the model frees it (the pandemic months).
The recursions follow SKF and FIS in the New York Fed code, so EM matches it step by step.
Arrays are indexed from 0: z[:, 0] is the initial state and z[:, t] the state of period t.

With the parameters and the pattern of missing data fixed, every smoothed state is linear
in the data. `impulse` uses that to give the exact effect of one datum on a smoothed state
without running the filter again, which is what the news decomposition needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Filtered:
    zm: Array  # (m, T) predicted states z_{t|t-1}
    vm: Array  # (T, m, m)
    zu: Array  # (m, T + 1) updated states z_{t|t}, column 0 the initial state
    vu: Array  # (T + 1, m, m)
    gain: Array  # (T, m, n) Kalman gain, zero columns where the datum is missing
    loglik: float
    last_gain_c: Array  # K_T C_T of the last period, for the lag-one smoother


@dataclass(frozen=True)
class Smoothed:
    z: Array  # (m, T + 1) smoothed states, column 0 the initial state
    v: Array  # (T + 1, m, m) smoothed covariances
    vv: Array  # (T, m, m): vv[t] = Cov(z_{t+1}, z_t | all data)
    loglik: float


def _symmetric(x: Array) -> Array:
    return 0.5 * (x + x.T)


def run_filter(
    y: Array,
    *,
    a: Array,
    c: Array,
    q: Array,
    r: Array,
    z0: Array,
    v0: Array,
    scale: Array | None = None,
) -> Filtered:
    """`y` is (n, T) with NaN for missing; `scale` is s_t for each period (default ones)."""
    n, periods = y.shape
    m = a.shape[0]
    s2 = np.ones(periods) if scale is None else np.asarray(scale, dtype=np.float64) ** 2
    zm = np.zeros((m, periods))
    vm = np.zeros((periods, m, m))
    zu_all = np.zeros((m, periods + 1))
    vu_all = np.zeros((periods + 1, m, m))
    gain = np.zeros((periods, m, n))
    zu, vu = z0.astype(np.float64).copy(), v0.astype(np.float64).copy()
    zu_all[:, 0], vu_all[0] = zu, vu
    loglik = 0.0
    last_gain_c = np.zeros((m, m))
    for t in range(periods):
        z = a @ zu
        v = _symmetric(a @ vu @ a.T + s2[t] * q)
        seen = ~np.isnan(y[:, t])
        if seen.any():
            ct = c[seen]
            vc = v @ ct.T
            inv_f = np.linalg.inv(ct @ vc + r[np.ix_(seen, seen)])
            vcf = vc @ inv_f
            innov = y[seen, t] - ct @ z
            zu = z + vcf @ innov
            vu = _symmetric(v - vcf @ vc.T)
            sign, logdet = np.linalg.slogdet(inv_f)
            loglik += 0.5 * (logdet - float(innov @ inv_f @ innov)) if sign > 0 else -np.inf
            gain[t][:, seen] = vcf
            last_gain_c = vcf @ ct
        else:
            zu, vu = z, v
            last_gain_c = np.zeros((m, m))
        zm[:, t], vm[t] = z, v
        zu_all[:, t + 1], vu_all[t + 1] = zu, vu
    return Filtered(
        zm=zm,
        vm=vm,
        zu=zu_all,
        vu=vu_all,
        gain=gain,
        loglik=float(loglik),
        last_gain_c=last_gain_c,
    )


def smoother_gains(f: Filtered, a: Array, first: int = 0) -> Array:
    """J_t = V_{t|t} A' V_{t+1|t}^{-1} for state columns first..T-1 (J_t links column t to
    column t + 1). The reference code uses a pseudo-inverse; the prior covariance is
    positive definite whenever the filter has run, so a batched solve gives the same gains
    far faster, with the pseudo-inverse kept for a singular prior."""
    vu = f.vu[first:-1]
    vm = f.vm[first:]
    rhs = np.matmul(a, vu)  # A V_{t|t}: the transpose of V_{t|t} A'
    try:
        gains_t = np.linalg.solve(vm, rhs)
    except np.linalg.LinAlgError:
        gains_t = np.matmul(np.linalg.pinv(vm), rhs)
    return np.asarray(np.transpose(gains_t, (0, 2, 1)), dtype=np.float64)


def smooth(
    y: Array,
    *,
    a: Array,
    c: Array,
    q: Array,
    r: Array,
    z0: Array,
    v0: Array,
    scale: Array | None = None,
) -> Smoothed:
    f = run_filter(y, a=a, c=c, q=q, r=r, z0=z0, v0=v0, scale=scale)
    periods = y.shape[1]
    m = a.shape[0]
    zs = np.zeros((m, periods + 1))
    vs = np.zeros((periods + 1, m, m))
    vvs = np.zeros((periods, m, m))
    zs[:, periods], vs[periods] = f.zu[:, periods], f.vu[periods]
    vvs[periods - 1] = (np.eye(m) - f.last_gain_c) @ a @ f.vu[periods - 1]
    gains = smoother_gains(f, a)
    for t in range(periods - 1, -1, -1):
        vu_t = f.vu[t]
        j1 = gains[t]
        zs[:, t] = f.zu[:, t] + j1 @ (zs[:, t + 1] - a @ f.zu[:, t])
        vs[t] = vu_t + j1 @ (vs[t + 1] - f.vm[t]) @ j1.T
        if t > 0:
            j2 = gains[t - 1]
            vvs[t - 1] = vu_t @ j2.T + j1 @ (vvs[t] - a @ vu_t) @ j2.T
    return Smoothed(z=zs, v=vs, vv=vvs, loglik=f.loglik)


def smoothed_means(f: Filtered, a: Array, *, down_to: int) -> Array:
    """Smoothed states z_{t|T} for state columns down_to..T; earlier columns stay zero.
    Only the backward steps that are needed are taken."""
    periods = f.zm.shape[1]
    zs = np.zeros_like(f.zu)
    zs[:, periods] = f.zu[:, periods]
    if down_to >= periods:
        return zs
    gains = smoother_gains(f, a, first=down_to)
    for t in range(periods - 1, down_to - 1, -1):
        zs[:, t] = f.zu[:, t] + gains[t - down_to] @ (zs[:, t + 1] - a @ f.zu[:, t])
    return zs


def impulse(f: Filtered, *, a: Array, c: Array, series: int, column: int, target: int) -> Array:
    """d z_{target|T} / d y: the exact effect of one datum, in the standardized units of y,
    on a smoothed state. `column` and `target` are state columns (1..T); the datum is series
    `series` in data period `column`."""
    periods = f.zm.shape[1]
    dz = np.zeros((a.shape[0], periods + 1))
    step = f.gain[column - 1][:, series].copy()
    dz[:, column] = step
    for t in range(column, periods):
        # z_{t|t} = A z_{t-1|t-1} + K_t (y_t - C A z_{t-1|t-1}); K_t is zero where y_t is
        # missing, so the full C can be used.
        pred = a @ step
        step = pred - f.gain[t] @ (c @ pred)
        dz[:, t + 1] = step
    low = min(target, column)
    gains = smoother_gains(f, a, first=low)
    ds = np.zeros_like(dz)
    ds[:, periods] = dz[:, periods]
    for t in range(periods - 1, low - 1, -1):
        ds[:, t] = dz[:, t] + gains[t - low] @ (ds[:, t + 1] - a @ dz[:, t])
    return ds[:, target]

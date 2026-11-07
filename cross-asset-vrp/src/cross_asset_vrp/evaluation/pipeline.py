"""Estimate stage: every analysis in order, written to data/runs/estimates.json and
reports/tables/*.csv. Each stage prints a progress line with the elapsed time."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import skew as stats_skew

from cross_asset_vrp.config import Config
from cross_asset_vrp.data import cboe
from cross_asset_vrp.data.build import MINUTES_PER_DAY, Tables
from cross_asset_vrp.models import futures, har, heston, predict, realized, replication, stats

INTRADAY = ("parkinson", "garman_klass", "rogers_satchell")
COVERAGE = 0.99  # a bitcoin day enters the calibration with at least 99% of its minutes


def _variance_window(config: Config) -> float:
    return config.windows.variance_days / config.windows.days_per_year


# 1. Range estimators on bitcoin, against minute data ------------------------------------------


def _calibration(config: Config, tables: Tables, tables_dir: Path) -> dict[str, Any]:
    d = tables.btc_daily.sort_values("date").set_index("date")
    d = d.loc[d["minutes"] >= COVERAGE * MINUTES_PER_DAY]
    reference = d["rv5"]
    window = config.realized.yang_zhang_window
    rows = []

    def score(name: str, level: str, *, est: pd.Series, ref: pd.Series) -> None:
        ok = est.notna() & ref.notna() & (ref > 0)
        e, r = est[ok].to_numpy(), ref[ok].to_numpy()
        rows.append(
            {
                "estimator": name,
                "level": level,
                "days": int(ok.sum()),
                "bias": float(e.mean() / r.mean() - 1.0),
                "rmse_ratio": float(np.sqrt(np.mean((e - r) ** 2)) / r.mean()),
                "correlation": float(np.corrcoef(e, r)[0, 1]),
            }
        )

    # Day level for the daily estimators; a window of `window` days for all of them, the only
    # level at which Yang-Zhang exists, so efficiencies are compared like for like.
    for name in config.realized.estimators:
        est = realized.estimate(name, d, window=window)
        if name != "yang_zhang":
            score(name, "day", est=est, ref=reference)
            est = est.rolling(window).mean()
        score(name, "window", est=est, ref=reference.rolling(window).mean())
    table = pd.DataFrame(rows)
    for level in ("day", "window"):
        at = table["level"] == level
        cc = table.loc[at & (table["estimator"] == "close_to_close"), "rmse_ratio"].iloc[0]
        table.loc[at, "efficiency_vs_cc"] = (cc / table.loc[at, "rmse_ratio"]) ** 2
    table.to_csv(tables_dir / "calibration.csv", index=False)
    records: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in table.to_dict("records")
    ]
    daily = table.loc[table["estimator"].isin(INTRADAY) & (table["level"] == "day")]
    chosen = str(daily.sort_values("rmse_ratio")["estimator"].iloc[0])
    ok = d["rk"].notna() & (d["rv5"] > 0)
    return {
        "rows": records,
        "chosen": chosen,
        "days": len(d),
        "first": str(d.index.min().date()),
        "last": str(d.index.max().date()),
        "kernel_over_rv5": float(d.loc[ok, "rk"].mean() / d.loc[ok, "rv5"].mean()),
        "rv1_over_rv5": float(d.loc[ok, "rv1"].mean() / d.loc[ok, "rv5"].mean()),
    }


# 2. Heston check of the estimators --------------------------------------------------------------


def _heston(config: Config, tables_dir: Path) -> dict[str, Any]:
    h = config.heston
    samplings = (1, *h.coarse_steps)
    params = heston.HestonParams(
        kappa=h.kappa,
        theta=h.theta,
        xi=h.xi,
        rho=h.rho,
        v0=h.v0,
        paths=h.paths,
        days=h.days,
        steps_per_day=h.steps_per_day,
        seed=h.seed,
    )
    sim = heston.simulate(params, samplings=samplings)
    rows = []
    for step in samplings:
        bars = sim.bars[step].copy()
        # Each simulated day starts at the previous close, like a market that never closes.
        for name in ("parkinson", "garman_klass", "rogers_satchell"):
            est = realized.DAILY_ESTIMATORS[name](bars).to_numpy()
            rows.append(
                {
                    "estimator": name,
                    "step_minutes": step * 1440 // h.steps_per_day,
                    "bias": float(np.mean(est) / np.mean(sim.integrated) - 1.0),
                }
            )
        oc = np.log(bars["close"].to_numpy() / bars["open"].to_numpy()) ** 2
        rows.append(
            {
                "estimator": "open_to_close_squared",
                "step_minutes": step * 1440 // h.steps_per_day,
                "bias": float(np.mean(oc) / np.mean(sim.integrated) - 1.0),
            }
        )
    deficits = []
    for coarse in h.coarse_steps:
        measured, predicted = heston.range_deficit(sim, fine=1, coarse=coarse)
        deficits.append(
            {
                "step_minutes": coarse * 1440 // h.steps_per_day,
                "measured": measured,
                "predicted": predicted,
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir / "heston.csv", index=False)
    return {"bias": rows, "range_deficit": deficits, "days": h.days * h.paths}


# 3. The premium per asset ------------------------------------------------------------------------


def _asset_frame(config: Config, tables: Tables, *, key: str, proxy: str) -> pd.DataFrame:
    """date, iv (points), target (daily variance), proxy (daily variance) for one asset.
    `proxy` names the intraday range estimator the bitcoin calibration chose."""
    a = config.asset(key)
    iv = tables.implied.loc[tables.implied["symbol"] == a.implied].set_index("date")["value"]
    if a.implied == config.crypto.implied:
        d = tables.btc_daily.sort_values("date").set_index("date")
        frame = pd.DataFrame({"target": d["rv5"], "proxy": d["rv5"]})
    else:
        bars = (
            tables.etf.loc[tables.etf["ticker"] == a.instrument]
            .sort_values("date")
            .set_index("date")
        )
        frame = pd.DataFrame(
            {
                "target": realized.close_to_close(bars),
                "proxy": realized.overnight(bars) + realized.estimate(proxy, bars, window=1),
            }
        )
        frame.loc[frame.index[0], ["target", "proxy"]] = np.nan  # no previous close
    frame["iv"] = iv.reindex(frame.index)
    first = pd.Timestamp(config.windows.first_date)
    frame = frame.loc[frame.index >= first]
    if config.windows.last_date:
        frame = frame.loc[frame.index <= pd.Timestamp(config.windows.last_date)]
    return frame


def _premium(
    config: Config, tables: Tables, *, proxy: str, tables_dir: Path, runs_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    w = config.windows
    scale = w.days_per_year / w.variance_days  # 30-day variance to annual
    summaries: dict[str, Any] = {}
    daily_frames = []
    for a in config.assets:
        frame = _asset_frame(config, tables, key=a.key, proxy=proxy).dropna(subset=["target"])
        lags = config.har.btc_lags if a.implied == config.crypto.implied else config.har.lags
        rv_fwd, rv_end = har.forward_sum(frame["target"], days=w.variance_days)
        past = frame["target"].rolling(f"{w.variance_days}D").sum()  # (t - 30 days, t]
        fc = har.walk_forward(
            frame["proxy"], rv_fwd, rv_end, lags=lags, min_train_days=config.har.min_train_days
        )
        out = pd.DataFrame(index=frame.index)
        out["iv_var"] = (frame["iv"] / 100.0) ** 2 * _variance_window(config)
        out["rv_fwd"] = rv_fwd
        out["rv_past"] = past
        out["har"] = fc["forecast"].astype(float).reindex(out.index)
        out["ex_post"] = out["iv_var"] - out["rv_fwd"]
        out["ex_ante"] = out["iv_var"] - out["har"]
        out["ex_ante_points"] = frame["iv"] - 100.0 * np.sqrt(out["har"].clip(lower=0) * scale)
        out["ex_post_points"] = frame["iv"] - 100.0 * np.sqrt(out["rv_fwd"].clip(lower=0) * scale)
        out["asset"] = a.key
        daily_frames.append(out.reset_index().rename(columns={"index": "date"}))
        have = out.dropna(subset=["iv_var", "rv_fwd"])
        ante = out.dropna(subset=["ex_ante"])
        skill_rows = out.dropna(subset=["har", "rv_fwd", "rv_past"])
        month_end = have.groupby(pd.DatetimeIndex(have.index).to_period("M")).tail(1)
        summaries[a.key] = {
            "label": a.label,
            "implied": a.implied,
            "instrument": a.instrument,
            "options_on": a.options_on,
            "reason": a.reason,
            "first": str(have.index.min().date()) if len(have) else None,
            "last": str(have.index.max().date()) if len(have) else None,
            "days": len(have),
            "mean_iv_var": float(have["iv_var"].mean() * scale) if len(have) else None,
            "mean_rv": float(have["rv_fwd"].mean() * scale) if len(have) else None,
            "ex_post_mean": float(have["ex_post"].mean() * scale) if len(have) else None,
            "ex_post_median": float(have["ex_post"].median() * scale) if len(have) else None,
            "ex_post_share_positive": float((have["ex_post"] > 0).mean()) if len(have) else None,
            "ex_post_ratio": float(have["iv_var"].mean() / have["rv_fwd"].mean())
            if len(have)
            else None,
            "ex_post_t": _monthly_t(month_end["ex_post"], config) if len(month_end) > 24 else None,
            "ex_ante_mean": float(ante["ex_ante"].mean() * scale) if len(ante) else None,
            "ex_ante_median": float(ante["ex_ante"].median() * scale) if len(ante) else None,
            "ex_ante_share_positive": float((ante["ex_ante"] > 0).mean()) if len(ante) else None,
            "ex_ante_first": str(ante.index.min().date()) if len(ante) else None,
            "vol_gap_points": float(out.loc[have.index, "ex_post_points"].mean())
            if len(have)
            else None,
            "ex_ante_points_mean": float(out["ex_ante_points"].mean()) if len(ante) else None,
            "har_r2_vs_persistence": har.oos_r2(
                skill_rows["har"].to_numpy(),
                skill_rows["rv_fwd"].to_numpy(),
                skill_rows["rv_past"].to_numpy(),
            )
            if len(skill_rows)
            else None,
            "har_days": len(skill_rows),
            "lags": list(lags),
        }
    panel = pd.concat(daily_frames, ignore_index=True)
    panel.to_parquet(runs_dir / "premium_daily.parquet", index=False)
    quantiles = []
    for key, group in panel.groupby("asset", sort=False):
        for measure in ("ex_ante_points", "ex_post_points"):
            x = group[measure].dropna()
            if len(x) < 20:
                continue
            q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
            quantiles.append(
                {
                    "asset": key,
                    "measure": measure,
                    "days": len(x),
                    "mean": float(x.mean()),
                    "p5": q[0],
                    "p25": q[1],
                    "p50": q[2],
                    "p75": q[3],
                    "p95": q[4],
                }
            )
            summaries[str(key)][f"{measure}_median"] = float(q[2])
    pd.DataFrame(quantiles).to_csv(tables_dir / "premium_distribution.csv", index=False)
    summary = pd.DataFrame(
        [
            {k: v for k, v in s.items() if not isinstance(v, list)} | {"asset": key}
            for key, s in summaries.items()
        ]
    )
    summary.to_csv(tables_dir / "premium.csv", index=False)
    return summaries, panel


def _monthly_t(series: pd.Series, config: Config) -> float:
    x = series.dropna().to_numpy(dtype=np.float64)
    lrv, _ = stats.ewc_long_run_variance(x, coefficient=config.predictability.ewc_coefficient)
    return float(x.mean() / np.sqrt(lrv / x.size)) if lrv > 0 else float("nan")


# 4. Return predictability ------------------------------------------------------------------------


def _predictability(
    config: Config, tables: Tables, *, panel: pd.DataFrame, tables_dir: Path
) -> list[dict[str, Any]]:
    p = config.predictability
    rf = tables.rf.set_index("date")["rate"].sort_index()
    rows = []
    for a in config.assets:
        prem = panel.loc[panel["asset"] == a.key].set_index("date")["ex_ante"].dropna()
        if prem.empty:
            continue
        if a.implied == config.crypto.implied:
            close = tables.btc_daily.set_index("date")["close"]
        else:
            close = tables.etf.loc[tables.etf["ticker"] == a.instrument].set_index("date")["close"]
        monthly_close = close.groupby(pd.DatetimeIndex(close.index).to_period("M")).last()
        monthly = pd.DataFrame({"ret": np.log(monthly_close).diff()})
        rf_by_month = rf.groupby(pd.DatetimeIndex(rf.index).to_period("M")).last()
        rf_month = rf_by_month.reindex(monthly.index).ffill() / 12.0
        monthly["rf"] = rf_month
        signal = prem.groupby(pd.DatetimeIndex(prem.index).to_period("M")).last()
        samples: list[tuple[str, pd.Period | None]] = [("full", None)]
        if a.implied == "VIX":
            samples.append(("after_btz", pd.Period(p.btz_sample_end, freq="M") + 1))
        for sample, start in samples:
            for h in p.horizons_months:
                y = predict.forward_excess(monthly, months=h)
                both = pd.DataFrame({"x": signal, "y": y}).dropna()
                if start is not None:
                    both = both.loc[both.index >= start]
                if len(both) < 24:
                    continue
                xs = both["x"].astype(float)
                x = ((xs - xs.mean()) / xs.std(ddof=1)).to_numpy(dtype=np.float64)
                reg = predict.ols_ewc(
                    x,
                    both["y"].to_numpy(dtype=np.float64),
                    coefficient=p.ewc_coefficient,
                    index=pd.PeriodIndex(both.index).to_timestamp(),
                )
                if reg is not None:
                    rows.append(
                        {
                            "asset": a.key,
                            "label": a.label,
                            "sample": sample,
                            "horizon": h,
                            **asdict(reg),
                        }
                    )
    pd.DataFrame(rows).to_csv(tables_dir / "predictability.csv", index=False)
    return rows


# 5. Short VIX futures ----------------------------------------------------------------------------


def _performance(
    returns: pd.Series, config: Config, *, ruined_on: pd.Timestamp | None = None
) -> dict[str, Any]:
    r = returns.dropna()
    sr = stats.sharpe(
        r.to_numpy(), periods_per_year=252.0, coefficient=config.predictability.ewc_coefficient
    )
    equity = (1.0 + r).cumprod()
    dd = stats.worst_drawdown(equity)
    tail = np.sort(r.to_numpy())[: max(1, int(np.ceil(0.05 * len(r))))]
    return {
        "first": str(r.index.min().date()),
        "last": str(r.index.max().date()),
        "days": len(r),
        "annual_return": float(equity.iloc[-1] ** (252.0 / len(r)) - 1.0)
        if equity.iloc[-1] > 0
        else -1.0,
        "annual_mean": sr.mean,
        "vol": sr.vol,
        "sharpe": sr.sharpe,
        "sharpe_se": sr.se,
        "skew": sr.skew,
        "expected_shortfall_5": float(tail.mean()),
        "max_drawdown": dd.depth,
        "drawdown_peak": str(dd.peak.date()),
        "drawdown_trough": str(dd.trough.date()),
        "recovered": str(dd.recovered.date()) if dd.recovered is not None else None,
        "recovery_days": float((dd.recovered - dd.trough).days)
        if dd.recovered is not None
        else None,
        "worst_day": float(r.min()),
        "worst_day_date": str(pd.Timestamp(str(r.idxmin())).date()),
        "ruined_on": str(ruined_on.date()) if ruined_on is not None else None,
    }


def _futures(
    config: Config, tables: Tables, *, panel: pd.DataFrame, tables_dir: Path
) -> dict[str, Any]:
    f = config.futures
    cm = futures.constant_maturity(
        tables.vx, target_days=f.target_days, cost_points=f.cost_points_per_side
    )
    # Every trading day needs a front and a second contract; a long stretch without a blended
    # return means a contract file is missing or ends early, and would read as flat P&L.
    holes = cboe.gaps(cm["ret"].dropna().index.to_series(), longer_than=f.max_gap_days)
    short = -cm["ret"]
    vix_key = next(a.key for a in config.assets if a.implied == "VIX")
    premium = panel.loc[panel["asset"] == vix_key].set_index("date")["ex_ante"]
    imp = tables.implied.pivot_table(index="date", columns="symbol", values="value", aggfunc="last")
    slope = 1.0 - imp[config.term_structure.short] / imp[config.term_structure.long]
    sizes = {
        "unconditional": pd.Series(1.0, index=short.index),
        "premium": futures.expanding_scale(
            premium.reindex(short.index),
            min_history=f.min_signal_history_days,
            cap=f.sizing_cap,
            positive_only=False,
        ),
        "slope": futures.expanding_scale(
            slope.reindex(short.index),
            min_history=f.min_signal_history_days,
            cap=f.sizing_cap,
            positive_only=True,
        ),
    }
    common_start = max(
        pd.Timestamp(str(s.first_valid_index()))
        for s in sizes.values()
        if s.first_valid_index() is not None
    )
    # A hole inside the compared sample stops the run. Holes before it (early serial contracts
    # whose files end before expiry) cannot be filled from any source, so the long unconditional
    # sample starts after the last of them.
    inside = [h for h in holes if h[1] > common_start]
    if inside:
        before, after, days = inside[0]
        raise ValueError(
            f"constant-maturity VX has no return for {days} calendar days, {before.date()} to "
            f"{after.date()}: a contract file is missing or starts late"
        )
    full_start = holes[-1][1] if holes else None
    strategies: dict[str, Any] = {}
    series = {}
    for name, size in sizes.items():
        ret = futures.strategy_returns(short, size)
        common, ruin = futures.limited_liability(ret.loc[ret.index >= common_start])
        series[name] = common
        strategies[name] = {
            "common": _performance(common, config, ruined_on=ruin),
            "mean_size": float(size.loc[size.index >= common_start].mean()),
            # The day's loss before the account's limit, as a share of the equity it held.
            "ruin_day_return": float(ret.loc[ruin]) if ruin is not None else None,
        }
        if name == "unconditional":
            kept = ret.loc[ret.index >= full_start] if full_start is not None else ret
            full, full_ruin = futures.limited_liability(kept)
            strategies[name]["full"] = _performance(full, config, ruined_on=full_ruin)
    episodes = []
    for ep in config.episodes:
        start, end = pd.Timestamp(ep.start), pd.Timestamp(ep.end)
        row: dict[str, Any] = {"name": ep.name, "start": ep.start, "end": ep.end}
        for name, size in sizes.items():
            ret = futures.strategy_returns(short, size)
            window, _ = futures.limited_liability(
                ret.loc[(ret.index >= start) & (ret.index <= end)]
            )
            growth = np.prod(1.0 + window.to_numpy(dtype=np.float64))
            row[name] = float(growth - 1.0) if len(window) else None
        episodes.append(row)
    equity = pd.DataFrame({k: (1.0 + v).cumprod() for k, v in series.items()})
    equity.to_csv(tables_dir / "futures_equity.csv", index_label="date")
    pd.DataFrame(episodes).to_csv(tables_dir / "futures_episodes.csv", index=False)
    return {
        "common_start": str(common_start.date()),
        "gaps_before_full": [
            {"before": str(b.date()), "after": str(a.date()), "days": d} for b, a, d in holes
        ],
        "strategies": strategies,
        "episodes": episodes,
        "contracts": int(tables.vx["expiry"].nunique()),
        "first_trade": str(tables.vx["trade_date"].min().date()),
        "last_trade": str(tables.vx["trade_date"].max().date()),
        "roll_yield_share": float((cm["ret"] < 0).mean()),
    }


# 6. Bitcoin variance swaps from the Deribit chain -------------------------------------------------


def _chain_stats(chains: pd.DataFrame) -> dict[str, Any]:
    """How much each rebuilt chain rests on: trades in the window, options per chain, how old
    the median mark is at the snapshot, and the effective half-spread near the money."""
    per = chains.groupby("snapshot")
    near = chains.loc[np.abs(np.log(chains["strike"] / chains["forward"])) <= 0.1]
    half = (near["ask"] - near["bid"]) / (near["ask"] + near["bid"])
    return {
        "window_trades_median": float(per["window_trades"].first().median()),
        "options_median": float(per.size().median()),
        "mark_age_minutes_median": float(per["age_minutes"].median().median()),
        "atm_half_spread_median": float(half.median()),
        # Deribit inverts each mark on its own forward; the gap measures the parity forward.
        "iv_gap_median_abs": float((near["iv"] - near["deribit_iv"]).abs().median()),
    }


def _replication(config: Config, tables: Tables, tables_dir: Path) -> dict[str, Any]:
    r = config.replication
    w = config.windows
    chains = tables.chains
    if chains.empty:
        return {"months": 0}
    five = tables.btc_5m.set_index("time")["ret"].sort_index()
    dvol = tables.dvol_snapshot.set_index("snapshot")["dvol"]
    rows = []
    for key, chain in chains.groupby("snapshot"):
        snapshot = pd.Timestamp(str(key))
        now = snapshot
        rates = {
            side: replication.thirty_day_rate(
                chain,
                now=now,
                side=side,
                target_days=w.variance_days,
                min_expiry_days=r.min_expiry_days,
                zero_bid_stop=r.zero_bid_stop,
            )
            for side in ("bid", "mid", "ask")
        }
        end = now + pd.Timedelta(days=w.variance_days)
        window = five.loc[(five.index > now) & (five.index <= end)]
        complete = len(five) and five.index.max() >= end
        rv = (
            float(np.sum(window.to_numpy() ** 2)) * w.days_per_year / w.variance_days
            if complete
            else None
        )
        rows.append(
            {
                "snapshot": str(snapshot.date()),
                **{f"rate_{k}": v for k, v in rates.items()},
                "realized": rv,
                "dvol": float(dvol.get(snapshot, np.nan)),
            }
        )
    frame = pd.DataFrame(rows)
    frame["vol_mid"] = np.sqrt(frame["rate_mid"]) * 100.0
    frame["gap_points"] = frame["vol_mid"] - frame["dvol"]
    frame["spread_points"] = (np.sqrt(frame["rate_ask"]) - np.sqrt(frame["rate_bid"])) * 100.0
    # A short variance swap entered at the bid, per unit of vega notional: (K - RV) / (2 sqrt K).
    frame["short_pnl_vega"] = (
        (frame["rate_bid"] - frame["realized"]) / (2.0 * np.sqrt(frame["rate_bid"])) * 100.0
    )
    frame.to_csv(tables_dir / "replication.csv", index=False)
    settled = frame.dropna(subset=["short_pnl_vega"])
    gap = frame.dropna(subset=["gap_points"])
    out: dict[str, Any] = {
        "months": len(frame),
        "priced": int(frame["rate_mid"].notna().sum()),
        "settled": len(settled),
        "first": frame["snapshot"].iloc[0],
        "last": frame["snapshot"].iloc[-1],
        "spread_points_median": float(frame["spread_points"].median()),
        **_chain_stats(chains),
    }
    if len(settled) >= 3:
        pnl = settled["short_pnl_vega"].to_numpy()
        out |= {
            "short_mean": float(pnl.mean()),
            "short_t": _monthly_t(settled["short_pnl_vega"], config) if len(settled) > 6 else None,
            "short_hit_rate": float((pnl > 0).mean()),
            "short_worst": float(pnl.min()),
            "short_worst_month": str(
                settled["snapshot"].iloc[int(np.argmin(settled["short_pnl_vega"].to_numpy()))]
            ),
            "short_skew": float(stats_skew(pnl)),
            "mid_premium_points": float(
                np.mean(np.sqrt(settled["rate_mid"]) - np.sqrt(settled["realized"])) * 100.0
            ),
        }
    if len(gap):
        out |= {
            "gap_mean": float(gap["gap_points"].mean()),
            "gap_mean_abs": float(gap["gap_points"].abs().mean()),
            "gap_months": len(gap),
            "gap_first": gap["snapshot"].iloc[0],
        }
    return out


# Run ---------------------------------------------------------------------------------------------


def run(
    *,
    config: Config,
    tables: Tables,
    tables_dir: Path,
    runs_dir: Path,
    say: Callable[[str], None] = print,
) -> dict[str, Any]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def step(line: str) -> None:
        say(f"  [{time.perf_counter() - started:5.0f} s] {line}")

    step("range estimators on bitcoin minute data")
    calibration = _calibration(config, tables, tables_dir)
    step("Heston check of the estimators")
    heston_out = _heston(config, tables_dir)
    step("variance premium and HAR forecasts per asset")
    premium, panel = _premium(
        config, tables, proxy=calibration["chosen"], tables_dir=tables_dir, runs_dir=runs_dir
    )
    step("return predictability")
    predictability = _predictability(config, tables, panel=panel, tables_dir=tables_dir)
    step("short VIX futures")
    futures_out = _futures(config, tables, panel=panel, tables_dir=tables_dir)
    step("bitcoin variance swaps from the option chain")
    replication_out = _replication(config, tables, tables_dir)
    step("done")
    return {
        "calibration": calibration,
        "heston": heston_out,
        "premium": premium,
        "predictability": predictability,
        "futures": futures_out,
        "replication": replication_out,
        "runtime_seconds": time.perf_counter() - started,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)}")


def write(estimates: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimates, indent=2, default=_json_default) + "\n", encoding="utf-8")

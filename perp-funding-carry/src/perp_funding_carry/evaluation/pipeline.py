"""Estimate stage: every analysis, from processed tables to estimates.json and tables/*.csv."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from perp_funding_carry.config import Config
from perp_funding_carry.data.build import Tables, screen
from perp_funding_carry.evaluation.prepare import interval_hours, prepare_bars, usd_per_usdt
from perp_funding_carry.models import costs, counterparty, forecast, funding, stats
from perp_funding_carry.models.carry import CarrySettings, simulate
from perp_funding_carry.models.carry import Path as MarginPath
from perp_funding_carry.models.portfolio import (
    Market,
    PortfolioResult,
    PortfolioSettings,
    simulate_portfolio,
)

PATHS: tuple[MarginPath, ...] = ("ideal", "topup", "liquidate")


def days_per_year(config: Config) -> float:
    """Crypto trades every day, so daily returns annualize over the calendar."""
    return config.evaluation.hours_per_year / 24.0


def _carry_settings(
    config: Config, *, leverage: float, path: MarginPath, symbol: str | None
) -> CarrySettings:
    m = config.margin
    mmr = (
        m.maintenance_rate.get(symbol, m.default_maintenance_rate)
        if symbol
        else m.default_maintenance_rate
    )
    return CarrySettings(
        leverage=leverage,
        reserve_share=config.carry.reserve_share,
        topup_multiple=config.carry.topup_leverage_multiple,
        maintenance_rate=mmr,
        spot_fee=config.costs.spot_taker,
        perp_fee=config.costs.perp_taker,
        path=path,
    )


def _performance(equity: pd.Series, *, config: Config) -> dict[str, Any]:
    daily = stats.daily_returns(equity)
    sr = stats.sharpe(
        daily.to_numpy(),
        periods_per_year=days_per_year(config),
        coefficient=config.evaluation.ewc_coefficient,
    )
    dd = stats.worst_drawdown(equity.resample("1D").last().dropna())
    years = (equity.index[-1] - equity.index[0]) / pd.Timedelta(
        hours=config.evaluation.hours_per_year
    )
    total = float(equity.iloc[-1] / equity.iloc[0])
    return {
        "annual_return": total ** (1.0 / years) - 1.0 if years > 0 and total > 0 else float("nan"),
        "total_return": total - 1.0,
        "years": float(years),
        "sharpe": sr.sharpe,
        "sharpe_se": sr.se,
        "sharpe_t": sr.t,
        "sharpe_p": sr.p_value,
        "vol": sr.vol,
        "skew": sr.skew,
        "kurtosis": sr.kurtosis,
        "days": sr.periods,
        "max_drawdown": dd.depth,
        "drawdown_days": dd.duration_days,
        "drawdown_peak": str(dd.peak.date()),
        "drawdown_trough": str(dd.trough.date()),
        "recovered": None if dd.recovered is None else str(dd.recovered.date()),
    }


def _indexed(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return bars.loc[bars["symbol"] == symbol].set_index("open_time").sort_index()


# 1. Funding rebuild --------------------------------------------------------------------------


def _rebuild(config: Config, tables: Tables, tables_dir: Path) -> dict[str, Any]:
    r = config.rebuild
    info = tables.funding_info
    caps = {
        str(sym): (float(lo), float(hi))
        for sym, lo, hi in zip(info["symbol"], info["floor"], info["cap"], strict=True)
    }
    prints = tables.funding.loc[tables.funding["symbol"].isin(r.symbols)]
    out = funding.rebuild(
        prints,
        tables.premium,
        caps=caps,
        settings=funding.RebuildSettings(
            interest_8h=r.interest_8h,
            clamp=r.clamp,
            equal_weight_interval_hours=r.equal_weight_interval_hours,
            conventions=r.conventions,
        ),
    )
    if out.empty:
        return {"n_prints": 0}
    out.to_csv(tables_dir / "rebuild.csv", index=False)
    summary: dict[str, Any] = {
        "n_prints": len(out),
        "symbols": sorted(out["symbol"].unique()),
        "first": str(out["time"].min().date()),
        "last": str(out["time"].max().date()),
        "conventions": {},
    }
    small = out["reported"].abs() < r.cap_test_threshold
    for convention in r.conventions:
        err = (out[convention] - out["reported"]).abs()
        by_interval = {
            str(int(k)): {"n": len(g), "share": float((g <= r.tolerance).mean())}
            for k, g in err.groupby(out["interval_h"])
        }
        summary["conventions"][convention] = {
            "share_within": float((err <= r.tolerance).mean()),
            "share_within_small": float((err[small] <= r.tolerance).mean())
            if small.any()
            else None,
            "share_within_large": float((err[~small] <= r.tolerance).mean())
            if (~small).any()
            else None,
            "median_abs_error_bp": float(err.median() * 1e4),
            "p95_abs_error_bp": float(err.quantile(0.95) * 1e4),
            "by_interval": by_interval,
            "by_symbol": {
                str(s): float((g <= r.tolerance).mean()) for s, g in err.groupby(out["symbol"])
            },
        }
    # The conventions only disagree where the interval is not 8 hours. Prints on which they
    # give different rates are the only ones that can tell them apart.
    rebuilt = out[list(r.conventions)].to_numpy()
    decisive = (rebuilt.max(axis=1) - rebuilt.min(axis=1)) > r.tolerance
    summary["decisive_prints"] = int(decisive.sum())
    summary["intervals"] = sorted(int(k) for k in out["interval_h"].unique())
    for convention in r.conventions:
        err = (out[convention] - out["reported"]).abs()[decisive]
        summary["conventions"][convention]["share_within_decisive"] = (
            float((err <= r.tolerance).mean()) if decisive.any() else None
        )
    key = "share_within_decisive" if decisive.any() else "share_within"
    best = max(r.conventions, key=lambda c: summary["conventions"][c][key])
    summary["best"] = best
    summary["decided"] = bool(decisive.any())
    summary["n_large"] = int((~small).sum())
    return summary


# 2. Core carry ---------------------------------------------------------------------------------


def _core(
    config: Config, *, core: pd.DataFrame, stable: pd.DataFrame, tables_dir: Path
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    series = []
    rows = []
    for symbol in config.universe.core_symbols:
        bars = _indexed(core, symbol)
        if bars.empty:
            continue
        usd = usd_per_usdt(stable, pd.DatetimeIndex(bars.index))
        block: dict[str, Any] = {}
        for leverage in config.carry.leverages:
            for path in PATHS:
                res = simulate(
                    bars,
                    settings=_carry_settings(config, leverage=leverage, path=path, symbol=symbol),
                )
                perf = _performance(res.equity, config=config)
                perf_usd = _performance(res.equity * usd, config=config)
                row = {
                    "symbol": symbol,
                    "leverage": leverage,
                    "path": path,
                    "liquidations": res.liquidations,
                    "topups": res.topups,
                    "forced_closes": res.forced_closes,
                    "funding_total": float(res.funding.iloc[-1]),
                    "costs_total": float(res.costs.iloc[-1]),
                    **perf,
                    "annual_return_usd": perf_usd["annual_return"],
                }
                rows.append(row)
                block[f"{leverage:g}|{path}"] = row
                if leverage == config.carry.headline_leverage:
                    series.append(
                        pd.DataFrame(
                            {
                                "symbol": symbol,
                                "path": path,
                                "time": res.equity.index,
                                "equity": res.equity.to_numpy(),
                                "funding": res.funding.to_numpy(),
                                "costs": res.costs.to_numpy(),
                            }
                        )
                    )
                    if path == "liquidate":
                        block["liquidation_times"] = [str(t) for t in res.liquidation_times]
        out[symbol] = block
    pd.DataFrame(rows).to_csv(tables_dir / "core_carry.csv", index=False)
    if series:
        # Cumulative series, so the last value of each UTC day loses nothing at daily
        # frequency and keeps the published table small at 1-hour bars.
        daily = (
            pd.concat(series)
            .assign(day=lambda f: f["time"].dt.floor("1D"))
            .groupby(["symbol", "path", "day"], sort=True)
            .last()
            .reset_index()
            .drop(columns="day")
        )
        daily.to_csv(tables_dir / "core_series.csv", index=False, float_format="%.10g")
    return out


# 3. Forecasts ----------------------------------------------------------------------------------


def _forecasts(
    config: Config, tables: Tables, tables_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    f = config.forecast
    panel = forecast.build_panel(
        tables.funding,
        tables.bars,
        bar_hours=interval_hours(config.universe.bar_interval),
        har_lags=f.har_lags,
        momentum_hours=f.momentum_hours,
        horizons=f.horizons,
    )
    models = forecast.model_features(har_lags=f.har_lags, momentum_hours=f.momentum_hours)
    preds = forecast.walk_forward(
        panel,
        horizons=f.horizons,
        models=models,
        min_train_months=f.min_train_months,
        winsor=f.winsor,
    )
    skills = forecast.skill(preds, models=list(models), coefficient=f.ewc_coefficient)
    pd.DataFrame([asdict(s) for s in skills]).to_csv(tables_dir / "forecast_skill.csv", index=False)
    out: dict[str, Any] = {
        "n_rows": len(panel),
        "n_symbols": int(panel["symbol"].nunique()),
        "oos_first": str(preds["t"].min().date()) if not preds.empty else None,
        "skill": {f"{s.model}|{s.horizon}": asdict(s) for s in skills},
    }
    # Open-interest change, for the core symbols over the window the daily files cover.
    if not tables.oi.empty:
        oi = (
            tables.oi.assign(t=tables.oi["time"].dt.ceil("8h"))
            .groupby(["symbol", "t"])["oi_usd"]
            .last()
        )
        core = panel.loc[panel["symbol"].isin(config.universe.core_symbols)].merge(
            oi.rename("oi").reset_index(), on=["symbol", "t"], how="inner"
        )
        core = core.sort_values(["symbol", "t"])
        core["oi_change"] = np.log(core["oi"]).groupby(core["symbol"]).diff(3)
        with_oi = forecast.model_features(
            har_lags=f.har_lags, momentum_hours=f.momentum_hours, extra=("oi_change",)
        )
        months = core["t"].dt.tz_localize(None).dt.to_period("M").nunique()
        train = max(1, min(f.min_train_months, months - 2))
        core_preds = forecast.walk_forward(
            core, horizons=f.horizons, models=with_oi, min_train_months=train, winsor=f.winsor
        )
        if not core_preds.empty:
            core_skill = forecast.skill(
                core_preds, models=list(with_oi), coefficient=f.ewc_coefficient
            )
            out["core_oi"] = {
                "train_months": train,
                "first": str(core_preds["t"].min().date()),
                "skill": {f"{s.model}|{s.horizon}": asdict(s) for s in core_skill},
            }
    return out, preds


# 4. Portfolio ----------------------------------------------------------------------------------


def _scores(preds: pd.DataFrame, model: str) -> pd.DataFrame:
    one = preds.loc[preds["horizon"] == preds["horizon"].min()]
    column = "persistence" if model == "persistence" else model
    wide: pd.DataFrame = one.pivot_table(
        index="t", columns="symbol", values=column, aggfunc="first"
    )
    return wide


def _portfolio(
    config: Config,
    *,
    bars: pd.DataFrame,
    preds: pd.DataFrame,
    stable: pd.DataFrame,
    tables_dir: Path,
) -> dict[str, Any]:
    p = config.portfolio
    market = Market.from_long(bars)
    scores = {m: _scores(preds, m) for m in p.models}
    rows = []
    trial_sharpes = []
    headline_key = (
        p.headline_model,
        p.headline_top_k,
        p.headline_min_quote_volume_usd,
        config.carry.headline_leverage,
    )
    results = {}
    for model, k, volume, leverage in itertools.product(
        p.models, p.top_k, p.min_quote_volume_usd, config.carry.leverages
    ):
        res = simulate_portfolio(
            market,
            scores[model],
            settings=PortfolioSettings(
                top_k=k,
                rebalance_hours=p.rebalance_hours,
                exit_multiple=p.exit_rank_multiple,
                carry=_carry_settings(config, leverage=leverage, path="topup", symbol=None),
            ),
            min_volume=volume,
            participation=p.participation,
            maintenance=dict(config.margin.maintenance_rate),
            default_maintenance=config.margin.default_maintenance_rate,
        )
        start = res.equity.index[res.holdings.to_numpy() > 0]
        equity = res.equity.loc[start[0] :] if len(start) else res.equity
        perf = _performance(equity, config=config)
        daily = stats.daily_returns(equity)
        trial_sharpes.append(
            float(daily.mean() / daily.std(ddof=1)) if daily.std(ddof=1) > 0 else 0.0
        )
        rows.append(
            {
                "model": model,
                "top_k": k,
                "min_volume": volume,
                "leverage": leverage,
                "liquidations": res.liquidations,
                "delisted_closes": res.delisted_closes,
                "turnover": res.turnover,
                **perf,
            }
        )
        results[(model, k, volume, leverage)] = (res, equity)
    variants = pd.DataFrame(rows)
    variants.to_csv(tables_dir / "portfolio_variants.csv", index=False)

    res, equity = results[headline_key]
    daily = stats.daily_returns(equity)
    sr = stats.sharpe(
        daily.to_numpy(),
        periods_per_year=days_per_year(config),
        coefficient=config.evaluation.ewc_coefficient,
    )
    dsr, sr0 = stats.deflated_sharpe(
        sharpe_per_period=sr.sharpe / np.sqrt(days_per_year(config)),
        trial_sharpes=np.array(trial_sharpes),
        periods=sr.periods,
        skew=sr.skew,
        kurtosis=sr.kurtosis,
    )
    # The headline variant on every margin path, for the headline figure.
    paths: dict[str, Any] = {}
    series: dict[str, pd.Series] = {}
    for path in PATHS:
        r = _headline_run(config, market=market, scores=scores[p.headline_model], path=path)
        e = r.equity.loc[equity.index[0] :]
        paths[path] = {**_performance(e, config=config), "liquidations": r.liquidations}
        series[path] = e
    # Gross funding on the headline path: what equity would have compounded to if each bar's
    # funding, as a share of that bar's opening equity, were the whole return.
    start = equity.index[0]
    opening = res.equity.loc[start:].shift(1).fillna(1.0)
    received = res.funding.loc[start:].diff().fillna(0.0)
    spent = (res.fees + res.spread).loc[start:].diff().fillna(0.0)
    series["funding"] = (1.0 + received / opening).cumprod()
    series["after_costs"] = (1.0 + (received - spent) / opening).cumprod()
    sensitivity = []
    for multiple in config.costs.cost_multiples:
        r = (
            res
            if multiple == 1.0
            else _headline_run(
                config,
                market=market,
                scores=scores[p.headline_model],
                path="topup",
                cost_multiple=multiple,
            )
        )
        perf = _performance(r.equity.loc[equity.index[0] :], config=config)
        sensitivity.append(
            {
                "cost_multiple": multiple,
                "annual_return": perf["annual_return"],
                "sharpe": perf["sharpe"],
            }
        )
    pd.DataFrame(sensitivity).to_csv(tables_dir / "cost_sensitivity.csv", index=False)
    decomposition = _decomposition(res, start=equity.index[0], config=config)
    usd = usd_per_usdt(stable, pd.DatetimeIndex(series["topup"].index))
    frame = pd.DataFrame(
        {
            "gross_funding": series["funding"],
            "after_costs": series["after_costs"],
            "net_ideal": series["ideal"],
            "net_topup": series["topup"],
            "net_liquidate": series["liquidate"],
            "net_topup_usd": series["topup"] * usd / usd.iloc[0],
        }
    )
    frame["drawdown"] = stats.drawdown_series(frame["net_topup"])
    frame.rename_axis("time").to_csv(tables_dir / "portfolio_series.csv")

    benchmarks = _benchmarks(config, bars=bars, start=equity.index[0], tables_dir=tables_dir)
    capacity = res.capacity_usd.loc[equity.index[0] :]
    return {
        "headline": {
            "model": p.headline_model,
            "top_k": p.headline_top_k,
            "min_volume": p.headline_min_quote_volume_usd,
            "leverage": config.carry.headline_leverage,
            **paths["topup"],
            "liquidations": res.liquidations,
            "delisted_closes": res.delisted_closes,
            "turnover": res.turnover,
            "first": str(equity.index[0].date()),
            "last": str(equity.index[-1].date()),
            "annual_return_usd": _performance(frame["net_topup_usd"], config=config)[
                "annual_return"
            ],
        },
        "paths": paths,
        "gross_funding_annual": _performance(series["funding"], config=config)["annual_return"],
        "after_costs_annual": _performance(series["after_costs"], config=config)["annual_return"],
        "deflated_sharpe": dsr,
        "sr0_annual": sr0 * np.sqrt(days_per_year(config)),
        "trials": len(trial_sharpes),
        "trial_sharpe_spread": float(
            np.std(trial_sharpes, ddof=1) * np.sqrt(days_per_year(config))
        ),
        "share_variants_positive": float((variants["annual_return"] > 0).mean()),
        "best_variant": variants.sort_values("sharpe").iloc[-1].to_dict(),
        "capacity_median_usd": float(capacity.median()) if len(capacity) else None,
        "capacity_p10_usd": float(capacity.quantile(0.1)) if len(capacity) else None,
        "benchmarks": benchmarks,
        "n_contracts_traded": int(bars["symbol"].nunique()),
        "cost_sensitivity": sensitivity,
        "decomposition": decomposition,
        "half_spread_bp": {
            leg: float(np.nanmedian(bars[f"{leg}_half"].to_numpy()) * 1e4)
            for leg in ("spot", "perp")
        },
    }


def _headline_run(
    config: Config,
    *,
    market: Market,
    scores: pd.DataFrame,
    path: MarginPath,
    cost_multiple: float = 1.0,
) -> PortfolioResult:
    """The headline variant on one margin path, with every cost scaled by `cost_multiple`."""
    p = config.portfolio
    carry = _carry_settings(config, leverage=config.carry.headline_leverage, path=path, symbol=None)
    if cost_multiple != 1.0:
        carry = replace(
            carry, spot_fee=carry.spot_fee * cost_multiple, perp_fee=carry.perp_fee * cost_multiple
        )
        market = replace(
            market,
            spot_half=market.spot_half * cost_multiple,
            perp_half=market.perp_half * cost_multiple,
        )
    return simulate_portfolio(
        market,
        scores,
        settings=PortfolioSettings(
            top_k=p.headline_top_k,
            rebalance_hours=p.rebalance_hours,
            exit_multiple=p.exit_rank_multiple,
            carry=carry,
        ),
        min_volume=p.headline_min_quote_volume_usd,
        participation=p.participation,
        maintenance=dict(config.margin.maintenance_rate),
        default_maintenance=config.margin.default_maintenance_rate,
    )


def _decomposition(
    res: PortfolioResult, *, start: pd.Timestamp, config: Config
) -> dict[str, float]:
    """Where the return came from, each part a year's sum of bar amounts over the equity at
    the bar's start. The parts add up to the arithmetic annual return: funding received,
    less fees, less spreads, plus everything else (basis moves, liquidation losses, margin)."""
    equity = res.equity.loc[start:]
    before = equity.shift(1).iloc[1:]
    years = (equity.index[-1] - equity.index[0]) / pd.Timedelta(
        hours=config.evaluation.hours_per_year
    )

    def rate(cumulative: pd.Series) -> float:
        step = cumulative.loc[start:].diff().iloc[1:]
        return float((step / before).sum() / years) if years > 0 else float("nan")

    total = rate(equity)
    funding, fees, spread = rate(res.funding), rate(res.fees), rate(res.spread)
    k = config.portfolio.headline_top_k
    return {
        "funding": funding,
        "fees": -fees,
        "spread": -spread,
        "other": total - funding + fees + spread,
        "total": total,
        "turnover": res.turnover,
        "holding_days": (
            k / res.turnover * config.portfolio.rebalance_hours / 24.0
            if res.turnover > 0
            else float("nan")
        ),
    }


def _benchmarks(
    config: Config, *, bars: pd.DataFrame, start: pd.Timestamp, tables_dir: Path
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    returns = {}
    for symbol in config.universe.core_symbols:
        b = _indexed(bars, symbol).loc[start:]
        if b.empty:
            continue
        res = simulate(
            b,
            settings=_carry_settings(
                config, leverage=config.carry.headline_leverage, path="topup", symbol=symbol
            ),
        )
        out[f"{symbol}_carry"] = _performance(res.equity, config=config)
        returns[symbol] = stats.daily_returns(res.equity)
        spot = b["spot_close"] / b["spot_close"].iloc[0]
        out[f"{symbol}_spot"] = _performance(spot, config=config)
    if len(returns) == 2:
        mix = pd.concat(returns, axis=1).dropna().mean(axis=1)
        equity = (1.0 + mix).cumprod()
        out["core_mix_carry"] = _performance(equity, config=config)
    return out


# 5. Episodes, venues, counterparty, stablecoin -------------------------------------------------


def _episodes(
    config: Config, *, tables: Tables, tables_dir: Path, core: dict[str, Any]
) -> list[dict[str, Any]]:
    series = pd.read_csv(tables_dir / "portfolio_series.csv", parse_dates=["time"]).set_index(
        "time"
    )
    rows = []
    flat = []
    for episode in config.episodes:
        start = pd.Timestamp(episode.start, tz="UTC")
        end = pd.Timestamp(episode.end, tz="UTC") + pd.Timedelta(days=1)
        row: dict[str, Any] = {"name": episode.name, "start": episode.start, "end": episode.end}
        row["in_sample"] = bool(
            start >= series.index[0] and end <= series.index[-1] + pd.Timedelta(days=1)
        )
        window = series.loc[(series.index >= start) & (series.index < end), "net_topup"]
        row["portfolio_return"] = (
            float(window.iloc[-1] / window.iloc[0] - 1.0) if len(window) > 1 else None
        )
        row["portfolio_drawdown"] = (
            float(stats.drawdown_series(window).min()) if len(window) > 1 else None
        )
        contracts = []
        for symbol in config.universe.core_symbols:
            prints = tables.funding.loc[
                (tables.funding["symbol"] == symbol)
                & (tables.funding["time"] >= start)
                & (tables.funding["time"] < end)
            ]
            rate = prints["rate"].to_numpy()
            times = core.get(symbol, {}).get("liquidation_times", [])
            contract = {
                "symbol": symbol,
                "prints": int(rate.size),
                "negative_share": float((rate < 0).mean()) if rate.size else None,
                "sign_flips": int((np.diff(np.sign(rate)) != 0).sum()) if rate.size > 1 else 0,
                "min_rate": float(rate.min()) if rate.size else None,
                "liquidations": sum(start <= pd.Timestamp(t) < end for t in times),
            }
            contracts.append(contract)
            flat.append(
                {k: row[k] for k in ("name", "portfolio_return", "portfolio_drawdown")} | contract
            )
        row["contracts"] = contracts
        rows.append(row)
    pd.DataFrame(flat).to_csv(tables_dir / "episodes.csv", index=False)
    return rows


def _mean_pairwise_correlation(frame: pd.DataFrame) -> float | None:
    if frame.shape[1] < 2:
        return None
    corr = frame.corr().to_numpy()
    return float(corr[np.triu_indices_from(corr, k=1)].mean())


def _venues(config: Config, venues: pd.DataFrame, tables_dir: Path) -> dict[str, Any]:
    windows_per_year = (
        config.evaluation.hours_per_year / 8.0
    )  # funding is summed into 8-hour windows
    out: dict[str, Any] = {}
    rows = []
    for symbol in config.universe.core_symbols:
        v = venues.loc[venues["symbol"] == symbol].copy()
        v["t"] = (v["time"] - pd.Timedelta(milliseconds=1)).dt.floor("8h") + pd.Timedelta(hours=8)
        wide = v.pivot_table(index="t", columns="venue", values="rate", aggfunc="sum")
        names = [n for n in config.sources.venue_names if n in wide.columns]
        common = wide[names].dropna()
        deep = [n for n in config.sources.full_history_venues if n in wide.columns]
        pair = wide[deep].dropna() if len(deep) > 1 else wide.iloc[0:0]
        block: dict[str, Any] = {}
        for label, frame in (("all", common), ("full_history", pair)):
            if frame.empty:
                continue
            block[label] = {
                "first": str(frame.index.min().date()),
                "last": str(frame.index.max().date()),
                "windows": len(frame),
                "annualized": {n: float(frame[n].mean() * windows_per_year) for n in frame.columns},
                "correlation": _mean_pairwise_correlation(frame),
            }
            means = block[label]["annualized"]
            if len(means) > 1:
                pairs = list(itertools.combinations(means, 2))
                a, b = max(pairs, key=lambda ab: abs(means[ab[0]] - means[ab[1]]))
                block[label]["widest_gap"] = {"venues": [a, b], "gap": abs(means[a] - means[b])}
            for n in frame.columns:
                rows.append(
                    {
                        "symbol": symbol,
                        "window": label,
                        "venue": n,
                        "first": block[label]["first"],
                        "last": block[label]["last"],
                        "annualized": block[label]["annualized"][n],
                    }
                )
        out[symbol] = block
    pd.DataFrame(rows).to_csv(tables_dir / "venues.csv", index=False)
    return out


def _counterparty(config: Config, annual_return: float, tables_dir: Path) -> dict[str, Any]:
    c = config.counterparty
    rows = []
    for scale, lgd in itertools.product(c.hazard_scales, c.loss_given_failure):
        hazards = {v: min(h * scale, 0.99) for v, h in c.venues.items()}
        split = counterparty.optimal_split(
            hazards, lgd=lgd, level=c.es_level, max_share=c.max_share
        )
        rows.append(
            {
                "hazard_scale": scale,
                "lgd": lgd,
                **{f"w_{v}": w for v, w in split.weights.items()},
                "expected_loss": split.expected_loss,
                "expected_shortfall": split.expected_shortfall,
                "single_venue_shortfall": split.single_venue_shortfall,
                "equal_split_shortfall": split.equal_split_shortfall,
                "any_failure_probability": split.any_failure_probability,
                "net_annual_return": annual_return - split.expected_loss,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(tables_dir / "counterparty.csv", index=False)
    head = frame.loc[
        (frame["hazard_scale"] == 1.0) & (frame["lgd"] == c.headline_loss_given_failure)
    ]
    return {
        "grid": rows,
        "headline": head.iloc[0].to_dict() if len(head) else None,
        "breakeven_scale": _breakeven(frame, annual_return, c.headline_loss_given_failure),
    }


def _breakeven(frame: pd.DataFrame, annual_return: float, lgd: float) -> float | None:
    """Smallest hazard scale on the grid at which expected counterparty loss eats the carry."""
    g = frame.loc[frame["lgd"] == lgd].sort_values("hazard_scale")
    hit = g.loc[g["expected_loss"] >= annual_return, "hazard_scale"]
    return float(hit.iloc[0]) if len(hit) else None


def _stablecoin(config: Config, tables: Tables, annual_return: float) -> dict[str, Any]:
    s = config.stablecoin
    usd = tables.stable.set_index("open_time")["usd_per_usdt"]
    worst = usd.idxmin() if len(usd) else None
    daily_carry = (
        (1.0 + annual_return) ** (1.0 / days_per_year(config)) - 1.0
        if annual_return > -1
        else float("nan")
    )
    return {
        "min_usd_per_usdt": float(usd.min()) if len(usd) else None,
        "min_date": str(pd.Timestamp(str(worst)).date()) if worst is not None else None,
        "max_usd_per_usdt": float(usd.max()) if len(usd) else None,
        "scenario_shock": s.shock,
        "scenario_days": s.days,
        "scenario_loss": s.shock,  # the trade's capital is USDT, so the whole equity is exposed
        "days_of_carry_to_recover": abs(s.shock) / daily_carry if daily_carry > 0 else None,
    }


def _cross_section(config: Config, tables: Tables, tables_dir: Path) -> dict[str, Any]:
    """Funding per contract-month, annualized over the hours its prints cover, and the
    monthly cross-section of all live contracts: the median contract and the share paying
    negative funding (longs are paid, so the carry trade loses)."""
    f = tables.funding.assign(month=tables.funding["time"].dt.tz_localize(None).dt.to_period("M"))
    cm = f.groupby(["symbol", "month"]).agg(rate=("rate", "sum"), hours=("interval_h", "sum"))
    cm = cm.reset_index()
    cm["annualized"] = cm["rate"] / cm["hours"] * config.evaluation.hours_per_year
    cm["month"] = cm["month"].astype(str)
    depth = cm.groupby("symbol").size().sort_values(ascending=False)
    keep = list(depth.index[: config.plot.heatmap_symbols])
    cm.loc[cm["symbol"].isin(keep)].to_csv(tables_dir / "heatmap.csv", index=False)
    monthly = (
        cm.groupby("month")["annualized"]
        .agg(
            contracts="size",
            median="median",
            share_negative=lambda x: float((x < 0).mean()),
        )
        .reset_index()
    )
    monthly.to_csv(tables_dir / "cross_section.csv", index=False)
    best = monthly.iloc[int(monthly["median"].to_numpy().argmax())]
    worst = monthly.iloc[int(monthly["median"].to_numpy().argmin())]
    widest = monthly.iloc[int(monthly["share_negative"].to_numpy().argmax())]
    return {
        "contract_months": len(cm),
        "months": len(monthly),
        "symbols_shown": len(keep),
        "median_annualized": float(cm["annualized"].median()),
        "share_contract_months_negative": float((cm["annualized"] < 0).mean()),
        "share_months_median_negative": float((monthly["median"] < 0).mean()),
        "best_month": str(best["month"]),
        "best_month_median": float(best["median"]),
        "worst_month": str(worst["month"]),
        "worst_month_median": float(worst["median"]),
        "most_negative_month": str(widest["month"]),
        "most_negative_share": float(widest["share_negative"]),
    }


def run(
    *,
    config: Config,
    tables: Tables,
    snapshots: pd.DataFrame,
    tables_dir: Path,
    log: bool = False,
) -> dict[str, Any]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def say(line: str) -> None:
        if log:
            print(f"  [{time.perf_counter() - started:5.0f} s] {line}", flush=True)

    tables, screened = screen(tables)
    say("screened rows: " + ", ".join(f"{k} {v:,}" for k, v in screened.items()))
    spreads = costs.fit_spread_model(
        snapshots, tables.contracts, clip_quantiles=config.costs.spread_clip_quantiles
    )
    say(
        f"spread model from {spreads.snapshots} snapshot(s) of {spreads.contracts} contracts: "
        f"perpetual slope {spreads.perp.slope:.2f}, spot slope {spreads.spot.slope:.2f}"
    )
    bars = prepare_bars(
        tables.bars,
        tables.funding,
        bar_hours=interval_hours(config.universe.bar_interval),
        spreads=spreads,
        volume_window_days=config.portfolio.volume_window_days,
    )
    core = prepare_bars(
        tables.core_bars,
        tables.funding,
        bar_hours=interval_hours(config.universe.core_bar_interval),
        spreads=spreads,
        volume_window_days=config.portfolio.volume_window_days,
    )
    say("bars prepared; rebuilding funding prints")
    rebuild = _rebuild(config, tables, tables_dir)
    say("simulating the core contracts")
    core_out = _core(config, core=core, stable=tables.stable, tables_dir=tables_dir)
    say("walk-forward funding forecasts")
    forecasts, preds = _forecasts(config, tables, tables_dir)
    say("portfolio variants")
    portfolio = _portfolio(
        config, bars=bars, preds=preds, stable=tables.stable, tables_dir=tables_dir
    )
    say("benchmarks, episodes, venues and counterparty risk")
    annual = portfolio["headline"]["annual_return"]
    contracts = tables.contracts
    return {
        "universe": {
            "contracts": len(contracts),
            "delisted": int(contracts["delisted"].sum()),
            "first": str(contracts["first_bar"].min().date()),
            "last": str(contracts["last_bar"].max().date()),
            "bars": len(bars),
            "funding_prints": len(tables.funding),
            "screened": screened,
            "spread_median_bp": float(bars["perp_half"].median() * 1e4),
        },
        "spread_model": spreads.as_dict(),
        "rebuild": rebuild,
        "core": core_out,
        "forecast": forecasts,
        "portfolio": portfolio,
        "episodes": _episodes(config, tables=tables, tables_dir=tables_dir, core=core_out),
        "venues": _venues(config, tables.venues, tables_dir),
        "counterparty": _counterparty(config, annual, tables_dir),
        "stablecoin": _stablecoin(config, tables, annual),
        "cross_section": _cross_section(config, tables, tables_dir),
        "runtime_seconds": time.perf_counter() - started,
    }


def write(estimates: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimates, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")

"""results.json: every number the README and paper quote, with its provenance."""

from __future__ import annotations

import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from perp_funding_carry.config import Config
from perp_funding_carry.data.manifest import Manifest


def _clean(value: Any) -> Any:
    """NaN and infinities become null so results.json is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def provenance_of(manifest: Manifest) -> str:
    origins = manifest.origins()
    if origins == {"official"}:
        return "official"
    return "synthetic" if origins else "empty"


def assemble(*, config: Config, estimates: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    p = estimates["portfolio"]
    bench = p["benchmarks"]
    core_symbols = list(config.universe.core_symbols)
    lev = f"{config.carry.headline_leverage:g}"
    core = {
        s: {
            path: estimates["core"].get(s, {}).get(f"{lev}|{path}")
            for path in ("ideal", "topup", "liquidate")
        }
        for s in core_symbols
    }
    liquidations = {
        s: {
            f"{x:g}": estimates["core"].get(s, {}).get(f"{x:g}|liquidate", {}).get("liquidations")
            for x in config.carry.leverages
        }
        for s in core_symbols
    }
    payload = {
        "title": "What funding carry actually pays",
        "author": "Ian Wachira",
        "provenance": provenance_of(manifest),
        "manifest_digest": manifest.digest(),
        "manifest_files": len(manifest.entries),
        "config_fingerprint": config.fingerprint(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "settings": {
            "core_symbols": core_symbols,
            "core_bases": [c.removesuffix(config.universe.quote) for c in core_symbols],
            "bar_interval": config.universe.bar_interval,
            "core_bar_interval": config.universe.core_bar_interval,
            "leverages": list(config.carry.leverages),
            "headline_leverage": config.carry.headline_leverage,
            "reserve_share": config.carry.reserve_share,
            "topup_multiple": config.carry.topup_leverage_multiple,
            "spot_taker": config.costs.spot_taker,
            "perp_taker": config.costs.perp_taker,
            "spread_clip_quantiles": list(config.costs.spread_clip_quantiles),
            "cost_multiples": list(config.costs.cost_multiples),
            "cap_test_threshold": config.rebuild.cap_test_threshold,
            "fees_retrieved": config.costs.retrieved,
            "top_k": list(config.portfolio.top_k),
            "headline_top_k": config.portfolio.headline_top_k,
            "exit_rank_multiple": config.portfolio.exit_rank_multiple,
            "volumes": list(config.portfolio.min_quote_volume_usd),
            "headline_volume": config.portfolio.headline_min_quote_volume_usd,
            "models": list(config.portfolio.models),
            "headline_model": config.portfolio.headline_model,
            "horizons": list(config.forecast.horizons),
            "har_lags": list(config.forecast.har_lags),
            "rebuild_symbols": list(config.rebuild.symbols),
            "rebuild_tolerance_bp": config.rebuild.tolerance * 1e4,
            "rebuild_months": config.rebuild.months,
            "participation": config.portfolio.participation,
            "es_level": config.counterparty.es_level,
            "venue_hazards": dict(config.counterparty.venues),
            "venue_names": dict(config.sources.venue_names),
            "full_history_venues": list(config.sources.full_history_venues),
            "headline_lgd": config.counterparty.headline_loss_given_failure,
            "episodes": [e.model_dump() for e in config.episodes],
            "stablecoin_pair": config.universe.stablecoin_pair,
        },
        "headline": {
            "portfolio": p["headline"],
            "paths": p["paths"],
            "gross_funding_annual": p["gross_funding_annual"],
            "after_costs_annual": p["after_costs_annual"],
            "deflated_sharpe": p["deflated_sharpe"],
            "trials": p["trials"],
            "btc_carry": bench.get(f"{core_symbols[0]}_carry"),
            "eth_carry": bench.get(f"{core_symbols[1]}_carry") if len(core_symbols) > 1 else None,
            "core_mix_carry": bench.get("core_mix_carry"),
            "rebuild_share": estimates["rebuild"]
            .get("conventions", {})
            .get(estimates["rebuild"].get("best", ""), {})
            .get("share_within"),
            "counterparty": estimates["counterparty"]["headline"],
        },
        "core": core,
        "liquidations": liquidations,
        "runtime": {"seconds": estimates["runtime_seconds"], "python": platform.python_version()},
        "estimates": estimates,
    }
    cleaned: dict[str, Any] = _clean(payload)
    return cleaned


def write(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(results), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded

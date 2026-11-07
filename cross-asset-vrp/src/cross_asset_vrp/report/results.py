"""results.json: every number the README and paper quote, with its provenance."""

from __future__ import annotations

import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cross_asset_vrp.config import Config
from cross_asset_vrp.data.manifest import Manifest


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


NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)


def _count_word(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)


def _headline(config: Config, estimates: dict[str, Any]) -> dict[str, Any]:
    premium = estimates["premium"]
    ranked = sorted(
        (k for k, v in premium.items() if v.get("vol_gap_points") is not None),
        key=lambda k: premium[k]["vol_gap_points"],
        reverse=True,
    )
    vix = next(a.key for a in config.assets if a.implied == "VIX")
    strategies = estimates["futures"]["strategies"]
    # A rule that wiped out its account is not a candidate, whatever its Sharpe ratio.
    solvent = [k for k in strategies if strategies[k]["common"].get("ruined_on") is None]
    best = max(solvent, key=lambda k: strategies[k]["common"]["sharpe"]) if solvent else None
    ruined = [k for k in strategies if k not in solvent]
    after = [
        r for r in estimates["predictability"] if r["asset"] == vix and r["sample"] == "after_btz"
    ]
    return {
        "largest_gap": ranked[0] if ranked else None,
        "smallest_gap": ranked[-1] if ranked else None,
        "ranked_by_gap": ranked,
        "positive_premium_assets": [
            k for k in ranked if premium[k]["ex_post_mean"] and premium[k]["ex_post_mean"] > 0
        ],
        "vix_asset": vix,
        "best_sizing": best,
        "ruined_sizing": ruined,
        "btz_after": after,
    }


def assemble(*, config: Config, estimates: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    payload = {
        "title": f"The variance risk premium across {_count_word(len(config.assets))} markets",
        "author": "Ian Wachira",
        "provenance": provenance_of(manifest),
        "manifest_digest": manifest.digest(),
        "manifest_files": len(manifest.entries),
        "config_fingerprint": config.fingerprint(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "settings": {
            "variance_days": config.windows.variance_days,
            "days_per_year": config.windows.days_per_year,
            "har_lags": list(config.har.lags),
            "btc_har_lags": list(config.har.btc_lags),
            "min_train_days": config.har.min_train_days,
            "btc_sample_minutes": config.realized.btc_sample_minutes,
            "yang_zhang_window": config.realized.yang_zhang_window,
            "horizons_months": list(config.predictability.horizons_months),
            "btz_sample_end": config.predictability.btz_sample_end,
            "ewc_coefficient": config.predictability.ewc_coefficient,
            "target_days": config.futures.target_days,
            "max_gap_days": config.futures.max_gap_days,
            "cost_points_per_side": config.futures.cost_points_per_side,
            "sizing_cap": config.futures.sizing_cap,
            "min_signal_history_days": config.futures.min_signal_history_days,
            "rescale_before": config.futures.rescale_before,
            "rescale_factor": config.futures.rescale_factor,
            "window_hours": config.replication.window_hours,
            "spread_buckets": list(config.replication.spread_buckets),
            "min_spread_trades": config.replication.min_spread_trades,
            "min_expiry_days": config.replication.min_expiry_days,
            "zero_bid_stop": config.replication.zero_bid_stop,
            "term_short": config.term_structure.short,
            "term_long": config.term_structure.long,
            "risk_free": config.risk_free.series,
            "heston": config.heston.model_dump(),
            "assets": [a.model_dump() for a in config.assets],
            "episodes": [e.model_dump() for e in config.episodes],
        },
        "headline": _headline(config, estimates),
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

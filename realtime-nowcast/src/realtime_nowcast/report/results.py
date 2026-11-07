"""results.json: every number the README and paper quote, with its provenance."""

from __future__ import annotations

import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from realtime_nowcast.config import Config
from realtime_nowcast.data.manifest import Manifest

SIGNIFICANCE = 0.05


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


def _verdict(metrics: dict[str, Any], key: str) -> str:
    """'better', 'worse' or 'level' for the model against GDPNow by a DM test at 5%."""
    stat, p = metrics.get(f"dm_{key}"), metrics.get(f"dm_{key}_p")
    if stat is None or p is None or not math.isfinite(p) or p >= SIGNIFICANCE:
        return "level"
    return "better" if stat < 0 else "worse"


def _headline(estimates: dict[str, Any], config: Config) -> dict[str, Any]:
    ev = estimates["evaluation"]
    final = ev["final"]
    groups = ev["groups"]
    beats = [g["group"] for g in groups if g["all"]["model_rmse"] < g["all"]["gdpnow_rmse"]]
    primary = estimates["market"][config.market.primary]["all"]
    return {
        "final_verdict_squared": _verdict(final["all"], "squared"),
        "final_verdict_absolute": _verdict(final["all"], "absolute"),
        "final_verdict_ex": _verdict(final["ex_excluded"], "squared"),
        "rmse_ratio_final": final["all"]["model_rmse"] / final["all"]["gdpnow_rmse"],
        "rmse_ratio_final_ex": final["ex_excluded"]["model_rmse"]
        / final["ex_excluded"]["gdpnow_rmse"],
        "groups_model_lower": beats,
        "groups_total": len(groups),
        "yield_significant": bool(primary["p"] < SIGNIFICANCE),
        "yield_positive": bool(primary["slope"] > 0),
    }


def assemble(*, config: Config, estimates: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    payload = {
        "title": "A real-time GDP nowcast against GDPNow",
        "author": "Ian Wachira",
        "provenance": provenance_of(manifest),
        "manifest_digest": manifest.digest(),
        "manifest_files": len(manifest.entries),
        "config_fingerprint": config.fingerprint(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "settings": {
            "spec_source": config.panel.spec_source,
            "observation_start": config.panel.observation_start,
            "estimation_start": config.panel.estimation_start,
            "blocks": list(config.panel.blocks),
            "series": [s.model_dump() for s in config.panel.series],
            "n_series": len(config.panel.series),
            "n_monthly": sum(s.frequency == "m" for s in config.panel.series),
            "n_quarterly": sum(s.frequency == "q" for s in config.panel.series),
            "target": config.panel.target,
            "threshold": config.model.threshold,
            "max_iter": config.model.max_iter,
            "measurement_variance": config.model.measurement_variance,
            "warm_start": config.model.warm_start,
            "pandemic_free": list(config.model.pandemic_free),
            "first_quarter": config.evaluation.first_quarter,
            "horizon_edges": list(config.evaluation.horizon_edges),
            "excluded_years": list(config.evaluation.excluded_years),
            "yields": list(config.market.yields),
            "primary_yield": config.market.primary,
        },
        "headline": _headline(estimates, config),
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

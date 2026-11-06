"""results.json: every number the README and paper quote, with its provenance."""

from __future__ import annotations

import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from factor_zoo_trial.config import Config
from factor_zoo_trial.data.manifest import Manifest


def _clean(value: Any) -> Any:
    """NaN and infinities become null so results.json is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    return value


def _download_mb(manifest: Manifest) -> dict[str, float]:
    """Megabytes downloaded per source (the first path component of each manifest key)."""
    totals: dict[str, float] = {}
    for key, entry in manifest.entries.items():
        source = key.split("/", 1)[0]
        totals[source] = totals.get(source, 0.0) + entry.bytes / 1e6
    return dict(sorted(totals.items()))


def provenance_of(manifest: Manifest) -> str:
    origins = manifest.origins()
    if origins == {"official"}:
        return "official"
    return "synthetic" if origins else "empty"


def _headline(config: Config, est: dict[str, Any]) -> dict[str, Any]:
    osap = est["osap"]
    shift = config.periods.headline_shift
    variant = config.osap.headline_variant
    basis = config.decay.headline_basis
    key = f"all|{variant}|{basis}|{shift}"
    decay = osap["decay"][key]
    clear = osap["decay"][f"clear|{variant}|{basis}|{shift}"]
    testing = osap["testing"][f"all|{variant}|raw"]
    jkp = est["jkp"]
    us = jkp["regions"][config.space.source_region]
    return {
        "decay": decay,
        "decay_clear": clear,
        "kept_after_publication": 1.0 + decay["share_post_publication"]["value"],
        "kept_after_sample": 1.0 + decay["share_post_sample"]["value"],
        "testing": testing,
        "us": us,
        "space": jkp["space"],
    }


def assemble(*, config: Config, estimates: dict[str, Any], manifest: Manifest) -> dict[str, Any]:
    shifts = {
        str(s): estimates["osap"]["decay"][
            f"all|{config.osap.headline_variant}|{config.decay.headline_basis}|{s}"
        ]
        for s in config.periods.publication_shifts
    }
    variants = {
        v: {
            "decay": estimates["osap"]["decay"][
                f"all|{v}|{config.decay.headline_basis}|{config.periods.headline_shift}"
            ],
            "testing": estimates["osap"]["testing"][f"all|{v}|raw"],
        }
        for v in config.osap.variants
    }
    bases = {
        b: estimates["osap"]["decay"][
            f"all|{config.osap.headline_variant}|{b}|{config.periods.headline_shift}"
        ]
        for b in config.decay.return_bases
    }
    payload = {
        "title": "The factor zoo on trial",
        "author": "Ian Wachira",
        "provenance": provenance_of(manifest),
        "manifest_digest": manifest.digest(),
        "manifest_files": len(manifest.entries),
        "download_mb": _download_mb(manifest),
        "config_fingerprint": config.fingerprint(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "osap_release": config.sources.osap_revision,
        "jkp_reference_commit": config.sources.jkp_reference_commit,
        "settings": {
            "level": config.multiple_testing.level,
            "ewc_coefficient": config.multiple_testing.ewc_coefficient,
            "storey_lambda": config.multiple_testing.storey_lambda,
            "hlz_threshold": config.multiple_testing.hlz_threshold,
            "romano_wolf_draws": config.multiple_testing.romano_wolf_draws,
            "romano_wolf_block_months": config.multiple_testing.romano_wolf_block_months,
            "publication_month": config.periods.publication_month,
            "publication_shifts": list(config.periods.publication_shifts),
            "headline_shift": config.periods.headline_shift,
            "min_months_in_sample": config.decay.min_months_in_sample,
            "eb_min_months": config.empirical_bayes.min_months,
            "vol_target": config.empirical_bayes.annual_vol_target,
            "interval": config.empirical_bayes.interval,
            "shrinkage": config.empirical_bayes.covariance_shrinkage,
            "space_min_factors": config.space.min_factors,
            "simulation": config.simulation.model_dump(mode="json"),
            "variants": list(config.osap.variants),
            "headline_variant": config.osap.headline_variant,
            "liquidity_variant": config.osap.liquidity_variant,
            "bases": list(config.decay.return_bases),
        },
        "headline": _headline(config, estimates),
        "shifts": shifts,
        "variants": variants,
        "bases": bases,
        "runtime": {
            "seconds": estimates["runtime_seconds"],
            "python": platform.python_version(),
        },
        "estimates": estimates,
    }
    cleaned: dict[str, Any] = _clean(payload)
    return cleaned


def write(results: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(results), indent=2, sort_keys=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def read(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from covariance_horse_race.config import ResearchConfig, load_config
from tests.conftest import ROOT


def _raw() -> dict:
    with (ROOT / "configs" / "research.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_research_config_loads(config: ResearchConfig) -> None:
    assert config.backtest.primary_window in config.backtest.windows
    assert len(config.fingerprint()) == 16


def test_override_merges_and_changes_fingerprint(
    config: ResearchConfig, ci_config: ResearchConfig
) -> None:
    assert [u.key for u in ci_config.universes] == ["ind10", "ind30"]
    assert ci_config.data == config.data
    assert ci_config.fingerprint() != config.fingerprint()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("backtest", "primary_window"), 999),
        (("data", "base_url"), "http://insecure.example.org/"),
        (("evaluation", "mcs", "alpha"), 1.5),
        (("report", "headline_universe"), "nope"),
        (("data", "files", "ind10"), "../escape.zip"),
    ],
)
def test_invalid_values_are_rejected(path: tuple[str, ...], value: object) -> None:
    raw = _raw()
    node = raw
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(raw)


def test_unknown_keys_are_rejected() -> None:
    raw = _raw()
    raw["backtest"]["window_typo"] = 10
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(raw)


def test_duplicate_model_names_rejected() -> None:
    raw = _raw()
    raw["models"].append(dict(raw["models"][0]))
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(raw)


def test_load_config_requires_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)


def test_fingerprint_ignores_presentation_only(config: ResearchConfig) -> None:
    relabelled = config.model_copy(
        update={
            "models": [m.model_copy(update={"label": m.label + "!"}) for m in config.models],
            "report": config.report.model_copy(update={"figure_dpi": 72}),
        }
    )
    assert relabelled.fingerprint() == config.fingerprint()
    windows = config.backtest.model_copy(update={"windows": [252], "primary_window": 252})
    assert config.model_copy(update={"backtest": windows}).fingerprint() != config.fingerprint()

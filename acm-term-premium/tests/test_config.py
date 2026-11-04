from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from acm_term_premium.config import ResearchConfig, load_config
from acm_term_premium.report import render
from tests.conftest import ROOT


def _raw() -> dict[str, Any]:
    with (ROOT / "configs" / "research.yaml").open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return raw


def test_research_config_loads(config: ResearchConfig) -> None:
    assert config.validation.headline_years in config.validation.published_years
    assert config.model.pc_maturities.months()[0] == config.model.pc_maturities.first
    assert len(config.fingerprint()) == 16


def test_override_merges_and_changes_fingerprint(config: ResearchConfig) -> None:
    ci = load_config(ROOT / "configs" / "research.yaml", [ROOT / "configs" / "ci.yaml"])
    assert ci.data.sources.acm.file == "ACMTermPremium.xlsx"
    assert ci.data.sources.gsw == config.data.sources.gsw, "untouched keys are kept"
    assert ci.model.n_factors == config.model.n_factors
    assert ci.fingerprint() != config.fingerprint()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("data", "sources", "gsw", "url"), "http://www.federalreserve.gov/feds200628.csv"),
        (("data", "sources", "gsw", "file"), "../escape.csv"),
        (("model", "convention"), "textbook"),
        (("model", "max_spectral_radius"), 1.2),
        (("model", "return_maturities"), [12, 6, 24, 36, 48, 60, 72]),
        (("model", "return_maturities"), [6, 12, 24]),
        (("model", "max_maturity"), 60),
        (("model", "short_rate", "method"), "cubic"),
        (("validation", "headline_years"), 30),
        (("uncertainty", "band"), 1.0),
        (("forecasting", "maturities"), [12]),
        (("events", "maturity_years"), 20),
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
    raw["model"]["n_factor"] = 5
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(raw)


def test_load_config_requires_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(bad)


def test_fingerprint_ignores_presentation_only(config: ResearchConfig) -> None:
    redrawn = config.model_copy(
        update={"report": config.report.model_copy(update={"figure_dpi": 72})}
    )
    assert redrawn.fingerprint() == config.fingerprint()
    model = config.model.model_copy(update={"max_spectral_radius": 0.99})
    assert config.model_copy(update={"model": model}).fingerprint() != config.fingerprint()


def test_template_filters() -> None:
    assert render._years_label(24) == "2-year"
    assert render._years_label(6) == "6-month"
    assert render._join_words(["2-year", "5-year", "10-year"]) == "2-year, 5-year and 10-year"
    # A monthly root of 0.5 halves in one month, 1/12 of a year.
    assert render._half_life_years(0.5) == pytest.approx(1 / 12)

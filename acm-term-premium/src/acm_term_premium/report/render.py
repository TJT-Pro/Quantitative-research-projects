"""Render README.md and the paper source from templates and results.json.

StrictUndefined turns a missing value into an error. Quarto uses `{#id}` for
cross-references, Jinja's comment opener, so .qmd templates use `<#-- --#>`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined


def _num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _join_words(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _years_label(months: int) -> str:
    """Bond maturity in months as a hyphenated label: 24 -> '2-year'."""
    return f"{months // 12}-year" if months % 12 == 0 else f"{months}-month"


def _half_life_years(root: float) -> float:
    """Half-life in years of a monthly AR root: log(1/2) / log(root) / 12."""
    return math.log(0.5) / math.log(root) / 12


def _environment(quarto: bool) -> Environment:
    comment = ("<#--", "--#>") if quarto else ("{#", "#}")
    env = Environment(
        undefined=StrictUndefined,
        # Renders Markdown from this project's own results.json, never HTML.
        autoescape=False,  # noqa: S701
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        comment_start_string=comment[0],
        comment_end_string=comment[1],
    )
    env.filters["num"] = _num
    env.filters["join_words"] = _join_words
    env.filters["years_label"] = _years_label
    env.filters["half_life_years"] = _half_life_years
    return env


def render(template: Path, results: dict[str, Any]) -> str:
    env = _environment(template.suffix == ".qmd")
    return env.from_string(template.read_text(encoding="utf-8")).render(r=results)


def write(template: Path, results: dict[str, Any], output: Path) -> None:
    output.write_text(render(template, results), encoding="utf-8")


def is_current(template: Path, results: dict[str, Any], output: Path) -> bool:
    return output.is_file() and output.read_text(encoding="utf-8") == render(template, results)

"""Parsers and panel construction against hand-checked cases in the published formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acm_term_premium.data.panel import complete_month_ends, policy_change_days, splice_short_rate
from acm_term_premium.data.parsers import (
    parse_acm_workbook,
    parse_fred,
    parse_gsw,
    svensson_yields,
)
from acm_term_premium.data.source import SourceError

GSW = """\
"Note: preamble lines vary between releases."

Series,Compounding Convention,Mnemonic(s)
Date,BETA0,BETA1,BETA2,BETA3,SVENY01,SVENY10,TAU1,TAU2
1975-01-02,7.5,-1.5,0.5,-999.99,NA,NA,1.2,-999.99
1990-06-29,8.0,-0.5,1.0,2.0,7.9,8.4,1.5,8.0
"""


def test_gsw_preamble_and_nelson_siegel_marker() -> None:
    params = parse_gsw(GSW)
    assert list(params.index) == [pd.Timestamp("1975-01-02"), pd.Timestamp("1990-06-29")]
    first = params.iloc[0]
    assert first["BETA3"] == 0.0, "-999.99 means no fourth term"
    assert np.isnan(first["TAU2"])
    assert params.iloc[1]["TAU2"] == 8.0


def test_gsw_without_header_is_rejected() -> None:
    with pytest.raises(SourceError, match="Date,"):
        parse_gsw("no table here\n1,2,3\n")


def test_svensson_matches_closed_form() -> None:
    params = parse_gsw(GSW)
    out = svensson_yields(params, np.array([12, 120]))
    b0, b1, b2, b3, t1, t2 = 8.0, -0.5, 1.0, 2.0, 1.5, 8.0
    m = 10.0
    f1 = (1 - np.exp(-m / t1)) / (m / t1)
    f2 = (1 - np.exp(-m / t2)) / (m / t2)
    expected = b0 + b1 * f1 + b2 * (f1 - np.exp(-m / t1)) + b3 * (f2 - np.exp(-m / t2))
    assert out.loc["1990-06-29", 120] == pytest.approx(expected / 100, abs=1e-14)
    # Nelson-Siegel row: no fourth term at any maturity.
    ns = svensson_yields(params.iloc[[0]].assign(BETA3=5.0), np.array([120]))
    assert ns.iloc[0, 0] == pytest.approx(
        svensson_yields(params.iloc[[0]], np.array([120])).iloc[0, 0]
    )


def test_fred_missing_marker_and_columns() -> None:
    series = parse_fred("observation_date,DFEDTAR\n2000-01-03,5.5\n2000-01-04,.\n", "DFEDTAR")
    assert list(series) == [5.5]
    with pytest.raises(SourceError, match="columns"):
        parse_fred("observation_date,OTHER\n2000-01-03,1\n", "DFEDTAR")


def test_acm_workbook_round_trip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "DATE": ["31-Jan-2000", "29-Feb-2000"],
            "ACMY10": [6.5, 6.4],
            "ACMTP10": [1.25, 1.2],
            "ACMRNY10": [5.25, 5.2],
        }
    )
    path = tmp_path / "acm.xlsx"
    frame.to_excel(path, sheet_name="ACM Monthly", index=False)
    long = parse_acm_workbook(path, "ACM Monthly")
    assert set(long.columns) == {"date", "maturity_years", "fitted", "risk_neutral", "term_premium"}
    row = long.iloc[0]
    assert row["date"] == pd.Timestamp("2000-01-31")
    assert row["term_premium"] == pytest.approx(0.0125)
    assert row["fitted"] == pytest.approx(row["term_premium"] + row["risk_neutral"])


def test_month_ends_drop_the_month_in_progress() -> None:
    days = pd.bdate_range("2024-01-02", "2024-03-14")
    ends = complete_month_ends(days)
    assert list(ends) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    full = pd.bdate_range("2024-01-02", "2024-03-29")
    assert complete_month_ends(full)[-1] == pd.Timestamp("2024-03-29")


def test_splice_replaces_only_early_months() -> None:
    dates = pd.date_range("1975-01-31", periods=240, freq="ME")
    rng = np.random.default_rng(1)
    ff = pd.Series(
        5 + np.cumsum(rng.normal(0, 0.2, 240)), index=dates.to_period("M").to_timestamp()
    )
    curve = pd.DataFrame(
        {1: 0.001 + 0.0095 * ff.to_numpy() + rng.normal(0, 1e-4, 240), 2: 0.05}, index=dates
    )
    curve.iloc[:60, 0] = 0.30  # broken early short end
    out, report = splice_short_rate(curve, ff, "1980-01", "1980-01")
    assert report.replaced_months == 60
    assert np.allclose(out.iloc[60:], curve.iloc[60:])
    assert np.abs(out.iloc[:60, 0] - (0.001 + 0.0095 * ff.iloc[:60].to_numpy())).max() < 1e-3
    # Fed funds enters in decimals, so a 0.0095 loading on percent is 0.95.
    assert report.slope == pytest.approx(0.95, rel=0.02)


def test_policy_days_ignore_the_switch_to_a_range() -> None:
    point = pd.Series(
        [5.0, 5.0, 4.75, 4.75],
        index=pd.to_datetime(["2008-12-10", "2008-12-11", "2008-12-12", "2008-12-15"]),
    )
    upper = pd.Series(
        [0.25, 0.25, 0.5], index=pd.to_datetime(["2008-12-16", "2008-12-17", "2015-12-17"])
    )
    days = policy_change_days(point, upper)
    assert list(days["date"]) == [pd.Timestamp("2008-12-12"), pd.Timestamp("2015-12-17")]
    assert list(days["change_pp"]) == pytest.approx([-0.25, 0.25])

"""Synthetic sources in the exact published formats, for offline runs and CI.

* feds200628.csv: free-text preamble, a `Date,` header, Nelson-Siegel rows
  (BETA3 = 0, TAU2 = -999.99) before 1980 and Svensson rows after, `NA` blanks.
* FEDFUNDS.csv, DFEDTAR.csv, DFEDTARU.csv: FRED graph CSVs (observation_date, value).
* ACMTermPremium.xlsx: `ACM Monthly` and `ACM Daily` sheets with DATE, ACMYxx,
  ACMTPxx and ACMRNYxx columns in percent. The values are this model's own
  output on the fixture curve, estimated with the same research config the
  pipeline will run, so a correct pipeline reproduces them exactly.

Numbers are synthetic and never feed a reported result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from acm_term_premium.config import ResearchConfig
from acm_term_premium.data.panel import complete_month_ends, splice_short_rate
from acm_term_premium.data.parsers import parse_gsw, svensson_yields
from acm_term_premium.models.configured import fit_model

START = "1975-01-02"
DAYS = 7_800
NELSON_SIEGEL_UNTIL = "1980-01-01"
RANGE_TARGET_FROM = "2000-12-18"


def _ar(n: int, mean: float, persistence: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n)
    level = mean
    for i in range(n):
        level = mean + persistence * (level - mean) + sd * rng.standard_normal()
        out[i] = level
    return out


def _gsw_text(dates: pd.DatetimeIndex, rng: np.random.Generator) -> str:
    n = len(dates)
    params = pd.DataFrame(
        {
            "BETA0": _ar(n, 6.0, 0.999, 0.04, rng),
            "BETA1": _ar(n, -2.0, 0.998, 0.05, rng),
            "BETA2": _ar(n, -1.0, 0.995, 0.08, rng),
            "BETA3": _ar(n, 1.0, 0.995, 0.06, rng),
            "TAU1": np.clip(_ar(n, 1.6, 0.99, 0.02, rng), 0.5, None),
            "TAU2": np.clip(_ar(n, 9.0, 0.99, 0.1, rng), 4.0, None),
        },
        index=dates,
    )
    early = dates < pd.Timestamp(NELSON_SIEGEL_UNTIL)
    params.loc[early, "BETA3"] = 0.0
    params.loc[early, "TAU2"] = -999.99
    zero = svensson_yields(
        params.assign(TAU2=params["TAU2"].where(~early)), np.array([12 * y for y in range(1, 31)])
    )
    preamble = [
        '"Note: synthetic fixture in the layout of the Federal Reserve Board GSW file."',
        "",
        "Series,Compounding Convention,Mnemonic(s)",
        "Zero-coupon yield,Continuously Compounded,SVENYXX",
        "Parameters,N/A,BETA0 to TAU2",
        "",
    ]
    columns = ["BETA0", "BETA1", "BETA2", "BETA3"]
    header = ["Date", *columns, *[f"SVENY{y:02d}" for y in range(1, 31)], "TAU1", "TAU2"]
    lines = [*preamble, ",".join(header)]
    for date, row, y in zip(dates, params.itertuples(index=False), zero.to_numpy(), strict=True):
        long_end = [
            "NA" if (early[dates.get_loc(date)] and k >= 20) else f"{100 * v:.4f}"
            for k, v in enumerate(y)
        ]
        lines.append(
            ",".join(
                [
                    date.strftime("%Y-%m-%d"),
                    *(f"{getattr(row, c):.12g}" for c in columns),
                    *long_end,
                    f"{row.TAU1:.12g}",
                    f"{row.TAU2:.12g}",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _fred(series: pd.Series, name: str) -> str:
    lines = [f"observation_date,{name}"]
    lines += [f"{d:%Y-%m-%d},{v:.2f}" for d, v in series.items()]
    return "\n".join(lines) + "\n"


def write_fixture(target: Path, config: ResearchConfig, seed: int) -> None:
    rng = np.random.default_rng(seed)
    target.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range(START, periods=DAYS)
    gsw = _gsw_text(dates, rng)
    (target / "feds200628.csv").write_text(gsw, encoding="utf-8")

    params = parse_gsw(gsw)
    maturities = np.arange(1, config.model.max_maturity + 1)
    daily = svensson_yields(params, maturities)
    one_month = daily[1] * 100
    fedfunds = one_month.groupby(pd.DatetimeIndex(one_month.index).to_period("M")).mean()
    # Rounded to the two decimals the CSV carries, so the workbook below is built
    # from exactly the inputs the pipeline will read back.
    fedfunds = (fedfunds + rng.normal(0.0, 0.15, len(fedfunds))).round(2)
    fedfunds.index = pd.PeriodIndex(fedfunds.index).to_timestamp()
    (target / "FEDFUNDS.csv").write_text(_fred(fedfunds, "FEDFUNDS"), encoding="utf-8")

    moves = rng.choice([-0.5, -0.25, 0.25, 0.5], size=DAYS) * (rng.random(DAYS) < 0.012)
    target_rate = pd.Series(5.0 + np.cumsum(moves), index=dates).clip(0.25, None)
    switch = pd.Timestamp(RANGE_TARGET_FROM)
    (target / "DFEDTAR.csv").write_text(
        _fred(target_rate[dates < switch], "DFEDTAR"), encoding="utf-8"
    )
    (target / "DFEDTARU.csv").write_text(
        _fred(target_rate[dates >= switch], "DFEDTARU"), encoding="utf-8"
    )

    ends = complete_month_ends(pd.DatetimeIndex(params.index))
    monthly = svensson_yields(params.loc[ends], maturities)
    short = config.model.short_rate
    if short.method == "fedfunds_splice":
        monthly, _ = splice_short_rate(monthly, fedfunds, short.fit_start, short.replace_before)
    model = fit_model(monthly, config.model)
    months = [12 * y for y in config.validation.published_years]
    with pd.ExcelWriter(target / "ACMTermPremium.xlsx", engine="openpyxl") as writer:
        for sheet, curve in (("ACM Monthly", monthly), ("ACM Daily", daily)):
            dec = model.decompose(curve, months)
            frame = pd.DataFrame({"DATE": [d.strftime("%d-%b-%Y") for d in curve.index]})
            for prefix, key in (
                ("ACMY", "fitted"),
                ("ACMTP", "term_premium"),
                ("ACMRNY", "risk_neutral"),
            ):
                for i, m in enumerate(months, start=1):
                    frame[f"{prefix}{i:02d}"] = dec[key][m].to_numpy() * 100
            frame.to_excel(writer, sheet_name=sheet, index=False)

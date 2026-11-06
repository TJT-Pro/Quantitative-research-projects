"""Synthetic sources in the exact upstream formats, laid out as a mirror folder.

The fixture drives CI and the end-to-end test. It has a known truth: predictors keep
75% of their in-sample mean after the sample and 50% after publication, a set share
are pure noise, and JKP alphas are drawn from the hierarchical prior. Every file is
written where `fzt fetch --mirror` looks for it: <mirror>/<host>/<url path>.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from factor_zoo_trial.config import Config
from factor_zoo_trial.data import sources
from factor_zoo_trial.data.download import mirror_path

SIGNAL_DOC_HEADER = [
    "Acronym",
    "Cat.Signal",
    "Predictability in OP",
    "Signal Rep Quality",
    "Authors",
    "Year",
    "LongDescription",
    "Journal",
    "Cat.Form",
    "Cat.Data",
    "Cat.Economic",
    "SampleStartYear",
    "SampleEndYear",
    "Acronym2",
    "Evidence Summary",
    "Key Table in OP",
    "Test in OP",
    "Sign",
    "Return",
    "T-Stat",
    "Stock Weight",
    "LS Quantile",
    "Quantile Filter",
    "Portfolio Period",
    "Start Month",
    "Filter",
    "Notes",
    "Detailed Definition",
    "GScholarCites202509",
]
FACTOR_DETAILS_HEADER = [
    "abr_hxz",
    "name",
    "name_new",
    "abr_jkp",
    "time_to_create",
    "lagged_denom",
    "cite",
    "variations",
    "group",
    "in-sample period",
    "t-stat",
    "p-value",
    "direction",
    "significance",
    "controls",
]
JKP_CLUSTERS = [
    "Accruals",
    "Debt Issuance",
    "Investment",
    "Low Leverage",
    "Low Risk",
    "Momentum",
    "Profit Growth",
    "Profitability",
    "Quality",
    "Seasonality",
    "Short-Term Reversal",
    "Size",
    "Value",
]
CATEGORIES = ["accruals", "momentum", "valuation", "profitability", "investment", "volatility"]


@dataclass(frozen=True)
class FixtureTruth:
    decay_post_sample: float = -0.25
    decay_post_publication: float = -0.50
    null_share: float = 0.3
    tau_c: float = 0.15
    tau_w: float = 0.10
    country_eta: float = 0.08


TRUTH = FixtureTruth()


def _write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _months(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="ME")


# OSAP -------------------------------------------------------------------------------------


def _osap(*, config: Config, mirror: Path, rng: np.random.Generator, market: pd.Series) -> None:
    n = 60
    months = market.index
    names = [f"Sig{i:02d}" for i in range(n)]
    start = rng.integers(1940, 1990, size=n)
    end = start + rng.integers(15, 30, size=n)
    year = end + rng.integers(2, 7, size=n)
    real = rng.random(n) >= TRUTH.null_share
    mu = np.where(real, rng.uniform(0.3, 0.9, size=n), 0.0)
    beta = rng.normal(0.0, 0.15, size=n)
    common = rng.standard_normal((months.size, 3))
    load = rng.normal(0.0, 1.2, size=(n, 3))
    noise = common @ load.T + rng.standard_normal((months.size, n)) * 2.5

    doc_rows = []
    for i, name in enumerate(names):
        row = dict.fromkeys(SIGNAL_DOC_HEADER, "")
        row.update(
            {
                "Acronym": name,
                "Cat.Signal": "Predictor",
                "Predictability in OP": "1_clear" if i % 4 else "2_likely",
                "Signal Rep Quality": "1_good",
                "Authors": f"Author {i}",
                "Year": str(year[i]),
                "LongDescription": f"Synthetic signal {i}",
                "Journal": "JF",
                "Cat.Form": "continuous",
                "Cat.Data": "Accounting",
                "Cat.Economic": CATEGORIES[i % len(CATEGORIES)],
                "SampleStartYear": str(start[i]),
                "SampleEndYear": str(end[i]),
                "Acronym2": name,
                "Sign": "1.0" if i % 3 else "-1.0",
                "Return": f"{mu[i]:.3f}",
                "T-Stat": f"{rng.uniform(2, 6):.2f}",
                "Stock Weight": "EW",
                "LS Quantile": "0.1",
                "Portfolio Period": "1.0",
                "Start Month": "6.0",
                "Detailed Definition": "A description, with a comma.",
                "GScholarCites202509": "100.0",
            }
        )
        doc_rows.append(row)
    for j in range(3):  # placebos share the file and must be filtered out
        row = dict.fromkeys(SIGNAL_DOC_HEADER, "")
        row.update(
            {
                "Acronym": f"Placebo{j}",
                "Cat.Signal": "Placebo",
                "Year": "2000",
                "SampleStartYear": "1970",
                "SampleEndYear": "1990",
                "Sign": "1.0",
            }
        )
        doc_rows.append(row)
    doc = pd.DataFrame(doc_rows, columns=SIGNAL_DOC_HEADER)
    buffer = io.StringIO()
    doc.to_csv(buffer, index=False)
    _write(
        mirror_path(mirror, sources.osap_target(config, config.osap.signal_doc).url),
        buffer.getvalue().encode(),
    )

    levels = np.zeros((months.size, n))
    yr = pd.DatetimeIndex(months).year.to_numpy()
    for i in range(n):
        in_s = (yr >= start[i]) & (yr <= end[i])
        post_s = (yr > end[i]) & (yr <= year[i])
        post_p = yr > year[i]
        levels[:, i] = mu[i] * (
            in_s
            + (1 + TRUTH.decay_post_sample) * post_s
            + (1 + TRUTH.decay_post_publication) * post_p
            + 1.0 * (yr < start[i])
        )
    multipliers = {
        "original": 1.0,
        "value_weighted": 0.6,
        "nyse_only": 0.7,
        "price_gt_5": 0.85,
        "me_gt_nyse20": 0.55,
    }
    for variant, path in config.osap.variants.items():
        scale = multipliers.get(variant, 0.8) * rng.uniform(0.5, 1.5, size=n)
        if variant == "original":
            scale = np.ones(n)
        ls = (
            levels * scale
            + noise * (1.0 if variant == "original" else 1.1)
            + np.outer(market.to_numpy(), beta)
        )
        if variant != "original":
            ls = ls + rng.standard_normal(ls.shape) * 0.5
        frames = []
        for i, name in enumerate(names):
            live = yr >= start[i] - 10  # data begin before the sample, as in OSAP
            for port, values in (("01", -ls[:, i] / 2), ("10", ls[:, i] / 2), ("LS", ls[:, i])):
                frames.append(
                    pd.DataFrame(
                        {
                            "signalname": name,
                            "port": port,
                            "date": months[live],
                            "ret": values[live],
                            "signallag": 6.0,
                            "Nlong": 120.0,
                            "Nshort": 115.0,
                        }
                    )
                )
        table = pd.concat(frames, ignore_index=True)
        buffer_b = io.BytesIO()
        table.to_parquet(buffer_b, index=False)
        _write(mirror_path(mirror, sources.osap_target(config, path).url), buffer_b.getvalue())


# JKP --------------------------------------------------------------------------------------


def _xlsx(frame: pd.DataFrame, sheet: str) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet, index=False)
    return buffer.getvalue()


def _zip_csv(name: str, frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.replace(".zip", ".csv"), frame.to_csv(index=False, na_rep="na"))
    return buffer.getvalue()


EMPTY_COUNTRY = "isl"
COUNTRIES = ("USA", "GBR", "JPN", "DEU", "FRA", "CAN", "AUS", "BRA", "IND", "ZAF", "KEN")
DEVELOPMENT = ("developed",) * 7 + ("emerging",) * 3 + ("frontier",)
GEO = (
    "north america",
    "europe",
    "asia",
    "europe",
    "europe",
    "north america",
    "asia",
    "south america",
    "asia",
    "africa",
    "africa",
)


def _jkp_reference(*, config: Config, mirror: Path) -> tuple[list[str], list[str]]:
    """Cluster labels, factor details and country classification at the pinned commit."""
    factors = [
        f"jkp_{c.lower().replace(' ', '_').replace('-', '_')}_{j}"
        for c in JKP_CLUSTERS
        for j in range(4)
    ]
    clusters = [c for c in JKP_CLUSTERS for _ in range(4)]
    n = len(factors)
    details = pd.DataFrame(
        {
            "abr_hxz": [f"H{i}" for i in range(n)],
            "name": factors,
            "name_new": factors,
            "abr_jkp": factors,
            "time_to_create": np.nan,
            "lagged_denom": np.nan,
            "cite": [f"Author ({1980 + i % 30})" for i in range(n)],
            "variations": 1.0,
            "group": clusters,
            "in-sample period": "1970 - 1990",
            "t-stat": 3.0,
            "p-value": np.nan,
            "direction": 1.0,
            "significance": (np.arange(n) % 5 != 0).astype(int),
            "controls": 0.0,
        }
    )
    files = {
        config.jkp.cluster_labels: pd.DataFrame({"characteristic": factors, "cluster": clusters})
        .to_csv(index=False)
        .encode(),
        config.jkp.factor_details: _xlsx(details, "details"),
        config.jkp.country_classification: _xlsx(
            pd.DataFrame(
                {
                    "excntry": [*COUNTRIES, EMPTY_COUNTRY.upper()],
                    "msci_development": [*DEVELOPMENT, "frontier"],
                    "region": [*GEO, "europe"],
                }
            ),
            "countries",
        ),
    }
    base = f"{config.sources.jkp_reference_base}/{config.sources.jkp_reference_commit}"
    for path, data in files.items():
        _write(mirror_path(mirror, f"{base}/{path}"), data)
    return factors, clusters


def _jkp_returns(
    *, factors: list[str], clusters: list[str], rng: np.random.Generator, us_market: pd.Series
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """US alphas from the hierarchical prior; every other country deviates by country_eta."""
    n = len(factors)
    cluster_names, cluster_index = np.unique(clusters, return_inverse=True)
    cluster_effect = rng.standard_normal(cluster_names.size) * TRUTH.tau_c
    alpha_us = 0.2 + cluster_effect[cluster_index] + rng.standard_normal(n) * TRUTH.tau_w
    target = 10.0 / np.sqrt(12.0)
    returns: dict[str, pd.DataFrame] = {}
    markets: dict[str, pd.Series] = {}
    for code in (c.lower() for c in COUNTRIES):
        usa = code == "usa"
        months = us_market.index if usa else _months("1990-01-31", "2025-12-31")
        mkt = (
            us_market.reindex(months)
            if usa
            else pd.Series(rng.normal(0.6, 5.0, size=months.size), index=months)
        ).astype(float)
        alpha = alpha_us if usa else alpha_us + rng.standard_normal(n) * TRUTH.country_eta
        common = rng.standard_normal((months.size, 4)) @ rng.normal(0, 0.5, (4, n))
        noise = (common + rng.standard_normal((months.size, n))) / np.sqrt(2.0) * target
        ret = alpha + noise + np.outer(mkt.to_numpy(), rng.normal(0.0, 0.2, size=n))
        live = np.ones_like(ret, dtype=bool)
        if not usa:
            live = np.arange(months.size)[:, None] >= rng.integers(0, 120, size=n)[None, :]
        if code == "ken":
            live[:, 10:] = False  # a country with too few factors for the space test
        returns[code] = pd.DataFrame(ret, index=months, columns=factors).where(live)
        markets[code] = mkt
    aggregates = {
        "developed": [
            c.lower() for c, d in zip(COUNTRIES, DEVELOPMENT, strict=True) if d == "developed"
        ],
        "emerging": ["bra", "ind", "zaf"],
        "frontier": ["ken", "zaf"],
        "world": [c.lower() for c in COUNTRIES],
    }
    months = _months("1990-01-31", "2025-12-31")
    for name, members in aggregates.items():
        returns[name] = (
            pd.concat([returns[c].reindex(months) for c in members]).groupby(level=0).mean()
        )
        markets[name] = (
            pd.concat([markets[c].reindex(months) for c in members]).groupby(level=0).mean()
        )
    return returns, markets


def _jkp_location_zip(
    *, config: Config, mirror: Path, location: str, selector: str, frame: pd.DataFrame
) -> None:
    name = sources.jkp_file_name(config, location=location, selector=selector)
    target = sources.jkp_location_target(config, location=location, selector=selector)
    _write(mirror_path(mirror, target.url), _zip_csv(name, frame))


def _jkp(*, config: Config, mirror: Path, rng: np.random.Generator, us_market: pd.Series) -> None:
    factors, clusters = _jkp_reference(config=config, mirror=mirror)
    returns, markets = _jkp_returns(
        factors=factors, clusters=clusters, rng=rng, us_market=us_market
    )
    availability: dict[str, list[str]] = {
        "all_countries": ["all_factors"],
        "all_regions": ["all_factors"],
    }
    regions = {"developed", "emerging", "frontier", "world"}
    for code, frame in returns.items():
        availability[code] = ["all_factors", "all_themes", "mkt", *factors]
        # Regional aggregates count contributing countries instead of stocks, as JKP publish them.
        counts = {"n_countries": 4} if code in regions else {"n_stocks": 60, "n_stocks_min": 25}
        long = (
            frame.rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="name", value_name="ret")
            .dropna(subset=["ret"])
        )
        factor_frame = pd.DataFrame(
            {
                "location": code,
                "name": long["name"],
                "freq": "monthly",
                "weighting": config.jkp.weighting,
                "direction": 1,
                **counts,
                "date": long["date"].dt.strftime("%Y-%m-%d"),
                "ret": long["ret"] / 100.0,
            }
        )
        _jkp_location_zip(
            config=config,
            mirror=mirror,
            location=code,
            selector=config.jkp.selector,
            frame=factor_frame,
        )
        mk = markets[code]
        market_frame = pd.DataFrame(
            {
                "location": code,
                "name": "market",
                "freq": "monthly",
                "weighting": config.jkp.weighting,
                "n_stocks": 500,
                "date": pd.DatetimeIndex(mk.index).strftime("%Y-%m-%d"),
                "ret": mk.to_numpy() / 100.0,
            }
        )
        _jkp_location_zip(
            config=config,
            mirror=mirror,
            location=code,
            selector=config.jkp.market_selector,
            frame=market_frame,
        )
    # A country whose files hold only the header row, as the smallest JKP countries do.
    availability[EMPTY_COUNTRY] = ["all_factors", "mkt"]
    header_only = pd.DataFrame(
        columns=[
            "location",
            "name",
            "freq",
            "weighting",
            "direction",
            "n_stocks",
            "n_stocks_min",
            "date",
            "ret",
        ]
    )
    for selector in (config.jkp.selector, config.jkp.market_selector):
        _jkp_location_zip(
            config=config,
            mirror=mirror,
            location=EMPTY_COUNTRY,
            selector=selector,
            frame=header_only,
        )
    availability["nzl"] = ["all_factors"]  # listed without a market file: must be skipped
    _write(
        mirror_path(mirror, sources.jkp_availability_target(config).url),
        json.dumps({"factors": availability}).encode(),
    )


# French -----------------------------------------------------------------------------------


def _french_csv(*, preamble: list[str], columns: list[str], frame: pd.DataFrame) -> bytes:
    lines = [*preamble, "", "," + ",".join(columns)]
    for month, row in frame.iterrows():
        values = ",".join(f"{v:8.2f}" for v in row.to_numpy())
        lines.append(f"{pd.Timestamp(str(month)).strftime('%Y%m')},{values}")
    annual = frame.groupby(pd.DatetimeIndex(frame.index).year).sum()
    lines += ["", " Annual Factors: January-December ", "," + ",".join(columns)]
    for year_value, row in annual.iterrows():
        lines.append(f"{year_value},{','.join(f'{v:8.2f}' for v in row.to_numpy())}")
    lines.append("")
    return "\r\n".join(lines).encode("latin-1")


def _french(*, config: Config, mirror: Path, rng: np.random.Generator, market: pd.Series) -> None:
    months = market.loc["1963-07-31":].index
    five = pd.DataFrame(
        {
            "Mkt-RF": market.loc[months].to_numpy(),
            **{c: rng.normal(0.25, 3.0, size=months.size) for c in ("SMB", "HML", "RMW", "CMA")},
            "RF": np.full(months.size, 0.35),
        },
        index=months,
    )
    mom_months = market.loc["1927-01-31":].index
    mom = pd.DataFrame({"Mom   ": rng.normal(0.6, 4.5, size=mom_months.size)}, index=mom_months)
    files = {
        config.french.five_factor_zip: _french_csv(
            preamble=[
                "This file was created using the 202509 CRSP database.",
                "The 1-month TBill rate data until 202405 are from Ibbotson Associates.",
            ],
            columns=list(five.columns),
            frame=five,
        ),
        config.french.momentum_zip: _french_csv(
            preamble=[
                "This file was created by CMPT_ME_PRIOR_RETURNS using the 202509 CRSP database.",
                "The monthly momentum factor, in percent.",
            ],
            columns=list(mom.columns),
            frame=mom,
        ),
    }
    for zip_name, data in files.items():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(zip_name.replace("_CSV.zip", ".csv"), data)
        _write(mirror_path(mirror, f"{config.sources.french_base}/{zip_name}"), buffer.getvalue())


def write_fixture(*, config: Config, mirror: Path, seed: int) -> FixtureTruth:
    rng = np.random.default_rng(seed)
    months = _months("1926-01-31", "2025-12-31")
    market = pd.Series(rng.normal(0.65, 5.3, size=months.size), index=months)
    _osap(config=config, mirror=mirror, rng=rng, market=market)
    _jkp(config=config, mirror=mirror, rng=rng, us_market=market.loc["1960-01-31":])
    _french(config=config, mirror=mirror, rng=rng, market=market)
    return TRUTH

"""Synthetic sources in the exact upstream formats, for offline runs, CI and the e2e test.

Planted structure, so the pipeline's answers can be checked:
- Each asset's daily variance follows a log-AR(1). Its implied index is the square root of
  (1 + premium) times the exact conditional expectation of the next 30 days' variance, so the
  ex-post premium ratio IV^2 / RV is (1 + premium) on average.
- VX futures converge to VIX at expiry from a contango curve; contracts traded before
  26 March 2007 are written at ten times the index, as the old contract was.
- Bitcoin minute bars carry a one-basis-point bid-ask bounce, so the minute-level realized
  variance sits above the 5-minute one and the realized kernel removes the gap.
- Each month's Deribit option trades carry marks at one flat volatility equal to DVOL at the
  snapshot, so the rebuilt chain replicates to DVOL squared, and print a planted half-spread
  from the mark (3% within 10% of the money, 6% beyond) that the rebuild must recover.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from cross_asset_vrp.config import Config
from cross_asset_vrp.data import cboe, crypto, prices
from cross_asset_vrp.data.download import mirror_path
from cross_asset_vrp.data.fetch import snapshot_days
from cross_asset_vrp.data.listing import page_url

START = pd.Timestamp("2005-01-03")
END = pd.Timestamp("2022-03-31")
BTC_START = pd.Timestamp("2020-01-01")
LISTED_FROM = {
    "VXN": "2009-09-14",
    "RVX": "2009-09-16",
    "GVZ": "2009-09-18",
    "OVX": "2009-09-18",
    "VXEEM": "2011-03-16",
    "VIX9D": "2011-01-03",
    "VIX3M": "2007-12-04",
}
ETF_FROM = {
    "SPY": "2005-01-03",
    "QQQ": "2005-01-03",
    "IWM": "2005-01-03",
    "GLD": "2005-01-03",
    "USO": "2006-04-10",
    "EEM": "2005-01-03",
}
SINGLE_VALUE = {"GVZ", "OVX"}


@dataclass(frozen=True)
class Planted:
    vol: float  # long-run annual volatility
    premium: float  # IV^2 = (1 + premium) E[RV]
    phi: float = 0.985
    eta: float = 0.12


PLANTED = {
    "spx": Planted(0.17, 0.35),
    "ndx": Planted(0.21, 0.30),
    "rut": Planted(0.23, 0.25),
    "gold": Planted(0.16, 0.15),
    "oil": Planted(0.34, 0.20),
    "em": Planted(0.21, 0.25),
    "btc": Planted(0.65, 0.20, phi=0.98, eta=0.10),
}
VX_CONTANGO = 1.5  # points at 30 days
VX_THETA = 19.0
VX_KAPPA = 4.0  # per year
BOUNCE = 1e-4


@dataclass
class Mirror:
    root: Path
    listed: dict[str, list[tuple[str, int, str]]] = field(default_factory=dict)

    def put(self, url: str, data: bytes) -> None:
        path = mirror_path(self.root, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    days = pd.date_range(start, end, freq="D")
    return pd.DatetimeIndex([d for d in days if cboe.is_business_day(d.date())])


def _log_variance(
    n: int, *, planted: Planted, periods: int, rng: np.random.Generator
) -> np.ndarray:
    mu = np.log(planted.vol**2 / periods)
    h = np.empty(n)
    h[0] = mu
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        h[t] = mu + planted.phi * (h[t - 1] - mu) + planted.eta * shocks[t]
    return h


def expected_sum(h: np.ndarray, *, steps: np.ndarray, planted: Planted, periods: int) -> np.ndarray:
    """E_t of the sum of exp(h) over the next `steps[t]` periods under the log-AR(1)."""
    mu = np.log(planted.vol**2 / periods)
    phi, eta = planted.phi, planted.eta
    out = np.zeros(h.size)
    for j in range(1, int(steps.max()) + 1):
        term = np.exp(mu + phi**j * (h - mu) + 0.5 * eta**2 * (1 - phi ** (2 * j)) / (1 - phi**2))
        out += np.where(steps >= j, term, 0.0)
    return out


def _steps_ahead(index: pd.DatetimeIndex, days: int) -> np.ndarray:
    values = index.to_numpy()
    ends = values + np.timedelta64(days, "D")
    return np.searchsorted(values, ends, side="right") - np.arange(1, values.size + 1)


def _index_csv(symbol: str, frame: pd.DataFrame) -> bytes:
    dates = pd.DatetimeIndex(frame.index).strftime("%m/%d/%Y")
    if symbol in SINGLE_VALUE:
        lines = [f"DATE,{symbol}"] + [
            f"{d},{v:.6f}" for d, v in zip(dates, frame["value"], strict=True)
        ]
    else:
        lines = ["DATE,OPEN,HIGH,LOW,CLOSE"] + [
            f"{d},{o:.6f},{h:.6f},{lo:.6f},{c:.6f}"
            for d, o, h, lo, c in zip(
                dates,
                frame["value"] * 1.01,
                frame["value"] * 1.04,
                frame["value"] * 0.97,
                frame["value"],
                strict=True,
            )
        ]
    return ("\n".join(lines) + "\n").encode()


def _etf_json(ticker: str, bars: pd.DataFrame) -> bytes:
    rows = []
    for day, (o, h, lo, c) in zip(
        pd.DatetimeIndex(bars.index), bars[["open", "high", "low", "close"]].to_numpy(), strict=True
    ):
        rows.append(
            {
                "date": f"{day:%Y-%m-%d}T00:00:00.000Z",
                "close": round(c * 1.02, 4),
                "high": round(h * 1.02, 4),
                "low": round(lo * 1.02, 4),
                "open": round(o * 1.02, 4),
                "volume": 1_000_000,
                "adjClose": c,
                "adjHigh": h,
                "adjLow": lo,
                "adjOpen": o,
                "adjVolume": 1_000_000,
                "divCash": 0.0,
                "splitFactor": 1.0,
            }
        )
    _ = ticker
    return json.dumps(rows).encode()


def _etf_bars(v: np.ndarray, *, rng: np.random.Generator, intraday_steps: int = 78) -> pd.DataFrame:
    """Daily bars whose overnight part carries 20% of the day's variance."""
    n = v.size
    on = rng.standard_normal(n) * np.sqrt(0.2 * v)
    steps = rng.standard_normal((n, intraday_steps)) * np.sqrt(0.8 * v / intraday_steps)[:, None]
    path = np.cumsum(steps, axis=1)
    log_close = np.cumsum(on + path[:, -1]) + np.log(100.0)
    log_open = log_close - path[:, -1]
    high = log_open + np.maximum(path.max(axis=1), 0.0)
    low = log_open + np.minimum(path.min(axis=1), 0.0)
    return pd.DataFrame(
        {
            "open": np.exp(log_open),
            "high": np.exp(high),
            "low": np.exp(low),
            "close": np.exp(log_close),
        }
    )


def _vx_files(config: Config, mirror: Mirror, vix: pd.Series) -> None:
    days = vix.index
    for expiry in cboe.expiries(config, last=END.date()):
        trade = pd.DatetimeIndex(
            days[
                (days <= pd.Timestamp(expiry))
                & (days >= pd.Timestamp(expiry) - pd.Timedelta(days=270))
            ]
        )
        if trade.empty:
            continue
        tau = (pd.Timestamp(expiry) - trade).days.to_numpy() / 365.0
        level = vix.loc[trade].to_numpy()
        settle = (
            level
            + (VX_THETA - level) * (1 - np.exp(-VX_KAPPA * tau))
            + VX_CONTANGO * np.sqrt(tau * 365 / 30)
        )
        settle[-1] = level[-1]  # final settlement at the index
        old = trade < pd.Timestamp(config.futures.rescale_before)
        settle = np.where(old, settle * config.futures.rescale_factor, settle)
        code = cboe.MONTH_CODES[expiry.month - 1]
        label = f"{code} ({expiry:%b %y})"
        rows = list(zip(trade, settle, strict=True))
        if cboe.in_new_files(config, expiry):
            # Like Cboe's, the per-contract files of the first expiries in the new layout start
            # partway through the contract's life; the archive holds the rest.
            begins = pd.Period(config.futures.new_files_from, freq="M").start_time
            mirror.put(
                cboe.futures_url(config, expiry),
                _vx_text([(d, x) for d, x in rows if d >= begins], label=label, archive=False),
            )
        if cboe.in_archive(config, expiry):
            mirror.put(cboe.archive_url(config, expiry), _vx_text(rows, label=label, archive=True))


def _vx_text(rows: list[tuple[pd.Timestamp, float]], *, label: str, archive: bool) -> bytes:
    header = "Trade Date,Futures,Open,High,Low,Close,Settle,Change,Total Volume,EFP,Open Interest"
    # The archive files open with a note above the header, and many rows end in a comma.
    lines = (
        ["CFE data is compiled for the convenience of site visitors", header]
        if archive
        else [header]
    )
    for n, (d, s) in enumerate(rows):
        stamp = f"{d:%m/%d/%Y}" if archive else f"{d:%Y-%m-%d}"
        tail = "," if archive and n % 3 == 0 else ""
        lines.append(
            f"{stamp},{label},{s:.4f},{s * 1.01:.4f},{s * 0.99:.4f},{s:.4f},{s:.4f},"
            f"0,100,0,1000{tail}"
        )
    return ("\n".join(lines) + "\n").encode()


def _cboe_and_etfs(config: Config, mirror: Mirror, rng: np.random.Generator) -> pd.Series:
    days = trading_days(START, END)
    steps = _steps_ahead(days, config.windows.variance_days)
    window = config.windows.variance_days / config.windows.days_per_year
    vix = pd.Series(dtype=float)
    for a in config.cboe_assets:
        planted = PLANTED[a.key]
        h = _log_variance(days.size, planted=planted, periods=252, rng=rng)
        expected = expected_sum(h, steps=steps, planted=planted, periods=252)
        iv = 100.0 * np.sqrt((1 + planted.premium) * expected / window)
        bars = _etf_bars(np.exp(h), rng=rng).set_axis(days)
        bars = bars.loc[bars.index >= pd.Timestamp(ETF_FROM[a.instrument])]
        mirror.put(prices.tiingo_url(config, a.instrument), _etf_json(a.instrument, bars))
        series = pd.Series(iv, index=days)
        listed = series.loc[series.index >= pd.Timestamp(LISTED_FROM.get(a.implied, "2005-01-03"))]
        mirror.put(
            cboe.index_url(config, a.implied), _index_csv(a.implied, listed.to_frame("value"))
        )
        if a.implied == "VIX":
            vix = series
    ratio = np.clip(0.9 + 0.25 * (vix / VX_THETA - 1.0), 0.7, 1.3)
    short = (vix * np.sqrt(ratio)).loc[lambda s: s.index >= pd.Timestamp(LISTED_FROM["VIX9D"])]
    long = (vix / np.sqrt(ratio)).loc[lambda s: s.index >= pd.Timestamp(LISTED_FROM["VIX3M"])]
    mirror.put(
        cboe.index_url(config, config.term_structure.short),
        _index_csv("VIX9D", short.to_frame("value")),
    )
    mirror.put(
        cboe.index_url(config, config.term_structure.long),
        _index_csv("VIX3M", long.to_frame("value")),
    )
    _vx_files(config, mirror, vix)
    return vix


def _fred(config: Config, mirror: Mirror, rng: np.random.Generator) -> None:
    days = trading_days(START, END)
    rate = np.clip(2.0 + np.cumsum(rng.normal(0, 0.02, days.size)), 0.01, 6.0)
    lines = [f"observation_date,{config.risk_free.series}"]
    for i, (d, r) in enumerate(zip(days, rate, strict=True)):
        lines.append(f"{d:%Y-%m-%d},{'.' if i % 97 == 5 else f'{r:.2f}'}")
    mirror.put(prices.fred_url(config), ("\n".join(lines) + "\n").encode())


def _zip(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def _btc(config: Config, mirror: Mirror, rng: np.random.Generator) -> tuple[pd.Series, pd.Series]:
    """Writes the minute files and the listing; returns daily DVOL and the daily close."""
    planted = PLANTED["btc"]
    days = pd.date_range(BTC_START, END, freq="D")
    h = _log_variance(days.size, planted=planted, periods=365, rng=rng)
    v = np.exp(h)
    steps = np.full(days.size, config.windows.variance_days)
    expected = expected_sum(h, steps=steps, planted=planted, periods=365)
    window = config.windows.variance_days / config.windows.days_per_year
    dvol = pd.Series(100.0 * np.sqrt((1 + planted.premium) * expected / window), index=days)
    minutes = rng.standard_normal((days.size, 1440)) * np.sqrt(v / 1440)[:, None]
    log_mid = np.log(8000.0) + np.cumsum(minutes.ravel())
    bounce = BOUNCE * np.where(rng.random(log_mid.size) < 0.5, -1.0, 1.0)
    close = np.exp(log_mid + bounce)
    open_ = np.concatenate([[close[0]], close[:-1]])
    wiggle = np.exp(
        np.abs(rng.standard_normal(close.size)) * 0.25 * np.sqrt(np.repeat(v, 1440) / 1440)
    )
    high = np.maximum(open_, close) * wiggle
    low = np.minimum(open_, close) / wiggle
    times = pd.date_range(BTC_START, periods=close.size, freq="min", tz="UTC")
    frame = pd.DataFrame({"time": times, "open": open_, "high": high, "low": low, "close": close})
    symbol = config.crypto.instrument
    folder = crypto.minute_folder(symbol)
    objects = []
    for month, part in frame.groupby(frame["time"].dt.strftime("%Y-%m")):
        ms = part["time"].dt.as_unit("ms").astype("int64").to_numpy()
        body = [",".join(crypto.KLINE_COLUMNS)] if month >= "2022-01" else []
        body += [
            f"{t},{o:.2f},{hi:.2f},{lo:.2f},{c:.2f},1.0,{t + 59_999},{c:.2f},10,0.5,{c / 2:.2f},0"
            for t, o, hi, lo, c in zip(
                ms, part["open"], part["high"], part["low"], part["close"], strict=True
            )
        ]
        name = f"{symbol}-1m-{month}"
        data = _zip(f"{name}.csv", "\n".join(body) + "\n")
        key = f"{folder}{name}.zip"
        mirror.put(f"{config.sources.vision_base}/{key}", data)
        objects.append((key, len(data), hashlib.md5(data, usedforsecurity=False).hexdigest()))
    xmlns = "http://s3.amazonaws.com/doc/2006-03-01/"
    body = [
        f'<?xml version="1.0" encoding="UTF-8"?>\n<ListBucketResult xmlns="{xmlns}">',
        f"<Name>data.binance.vision</Name><Prefix>{folder}</Prefix><Marker></Marker>",
        "<MaxKeys>1000</MaxKeys><Delimiter>/</Delimiter><IsTruncated>false</IsTruncated>",
    ]
    body += [
        f'<Contents><Key>{k}</Key><LastModified>2024-01-02T00:00:00.000Z</LastModified><ETag>"{m}"</ETag>'
        f"<Size>{s}</Size><StorageClass>STANDARD</StorageClass></Contents>"
        for k, s, m in objects
    ]
    body.append("</ListBucketResult>")
    mirror.put(
        page_url(config.sources.vision_list, prefix=folder, marker=""), "\n".join(body).encode()
    )
    daily_close = frame.set_index("time")["close"].resample("1D").last()
    return dvol, daily_close


def _dvol_pages(config: Config, mirror: Mirror, dvol: pd.Series, *, months: list[str]) -> None:
    start = int(pd.Timestamp(config.windows.first_date, tz="UTC").value // 1_000_000)
    end = int(
        pd.Period(months[-1], freq="M").end_time.tz_localize("UTC").ceil("D").value // 1_000_000
    )
    ms = pd.DatetimeIndex(dvol.index).tz_localize("UTC").as_unit("ms").asi8
    candles = [[int(t), v, v * 1.02, v * 0.98, v] for t, v in zip(ms, dvol.to_numpy(), strict=True)]
    size = 300  # pages of 300 candles, newest first, so the backward paging is exercised
    remaining = [c for c in candles if c[0] <= end]
    current = end
    while True:
        page = [c for c in remaining if c[0] <= current][-size:]
        older = [c for c in remaining if c[0] < page[0][0]] if page else []
        continuation = page[0][0] - 1 if older else None
        body = {"jsonrpc": "2.0", "id": 1, "result": {"data": page, "continuation": continuation}}
        mirror.put(
            crypto.dvol_url(config, start_ms=start, end_ms=current, resolution="1D"),
            json.dumps(body).encode(),
        )
        if continuation is None:
            break
        current = continuation


def _before(series: pd.Series, moment: pd.Timestamp) -> float:
    """The last value at or before the day before `moment` (the first value if none): the
    level a snapshot at the start of a day sees."""
    earlier = series.loc[series.index <= moment - pd.Timedelta(days=1)]
    return float(earlier.iloc[-1] if len(earlier) else series.iloc[0])


def _black(*, forward: float, strike: float, years: float, vol: float, kind: str) -> float:
    s = vol * np.sqrt(years)
    d1 = (np.log(forward / strike) + 0.5 * s * s) / s
    d2 = d1 - s
    if kind == "call":
        return float(forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return float(strike * norm.cdf(-d2) - forward * norm.cdf(-d1))


def _expiries(day: pd.Timestamp) -> list[pd.Timestamp]:
    fridays = pd.date_range(day + pd.Timedelta(days=1), day + pd.Timedelta(days=200), freq="W-FRI")
    weekly = list(fridays[:4])
    month_ends = [f for f in fridays if (f + pd.Timedelta(days=7)).month != f.month][:4]
    return sorted({(f + pd.Timedelta(hours=8)) for f in weekly + month_ends})


ATM_HALF_SPREAD = 0.03  # planted effective half-spreads, recovered by the rebuild
WING_HALF_SPREAD = 0.06
TICK = 0.0001


def trade_pages(
    config: Config, mirror: Mirror, *, day: pd.Timestamp, trades: list[dict[str, object]]
) -> None:
    """Serve `trades` the way the history server pages them: from `start_timestamp` on, at most
    `page_size` per page, the next page starting at the last timestamp received."""
    trades.sort(key=lambda t: int(str(t["timestamp"])))
    stamps = [int(str(t["timestamp"])) for t in trades]
    start = int((day - pd.Timedelta(hours=config.replication.window_hours)).value // 1_000_000)
    end = int(day.value // 1_000_000)
    size = config.replication.page_size
    while True:
        first = next((i for i, x in enumerate(stamps) if x >= start), len(stamps))
        page = trades[first : first + size]
        more = first + size < len(trades)
        body = {"jsonrpc": "2.0", "id": 1, "result": {"trades": page, "has_more": more}}
        mirror.put(crypto.trades_url(config, start_ms=start, end_ms=end), json.dumps(body).encode())
        last = int(str(page[-1]["timestamp"])) if page else start
        if not more or last <= start:
            return
        start = last


def _chains(
    config: Config,
    mirror: Mirror,
    *,
    dvol: pd.Series,
    close: pd.Series,
    days: list[pd.Timestamp],
    rng: np.random.Generator,
) -> None:
    """A day of option trades before each snapshot. Every mark is Black-76 at DVOL on a forward
    equal to the index, so the rebuilt chain is flat at DVOL; trades print a planted
    half-spread away from the mark on the taker's side."""
    window_ms = config.replication.window_hours * 3_600_000
    serial = 0
    for day in days:
        vol = _before(dvol, day.tz_localize(None)) / 100.0
        spot = _before(close, day)
        end = int(day.value // 1_000_000)
        trades: list[dict[str, object]] = []
        for expiry in _expiries(day):
            step = max(250.0, round(spot * 0.025 / 250) * 250)
            strikes = np.arange(max(step, spot * 0.2 // step * step), spot * 3.0, step)
            code = f"{expiry.day}{expiry:%b%y}".upper()
            for k in strikes:
                half = ATM_HALF_SPREAD if abs(np.log(k / spot)) <= 0.1 else WING_HALF_SPREAD
                for kind in ("call", "put"):
                    for at in np.sort(rng.choice(window_ms - 1, size=3, replace=False)):
                        stamp = end - window_ms + int(at)
                        moment = pd.Timestamp(stamp, unit="ms", tz="UTC")
                        years = (expiry - moment).total_seconds() / (365 * 86400)
                        mark = (
                            _black(forward=spot, strike=float(k), years=years, vol=vol, kind=kind)
                            / spot
                        )
                        if mark < TICK:
                            continue
                        side = "buy" if rng.random() < 0.5 else "sell"
                        price = mark * (1.0 + half if side == "buy" else 1.0 - half)
                        serial += 1
                        trades.append(
                            {
                                "trade_id": str(serial),
                                "instrument_name": f"BTC-{code}-{int(k)}-{kind[0].upper()}",
                                "timestamp": stamp,
                                "price": price,
                                "mark_price": mark,
                                "index_price": spot,
                                "direction": side,
                                "amount": 1.0,
                                "iv": vol * 100.0,
                            }
                        )
        trade_pages(config, mirror, day=day, trades=trades)


def _dvol_minutes(
    config: Config, mirror: Mirror, *, dvol: pd.Series, days: list[pd.Timestamp]
) -> None:
    for day in days:
        value = _before(dvol, day.tz_localize(None))
        start = int((day - pd.Timedelta(minutes=5)).value // 1_000_000)
        end = int((day + pd.Timedelta(minutes=1)).value // 1_000_000)
        candles = [[start + 60_000 * i, value, value, value, value] for i in range(7)]
        body = {"jsonrpc": "2.0", "id": 1, "result": {"data": candles, "continuation": None}}
        mirror.put(
            crypto.dvol_url(config, start_ms=start, end_ms=end, resolution="60"),
            json.dumps(body).encode(),
        )


def write_fixture(*, config: Config, mirror: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    m = Mirror(root=mirror)
    _cboe_and_etfs(config, m, rng)
    _fred(config, m, rng)
    dvol, close = _btc(config, m, rng)
    months = sorted({f"{d:%Y-%m}" for d in pd.date_range(BTC_START, END, freq="D")})
    _dvol_pages(config, m, dvol, months=months)
    days = snapshot_days(months)
    _chains(config, m, dvol=dvol, close=close, days=days, rng=rng)
    _dvol_minutes(config, m, dvol=dvol, days=days)
    _ = date

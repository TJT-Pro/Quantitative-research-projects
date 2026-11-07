"""Synthetic sources in the exact upstream formats, laid out as a mirror folder.

The market has a known structure: funding is an AR(1) in each contract's premium, so the
HAR forecast beats persistence; the rebuild symbols' prints are computed from their own
1-minute premium with the per-interval convention, so that convention reproduces every
print; one contract is delisted halfway; a 1000-unit contract maps to its spot pair; one
alt spikes 60% inside a bar in November 2022, liquidating leveraged shorts; USDC
depegs in March 2023; the live order books follow a planted spread-volume law. Every
response is written where `pfc fetch --mirror` looks for it.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from perp_funding_carry.config import Config
from perp_funding_carry.data import books, rest, vision
from perp_funding_carry.data.download import mirror_path
from perp_funding_carry.data.listing import page_url
from perp_funding_carry.models.funding import average_premium, funding_rate

START = pd.Timestamp("2022-01-01", tz="UTC")
END = pd.Timestamp("2024-01-01", tz="UTC")
HEADER_FROM = pd.Period("2023-01", "M")  # Vision added header rows; older files have none


@dataclass(frozen=True)
class FixtureContract:
    symbol: str
    spot: str
    multiplier: int
    price: float
    vol_daily: float
    premium_mean: float
    volume_daily: float
    end: pd.Timestamp = END
    interval_switch: tuple[str, int] | None = None  # from month, interval hours


CONTRACTS = (
    FixtureContract("BTCUSDT", "BTCUSDT", 1, 40000.0, 0.03, 0.00008, 8e9),
    FixtureContract("ETHUSDT", "ETHUSDT", 1, 2500.0, 0.04, 0.00009, 4e9),
    FixtureContract(
        "SOLUSDT", "SOLUSDT", 1, 90.0, 0.06, 0.00012, 6e8, interval_switch=("2023-07", 4)
    ),
    FixtureContract("1000PEPEUSDT", "PEPEUSDT", 1000, 0.002, 0.09, 0.0002, 2e8),
    FixtureContract(
        "OLDUSDT", "OLDUSDT", 1, 3.0, 0.07, 0.00025, 9e7, end=pd.Timestamp("2023-01-01", tz="UTC")
    ),
    *(
        FixtureContract(
            f"ALT{i}USDT",
            f"ALT{i}USDT",
            1,
            5.0 + i,
            0.04 + 0.004 * i,
            0.00004 * i,
            3e6 * (i + 1) ** 2,
        )
        for i in range(1, 7)
    ),
)
EXTRA_PERPS = ("BTCUSDC", "NOSPOTUSDT")  # filtered: not USDT-quoted / no spot pair
EXTRA_SPOT = ("BNBUSDT", "USDCUSDT")
SPIKE = ("ALT3USDT", pd.Timestamp("2022-11-09 12:00", tz="UTC"), 0.6)
# Planted spread law: perpetual half-spread = PERP_HALF_AT * (SPREAD_VOLUME / V) ** SPREAD_POWER,
# spot half-spread SPOT_RATIO times that, each with lognormal noise. The fit must recover it.
PERP_HALF_AT = 1.5e-4
SPREAD_VOLUME = 1e8
SPREAD_POWER = 0.4
SPOT_RATIO = 1.5
SPREAD_NOISE = 0.15


@dataclass
class Mirror:
    root: Path
    listed: dict[str, list[tuple[str, int, str]]] = field(default_factory=dict)  # folder -> objects

    def put(self, url: str, data: bytes) -> None:
        path = mirror_path(self.root, url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def put_vision(self, config: Config, key: str, data: bytes) -> None:
        self.put(f"{config.sources.vision_base}/{key}", data)
        folder = key.rsplit("/", 1)[0] + "/"
        self.listed.setdefault(folder, []).append(
            (key, len(data), hashlib.md5(data, usedforsecurity=False).hexdigest())
        )


def epoch_ms(values: pd.Series) -> pd.Series:
    """Milliseconds since the epoch, whatever the timestamps' stored resolution."""
    ms: pd.Series = values.dt.as_unit("ms").astype("int64")
    return ms


def _zip(name: str, text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, text)
    return buffer.getvalue()


def _kline_csv(frame: pd.DataFrame, *, header: bool) -> str:
    ms = epoch_ms(frame["open_time"])
    width = (
        int(
            (frame["open_time"].iloc[1] - frame["open_time"].iloc[0]) / pd.Timedelta(milliseconds=1)
        )
        if len(frame) > 1
        else 3_600_000
    )
    out = pd.DataFrame(
        {
            "open_time": ms,
            "open": frame["open"].round(8),
            "high": frame["high"].round(8),
            "low": frame["low"].round(8),
            "close": frame["close"].round(8),
            "volume": frame["volume"].round(4),
            "close_time": ms + width - 1,
            "quote_volume": frame["quote_volume"].round(4),
            "count": 100,
            "taker_buy_volume": (frame["volume"] / 2).round(4),
            "taker_buy_quote_volume": (frame["quote_volume"] / 2).round(4),
            "ignore": 0,
        }
    )
    return out.to_csv(index=False, header=header)


def _ohlc(close: pd.Series, *, rule: str, rng: np.random.Generator, wiggle: float) -> pd.DataFrame:
    """Aggregate hourly closes to `rule` bars with synthetic intrabar highs and lows."""
    grouped = close.resample(rule, label="left", closed="left")
    o = grouped.first()
    c = grouped.last()
    h = grouped.max() * np.exp(np.abs(rng.normal(0, wiggle, len(o))))
    lo = grouped.min() * np.exp(-np.abs(rng.normal(0, wiggle, len(o))))
    return pd.DataFrame(
        {
            "open_time": o.index,
            "open": o.to_numpy(),
            "high": np.maximum(h.to_numpy(), np.maximum(o.to_numpy(), c.to_numpy())),
            "low": np.minimum(lo.to_numpy(), np.minimum(o.to_numpy(), c.to_numpy())),
            "close": c.to_numpy(),
        }
    )


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Period]:
    return list(
        pd.period_range(
            start.tz_localize(None), (end - pd.Timedelta(seconds=1)).tz_localize(None), freq="M"
        )
    )


def _write_klines(
    config: Config, mirror: Mirror, *, folder: str, symbol: str, interval: str, frame: pd.DataFrame
) -> None:
    month = frame["open_time"].dt.tz_localize(None).dt.to_period("M")
    for m, g in frame.groupby(month):
        name = f"{symbol}-{interval}-{m}"
        key = vision.monthly_key(folder, symbol=symbol, interval=interval, month=str(m))
        mirror.put_vision(config, key, _zip(f"{name}.csv", _kline_csv(g, header=m >= HEADER_FROM)))


def _intervals(contract: FixtureContract, times: pd.DatetimeIndex) -> np.ndarray:
    hours = np.full(times.size, 8)
    if contract.interval_switch is not None:
        month, value = contract.interval_switch
        hours[times >= pd.Timestamp(month + "-01", tz="UTC")] = value
    return hours


@dataclass(frozen=True)
class Paths:
    hours: pd.DatetimeIndex
    spot: np.ndarray  # per contract unit, multiplier included
    premium: np.ndarray
    hourly_vol: float


def _paths(contract: FixtureContract, rng: np.random.Generator) -> Paths:
    hours = pd.date_range(START, contract.end, freq="1h", inclusive="left")
    n = hours.size
    vol = contract.vol_daily / np.sqrt(24)
    spot = np.exp(np.log(contract.price) + np.cumsum(rng.normal(0, vol, n)))
    premium = np.empty(n)
    premium[0] = contract.premium_mean
    shocks = rng.normal(0, 0.00012, n)
    for i in range(1, n):  # premium is an AR(1): funding is persistent but mean-reverting
        premium[i] = (
            contract.premium_mean + 0.995 * (premium[i - 1] - contract.premium_mean) + shocks[i]
        )
    if contract.symbol == SPIKE[0]:
        k = int(hours.searchsorted(SPIKE[1]))
        spot[k:] *= 1.0 + SPIKE[2] * np.exp(-np.arange(n - k) / 6.0)  # decays within hours
    return Paths(hours=hours, spot=spot, premium=premium, hourly_vol=vol)


def _write_price_bars(
    config: Config,
    mirror: Mirror,
    *,
    contract: FixtureContract,
    paths: Paths,
    rng: np.random.Generator,
) -> None:
    spot_series = pd.Series(paths.spot / contract.multiplier, index=paths.hours)
    mark_series = pd.Series(paths.spot * (1.0 + paths.premium), index=paths.hours)
    intervals = {config.universe.bar_interval}
    if contract.symbol in config.universe.core_symbols:
        intervals.add(config.universe.core_bar_interval)
    wiggle = paths.hourly_vol / 2
    for interval in sorted(intervals):
        bar_hours = int(interval[:-1])
        mark_bars = _ohlc(mark_series, rule=interval, rng=rng, wiggle=wiggle)
        perp_bars = mark_bars.copy()
        perp_bars["close"] = perp_bars["close"] * (1 + rng.normal(0, 0.0002, len(perp_bars)))
        quote = contract.volume_daily / 24 * bar_hours * np.exp(rng.normal(0, 0.3, len(perp_bars)))
        perp_bars["quote_volume"] = quote
        perp_bars["volume"] = quote / perp_bars["close"]
        mark_bars["quote_volume"] = 0.0
        mark_bars["volume"] = 0.0
        spot_bars = _ohlc(spot_series, rule=interval, rng=rng, wiggle=wiggle)
        spot_bars["quote_volume"] = quote * 0.6
        spot_bars["volume"] = spot_bars["quote_volume"] / spot_bars["close"]
        if contract.symbol == SPIKE[0] and interval == config.universe.bar_interval:
            hit = mark_bars["open_time"] == SPIKE[1].floor(f"{bar_hours}h")
            mark_bars.loc[hit, "high"] = mark_bars.loc[hit, "high"] * (1 + SPIKE[2] * 0.6)
        for folder, symbol, frame in (
            (vision.MARK, contract.symbol, mark_bars),
            (vision.PERP, contract.symbol, perp_bars),
            (vision.SPOT, contract.spot, spot_bars),
        ):
            _write_klines(
                config, mirror, folder=folder, symbol=symbol, interval=interval, frame=frame
            )


def _minute_premium(contract: FixtureContract, paths: Paths, rng: np.random.Generator) -> pd.Series:
    minutes = pd.date_range(START, contract.end, freq="1min", inclusive="left")
    hourly = pd.Series(paths.premium, index=paths.hours).reindex(minutes, method="ffill").to_numpy()
    return pd.Series(hourly + rng.normal(0, 0.00003, minutes.size), index=minutes)


def _funding_prints(
    config: Config, *, contract: FixtureContract, paths: Paths, minute_premium: pd.Series | None
) -> pd.DataFrame:
    """Prints from the premium at each contract's interval, with the per-interval convention."""
    if minute_premium is not None:
        index, values = minute_premium.index, minute_premium.to_numpy()
    else:
        index, values = paths.hours, paths.premium
    prints: list[dict[str, object]] = []
    t = START
    while True:
        step = int(_intervals(contract, pd.DatetimeIndex([t]))[0])
        lo, t = t, t + pd.Timedelta(hours=step)
        if t >= contract.end:
            return pd.DataFrame(prints)
        samples = values[index.searchsorted(lo) : index.searchsorted(t)]
        p = average_premium(
            np.asarray(samples, dtype=np.float64),
            linear=step not in config.rebuild.equal_weight_interval_hours,
        )
        rate = funding_rate(
            premium=p,
            interval_h=step,
            interest_8h=config.rebuild.interest_8h,
            clamp=config.rebuild.clamp,
            convention="per_interval",
        )
        prints.append({"time": t, "interval_h": step, "rate": float(np.clip(rate, -0.02, 0.02))})


def _write_funding(
    config: Config,
    mirror: Mirror,
    *,
    contract: FixtureContract,
    funding: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    month = funding["time"].dt.tz_localize(None).dt.to_period("M")
    for m, g in funding.groupby(month):
        ms = epoch_ms(g["time"]) + rng.integers(0, 3, len(g))  # settlement jitter
        text = pd.DataFrame(
            {
                "calc_time": ms,
                "funding_interval_hours": g["interval_h"],
                "last_funding_rate": g["rate"].map(lambda x: f"{x:.8f}"),
            }
        ).to_csv(index=False)
        key = vision.monthly_key(
            vision.FUNDING, symbol=contract.symbol, interval=None, month=str(m)
        )
        mirror.put_vision(config, key, _zip(f"{contract.symbol}-fundingRate-{m}.csv", text))


def _write_premium(
    config: Config,
    mirror: Mirror,
    *,
    contract: FixtureContract,
    minute_premium: pd.Series,
    months: set[pd.Period],
) -> None:
    values = minute_premium.to_numpy()
    frame = pd.DataFrame(
        {
            "open_time": minute_premium.index,
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": 0.0,
            "quote_volume": 0.0,
        }
    )
    month = frame["open_time"].dt.tz_localize(None).dt.to_period("M")
    _write_klines(
        config,
        mirror,
        folder=vision.PREMIUM,
        symbol=contract.symbol,
        interval="1m",
        frame=frame.loc[month.isin(months)],
    )


def _write_metrics(
    config: Config, mirror: Mirror, *, contract: FixtureContract, rng: np.random.Generator
) -> None:
    first = (contract.end - pd.Timedelta(days=config.universe.oi_window_days)).floor("1D")
    level = contract.volume_daily * 0.8
    for day in pd.date_range(first, contract.end, freq="1D", inclusive="left"):
        stamps = pd.date_range(day, day + pd.Timedelta(days=1), freq="5min", inclusive="left")
        oi_value = level * np.exp(np.cumsum(rng.normal(0, 0.002, stamps.size)))
        text = pd.DataFrame(
            {
                "create_time": stamps.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": contract.symbol,
                "sum_open_interest": oi_value / contract.price,
                "sum_open_interest_value": oi_value,
                "count_toptrader_long_short_ratio": 1.1,
                "sum_toptrader_long_short_ratio": 1.2,
                "count_long_short_ratio": 1.05,
                "sum_taker_long_short_vol_ratio": 0.98,
            }
        ).to_csv(index=False)
        stamp = day.strftime("%Y-%m-%d")
        mirror.put_vision(
            config,
            vision.metrics_key(contract.symbol, stamp),
            _zip(f"{contract.symbol}-metrics-{stamp}.csv", text),
        )


def _simulate_contract(
    config: Config,
    mirror: Mirror,
    *,
    contract: FixtureContract,
    rng: np.random.Generator,
    rebuild_months: set[pd.Period],
) -> pd.DataFrame:
    paths = _paths(contract, rng)
    _write_price_bars(config, mirror, contract=contract, paths=paths, rng=rng)
    minutes = (
        _minute_premium(contract, paths, rng) if contract.symbol in config.rebuild.symbols else None
    )
    funding = _funding_prints(config, contract=contract, paths=paths, minute_premium=minutes)
    _write_funding(config, mirror, contract=contract, funding=funding, rng=rng)
    if minutes is not None:
        _write_premium(
            config, mirror, contract=contract, minute_premium=minutes, months=rebuild_months
        )
    if contract.symbol in config.universe.core_symbols:
        _write_metrics(config, mirror, contract=contract, rng=rng)
    return funding.assign(symbol=contract.symbol)


def _stablecoin(config: Config, mirror: Mirror, rng: np.random.Generator) -> None:
    hours = pd.date_range(START, END, freq="1h", inclusive="left")
    price = 1.0 + rng.normal(0, 0.0003, hours.size)
    depeg = (hours >= pd.Timestamp("2023-03-10", tz="UTC")) & (
        hours < pd.Timestamp("2023-03-14", tz="UTC")
    )
    price[depeg] -= 0.06  # USDC trades at 0.94 USDT for four days
    bars = _ohlc(
        pd.Series(price, index=hours), rule=config.universe.bar_interval, rng=rng, wiggle=0.0002
    )
    bars["quote_volume"] = 1e8
    bars["volume"] = 1e8
    _write_klines(
        config,
        mirror,
        folder=vision.SPOT,
        symbol=config.universe.stablecoin_pair,
        interval=config.universe.bar_interval,
        frame=bars,
    )


def _write_listings(config: Config, mirror: Mirror) -> None:
    base = config.sources.vision_list
    xmlns = "http://s3.amazonaws.com/doc/2006-03-01/"

    def page(
        prefix: str,
        *,
        prefixes: list[str],
        objects: list[tuple[str, int, str]],
        marker: str,
        truncated: bool,
        next_marker: str | None,
    ) -> None:
        body = [
            f'<?xml version="1.0" encoding="UTF-8"?>\n<ListBucketResult xmlns="{xmlns}">',
            "<Name>data.binance.vision</Name>",
            f"<Prefix>{prefix}</Prefix>",
            f"<Marker>{marker}</Marker>",
        ]
        if next_marker:
            body.append(f"<NextMarker>{next_marker}</NextMarker>")
        body += [
            "<MaxKeys>1000</MaxKeys>",
            "<Delimiter>/</Delimiter>",
            f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>",
        ]
        for key, size, md5 in objects:
            body.append(
                f"<Contents><Key>{key}</Key><LastModified>2024-01-02T00:00:00.000Z</LastModified>"
                f'<ETag>"{md5}"</ETag><Size>{size}</Size><StorageClass>STANDARD</StorageClass></Contents>'
            )
        body += [f"<CommonPrefixes><Prefix>{p}</Prefix></CommonPrefixes>" for p in prefixes]
        body.append("</ListBucketResult>")
        mirror.put(page_url(base, prefix=prefix, marker=marker), "\n".join(body).encode())

    def paged(prefix: str, items: list[str], *, size: int, use_next_marker: bool) -> None:
        chunks = [items[i : i + size] for i in range(0, len(items), size)] or [[]]
        marker = ""
        for n, chunk in enumerate(chunks):
            last = n == len(chunks) - 1
            page(
                prefix,
                prefixes=chunk,
                objects=[],
                marker=marker,
                truncated=not last,
                next_marker=chunk[-1] if (use_next_marker and not last) else None,
            )
            marker = chunk[-1] if chunk else ""

    perps = sorted({c.symbol for c in CONTRACTS} | set(EXTRA_PERPS))
    paged(
        vision.FUNDING + "/",
        [f"{vision.FUNDING}/{s}/" for s in perps],
        size=4,
        use_next_marker=True,
    )
    spots = sorted({c.spot for c in CONTRACTS} | set(EXTRA_SPOT))
    paged(vision.SPOT + "/", [f"{vision.SPOT}/{s}/" for s in spots], size=5, use_next_marker=False)
    # Folders without objects still answer with an empty listing.
    for contract in CONTRACTS:
        for _, folder, _ in vision.groups_for(
            config, vision.Contract(contract.symbol, contract.spot, contract.multiplier)
        ):
            mirror.listed.setdefault(folder, [])
    mirror.listed.setdefault(
        f"{vision.SPOT}/{config.universe.stablecoin_pair}/{config.universe.bar_interval}/", []
    )
    for folder, unsorted in mirror.listed.items():
        objects = sorted(unsorted)
        if len(objects) > 400:  # long folders page without NextMarker: the marker is the last key
            for n in range(0, len(objects), 400):
                chunk = objects[n : n + 400]
                marker = objects[n - 1][0] if n else ""
                page(
                    folder,
                    prefixes=[],
                    objects=chunk,
                    marker=marker,
                    truncated=n + 400 < len(objects),
                    next_marker=None,
                )
        else:
            page(folder, prefixes=[], objects=objects, marker="", truncated=False, next_marker=None)


def _rest(
    config: Config, mirror: Mirror, *, funding: pd.DataFrame, rng: np.random.Generator
) -> None:
    info = [
        {
            "symbol": "SOLUSDT",
            "adjustedFundingRateCap": "0.02000000",
            "adjustedFundingRateFloor": "-0.02000000",
            "fundingIntervalHours": 4,
            "disclaimer": False,
        }
    ]
    mirror.put(rest.funding_info_url(config), json.dumps(info).encode())
    for symbol in config.universe.core_symbols:
        f = funding.loc[funding["symbol"] == symbol].sort_values("time")
        ms = epoch_ms(f["time"]).to_numpy()
        bybit = f["rate"].to_numpy() + rng.normal(0, 0.00002, len(f))
        # Bybit pages backward from the newest print, 200 per page, newest first.
        end: int | None = None
        remaining = np.arange(len(f))
        while True:
            chosen = remaining if end is None else remaining[ms[remaining] <= end]
            page = chosen[-200:][::-1]
            body = {
                "retCode": 0,
                "retMsg": "OK",
                "result": {
                    "category": "linear",
                    "list": [
                        {
                            "symbol": symbol,
                            "fundingRate": f"{bybit[i]:.8f}",
                            "fundingRateTimestamp": str(ms[i]),
                        }
                        for i in page
                    ],
                },
                "retExtInfo": {},
                "time": int(ms[-1]),
            }
            mirror.put(
                rest.bybit_page_url(config, symbol=symbol, end=end), json.dumps(body).encode()
            )
            if page.size == 0:
                break
            end = int(ms[page].min()) - 1
        # OKX keeps the last 93 days, 100 per page, newest first, paged with `after`.
        instrument = rest.okx_instrument(config, symbol)
        keep = ms >= ms[-1] - 93 * 86_400_000
        okx_ms = ms[keep]
        okx = f["rate"].to_numpy()[keep] + rng.normal(0, 0.00003, keep.sum())
        after: int | None = None
        while True:
            idx = np.arange(okx_ms.size) if after is None else np.nonzero(okx_ms < after)[0]
            page = idx[-100:][::-1]
            body = {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "instType": "SWAP",
                        "instId": instrument,
                        "fundingRate": f"{okx[i]:.8f}",
                        "realizedRate": f"{okx[i]:.8f}",
                        "fundingTime": str(okx_ms[i]),
                        "method": "current_period",
                        "formulaType": "withRate",
                    }
                    for i in page
                ],
            }
            mirror.put(
                rest.okx_page_url(config, instrument=instrument, after=after),
                json.dumps(body).encode(),
            )
            if page.size == 0:
                break
            after = int(okx_ms[page].min())


def _books(config: Config, mirror: Mirror, rng: np.random.Generator) -> None:
    """Live books: every contract still trading, both legs, plus pairs the model must ignore."""
    live = [c for c in CONTRACTS if c.end >= END]
    rows: dict[str, list[dict[str, str]]] = {"spot": [], "perp": []}
    tickers: dict[str, list[dict[str, str]]] = {"spot": [], "perp": []}
    for c in live:
        perp_half = PERP_HALF_AT * (SPREAD_VOLUME / c.volume_daily) ** SPREAD_POWER
        for market, symbol, price, half in (
            ("perp", c.symbol, c.price * c.multiplier, perp_half),
            ("spot", c.spot, c.price, perp_half * SPOT_RATIO),
        ):
            h = half * float(np.exp(rng.normal(0.0, SPREAD_NOISE)))
            rows[market].append(
                {
                    "symbol": symbol,
                    "bidPrice": f"{price * (1 - h):.10g}",
                    "bidQty": "10",
                    "askPrice": f"{price * (1 + h):.10g}",
                    "askQty": "10",
                }
            )
            volume = c.volume_daily * (1.0 if market == "perp" else 0.6)
            tickers[market].append({"symbol": symbol, "quoteVolume": f"{volume:.2f}"})
    # Ignored: another quote currency, and a crossed book.
    rows["spot"].append(
        {"symbol": "ETHBTC", "bidPrice": "0.05", "bidQty": "1", "askPrice": "0.0501", "askQty": "1"}
    )
    tickers["spot"].append({"symbol": "ETHBTC", "quoteVolume": "100"})
    rows["perp"].append(
        {"symbol": "NOSPOTUSDT", "bidPrice": "2", "bidQty": "1", "askPrice": "1", "askQty": "1"}
    )
    tickers["perp"].append({"symbol": "NOSPOTUSDT", "quoteVolume": "100"})
    for market, (book_url, ticker_url) in books.endpoints(config).items():
        mirror.put(book_url, json.dumps(rows[market]).encode())
        mirror.put(ticker_url, json.dumps(tickers[market]).encode())


def write_fixture(*, config: Config, mirror: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    m = Mirror(root=mirror)
    last = _months(START, END)
    rebuild_months = set(last[-config.rebuild.months :])
    frames = [
        _simulate_contract(config, m, contract=c, rng=rng, rebuild_months=rebuild_months)
        for c in CONTRACTS
    ]
    _stablecoin(config, m, rng)
    _write_listings(config, m)
    _rest(config, m, funding=pd.concat(frames, ignore_index=True), rng=rng)
    _books(config, m, rng)

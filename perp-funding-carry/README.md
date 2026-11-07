# What funding carry actually pays

Long spot, short perpetual, rebalanced into the 10 contracts with the highest forecast funding at 2x: -31.0% a year from 2020-07-01 to 2026-10-31 after fees, spreads, basis and margin, against 12.2% from funding alone. The Sharpe ratio is -12.51 (20 liquidations), and its deflated Sharpe ratio across 72 variants is 0.00. Measured against equity each year, funding paid 11.5%, fees took 53.4%, spreads 10.1%, and basis, liquidations and margin 14.9%. The same trade in BTC alone returned 5.4% a year.
![The headline portfolio: funding alone, after fees and spreads, and net](reports/figures/headline.png)

| Question | Answer | Detail |
| --- | --- | --- |
| Can Binance funding be rebuilt from its premium index? | 100.0% of 3255 prints within 1 bp | both conventions (only 8-hour prints), BTCUSDT, ETHUSDT, SOLUSDT, 2025-11-01 to 2026-10-31; median error 0.03 bp |
| Does the carry portfolio beat BTC and ETH alone? | -31.0% a year, Sharpe -12.51 ± 3.56 | BTC + ETH carry: 5.6%, Sharpe 8.07; max drawdown -92.1% |
| Can funding be forecast? | 1: 18%; 3: 33%; 21: 66% | Out-of-sample MSE reduction against persistence, by horizon in 8-hour periods |

## Question, data, method

The long-spot, short-perpetual trade is sold as a riskless yield. This project measures what it earns after fees, margin, liquidation and basis moves, whether funding can be forecast, and whether a cross-sectional carry portfolio beats BTC and ETH alone. Binance Vision supplies 477 USDT perpetuals with a spot pair, 118 of them since delisted, with funding, mark and spot prices from 2020-01-01 to 2026-10-31. Each funding print is first rebuilt from the 1-minute premium index. The trade is then simulated bar by bar, with isolated margin checked against mark-price highs and three margin paths. Funding forecasts are fitted walk-forward and scored against persistence. Counterparty failure and stablecoin depeg risk are priced as scenarios, since price data cannot show them.

## Supporting exhibits

![Funding of the median contract each month](reports/figures/cross_section.png)

Across 17024 contract-months the median paid 6.8% a year, and 27% of contract-months had negative funding, which the short pays. The median contract was negative in 9% of 82 months; negative funding was most widespread in 2022-05, when 80% of contracts paid longs.

![Annualized funding by contract and month](reports/figures/heatmap.png)

The same funding contract by contract, for the 30 contracts with the longest history.

![Forecast skill against persistence](reports/figures/forecast.png)

Out-of-sample skill over 477 contracts from 2020-07-01. Positive bars beat the last print.

![Drawdown with the stress episodes labelled](reports/figures/drawdown.png)

The worst drawdown is -92.1%, from 2021-04-17 to 2026-10-30, not yet recovered.

<details>
<summary>More exhibits: rebuild accuracy, liquidations, every variant, venues, counterparty risk, benchmarks</summary>

![Funding rebuild accuracy](reports/figures/rebuild.png)

The two readings of Binance's formula agree for 8-hour contracts and differ for shorter ones. Every print in this window is 8-hourly, so the window cannot tell them apart. Caps and floors can bind on the 0 prints of 0.1% or more, where n/a are reproduced (none in this sample).

![Liquidations by leverage](reports/figures/liquidations.png)

With no top-ups, liquidations at each leverage over the full sample, checked on 1h mark-price highs.

![Every variant tried](reports/figures/variants.png)

72 variants: 3 forecast models, top 5, 10 and 20, 2 liquidity filters and 4 leverages. 0% have a positive return.

![Funding by venue](reports/figures/venues.png)

Funding on dates every venue covers. OKX's public history is about three months deep, so its prints accumulate in a local store with each run.

![Counterparty risk](reports/figures/counterparty.png)

At the prior hazards (Binance 1.0%, Bybit 1.5%, OKX 1.5% a year) and 50% loss given failure, the split that minimizes 99% expected shortfall is Binance 33%, Bybit 33%, OKX 33%, with a shortfall of 17.5% against 50.0% with everything on the safest venue. Expected loss is 0.67% a year, which leaves -31.7% of net carry. Carry is gone once hazards reach 0.2 times the prior.
![Benchmarks](reports/figures/benchmarks.png)

</details>

## Limitations, and the workaround built for each

- **Exchange failure cannot be backtested from prices.** A scenario layer gives each venue an annual failure hazard (a prior swept from 0.25 to 4 times, with FTX in November 2022 as the reference event) and a loss given failure. It then picks the collateral split that minimizes expected shortfall. Carry is reported net of the expected loss.
- **USDT is not USD.** Equity is marked to USD through USDCUSDT at every bar: -31.0% a year in USD against -31.0% in USDT.- **OKX history is about 93 days deep.** Every run appends OKX prints to `data/collected/okx_funding.csv`, which is kept under version control. Running and committing at least once every three months leaves no gaps in the store. Cross-venue figures use only dates all venues cover, and the dates are stated with them.
- **Spreads cannot be read from price bars, and historical quotes are not public for every pair.** `pfc snapshot` measures the best bid and ask of every spot and perpetual pair, each run adds a snapshot to `data/collected/book_snapshots.csv`, and half-spreads are fitted to volume across contracts (perpetual slope -0.34, spot -0.44, from 1 snapshot). Past bars get the half-spread their volume implies. Because that assumes the relation held in the past, the headline is rerun with all costs at 0.5, 1.0, 2.0 times: 0.5x -5.3%, 1.0x -31.0%, 2.0x -63.4% a year.
- **Binance samples the premium every 5 seconds; the public record is 1-minute klines.** The rebuild uses each minute's close, and both the share within 1 bp and the error distribution are reported.
- **Leverage brackets need a signed request.** Tier-1 maintenance rates are kept in the config with their source, and the positions stay inside tier 1 at the capacity reported: the median rebalance could absorb 2,290,733 USD at 1% of daily volume.
- **Unlimited margin is not a tradable path.** It is shown as the frictionless bound. The headline uses top-ups from a reserve of 20%, and the no-top-up path is reported alongside.

## Reproduce

Windows PowerShell, one command per line:

```powershell
cd perp-funding-carry
uv sync
uv run pfc results
uv run pfc check
uv run pfc check-provenance
```

The first run lists the Binance Vision bucket and downloads every monthly file for each contract: funding, mark price, perpetual and spot klines, the 1-minute premium index for the rebuild symbols and daily open interest for BTC and ETH. If the connection drops, run it again: files whose MD5 matches the bucket listing are skipped. `uv run pfc results --skip-paper` skips the PDF, and `uv run pfc --help` lists each stage. Without network access, run `uv run pfc fixture mirror` and then `uv run pfc results -o configs/ci.yaml --mirror mirror` to run the whole pipeline on synthetic data.

There is no scheduled refresh workflow. GitHub's hosted runners are in the United States, and the Binance and Bybit APIs refuse requests from there, so a scheduled run would fail every time. Refresh from your own machine instead: `uv run pfc results` downloads only what changed, adds an OKX batch and an order-book snapshot, and the run must happen at least once every three months to keep the OKX store gap-free. Commit `data/collected/` with the results.

## Layout

| Path | Contents |
| --- | --- |
| [`configs/research.yaml`](configs/research.yaml) | Every research choice, fee and margin rate, each with its reason |
| [`src/perp_funding_carry/data`](src/perp_funding_carry/data) | Bucket listing, resumable downloads, REST collectors, parsers, synthetic fixture |
| [`src/perp_funding_carry/models`](src/perp_funding_carry/models) | Funding rebuild, liquidation price, carry and portfolio simulators, forecasts, statistics, counterparty split |
| [`src/perp_funding_carry/evaluation`](src/perp_funding_carry/evaluation) | Bar preparation and every analysis |
| [`reports/`](reports) | Figures, tables and `results.json` |
| [`paper/`](paper) | Full write-up (Quarto, Typst PDF and HTML) |

## References

- Abdi, F. and Ranaldo, A. (2017). A simple estimation of bid-ask spreads from daily close, high, and low prices. *Review of Financial Studies* 30(12), 4437-4480.
- Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management* 40(5), 94-107.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics* 7(2), 174-196.
- Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics* 13(3), 253-263.
- Duan, N. (1983). Smearing estimate: a nonparametric retransformation method. *Journal of the American Statistical Association* 78(383), 605-610.
- Lazarus, E., Lewis, D. J., Stock, J. H. and Watson, M. W. (2018). HAR inference: recommendations for practice. *Journal of Business and Economic Statistics* 36(4), 541-559.
- Rockafellar, R. T. and Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk* 2(3), 21-41.

Data: Binance Vision, the Binance, Bybit and OKX public APIs. Fees from the Binance fee schedule, retrieved 2026-11-04. This project is not affiliated with any exchange.

<sub>Generated 2026-11-04 11:44 UTC from configuration `04958e0f41bd` and inputs `7034dd1cab00` (official).</sub>

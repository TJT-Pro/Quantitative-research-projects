{% set e = r.estimates %}
{% set s = r.settings %}
{% set h = r.headline %}
{% set prem = e.premium %}
{% set fut = e.futures %}
{% set rep = e.replication %}
{% set cal = e.calibration %}
{% set names = {"close_to_close": "squared close-to-close returns", "parkinson": "Parkinson", "garman_klass": "Garman-Klass", "rogers_satchell": "Rogers-Satchell", "yang_zhang": "Yang-Zhang"} %}
{% set sizing = {"unconditional": "Unconditional", "premium": "Sized by the forecast premium", "slope": "Sized by the " ~ s.term_short ~ " / " ~ s.term_long ~ " slope"} %}
# {{ r.title }}

{% if r.provenance != "official" %}
> **These numbers come from the synthetic fixture, not from market data.** They check that the pipeline runs and recovers a planted truth, and they say nothing about real markets. Run `uv run vrp results` to build them from the sources.

{% endif %}
Options imply more variance than markets go on to deliver in {{ h.positive_premium_assets | length }} of {{ prem | length }} markets. Over the next {{ s.variance_days }} days, implied volatility exceeded the volatility that followed by {{ prem[h.largest_gap].vol_gap_points | num(1) }} points on average in {{ prem[h.largest_gap].label | inline }}, the widest gap, and by {{ prem[h.smallest_gap].vol_gap_points | num(1) }} in {{ prem[h.smallest_gap].label | inline }}, the narrowest. A short position in {{ s.target_days }}-day VIX futures {% set u = fut.strategies.unconditional.common %}returned {{ u.annual_return | pct(1) }} a year compounded from {{ u.first }} to {{ u.last }}. Its arithmetic mean was {{ u.annual_mean | pct(1) }} a year at {{ u.vol | pct(0) }} volatility, so the Sharpe ratio is {{ u.sharpe | num }} while the compounded return is {{ "negative" if u.annual_return < 0 else "positive" }}; skewness was {{ u.skew | num }} and the worst drawdown {{ u.max_drawdown | pct(1) }}. {% for k in h.ruined_sizing %}{{ sizing[k] }}, the position lost {{ (-fut.strategies[k].ruin_day_return) | pct(0) }} of the account's equity on {{ fut.strategies[k].common.ruined_on }} and wiped it out.{% if not loop.last %} {% endif %}{% endfor %}{% if h.ruined_sizing %} {% endif %}{% if h.best_sizing and h.best_sizing != "unconditional" %}{% set b = fut.strategies[h.best_sizing].common %}Sizing it {{ sizing[h.best_sizing] | inline | replace("sized ", "", 1) }} raised the Sharpe ratio to {{ b.sharpe | num }} on the same days.{% else %}No {{ "surviving " if h.ruined_sizing else "" }}sizing rule beat the unconditional position on the same days.{% endif %}


| Market | Implied index | Options on | Sample | Implied minus realized vol, points | Implied over realized variance | Days with a positive premium | HAR forecast against persistence |
| --- | --- | --- | --- | --- | --- | --- | --- |
{% for key, p in prem.items() %}
| {{ p.label }} | {{ p.implied }} | {{ p.options_on }} | {{ p.first }} to {{ p.last }} | {{ p.vol_gap_points | signed(1) }} | {{ p.ex_post_ratio | num(2) }} | {{ p.ex_post_share_positive | pct }} | {{ p.har_r2_vs_persistence | pct(0) }} |
{% endfor %}

The premium is measured over {{ s.variance_days }} calendar days, the window every index here quotes. "Against persistence" is the out-of-sample cut in squared forecast error from the HAR model over the realized variance of the past {{ s.variance_days }} days.

![Ex-ante variance risk premium by market](reports/figures/premium.png)

## Question, data, method

Selling volatility earns a premium in calm years and gives it back in crashes. This project measures that premium in {{ s.assets | length }} markets, tests what predicts it, and asks whether sizing a short volatility position by a forecast of the premium beats selling it unconditionally once the crashes are included.

Each implied volatility index is matched to the instrument its options are written on: {% for key, p in prem.items() %}{{ p.implied }} with {{ p.instrument }}{{ "; " if not loop.last else "." }}{% endfor %} Daily ETF prices come from Tiingo, adjusted for dividends and splits; Cboe publishes the indices and the VX futures; bitcoin's minute bars come from Binance Vision, DVOL from Deribit and the option chains are rebuilt from Deribit's public trade history. Each index is squared and scaled to a {{ s.variance_days }}-day variance, and realized variance is summed over the same {{ s.variance_days }} calendar days, overnight moves included. The ex-ante premium replaces realized variance with a HAR forecast (Corsi 2009) refit each month on data that had already settled.

The range estimators were calibrated on bitcoin first, against realized variance from {{ s.btc_sample_minutes }}-minute returns. {{ names[cal.chosen] }} had the smallest error of the three intraday estimators, so it measures the open-to-close part of each ETF day in the HAR forecasts, with the overnight move added. The forecast target stays the unbiased sum of squared daily returns.

<details>
<summary>More exhibits: estimator calibration, the Heston check, VIX futures, bitcoin variance swaps, return predictability</summary>

![Range estimators against minute data](reports/figures/calibration.png)

Over {{ cal.days }} bitcoin days from {{ cal.first }} to {{ cal.last }}: {% for row in cal.rows if row.level == "day" %}{{ names[row.estimator] }} {{ row.bias | signed(3) }}{{ ", " if not loop.last }}{% endfor %} (bias of the daily estimate). The realized kernel sits at {{ cal.kernel_over_rv5 | num(3) }} times {{ s.btc_sample_minutes }}-minute variance and 1-minute variance at {{ cal.rv1_over_rv5 | num(3) }} times, which measures the microstructure noise at the finest sampling.

![Estimator bias on Heston paths](reports/figures/heston.png)

![Short VIX futures](reports/figures/futures.png)

| Strategy, {{ fut.common_start }} on | Annual return, compounded | Sharpe | Skewness | Expected shortfall, worst 5% of days | Maximum drawdown | Days from trough to recovery |
| --- | --- | --- | --- | --- | --- | --- |
{% for name, st in fut.strategies.items() %}
| {{ sizing[name] }} | {{ st.common.annual_return | pct(1) }} | {{ st.common.sharpe | num }} ± {{ (1.96 * st.common.sharpe_se) | num }} | {{ st.common.skew | num }} | {{ st.common.expected_shortfall_5 | pct(1) }} | {{ st.common.max_drawdown | pct(1) }} | {{ ("account lost on " ~ st.common.ruined_on) if st.common.ruined_on else (st.common.recovery_days | num(0) if st.common.recovery_days is not none else "not recovered") }} |
{% endfor %}

| Episode | {% for name in fut.strategies %}{{ sizing[name] }} | {% endfor %}

| --- | {% for name in fut.strategies %}--- | {% endfor %}

{% for ep in fut.episodes %}
| {{ ep.name }} ({{ ep.start }} to {{ ep.end }}) | {% for name in fut.strategies %}{{ ep[name] | pct(1) if ep[name] is not none else "outside the sample" }} | {% endfor %}

{% endfor %}

{% if rep.months %}
![Bitcoin variance swaps](reports/figures/replication.png)

{% if rep.gap_months %}On {{ rep.gap_months }} monthly snapshots from {{ rep.gap_first }}, the replicated {{ s.variance_days }}-day volatility sat {{ rep.gap_mean | signed(2) }} points from DVOL on average ({{ rep.gap_mean_abs | num(2) }} in absolute terms). {% endif %}{% if rep.short_mean is defined %}A {{ s.variance_days }}-day variance swap sold at the bid strip earned {{ rep.short_mean | num(2) }} volatility points per unit of vega on average over {{ rep.settled }} months, positive in {{ rep.short_hit_rate | pct }} of them; the worst month, {{ rep.short_worst_month }}, lost {{ (-rep.short_worst) | num(1) }}.{% endif %}

{% endif %}
![Return predictability](reports/figures/predictability.png)

</details>

## Limitations, and the workaround built for each

- **The ETF implied indices are not tradable variance swaps, so their premiums are measurements, not returns.** For bitcoin the {{ s.variance_days }}-day variance swap rate is replicated from the Deribit chain itself (Carr and Madan 1998, in the discrete form of the Cboe method) at the bid, the mid and the ask, and a short position is priced at the bid each month: one premium becomes a strategy return with real costs, next to the VIX futures test.
- **There are no free daily option chains for the ETFs.** The replicated bitcoin rate is checked against DVOL on the same minute{% if rep.gap_mean_abs is defined %} ({{ rep.gap_mean_abs | num(2) }} points apart on average){% endif %}, which bounds how far a published index can sit from a rate built from the options themselves.
- **ETF variance can only be measured from daily bars.** The range estimators are calibrated on bitcoin, where minute data give the truth, and checked first on Heston paths whose integrated variance is known; the discrete-sampling bias they show follows Broadie, Glasserman and Kou (1997).
- **VX traded on ten times the index before {{ s.rescale_before }}.** Those settles are divided by {{ s.rescale_factor | num(0) }}, and the build checks the nearest contract against VIX year by year and stops if the ratio leaves its band.
- **No free source keeps historical Deribit order books.** Deribit's history server keeps every trade, each with the option's mark price and the index at that instant, so each month's chain is rebuilt from the {{ s.window_hours }} hours of trades before 00:00 UTC on the 1st: put-call parity on the marks gives each expiry's forward, each option's last mark gives its implied volatility{% if rep.iv_gap_median_abs is defined %} (a median {{ rep.iv_gap_median_abs | num(2) }} points from the volatility Deribit reports for the same mark near the money){% endif %}, and every option is repriced at the snapshot from that volatility. Bid and ask are the mark plus and minus the effective half-spread, the median distance between trade prices and the mark on the taker's side, by moneyness bucket{% if rep.atm_half_spread_median is defined %} ({{ rep.atm_half_spread_median | pct(1) }} of the premium near the money){% endif %}. These are costs paid by real trades rather than quoted books; a mark{% if rep.mark_age_minutes_median is defined %} is a median {{ (rep.mark_age_minutes_median / 60) | num(1) }} hours old at the snapshot, and its{% else %}'s{% endif %} volatility is carried to the snapshot unchanged.

## Reproduce

Tiingo needs a free API key from [tiingo.com](https://www.tiingo.com). In Windows PowerShell, one command per line:

```powershell
cd cross-asset-vrp
$env:TIINGO_API_KEY = "<your key>"
uv sync
uv run vrp results
uv run vrp check
uv run vrp check-provenance
```

The key lives only in that PowerShell window and is sent in a request header, never written to a file. The first run downloads every VX contract, the six ETFs, bitcoin's minute bars since {{ cal.first[:4] }} and one option-chain snapshot a month; later runs download only what changed. `uv run vrp check-provenance --rehash` also hashes every downloaded file again. Without network access, `uv run vrp fixture mirror -o configs/ci.yaml` and then `uv run vrp results -o configs/ci.yaml --mirror mirror` run the whole pipeline on synthetic data.

## Layout

| Path | What it holds |
| --- | --- |
| [`configs/research.yaml`](configs/research.yaml) | Every choice with its reason: the index-instrument pairs, windows, HAR lags, futures rules, episodes |
| [`src/cross_asset_vrp/data`](src/cross_asset_vrp/data) | Sources, parsers, the resumable downloader with its SHA-256 manifest, the synthetic fixture |
| [`src/cross_asset_vrp/models`](src/cross_asset_vrp/models) | Realized measures and range estimators, HAR, variance swap replication, VX futures, Heston, regressions |
| [`src/cross_asset_vrp/evaluation`](src/cross_asset_vrp/evaluation) | The estimate stage |
| [`reports`](reports) | results.json, figures and tables, all rebuilt by `uv run vrp results` |
| [`paper`](paper) | The paper template and the built PDF |
| [`tests`](tests) | Estimator theory, HAR recovery, replication on a known smile, file formats, the end-to-end fixture run |

## References

- Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2008). Designing realized kernels to measure the ex post variation of equity prices in the presence of noise. *Econometrica* 76(6), 1481-1536.
- Bollerslev, T., Tauchen, G. and Zhou, H. (2009). Expected stock returns and variance risk premia. *Review of Financial Studies* 22(11), 4463-4492.
- Broadie, M., Glasserman, P. and Kou, S. (1997). A continuity correction for discrete barrier options. *Mathematical Finance* 7(4), 325-349.
- Carr, P. and Madan, D. (1998). Towards a theory of volatility trading. In *Volatility: New Estimation Techniques for Pricing Derivatives*, 417-427.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics* 7(2), 174-196.
- Garman, M. B. and Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business* 53(1), 67-78.
- Lazarus, E., Lewis, D. J., Stock, J. H. and Watson, M. W. (2018). HAR inference: recommendations for practice. *Journal of Business and Economic Statistics* 36(4), 541-559.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business* 53(1), 61-65.
- Rogers, L. C. G. and Satchell, S. E. (1991). Estimating variance from high, low and closing prices. *Annals of Applied Probability* 1(4), 504-512.
- Yang, D. and Zhang, Q. (2000). Drift-independent volatility estimation based on high, low, open, and close prices. *Journal of Business* 73(3), 477-491.

Author: {{ r.author }}. Built {{ r.generated_at }} from {{ r.manifest_files }} source files (manifest digest `{{ r.manifest_digest[:12] }}`, config `{{ r.config_fingerprint }}`).

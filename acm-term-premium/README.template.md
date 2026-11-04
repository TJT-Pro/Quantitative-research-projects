{#- Rendered by `acm report` from reports/results.json. Edit this file, never README.md. -#}
{% set v = r.validation %}
{% set rt = r.realtime %}
# The term premium, rebuilt and stress-tested

{% if r.provenance != "official" %}
> **Synthetic data.** These numbers come from the CI fixture, not the Federal Reserve and NY Fed files. Run `uv run acm results` to regenerate them from the official sources.

{% endif %}
Can the NY Fed's Adrian-Crump-Moench Treasury term premium be rebuilt exactly from public data, how different did it look to someone estimating it at the time, and does it predict bond returns?

{% set acm_fc = r.forecasts | selectattr("predictor", "eq", "acm") | list if r.forecasts is defined else [] %}
{% set cp_fc = r.forecasts | selectattr("predictor", "eq", "cp") | list if r.forecasts is defined else [] %}
| | |
| --- | --- |
| Replication | Every published series, every maturity from {{ r.settings.published_years | min }} to {{ r.settings.published_years | max }} years, all {{ v.months }} months from {{ v.start[:4] }} to {{ v.end[:4] }}: largest gap **{{ v.max_abs_all_series_bp | num(4) }} bp** |
| Textbook equations instead | {{ v.headline_years }}-year term premium off by {{ v.convention_gap_max_abs_bp | num(1) }} bp |
| Estimated in real time | {{ v.headline_years }}-year term premium {{ rt.mean_abs_revision_bp | num(0) }} bp from the published history on average, up to {{ rt.max_abs_revision_bp | num(0) }} bp ({{ rt.max_revision_date[:7] }}) |
{% if acm_fc %}
| Forecasting 12-month bond returns | Real-time term premium: out-of-sample R² {{ (100 * (acm_fc | map(attribute="oos_r2") | min)) | num(2) }}% to {{ (100 * (acm_fc | map(attribute="oos_r2") | max)) | num(2) }}%, {% set sig = acm_fc | selectattr("p_value", "lt", 0.05) | list | length %}{{ "none" if sig == 0 else sig ~ " of " ~ (acm_fc | length) }} significant at 5%. Cochrane-Piazzesi: {{ (100 * (cp_fc | map(attribute="oos_r2") | min)) | num(1) }}% to {{ (100 * (cp_fc | map(attribute="oos_r2") | max)) | num(1) }}% |
{% endif %}
{% if r.events is defined %}
| Policy-rate change days | Term premium carries {{ (100 * r.events.term_premium_variance_share) | num(0) }}% of the {{ v.headline_years }}-year yield's variance ({{ r.events.events }} days) |
{% endif %}
| Latest {{ v.headline_years }}-year term premium | {{ v.latest_tp_pct | num(2) }}% on {{ v.latest_date }} (90% band {{ r.uncertainty.latest_lower_pct | num(2) }}% to {{ r.uncertainty.latest_upper_pct | num(2) }}%) |
| Data | Federal Reserve Board, NY Fed and FRED files, every download SHA-256 hashed in [`data/manifest.json`](data/manifest.json) |

![Rebuilt term premium and its gap to the published series]({{ r.figures.replication[0] }})

*Top: {{ v.headline_years }}-year term premium from this code, end of month, {{ v.start[:4] }} to {{ v.end[:4] }}. Bottom: the largest absolute gap to the NY Fed's published series in each year. RMSE {{ v.tp_rmse_bp | num(4) }} bp, largest {{ v.tp_max_abs_bp | num(4) }} bp. On the daily series ({{ "{:,}".format(v.daily_days) }} days) the largest gap is {{ v.daily_max_abs_bp | num(4) }} bp.*

## Question, data, method

The term premium is the extra yield investors demand for holding a long bond instead of rolling short ones. It cannot be observed, so it has to be estimated, and the NY Fed's ACM series has become the reference estimate.

Inputs are the Gurkaynak-Sack-Wright daily Svensson curve from the Federal Reserve Board and, to repair the curve's unreliable short end before {{ r.settings.short_rate.replace_before[:4] }}, the effective fed funds rate. The model is ACM (2013): {{ r.settings.n_factors }} principal components of yields, a VAR for their dynamics, a regression of one-month bond excess returns on the VAR innovations and lagged factors, prices of risk from that regression's cross-section, and affine pricing recursions run once with those prices of risk and once without. The term premium is the difference.

## What it takes to match the published series

The textbook estimator gets the fitted yields right but not the term premium. Matching the published numbers needs two conventions that the paper does not spell out, found by testing candidate definitions against the published series:

1. The VAR intercept is folded into the innovations, so the pricing recursions run with a zero drift. Only the drift minus the price of risk enters fitted yields, so they are unchanged; the risk-neutral leg moves.
2. The recursion starts at the one-month bond with no convexity term, and the factor covariance uses T - 1 degrees of freedom.

With the textbook equations instead, the {{ v.headline_years }}-year term premium is {{ (v.convention_gap_mean_bp | abs) | num(2) }} bp {{ 'lower' if v.convention_gap_mean_bp < 0 else 'higher' }} on average, with a largest gap of {{ v.convention_gap_max_abs_bp | num(2) }} bp. Using the raw GSW one-month yield instead of the fed funds repair moves it by {{ v.splice_gap_mean_abs_bp | num(1) }} bp on average and up to {{ v.splice_gap_max_abs_bp | num(0) }} bp. Every run estimates both conventions and both short-rate treatments.

{{ r.tables.validation_decades }}

*{{ v.headline_years }}-year term premium against the NY Fed by decade. The full table for every series and maturity is in [`reports/tables/validation.md`](reports/tables/validation.md).*

## What a user could have known at the time

The published series is re-estimated on the full sample every month, so its history is rewritten as data arrive. Here the model is re-estimated on an expanding window ending at each month from {{ rt.start[:4] }}, factors included, and only that month's estimate is kept.

![Real-time revisions]({{ r.figures.revisions[0] }})

*Real-time minus full-sample {{ v.headline_years }}-year term premium. Correlation between the two series: {{ rt.correlation_with_full_sample | num(2) }}.*

{{ r.tables.revisions }}

{% if rt.constrained_months > 0 %}
In {{ rt.constrained_months }} windows ending between {{ rt.constrained_first[:7] }} and {{ rt.constrained_last[:7] }}, the estimated VAR had a root at or above the bound of {{ r.settings.max_spectral_radius }} (largest {{ rt.largest_ols_spectral_radius | num(4) }}). Near or past a unit root, {{ v.headline_years }}-year expectations barely mean-revert, and the unconstrained real-time term premium misses the published one by up to {{ rt.unconstrained_max_abs_revision_bp | num(0) }} bp. Those windows are re-estimated with every root above the bound pulled down to it, a half-life of about {{ (r.settings.max_spectral_radius | half_life_years) | num(1) }} years. The full-sample model is never constrained, so the replication above is unaffected.
{% else %}
No real-time window had a VAR root above the bound of {{ r.settings.max_spectral_radius }} (largest {{ rt.largest_ols_spectral_radius | num(4) }}), so no estimate was constrained.
{% endif %}

## The decomposition, with its uncertainty

![{{ v.headline_years }}-year yield decomposition]({{ r.figures.decomposition[0] }})

*Fitted {{ v.headline_years }}-year yield, expected average short rate and term premium, with a 90% band from a {{ r.settings.bootstrap.replications }}-draw moving-block bootstrap ({{ r.settings.bootstrap.block_months }}-month blocks) of the whole estimator. Median band width: {{ r.uncertainty.median_band_width_bp | num(0) }} bp.*

## Does the term premium predict bond returns?

{% if r.forecasts is defined %}
Every month from {{ r.forecast_sample.start[:4] }} to {{ r.forecast_sample.end[:4] }}, {{ r.settings.forecasting.horizon_months }}-month excess returns on {{ r.settings.forecasting.maturities | map('years_label') | list | join_words }} bonds are forecast with the real-time term premium of the same maturity, with the Cochrane-Piazzesi (2005) forward-rate factor, and with the historical mean. Each forecast uses only returns realized by its date.

![Out-of-sample forecasts]({{ r.figures.forecasts[0] }})

{{ r.tables.forecasts }}

*Out-of-sample R² is relative to the historical mean (Campbell and Thompson, 2008). Clark-West (2007) statistics use Newey-West errors with {{ r.settings.forecasting.newey_west_lags }} lags because the {{ r.settings.forecasting.horizon_months }}-month returns overlap.*
{% endif %}

{% if r.events is defined %}
## Policy-rate change days

{% set e = r.events %}
On the {{ e.events }} days from {{ e.first[:4] }} to {{ e.last[:4] }} when the federal funds target or the top of its range changed ({{ e.hikes }} increases, {{ e.cuts }} cuts), the daily {{ v.headline_years }}-year yield is decomposed with the model a user had at the end of the previous month.

![Policy days]({{ r.figures.policy_days[0] }})

{% set a = e.all_days %}
*Policy-change days against all {{ "{:,}".format(a.days) }} trading days of the same period. Adding twice the covariance of the two components ({{ (100 * e.covariance_share) | num(0) }}% on change days, {{ (100 * a.covariance_share) | num(0) }}% on all days) to the two shares gives {{ (100 * (e.expectations_variance_share + e.term_premium_variance_share + e.covariance_share)) | num(0) }}% and {{ (100 * (a.expectations_variance_share + a.term_premium_variance_share + a.covariance_share)) | num(0) }}%; the model's fitting error accounts for the difference from 100%.*

| | Change days | All days |
| --- | ---: | ---: |
| Mean absolute change in the yield | {{ e.mean_abs_yield_bp | num(1) }} bp | {{ a.mean_abs_yield_bp | num(1) }} bp |
| ... in expected short rates | {{ e.mean_abs_expectations_bp | num(1) }} bp | {{ a.mean_abs_expectations_bp | num(1) }} bp |
| ... in the term premium | {{ e.mean_abs_term_premium_bp | num(1) }} bp | {{ a.mean_abs_term_premium_bp | num(1) }} bp |
| Share of yield-change variance from expected short rates | {{ (100 * e.expectations_variance_share) | num(0) }}% | {{ (100 * a.expectations_variance_share) | num(0) }}% |
| Share of yield-change variance from the term premium | {{ (100 * e.term_premium_variance_share) | num(0) }}% | {{ (100 * a.term_premium_variance_share) | num(0) }}% |
{% endif %}

## Limitations, and what was built around them

- **The GSW curve's short end is an extrapolation before {{ r.settings.short_rate.replace_before[:4] }}.** The one-month yield is repaired from the fed funds rate, as in the published series, and the raw-curve alternative is run every time with its effect reported above.
- **Small real-time samples can produce a VAR with no mean reversion.** Those windows are constrained to a stated bound, counted and shown, with the unconstrained path plotted in its own panel.
- **The bootstrap holds the factor definition fixed.** The ACM estimator is invariant to invertible affine changes of the factors, so this loses nothing for yields and term premia; {{ r.settings.bootstrap.block_months }}-month blocks keep the joint dynamics together.
- **Policy days before {{ r.settings.events_start[:4] }} are excluded.** Until then decisions were not announced, so the change day was not an information event.
- **The sources revise history.** Every download is hashed in [`data/manifest.json`](data/manifest.json). `uv run acm fetch` revalidates each file with a conditional GET and lists any file whose content changed in `data/changed.txt`; `uv run acm results` then rebuilds every number on this page from the revised data.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and, for the PDF, [Quarto](https://quarto.org). Run from this folder, one line at a time; the same commands work in PowerShell, cmd, bash and zsh:

```
uv sync
uv run acm results
uv run acm check
uv run acm check-provenance
```

`acm results` downloads and verifies the sources, builds the curves, runs every estimate and rebuilds the figures, this README and the paper. `acm check` runs lint, strict type checking, the tests and the README drift check. `acm check-provenance` fails if the results came from the synthetic CI fixture. `uv run acm --help` lists each stage.
{% if r.runtime is defined and r.runtime.estimate is defined %}
The estimation behind this page took {{ r.runtime.estimate.seconds | num(0) }} seconds (Python {{ r.runtime.estimate.python }}, {{ r.runtime.estimate.cpu_count }} CPUs).
{% endif %}

## Layout

| Path | Contents |
| --- | --- |
| [`configs/research.yaml`](configs/research.yaml) | Every research choice with its reason |
| [`src/acm_term_premium/data`](src/acm_term_premium/data) | Resumable downloads with a SHA-256 manifest; parsers; the Svensson curve; the short-end repair |
| [`src/acm_term_premium/models/acm.py`](src/acm_term_premium/models/acm.py) | The estimator, both conventions, the stationarity bound and the bootstrap |
| [`src/acm_term_premium/evaluation`](src/acm_term_premium/evaluation) | Real-time estimates, out-of-sample forecasts, policy days |
| [`reports/`](reports) | Figures, tables and `results.json` |
| [`paper/`](paper) | Full write-up |

## References

- Adrian, T., Crump, R. K. and Moench, E. (2013). Pricing the term structure with linear regressions. *Journal of Financial Economics* 110(1), 110-138.
- Campbell, J. Y. and Thompson, S. B. (2008). Predicting excess stock returns out of sample. *Review of Financial Studies* 21(4), 1509-1531.
- Clark, T. E. and West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics* 138(1), 291-311.
- Cochrane, J. H. and Piazzesi, M. (2005). Bond risk premia. *American Economic Review* 95(1), 138-160.
- Gurkaynak, R. S., Sack, B. and Wright, J. H. (2007). The U.S. Treasury yield curve: 1961 to the present. *Journal of Monetary Economics* 54(8), 2291-2304.

Data: Board of Governors of the Federal Reserve System (GSW curve, H.15); Federal Reserve Bank of New York (ACM term premia, used only as the validation target); Federal Reserve Bank of St. Louis (FRED). This project is not affiliated with any of them.

<sub>Generated {{ r.generated_at }} from configuration `{{ r.config_fingerprint }}`.</sub>

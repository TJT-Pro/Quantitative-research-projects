{#- Rendered by `chr report` from reports/results.json. Edit this file, never README.md. -#}
{% set h = r.headline %}
# Which covariance forecast survives a century?

{% if r.provenance != "library" %}
> **Synthetic data.** The numbers on this page come from the CI fixture, not the Kenneth French data. Run `uv run chr results` to regenerate them from the library before reading anything into them.

{% endif %}
Which of {{ r.settings.models | length }} covariance estimators, from the sample covariance to dynamic conditional correlation, forecasts risk best? Each one builds a minimum-variance portfolio every month on {{ h.start_year }} to {{ h.end_year }} daily returns, held for the next month. Lowest realized risk wins.

| | |
| --- | --- |
| Winner | **{{ h.best_label }}**, {{ h.best_vol_pct | num }}% annualized GMV volatility |
{% if h.sample_vol_pct is defined %}
| Sample covariance | {{ h.sample_vol_pct | num }}%; the winner cuts risk {{ h.reduction_vs_sample_pct | num(1) }}% below it |
{% endif %}
| Worst | {{ h.worst_label }}, {{ h.worst_vol_pct | num }}% |
{% if h.mcs_full_size is defined %}
| Statistically tied for best | {{ h.mcs_full_size }} of {{ h.mcs_full_models }}: {{ h.mcs_full_included | join_words }} ({{ h.mcs_confidence_pct }}% Model Confidence Set) |
{% endif %}
| Test bed | {{ h.universe_label }}, {{ h.months }} monthly rebalances, {{ h.window }}-day window{% if h.n_models < r.settings.models | length %}, the {{ h.n_models }} models defined in every month{% endif %} |
| Data | Kenneth R. French Data Library, every file SHA-256 hashed in [`data/manifest.json`](data/manifest.json) |

{% if r.figures.mcs_heatmap is defined %}
![Model Confidence Set membership]({{ r.figures.mcs_heatmap[0] }})

*MCS p-values {{ 'by decade' if r.settings.mcs_period == 'decade' else 'over the full sample' }} for {{ h.universe_label }}, loss = realized variance of the unconstrained global minimum variance portfolio. A shaded cell means the model cannot be statistically separated from the best model in that period at the {{ h.mcs_confidence_pct }}% level.*
{% endif %}

{{ r.tables.headline }}

*Full-sample results for {{ h.universe_label }}, {{ h.window }}-day window, common sample of {{ h.months }} months. Volatility is the annualized standard deviation of daily out-of-sample returns. QLIKE is averaged over months against the realized covariance; lower is better.*

## Question, data, method

Every risk model and every mean-variance optimizer depends on a covariance forecast, and the textbook choice, the sample covariance, is known to be badly conditioned once the number of assets is a meaningful fraction of the sample size. This project asks which of {{ r.settings.models | length }} estimators delivers the lowest realized risk out of sample, how that answer changes with the ratio of assets to observations, and which models cannot be told apart once sampling noise is accounted for.

The data are daily value-weighted returns on {{ r.settings.universes | map(attribute="label") | list | join_words }} from the Kenneth R. French Data Library, plus the Fama-French factors. At the end of every month each model is fitted on a trailing window, a global minimum variance portfolio is formed, and its return is recorded over the next month. Models are compared on realized portfolio variance, QLIKE and Frobenius loss against realized covariance, and turnover, and separated statistically with the Model Confidence Set of Hansen, Lunde and Nason (2011).

## The estimators

| Model | What it does |
| --- | --- |
{% for m in r.settings.models %}
| {{ m.label }} | {{ m.description }} |
{% endfor %}

Every estimator is implemented from scratch in [`src/covariance_horse_race/models`](src/covariance_horse_race/models) behind one `fit` / `forecast` interface, and each has a test with a known answer: recovery of GARCH and DCC parameters from simulated data, recovery of the number of factors, the closed-form behaviour of shrinkage on a constant-correlation truth, and positive definiteness on randomized inputs.

## What the evidence shows

{% if r.figures.dimension_sweep is defined %}
![Excess volatility by universe and window]({{ r.figures.dimension_sweep[0] }})

*Each column is one universe and estimation window; each cell is the out-of-sample GMV volatility above the best model in that column, in percent. Darker means further behind.*
{% endif %}

{% set partial = r.settings.models | selectattr("windows", "ne", "all") | list %}
{% if partial %}
{% set spans = partial | map(attribute="windows") | map("join", ", ") | unique | list %}
*Not every model runs at every window: {% if spans | length == 1 %}{{ partial | map(attribute="label") | list | join_words }} run only on the {{ spans[0] }}-day window{% else %}{% for m in partial %}{{ m.label }} runs on {{ m.windows | join(", ") }} days{{ "; " if not loop.last else "" }}{% endfor %}{% endif %}, because the conditional-versus-unconditional question needs only the primary window and these are the slowest estimators. Annualized volatility levels for every cell are in [`reports/tables/sweep_vol.md`](reports/tables/sweep_vol.md).*
{% else %}
*Annualized volatility levels for every cell are in [`reports/tables/sweep_vol.md`](reports/tables/sweep_vol.md).*
{% endif %}

{% if r.excluded %}
Left out of a comparison because they are not defined in every month of it:

| Model | Window | Universes | Reason |
| --- | ---: | --- | --- |
{% for wgroup in r.excluded | groupby("window") %}
{% for rgroup in wgroup.list | groupby("reason") %}
{% for mgroup in rgroup.list | groupby("label") %}
| {{ mgroup.grouper }} | {{ wgroup.grouper }}d | {{ mgroup.list | map(attribute="universe_label") | list | join_words }} | {{ rgroup.grouper }} |
{% endfor %}
{% endfor %}
{% endfor %}
{% endif %}

{% if r.figures.turnover_vs_vol is defined %}
![Turnover against volatility]({{ r.figures.turnover_vs_vol[0] }})

*Lower risk is only worth having if it does not come from trading far more. Left: realized GMV volatility per model. Right: average monthly turnover, log scale. Models sorted by volatility, lowest at the top.*
{% endif %}

{% if r.spectrum is defined %}
![Sample and shrunk eigenvalues]({{ r.figures.eigen_spectrum[0] }})

*Why shrinkage helps: on the last {{ r.spectrum.window }} days of {{ r.spectrum.n_assets }} portfolios ending {{ r.spectrum.end }}, the sample covariance has a condition number of {{ r.spectrum.sample_condition_number | num(0) }}; nonlinear shrinkage brings it to {{ r.spectrum.shrunk_condition_number | num(0) }} by pulling the extreme eigenvalues toward the bulk.*
{% endif %}

## A check with a known answer

Real data cannot say which forecast was right, only which one did better. The simulation study removes that ambiguity: returns are drawn from multivariate Student-t ({{ r.settings.simulation.degrees_of_freedom }} degrees of freedom) with a fixed {{ r.settings.simulation.truth.source }} covariance, so the true risk of every estimated portfolio is known exactly and can be compared with the portfolio built from the true covariance.

{% if r.figures.simulation is defined %}
![Simulation study]({{ r.figures.simulation[0] }})

*Models in the simulation: {{ r.settings.models | selectattr("name", "in", r.settings.simulation.models) | map(attribute="label") | list | join_words }}. True GMV variance above the oracle portfolio, in percent, averaged over {{ r.settings.simulation.replications }} replications per cell. Standard errors are in [`reports/results.json`](reports/results.json) and the full table in [`reports/tables/simulation_excess_variance.md`](reports/tables/simulation_excess_variance.md).*
{% endif %}

## Limitations, and what was built around them

- **Industry portfolios are diversified**, so estimation noise is smaller than in a single-stock universe and the gaps between models understate what a stock-level desk would see. The sweep therefore runs up to {{ h.max_ratio_universe_label }}, where the {{ h.max_ratio_window }}-day window gives {{ h.max_assets_per_observation | num(2) }} assets per observation, the regime in which the sample covariance is closest to singular.
- **Real data never reveal the true covariance.** The simulation above is the control: the same estimators and the same GMV evaluation where the truth is known.
- **Some models are undefined in some cells.** The sample covariance is singular when assets reach the window length, and the extended factor data begin on {{ r.data.extended_factor_start }}. Rather than patch these cases, each comparison uses only months where every included model is defined, and every exclusion is listed with its reason.
- **Weights are held fixed through each month**, as in Ledoit and Wolf (2017), so realized variance measures forecast quality rather than drift. Turnover is measured against the drifted weights a monthly rebalancer would actually trade from.
- **Transaction costs are not charged.** Turnover is reported next to risk so the trade-off is visible; the portfolios are a lens on forecast quality, not a strategy.
- **The library revises history.** It moved from CRSP's legacy files to the CIZ format in January 2025. Every download is hashed in [`data/manifest.json`](data/manifest.json). `uv run chr fetch` revalidates each file with a conditional GET and lists any file whose content changed in `data/changed.txt`; `uv run chr results` then rebuilds every number on this page from the revised data.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and, for the PDF, [Quarto](https://quarto.org). The same commands work in PowerShell, cmd, bash and zsh; run them one at a time from this folder.

```
uv sync
uv run chr results
uv run chr check
uv run chr check-provenance
```

`chr results` downloads and verifies the data, runs every stage and rebuilds the figures, this README and the paper. `chr check` runs lint, strict type checking, the test suite and the README drift check. `chr check-provenance` fails if the results came from the synthetic CI fixture. `uv run chr --help` lists every stage.

{% if r.runtime is defined and r.runtime.backtest is defined %}
The backtest behind this page took {{ (r.runtime.backtest.seconds / 60) | num(1) }} minutes on {{ r.runtime.backtest.workers }} worker processes ({{ r.runtime.backtest.cpu_count }} CPUs, Python {{ r.runtime.backtest.python }}).
{% endif %}

## Layout

| Path | Contents |
| --- | --- |
| [`configs/research.yaml`](configs/research.yaml) | Every research choice with its reason; validated on load |
| [`src/covariance_horse_race/data`](src/covariance_horse_race/data) | Download with SHA-256 manifest, archive safety checks, parser for the library format |
| [`src/covariance_horse_race/models`](src/covariance_horse_race/models) | The {{ r.settings.models | length }} estimators |
| [`src/covariance_horse_race/evaluation`](src/covariance_horse_race/evaluation) | Point-in-time backtest, GMV solvers, losses, Model Confidence Set, simulation |
| [`reports/`](reports) | Figures (PNG and SVG), tables (CSV and Markdown) and `results.json` |
| [`paper/`](paper) | Write-up with full method and references |

## References

- Bai, J. and Ng, S. (2002). Determining the number of factors in approximate factor models. *Econometrica* 70(1).
- Engle, R. (2002). Dynamic conditional correlation. *Journal of Business and Economic Statistics* 20(3).
- Engle, R., Ledoit, O. and Wolf, M. (2019). Large dynamic covariance matrices. *Journal of Business and Economic Statistics* 37(2).
- Engle, R., Shephard, N. and Sheppard, K. (2008). Fitting vast dimensional time-varying covariance models. Working paper, Oxford.
- Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set. *Econometrica* 79(2).
- Laurent, S., Rombouts, J. and Violante, F. (2013). On loss functions and ranking forecasting performances of multivariate volatility models. *Journal of Econometrics* 173(1).
- Ledoit, O. and Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management* 30(4).
- Ledoit, O. and Wolf, M. (2017). Nonlinear shrinkage of the covariance matrix for portfolio selection: Markowitz meets Goldilocks. *Review of Financial Studies* 30(12).
- Ledoit, O. and Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics* 48(5).
- Politis, D. and Romano, J. (1994). The stationary bootstrap. *Journal of the American Statistical Association* 89(428).

<sub>Generated {{ r.generated_at }} from configuration `{{ r.config_fingerprint }}`.</sub>

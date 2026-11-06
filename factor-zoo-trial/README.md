# The factor zoo on trial

After publication, the average of 212 published US equity predictors earns 0.31% a month against 0.68% in sample, a change of -55% (95% interval -68% to -41%). 96 of 212 survive a false discovery control valid under any dependence, and 84.0% of the originally significant JKP factors replicate in the US.

![Average long-short return in sample, after the sample and after publication](reports/figures/headline.png)

| Test | This study | Published benchmark |
| --- | --- | --- |
| Decay after publication | -55% of the in-sample mean (-68% to -41%); after the sample, before publication: -34% | McLean and Pontiff (2016): -58% and -26%; on their window this study gets -48% and -33% |
| In-sample discoveries out of 212 | Unadjusted 181, Holm 58, BHY 96, Romano-Wolf 76, t > 3.0: 101; Storey share of true nulls 3% | Harvey, Liu and Zhu (2016) hurdle t > 3.0 |
| Empirical Bayes replication, US | 84.0% of 119 factors (76% to 87%); the US posterior out-predicts no skill in 81% of 52 countries, but a prior with the US spread and no US direction in only 40% | Jensen, Kelly and Pedersen (2023): 82.4% of 119; on their window this study gets 83.2% of 119 |

## Question, data, method

Of the published US equity return predictors, how many hold up once the tests account for how many were tried? How much of their return is left after the original sample and after publication? Do they replicate in other countries? Open Source Asset Pricing (release Data-Release-2026.10) provides 212 long-short predictors from 1926 to 2025, each with its paper's sample period and publication year. Global Factor Data provides 153 factors in 71 countries. The decay is the McLean-Pontiff panel regression with predictor fixed effects and month-clustered errors. In-sample t-statistics go through six multiple-testing procedures, one of them a bootstrap that resamples whole months. Alphas are shrunk by the JKP hierarchical empirical Bayes model. The US posterior is then scored as a forecast of each other country's alphas.

## Supporting exhibits

![In-sample t-statistics with each correction's discoveries shaded](reports/figures/discoveries.png)

The same 212 t-statistics in every panel. Each correction shades a different set of discoveries: 181 with no correction, 55 under Bonferroni.

![Replication rate by region with posterior intervals](reports/figures/regions.png)

Replication rate in each JKP region, counting factors that were significant in the original paper.

![US posterior alphas by theme cluster](reports/figures/clusters.png)

US posterior alphas with 95% intervals, one panel per theme cluster. Significant: the 95% posterior interval lies above zero. The prior has mean zero, so every alpha is pulled toward zero.

![The US posterior as a forecast of other countries](reports/figures/space.png)

Out of sample in space: per-factor log predictive gain of the US posterior over two alternatives, for each country with at least 20 factors. The median country gains 0.235 over no skill and -0.022 over a prior with the US spread but no US direction.

<details>
<summary>More exhibits: benchmarks, publication shifts, tradability, simulation</summary>

![Benchmarks](reports/figures/benchmarks.png)

McLean-Pontiff, on returns to 2013-12 and predictors published by then: post-publication change -48% against -58% published, a difference of 10 percentage points. This study's sample is a different set of predictors, rebuilt by OSAP rather than by the original authors, so some gap is expected. JKP, US to 2019-12: 83.2% against 82.4%, with 119 factors counted against their 119. Benjamini-Yekutieli gives 79.0% against their 75.6%.

![Decay as a function of the publication shift](reports/figures/shifts.png)

Moving each publication date back by up to 3 years, to allow for working-paper circulation, changes the post-publication decline from -55% to -53%.

![Tradability](reports/figures/tradability.png)

The same decay and discovery counts on value-weighted, NYSE-only, price-screened and microcap-free portfolios. Without microcaps the in-sample return is 0.50% a month against 0.68%, and BHY keeps 54 predictors against 96.

![Simulation with known truth](reports/figures/simulation.png)

200 simulated zoos of 200 factors from a 5-factor structure fitted to the real panel. Error rates against the 5% target, and power, by share of true nulls. Holm with 60% nulls (6.0%) and Romano-Wolf with 60% nulls (5.5%) are above the target but within two Monte Carlo standard errors of it (3.1 percentage points with 200 replications). Empirical Bayes 95% intervals cover the true alpha 93.9% of the time. Estimated tau_c is 0.248 against a true 0.257, and tau_w 0.135 against 0.131.

</details>

## Limitations, and the workaround built for each

- **OSAP rebuilds each predictor from public data, so some returns differ from the papers.** Every result is also run on the 165 predictors OSAP classifies as clear replications. Their post-publication change is -58% against -55% for all predictors. The full grid is in [`reports/tables/decay_grid.csv`](reports/tables/decay_grid.csv).
- **The journal year is not when the market learned of a predictor.** Publication dates are shifted back by 0, 1, 2 and 3 years and the decay is reported for each shift (figure above).
- **Predictor returns are strongly cross-correlated.** Errors are clustered by month. False discoveries are controlled with BHY, which holds under any dependence. The Romano-Wolf bootstrap resamples whole months in 12-month circular blocks, so the cross-section and serial dependence survive in each of 5000 draws.
- **Both sources publish revised releases.** The OSAP release and the JKP reference commit are pinned in [`configs/research.yaml`](configs/research.yaml), and every file is hashed in `data/manifest.json`. `uv run fzt check-release` compares the pin with the newest release on Hugging Face, and `uv run fzt check-release --write` moves the pin; `uv run fzt results` then downloads only the files that changed and rebuilds every number on this page.
- **Factors outnumber months in many countries, so sample covariances are singular.** Correlations are estimated pairwise and shrunk toward the identity with the Schafer-Strimmer intensity (0.007 in the US).
- **Pairwise correlations from factors with different start dates do not form a valid correlation matrix.** Raising its negative eigenvalues to a numerical zero would treat some combinations of alphas as known exactly, and every likelihood along them explodes. Eigenvalues are instead raised to the Marchenko-Pastur lower edge, (1 - sqrt(K/n))^2 for K factors and the median pairwise overlap n: 0.34 in the US, where 106 of 153 eigenvalues are raised. A test with zero true alphas on a staggered panel pins the effect: the per-factor chi-square statistic stays below 3 with the floor and is more than a hundred times larger with a numerical floor. Countries sharing fewer than 20 factors with the US are left out of the space test and listed in `results.json`.
- **CAPM and FF5 + momentum adjustments use full-sample betas.** The headline uses raw returns, as McLean and Pontiff do. On adjusted returns the post-publication change is -46% (CAPM) and -35% (FF5 + momentum).

## Reproduce

Windows PowerShell, one command per line:

```powershell
cd factor-zoo-trial
uv sync
uv run fzt results
uv run fzt check
uv run fzt check-provenance
```

On the first run `fzt results` downloads 100 MB of OSAP portfolio files and 51 MB of JKP files (162 files in all). If the connection drops, run it again: downloads resume and skip files that have not changed. `uv run fzt results --skip-paper` skips the PDF. `uv run fzt --help` lists each stage. Without network access, `uv run fzt fixture mirror` followed by `uv run fzt results -o configs/ci.yaml --mirror mirror` runs the whole pipeline on synthetic data.

## Layout

| Path | Contents |
| --- | --- |
| [`configs/research.yaml`](configs/research.yaml) | Every research choice, each with its reason |
| [`src/factor_zoo_trial/data`](src/factor_zoo_trial/data) | Resumable downloads, SHA-256 manifest, parsers, synthetic fixture |
| [`src/factor_zoo_trial/models`](src/factor_zoo_trial/models) | Fixed-effects panel, multiple testing, covariance shrinkage, empirical Bayes, transfer scoring |
| [`src/factor_zoo_trial/evaluation`](src/factor_zoo_trial/evaluation) | Periods, decay, discoveries, replication, space test, simulation |
| [`reports/`](reports) | Figures, tables and `results.json` |
| [`paper/`](paper) | Full write-up (Quarto, Typst PDF and HTML) |

## References

- Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics* 29(4), 1165-1188.
- Chen, A. Y. and Zimmermann, T. (2022). Open source cross-sectional asset pricing. *Critical Finance Review* 11(2), 207-264.
- Harvey, C. R., Liu, Y. and Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies* 29(1), 5-68.
- Jensen, T. I., Kelly, B. and Pedersen, L. H. (2023). Is there a replication crisis in finance? *Journal of Finance* 78(5), 2465-2518.
- Lazarus, E., Lewis, D. J., Stock, J. H. and Watson, M. W. (2018). HAR inference: recommendations for practice. *Journal of Business and Economic Statistics* 36(4), 541-559.
- McLean, R. D. and Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance* 71(1), 5-32.
- Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica* 73(4), 1237-1282.
- Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (1999). Noise dressing of financial correlation matrices. *Physical Review Letters* 83(7), 1467-1470.
- Schafer, J. and Strimmer, K. (2005). A shrinkage approach to large-scale covariance matrix estimation. *Statistical Applications in Genetics and Molecular Biology* 4(1).
- Storey, J. D. (2002). A direct approach to false discovery rates. *Journal of the Royal Statistical Society B* 64(3), 479-498.
- Storey, J. D., Taylor, J. E. and Siegmund, D. (2004). Strong control, conservative point estimation and simultaneous conservative consistency of false discovery rates. *Journal of the Royal Statistical Society B* 66(1), 187-205.

Data: Open Source Asset Pricing (CC BY 4.0), Global Factor Data (CC BY-NC 4.0) and the Kenneth R. French data library. This project is not affiliated with any of them.

<sub>Generated 2026-11-06 05:29 UTC from configuration `c83b9ce8cb70` and inputs `3a8ed902c8b6` (official).</sub>

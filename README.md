# Quantitative research projects

Empirical finance and econometrics projects by Ian Wachira. Each folder is a standalone project with its own code, tests, CI, results and paper. Every number in a project README is rendered from that project's `results.json`, built from the public source data, and CI fails if the two disagree or if the results came from synthetic test data.

## Projects

| Project | Question | CI |
| --- | --- | --- |
| [Covariance horse race](covariance-horse-race) | Which covariance forecast gives the lowest out-of-sample risk on a century of daily US portfolio returns, and which models are statistically tied? | [![covariance-horse-race](https://github.com/TJT-Pro/Quantitative-research-projects/actions/workflows/covariance-horse-race.yml/badge.svg)](https://github.com/TJT-Pro/Quantitative-research-projects/actions/workflows/covariance-horse-race.yml) |
| [The term premium, rebuilt](acm-term-premium) | Can the NY Fed's ACM Treasury term premium be reproduced exactly from public data, how different was it in real time, and does it predict bond returns? | [![acm-term-premium](https://github.com/TJT-Pro/Quantitative-research-projects/actions/workflows/acm-term-premium.yml/badge.svg)](https://github.com/TJT-Pro/Quantitative-research-projects/actions/workflows/acm-term-premium.yml) |
| [The factor zoo on trial](factor-zoo-trial) | How much of a published equity anomaly's return survives publication, how many survive multiple-testing corrections, and do they replicate outside the US? | [![factor-zoo-trial](https://github.com/TJT-Pro/Quantitative-research-projects/actions/workflows/factor-zoo-trial.yml/badge.svg)](https://github.com/TJT-Pro/Quantitative-research-projects/actions/workflows/factor-zoo-trial.yml) |

### [Covariance horse race](covariance-horse-race)

Estimators from the sample covariance through Ledoit-Wolf shrinkage, factor models and DCC-GARCH forecast next-month risk for minimum-variance portfolios on Kenneth French daily data. Models are separated with the Model Confidence Set and checked against a simulation where the true covariance is known. Read the [results](covariance-horse-race#readme) or the [paper (PDF)](covariance-horse-race/paper/_output/paper/paper.pdf).

[![Out-of-sample GMV volatility above the best model, by universe and window](covariance-horse-race/reports/figures/dimension_sweep.png)](covariance-horse-race#readme)

### [The term premium, rebuilt](acm-term-premium)

The Adrian-Crump-Moench model is rebuilt from the Federal Reserve Board's Gurkaynak-Sack-Wright yield curve and checked against every series the NY Fed publishes. It is then re-estimated month by month as a user would have seen it at the time, tested as an out-of-sample predictor of bond excess returns against the Cochrane-Piazzesi factor, and used to split long-yield moves on policy-rate change days into expected short rates and term premium. Read the [results](acm-term-premium#readme) or the [paper (PDF)](acm-term-premium/paper/_output/paper/paper.pdf).

[![Term premium rebuilt from public inputs, and its gap to the published series](acm-term-premium/reports/figures/replication.png)](acm-term-premium#readme)

### [The factor zoo on trial](factor-zoo-trial)

Every published US predictor in Open Source Asset Pricing is put through the McLean-Pontiff decay regression, six multiple-testing procedures including a whole-month bootstrap, and the Jensen-Kelly-Pedersen empirical Bayes model on Global Factor Data in every country. The US posterior is then scored as a forecast of each other country's alphas, and every procedure is checked in a simulation with known truth. Read the [results](factor-zoo-trial#readme) or the [paper (PDF)](factor-zoo-trial/paper/_output/paper/paper.pdf).

[![Average long-short return in sample, after the sample and after publication](factor-zoo-trial/reports/figures/headline.png)](factor-zoo-trial#readme)

## Running a project

Each project uses [uv](https://docs.astral.sh/uv/). From the repository root, one command at a time, with the project's folder and command name:

| Project | Folder | Command |
| --- | --- | --- |
| Covariance horse race | `covariance-horse-race` | `chr` |
| The term premium, rebuilt | `acm-term-premium` | `acm` |
| The factor zoo on trial | `factor-zoo-trial` | `fzt` |

```
cd acm-term-premium
uv sync
uv run acm results
uv run acm check
uv run acm check-provenance
```

The project README lists what each command does.

## License

MIT. See [LICENSE](LICENSE).

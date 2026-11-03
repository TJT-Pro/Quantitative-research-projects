# Quantitative research projects

Empirical finance and econometrics projects by Ian Wachira, AKA TJT-Pro. Each folder is a standalone project with its own code, tests, CI, results and paper. Every number in a project README is rendered from that project's `results.json`, built from the public source data, and CI fails if the two disagree or if the results came from synthetic test data.

## Projects

| Project | Question | CI |
| --- | --- | --- |
| [Covariance horse race](covariance-horse-race) | Which covariance forecast gives the lowest out-of-sample risk on a century of daily US portfolio returns, and which models are statistically tied? | [![covariance-horse-race](https://github.com/TJT-Pro/Quantitative-research-projects/actions/workflows/covariance-horse-race.yml/badge.svg)](https://github.com/TJT-Pro/Quantitative-research-projects/actions/workflows/covariance-horse-race.yml) |

### [Covariance horse race](covariance-horse-race)

Estimators from the sample covariance through Ledoit-Wolf shrinkage, factor models and DCC-GARCH forecast next-month risk for minimum-variance portfolios on Kenneth French daily data. Models are separated with the Model Confidence Set and checked against a simulation where the true covariance is known. Read the [results](covariance-horse-race#readme) or the [paper (PDF)](covariance-horse-race/paper/_output/paper/paper.pdf).

[![Out-of-sample GMV volatility above the best model, by universe and window](covariance-horse-race/reports/figures/dimension_sweep.png)](covariance-horse-race#readme)

## Running a project

Each project uses [uv](https://docs.astral.sh/uv/). From the repository root, one command at a time:

```
cd covariance-horse-race
uv sync
uv run chr results
uv run chr check
uv run chr check-provenance
```

The project README lists what each command does.

## License

MIT. See [LICENSE](LICENSE).

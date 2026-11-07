# New York Fed reference files

From [FRBNY-TimeSeriesAnalysis/Nowcasting](https://github.com/FRBNY-TimeSeriesAnalysis/Nowcasting) at commit `19f365cab8269e3aac3faa11ad091d6e913c5c43`, under the BSD 3-Clause licence in `LICENSE`.

- `ResDFM.mat`: the estimates their `example_DFM.m` saved, unchanged.
- `Spec_US_example.csv`: their `Spec_US_example.xls`, saved as CSV with the values unchanged.
- `2016-06-29.csv`: the sheet `data` of their `data/US/2016-06-29.xls`, saved as CSV with every value written to full precision.

`tests/test_methods.py::test_em_matches_the_new_york_fed_code` estimates the model on this vintage exactly as `example_DFM.m` does (sample from 2000-01, threshold 1e-4) and compares every matrix with `ResDFM.mat`.

# FinTech 545 - Assignment 2

The written submission is **Assignment2_Report.pdf**. It answers all five
problems, including the Predict / Fit / Reconcile parts, with numerical tables
and plots. This directory is self-contained; no sibling repository is needed.

## Files

- `Assignment2_Report.pdf`: written answers.
- `report.md`: readable Markdown version generated with the PDF.
- `predictions.md`: original expectations recorded before fitting.
- `prepare.py`: raw-data counts, moments, outlier diagnostics and plots.
- `analysis.py`: all model fits, risk estimates, repairs and simulations.
- `build_report.py`: generates the PDF and Markdown using current outputs.
- `data/problem1.csv` through `data/problem5.csv`: unmodified supplied data.
- `tables/`: full-precision CSV outputs, including each day's copula likelihood.
- `figures/`: generated plots.
- `results.json`: remaining scalar results and reproducibility settings.
- `requirements.txt`: Python dependencies.

## Reproduce everything

From this `Assignment2` directory, use Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
python build_report.py
```

The first script also calls `prepare.py`, writes every result table, and
regenerates all figures. The second rebuilds the PDF and Markdown. A separate
LaTeX or Quarto installation is not required. To inspect just the raw data before
fitting, run `python prepare.py`. The PDF's narrative retains the original
predictions rather than rewriting them to match the fits. `build_report.py` is
written for the supplied dataset: re-analyze and revise its narrative if inputs
change. Edit that file, rather than the generated `report.md`, to revise the PDF.

## Numerical conventions

- Returns: `price[t] / price[t-1] - 1`. Only Problem 2 removes the sample mean;
  Problem 5 enforces zero means during simulation and delta-normal valuation.
- Descriptive moments use denominator `n`, with excess kurtosis. Ordinary sample
  SD and covariance use `n-1`; normal marginal MLE uses denominator `n`.
- Historical VaR uses the empirical inverse CDF: minus the observation at
  rank `ceil(n * alpha)` in ascending P&L order. ES averages exactly the worst
  `alpha` probability mass, using fractional boundary mass when necessary.
  With these datasets, all relevant tail counts are integers. A negative VaR
  denotes a profitable cutoff; it is neither floored at zero nor absolute-valued.
- Problem 1 repairs correlations before restoring the original full-history
  standard deviations. Higham uses Dykstra correction, unit diagonal and
  tolerance `1e-12`. Numerical eigenvalues within `1e-10` of zero count as PSD.
- Problem 2 uses the last 40 returns as a visually chosen recent regime.
  EW weights are normalized over 500 returns, newest weight largest. Weighted
  squared globally demeaned returns have no Bessel correction or second demeaning.
- Student-t fits estimate location, scale and df. Standardization and multiple
  starting df values stabilize the numerical MLE; t scale is not its SD.
- Problem 4 fits normal/t margins with AICc, keeps all observations, and freezes
  selected margins and Kendall `R` for copula comparison. Following Week 5,
  copula-only parameter counts are 0 (Gaussian) and 1 (t); the marginal AICc
  counts are 2 and 3. Empirical ranks define observed tails and central days;
  fitted CDF values define copula likelihood inputs. Both central-day conventions
  are reported for sensitivity. Each t scenario shares one chi-square scale.
- Problem 5 fits OLS with an intercept but omits the estimated alphas from
  simulation to impose zero expected returns. Residual covariance uses `n-1`
  to preserve the sample OLS decomposition; `n-2` SDs are reported separately.
- 100,000 draws per simulation; NumPy `default_rng` seeds 54506 (Problem 4)
  and 54507 (Problem 5). Common base draws are reused across model comparisons.
  Ten-batch Monte Carlo SEs describe simulation noise, not model uncertainty.
- Assertions check repaired PSD matrices, Higham's distance relative to RJ,
  empirical ES subadditivity, OLS orthogonality, and the covariance decomposition.

Floating-point results and final simulation digits can vary slightly with
NumPy/SciPy versions. All CSV tables retain full output precision; the report
rounds money and selected statistics for readability.

## Submission

The assignment separately asks the student to respond `done` in Canvas after
finishing. No Canvas response or repository push is performed by these scripts.

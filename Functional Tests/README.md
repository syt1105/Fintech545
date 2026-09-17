# Functional Tests

Run all completed tests from this folder:

```bash
python3 -m pytest test_1.py test_2.py test_3.py test_4.py test_5.py test_6.py test_7.py -q
```

## Test 1.1–1.4

All four tests read `class/testfiles/data/test1.csv` (10 rows, 5 columns)
and compare a 5×5 result with the matching `testout_1.x.csv` file.

| Test | Result | Missing-data rule |
| --- | --- | --- |
| 1.1 | Sample covariance | Drop every row containing a missing value |
| 1.2 | Correlation | Drop every row containing a missing value |
| 1.3 | Sample covariance | For each column pair, keep rows where both are present |
| 1.4 | Correlation | For each column pair, keep rows where both are present |

The `class` and `homework` folders need to be side by side in `GitHubs`.

## Test 2.1–2.3

Read `test2.csv`. Test 2.1 calculates exponentially weighted covariance
with λ=0.97. Test 2.2 calculates correlation with λ=0.94. Test 2.3 uses
variance from λ=0.97 and correlation from λ=0.94.

## Test 3.1–3.4

Tests 3.1 and 3.2 use `near_psd`; tests 3.3 and 3.4 use Higham's method.
Each method is tested once on a covariance matrix (`testout_1.3.csv`) and
once on a correlation matrix (`testout_1.4.csv`).

## Test 4.1

Read the repaired PSD matrix from `testout_3.1.csv`, calculate its
lower-triangular Cholesky-style factor, and compare it with `testout_4.1.csv`.
The test also checks that `L @ L.T` reconstructs the input matrix.

## Test 5.1–5.5

Simulate 100,000 zero-mean observations and compare their sample covariance
with the teacher's output. Tests 5.1 and 5.2 use positive-definite and
positive-semidefinite input. Tests 5.3 and 5.4 repair a non-PSD input with
`near_psd` and Higham's method. Test 5.5 uses enough PCA components to explain
99% of variance. Julia and Python use different random-number generators, so
these tests allow normal sampling variation rather than requiring exact equality.

## Test 6.1–6.2

Read daily prices from `test6.csv`. Test 6.1 calculates arithmetic returns
(`new price / old price - 1`); test 6.2 calculates log returns
(`log(new price / old price)`). The first date is dropped because it has no
previous price. Both tests compare dates, asset columns, and all return values
with the teacher's `testout6_1.csv` and `testout6_2.csv` files.

## Test 7.1–7.6

Test 7.1 fits a normal distribution. Tests 7.2–7.4 fit a Student-t
distribution, fit a regression with t errors, and calculate AICc. Tests
7.5–7.6 fit a Normal Inverse Gaussian (NIG) distribution by moments and
maximum likelihood. The latter converts SciPy's parameters to the teacher's
`mu, alpha, beta, delta` order. Numerical optimizers can return slightly
different values, so the t-fit tests use a small comparison tolerance.

If needed, install the Python packages with `python3 -m pip install numpy scipy pytest`.

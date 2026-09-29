# FinTech 545 Python library

The `fintech545/` folder contains reusable calculation functions. Assignment
scripts and functional tests can import them instead of copying the same code.
`Assignment1/` remains a separate submission with its own instructions.

## Install

From the `homework` folder:

```bash
python3 -m pip install -e ".[test]"
```

The `-e` option installs the library in editable mode: changes to the source
files are available without reinstalling it.

## Use

```python
import numpy as np
from fintech545.missing_covariance import missing_covariance_skip_rows

data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
covariance = missing_covariance_skip_rows(data)
print(covariance)
```

The modules cover missing-data covariance and correlation, exponentially
weighted covariance, PSD repair, Cholesky factorization, simulation, returns,
and distribution fitting.

## Run the functional tests

Keep the teacher's `class` repository next to `homework`, since the tests
read input and expected-output CSV files from `class/testfiles/data`.
From the `homework` folder run:

```bash
python3 -m pytest -q "Functional Tests"
```

The tests are separate from the library: `fintech545/` contains functions,
and `Functional Tests/test_1.py` through `test_7.py` check their results.

"""Covariance and correlation tests with missing data."""

import numpy as np


def _missing_matrix(data, pairwise=False, correlation=False):
    """Choose usable rows, then calculate covariance or correlation."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must have rows and columns")

    if not pairwise:
        complete_rows = values[~np.isnan(values).any(axis=1)]
        if len(complete_rows) < 2:
            raise ValueError("at least two complete rows are needed")
        if correlation:
            return np.corrcoef(complete_rows, rowvar=False)
        return np.cov(complete_rows, rowvar=False, ddof=1)

    # For each pair of columns, keep rows where both values are present.
    column_count = values.shape[1]
    result = np.empty((column_count, column_count))
    for i in range(column_count):
        for j in range(i + 1):
            usable = ~np.isnan(values[:, i]) & ~np.isnan(values[:, j])
            pair = values[usable][:, [i, j]]
            if len(pair) < 2:
                raise ValueError("at least two usable rows are needed for each pair")
            if correlation:
                result[i, j] = np.corrcoef(pair, rowvar=False)[0, 1]
            else:
                result[i, j] = np.cov(pair, rowvar=False, ddof=1)[0, 1]
            result[j, i] = result[i, j]
    return result


def missing_covariance_skip_rows(data):
    return _missing_matrix(data)


def missing_correlation_skip_rows(data):
    return _missing_matrix(data, correlation=True)


def missing_covariance_pairwise(data):
    return _missing_matrix(data, pairwise=True)


def missing_correlation_pairwise(data):
    return _missing_matrix(data, pairwise=True, correlation=True)

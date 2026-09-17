"""Exponentially weighted covariance and correlation."""

import numpy as np


def ew_covariance(data, decay):
    """Give newer rows more weight; the last row is the newest."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("data must have at least two rows and two dimensions")
    if not 0 < decay < 1:
        raise ValueError("decay must be between 0 and 1")

    row_count = len(values)
    weights = decay ** np.arange(row_count - 1, -1, -1)
    weights = weights / weights.sum()

    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    return (centered * weights[:, None]).T @ centered


def ew_correlation(data, decay):
    covariance = ew_covariance(data, decay)
    standard_deviation = np.sqrt(np.diag(covariance))
    return covariance / np.outer(standard_deviation, standard_deviation)


def ew_covariance_mixed(data, variance_decay=0.97, correlation_decay=0.94):
    """Use variances from one decay and correlations from another."""
    variance_covariance = ew_covariance(data, variance_decay)
    standard_deviation = np.sqrt(np.diag(variance_covariance))
    correlation = ew_correlation(data, correlation_decay)
    return correlation * np.outer(standard_deviation, standard_deviation)

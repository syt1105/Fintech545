"""Two ways to repair a covariance or correlation matrix."""

import numpy as np


def _as_correlation(matrix):
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("input must be a square matrix")

    # Covariance matrices need to be scaled to correlation matrices first.
    if np.allclose(np.diag(values), 1.0):
        standard_deviation = np.ones(len(values))
    else:
        standard_deviation = np.sqrt(np.diag(values))
    correlation = values / np.outer(standard_deviation, standard_deviation)
    return correlation, standard_deviation


def near_psd(matrix):
    """Replace negative eigenvalues with zero and restore the diagonal."""
    correlation, standard_deviation = _as_correlation(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Scaling each row restores a diagonal of 1 after clipping eigenvalues.
    scale = 1 / np.sqrt((eigenvectors**2) @ eigenvalues)
    b = (eigenvectors * np.sqrt(eigenvalues)) * scale[:, None]
    repaired = b @ b.T
    return repaired * np.outer(standard_deviation, standard_deviation)


def higham_nearest_psd(matrix, max_iterations=100, tolerance=1e-9):
    """Alternate between a PSD matrix and a matrix with diagonal 1."""
    correlation, standard_deviation = _as_correlation(matrix)
    original = correlation.copy()
    delta = np.zeros_like(correlation)
    previous_norm = np.inf

    for _ in range(max_iterations):
        r = correlation - delta
        eigenvalues, eigenvectors = np.linalg.eigh(r)
        positive = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        delta = positive - r

        correlation = positive.copy()
        np.fill_diagonal(correlation, 1.0)
        norm = np.sum((correlation - original) ** 2)
        if abs(norm - previous_norm) < tolerance and np.min(np.linalg.eigvalsh(correlation)) > -1e-9:
            break
        previous_norm = norm

    return correlation * np.outer(standard_deviation, standard_deviation)

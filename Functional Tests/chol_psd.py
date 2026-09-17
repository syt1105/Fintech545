"""Cholesky-style factorization for a positive semidefinite matrix."""

import numpy as np


def chol_psd(matrix):
    """Return a lower-triangular L such that matrix is approximately L @ L.T."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("input must be a square matrix")

    n = len(values)
    root = np.zeros((n, n))
    for j in range(n):
        previous = np.dot(root[j, :j], root[j, :j])
        remaining = values[j, j] - previous

        # A tiny negative number can be caused by floating-point rounding.
        if -1e-8 <= remaining <= 0:
            remaining = 0.0
        if remaining < 0:
            raise ValueError("input is not positive semidefinite")
        root[j, j] = np.sqrt(remaining)

        if root[j, j] == 0:
            continue
        for i in range(j + 1, n):
            earlier = np.dot(root[i, :j], root[j, :j])
            root[i, j] = (values[i, j] - earlier) / root[j, j]

    return root

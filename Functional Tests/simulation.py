"""Normal and PCA simulations for Test 5."""

import numpy as np

from chol_psd import chol_psd


def simulate_normal(covariance, count, seed=1234, fix_method=None):
    """Simulate zero-mean normal data with the requested covariance."""
    covariance = np.asarray(covariance, dtype=float)
    try:
        root = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        try:
            root = chol_psd(covariance)
        except ValueError:
            if fix_method is None:
                raise
            root = chol_psd(fix_method(covariance))

    rng = np.random.default_rng(seed)
    standard_normal = rng.standard_normal((count, len(covariance)))
    return standard_normal @ root.T


def pca_loadings(covariance, explained=0.99):
    """Keep enough principal components to explain the requested variance."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    positive = eigenvalues >= 1e-8
    eigenvalues = eigenvalues[positive]
    eigenvectors = eigenvectors[:, positive]

    cumulative = np.cumsum(eigenvalues) / np.sum(np.linalg.eigvalsh(covariance))
    component_count = np.searchsorted(cumulative, explained) + 1
    return eigenvectors[:, :component_count] * np.sqrt(eigenvalues[:component_count])


def simulate_pca(covariance, count, explained=0.99, seed=1234):
    loadings = pca_loadings(covariance, explained)
    rng = np.random.default_rng(seed)
    standard_normal = rng.standard_normal((count, loadings.shape[1]))
    return standard_normal @ loadings.T

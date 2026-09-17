"""Functional tests 5.1–5.5 from the class Tests.xlsx workbook."""

from pathlib import Path

import numpy as np

from psd_repair import near_psd, higham_nearest_psd
from simulation import simulate_normal, simulate_pca, pca_loadings


DATA_DIR = Path(__file__).resolve().parents[2] / "class" / "testfiles" / "data"
SIMULATION_COUNT = 100_000


def read_csv(filename):
    return np.genfromtxt(DATA_DIR / filename, delimiter=",", skip_header=1)


def check_simulation(samples, expected, target):
    """Check sample covariance, allowing for random sampling differences."""
    actual = np.cov(samples, rowvar=False, ddof=1)
    assert samples.shape == (SIMULATION_COUNT, 5)
    assert actual.shape == expected.shape == target.shape == (5, 5)

    # A sample covariance varies around its true value. This is its standard
    # error for normal data; Julia and NumPy produce different random samples.
    variances = np.diag(target)
    mean_standard_error = np.sqrt(variances / SIMULATION_COUNT)
    assert np.all(np.abs(np.mean(samples, axis=0)) < 5 * mean_standard_error)

    standard_error = np.sqrt((target**2 + np.outer(variances, variances)) / (SIMULATION_COUNT - 1))
    assert np.all(np.abs(actual - target) < 5 * standard_error)
    assert np.all(np.abs(actual - expected) < 5 * np.sqrt(2) * standard_error)


def test_5_1_normal_pd():
    covariance = read_csv("test5_1.csv")
    expected = read_csv("testout_5.1.csv")
    samples = simulate_normal(covariance, SIMULATION_COUNT)
    check_simulation(samples, expected, covariance)


def test_5_2_normal_psd():
    covariance = read_csv("test5_2.csv")
    expected = read_csv("testout_5.2.csv")
    samples = simulate_normal(covariance, SIMULATION_COUNT)
    check_simulation(samples, expected, covariance)


def test_5_3_normal_near_psd_repair():
    covariance = read_csv("test5_3.csv")
    expected = read_csv("testout_5.3.csv")
    repaired = near_psd(covariance)
    samples = simulate_normal(covariance, SIMULATION_COUNT, fix_method=near_psd)
    check_simulation(samples, expected, repaired)


def test_5_4_normal_higham_repair():
    covariance = read_csv("test5_3.csv")
    expected = read_csv("testout_5.4.csv")
    repaired = higham_nearest_psd(covariance)
    samples = simulate_normal(covariance, SIMULATION_COUNT, fix_method=higham_nearest_psd)
    check_simulation(samples, expected, repaired)


def test_5_5_pca_99_percent():
    covariance = read_csv("test5_2.csv")
    expected = read_csv("testout_5.5.csv")
    loadings = pca_loadings(covariance, explained=0.99)
    assert loadings.shape == (5, 2)
    target = loadings @ loadings.T
    samples = simulate_pca(covariance, SIMULATION_COUNT, explained=0.99)
    check_simulation(samples, expected, target)

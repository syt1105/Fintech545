"""Functional tests 2.1–2.3 from the class Tests.xlsx workbook.

Run from this folder: python3 -m pytest test_2.py -q
The class and homework folders need to be side by side in GitHubs.
"""

from pathlib import Path

import numpy as np

from ew_covariance import ew_covariance, ew_correlation, ew_covariance_mixed


DATA_DIR = Path(__file__).resolve().parents[2] / "class" / "testfiles" / "data"


def read_csv(filename):
    return np.genfromtxt(DATA_DIR / filename, delimiter=",", skip_header=1)


def test_2_1_ew_covariance():
    data = read_csv("test2.csv")
    expected = read_csv("testout_2.1.csv")

    assert data.shape == (40, 5)
    actual = ew_covariance(data, 0.97)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_2_2_ew_correlation():
    data = read_csv("test2.csv")
    expected = read_csv("testout_2.2.csv")

    actual = ew_correlation(data, 0.94)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_2_3_mixed_ew_covariance():
    data = read_csv("test2.csv")
    expected = read_csv("testout_2.3.csv")

    actual = ew_covariance_mixed(data, variance_decay=0.97, correlation_decay=0.94)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

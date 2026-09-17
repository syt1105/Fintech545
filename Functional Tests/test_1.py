"""Functional tests 1.1–1.4 from the class Tests.xlsx workbook."""

from pathlib import Path

import numpy as np

from missing_covariance import (
    missing_covariance_skip_rows,
    missing_correlation_skip_rows,
    missing_covariance_pairwise,
    missing_correlation_pairwise,
)


# homework and class are sibling folders inside GitHubs.
DATA_DIR = Path(__file__).resolve().parents[2] / "class" / "testfiles" / "data"


def test_1_1_covariance_skip_missing_rows():
    input_data = np.genfromtxt(DATA_DIR / "test1.csv", delimiter=",", skip_header=1)
    expected = np.genfromtxt(DATA_DIR / "testout_1.1.csv", delimiter=",", skip_header=1)

    # Test 1.1 uses all five columns, but only rows with no missing values.
    assert input_data.shape == (10, 5)
    assert np.sum(~np.isnan(input_data).any(axis=1)) == 4

    actual = missing_covariance_skip_rows(input_data)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_1_2_correlation_skip_missing_rows():
    input_data = np.genfromtxt(DATA_DIR / "test1.csv", delimiter=",", skip_header=1)
    expected = np.genfromtxt(DATA_DIR / "testout_1.2.csv", delimiter=",", skip_header=1)

    actual = missing_correlation_skip_rows(input_data)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_1_3_covariance_pairwise():
    input_data = np.genfromtxt(DATA_DIR / "test1.csv", delimiter=",", skip_header=1)
    expected = np.genfromtxt(DATA_DIR / "testout_1.3.csv", delimiter=",", skip_header=1)

    actual = missing_covariance_pairwise(input_data)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_1_4_correlation_pairwise():
    input_data = np.genfromtxt(DATA_DIR / "test1.csv", delimiter=",", skip_header=1)
    expected = np.genfromtxt(DATA_DIR / "testout_1.4.csv", delimiter=",", skip_header=1)

    actual = missing_correlation_pairwise(input_data)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

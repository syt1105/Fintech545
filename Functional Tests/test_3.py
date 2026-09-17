"""Functional tests 3.1–3.4 from the class Tests.xlsx workbook."""

from pathlib import Path

import numpy as np

from psd_repair import near_psd, higham_nearest_psd


DATA_DIR = Path(__file__).resolve().parents[2] / "class" / "testfiles" / "data"


def read_csv(filename):
    return np.genfromtxt(DATA_DIR / filename, delimiter=",", skip_header=1)


def test_3_1_near_psd_covariance():
    input_matrix = read_csv("testout_1.3.csv")
    expected = read_csv("testout_3.1.csv")

    actual = near_psd(input_matrix)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_3_2_near_psd_correlation():
    input_matrix = read_csv("testout_1.4.csv")
    expected = read_csv("testout_3.2.csv")

    actual = near_psd(input_matrix)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_3_3_higham_covariance():
    input_matrix = read_csv("testout_1.3.csv")
    expected = read_csv("testout_3.3.csv")

    actual = higham_nearest_psd(input_matrix)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_3_4_higham_correlation():
    input_matrix = read_csv("testout_1.4.csv")
    expected = read_csv("testout_3.4.csv")

    actual = higham_nearest_psd(input_matrix)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

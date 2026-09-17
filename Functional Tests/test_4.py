"""Functional test 4.1 from the class Tests.xlsx workbook."""

from pathlib import Path

import numpy as np

from chol_psd import chol_psd


DATA_DIR = Path(__file__).resolve().parents[2] / "class" / "testfiles" / "data"


def test_4_1_chol_psd():
    input_matrix = np.genfromtxt(DATA_DIR / "testout_3.1.csv", delimiter=",", skip_header=1)
    expected = np.genfromtxt(DATA_DIR / "testout_4.1.csv", delimiter=",", skip_header=1)

    actual = chol_psd(input_matrix)
    assert actual.shape == expected.shape == (5, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual @ actual.T, input_matrix, rtol=1e-10, atol=1e-12)

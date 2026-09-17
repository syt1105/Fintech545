"""Functional tests 7.1–7.6 from the class Tests.xlsx workbook."""

import csv
from pathlib import Path

import numpy as np

from fitting import (
    aicc_t,
    fit_nig_mle,
    fit_nig_moments,
    fit_normal,
    fit_t,
    fit_t_regression,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "class" / "testfiles" / "data"


def read_csv(filename):
    with open(DATA_DIR / filename, newline="") as file:
        rows = list(csv.reader(file))
    headers = rows[0]
    values = np.array([[float(value) for value in row] for row in rows[1:]])
    return headers, values


def test_7_1_normal_fit():
    _, data = read_csv("test7_1.csv")
    headers, expected = read_csv("testout7_1.csv")
    assert headers == ["mu", "sigma"]
    actual = fit_normal(data[:, 0])
    np.testing.assert_allclose(actual, expected[0], rtol=1e-10, atol=1e-12)


def test_7_2_t_fit():
    _, data = read_csv("test7_2.csv")
    headers, expected = read_csv("testout7_2.csv")
    assert headers == ["mu", "sigma", "nu"]
    actual = fit_t(data[:, 0])
    np.testing.assert_allclose(actual, expected[0], rtol=1e-5, atol=1e-7)


def test_7_3_t_regression():
    headers, data = read_csv("test7_3.csv")
    output_headers, expected = read_csv("testout7_3.csv")
    assert headers == ["x1", "x2", "x3", "y"]
    assert output_headers == ["mu", "sigma", "nu", "Alpha", "B1", "B2", "B3"]
    actual = fit_t_regression(data[:, :3], data[:, 3])
    np.testing.assert_allclose(actual, expected[0], rtol=1e-5, atol=1e-6)


def test_7_4_t_aicc():
    _, data = read_csv("test7_2.csv")
    headers, expected = read_csv("testout7_4.csv")
    assert headers == ["AICC"]
    actual = aicc_t(data[:, 0], fit_t(data[:, 0]))
    np.testing.assert_allclose(actual, expected[0, 0], rtol=1e-9, atol=1e-6)


def test_7_5_nig_method_of_moments():
    _, data = read_csv("test7_5.csv")
    headers, expected = read_csv("testout7_5.csv")
    assert headers == ["mu", "alpha", "beta", "delta"]
    actual = fit_nig_moments(data[:, 0])
    np.testing.assert_allclose(actual, expected[0], rtol=1e-10, atol=1e-12)


def test_7_6_nig_maximum_likelihood():
    _, data = read_csv("test7_5.csv")
    headers, expected = read_csv("testout7_6.csv")
    assert headers == ["mu", "alpha", "beta", "delta"]
    actual = fit_nig_mle(data[:, 0])
    np.testing.assert_allclose(actual, expected[0], rtol=1e-10, atol=1e-12)

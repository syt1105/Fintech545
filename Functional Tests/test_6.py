"""Functional tests 6.1 and 6.2 from the class Tests.xlsx workbook."""

import csv
from pathlib import Path

import numpy as np

from returns import calculate_returns


DATA_DIR = Path(__file__).resolve().parents[2] / "class" / "testfiles" / "data"


def read_csv(filename):
    with open(DATA_DIR / filename, newline="") as file:
        rows = list(csv.reader(file))
    headers = rows[0]
    dates = [row[0] for row in rows[1:]]
    values = np.array([[float(value) for value in row[1:]] for row in rows[1:]])
    return headers, dates, values


def test_6_1_arithmetic_returns():
    headers, dates, prices = read_csv("test6.csv")
    expected_headers, expected_dates, expected = read_csv("testout6_1.csv")

    actual_dates, actual = calculate_returns(dates, prices)
    assert headers == expected_headers
    assert actual_dates == expected_dates
    assert prices.shape == (266, 101)
    assert actual.shape == expected.shape == (265, 101)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_6_2_log_returns():
    headers, dates, prices = read_csv("test6.csv")
    expected_headers, expected_dates, expected = read_csv("testout6_2.csv")

    actual_dates, actual = calculate_returns(dates, prices, method="LOG")
    assert headers == expected_headers
    assert actual_dates == expected_dates
    assert actual.shape == expected.shape == (265, 101)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

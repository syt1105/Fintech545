"""Calculate returns from daily prices."""

import numpy as np


def calculate_returns(dates, prices, method="DISCRETE"):
    """Return dates and arithmetic or log returns for each price column."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 2 or len(prices) < 2 or len(dates) != len(prices):
        raise ValueError("dates and prices must contain at least two matching rows")

    ratio = prices[1:] / prices[:-1]
    if method.upper() == "DISCRETE":
        returns = ratio - 1.0
    elif method.upper() == "LOG":
        returns = np.log(ratio)
    else:
        raise ValueError("method must be DISCRETE or LOG")

    # The first date has no previous price, so it has no return.
    return dates[1:], returns

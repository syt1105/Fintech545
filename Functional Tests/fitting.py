"""Distribution fitting functions for Test 7."""

import numpy as np
from scipy import optimize, stats


def fit_normal(data):
    """Return mean and sample standard deviation."""
    return np.mean(data), np.std(data, ddof=1)


def fit_t(data):
    """Return location, scale, and degrees of freedom of a fitted t."""
    degrees, location, scale = stats.t.fit(data)
    return location, scale, degrees


def fit_t_regression(features, y):
    """Fit an intercept and slopes with Student-t errors centered at zero."""
    x = np.column_stack((np.ones(len(y)), features))
    start_beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ start_beta
    start_degrees, _, start_scale = stats.t.fit(residuals, floc=0)

    def negative_log_likelihood(parameters):
        beta = parameters[: x.shape[1]]
        scale, degrees = parameters[-2:]
        errors = y - x @ beta
        return -np.sum(stats.t.logpdf(errors, df=degrees, loc=0, scale=scale))

    start = np.r_[start_beta, start_scale, start_degrees]
    bounds = [(None, None)] * x.shape[1] + [(1e-6, None), (2.0001, None)]
    result = optimize.minimize(
        negative_log_likelihood,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError("t regression optimization failed")

    beta = result.x[: x.shape[1]]
    scale, degrees = result.x[-2:]
    return np.r_[0.0, scale, degrees, beta]


def aicc_t(data, fit):
    """AIC with the small-sample correction for three t parameters."""
    location, scale, degrees = fit
    log_likelihood = np.sum(stats.t.logpdf(data, df=degrees, loc=location, scale=scale))
    n = len(data)
    k = 3
    return -2 * log_likelihood + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_nig_moments(data):
    """Return NIG mu, alpha, beta, delta from sample moments."""
    mean = np.mean(data)
    variance = np.var(data, ddof=1)
    skewness = stats.skew(data, bias=True)
    excess_kurtosis = stats.kurtosis(data, bias=True)
    if excess_kurtosis <= 0:
        raise ValueError("NIG moments need positive excess kurtosis")

    t = skewness**2 / excess_kurtosis
    if t >= 3 / 5:
        raise ValueError("sample moments are outside the NIG region")
    rho_squared = t / (3 - 4 * t)
    rho = np.sign(skewness) * np.sqrt(rho_squared)
    delta_gamma = 3 * (1 + 4 * rho_squared) / excess_kurtosis
    alpha = np.sqrt(delta_gamma / (variance * (1 - rho_squared) ** 2))
    beta = rho * alpha
    gamma = alpha * np.sqrt(1 - rho_squared)
    delta = delta_gamma / gamma
    mu = mean - delta * beta / gamma
    return np.array([mu, alpha, beta, delta])


def fit_nig_mle(data):
    """Fit NIG by likelihood and convert SciPy's parameterization."""
    a, b, location, scale = stats.norminvgauss.fit(data)
    # SciPy uses a=alpha*delta, b=beta*delta, loc=mu, scale=delta.
    return np.array([location, a / scale, b / scale, scale])

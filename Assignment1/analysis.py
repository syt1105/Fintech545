"""FinTech 545 - Assignment 1 reproducible analysis.

Run from the Assignment1 directory with:

    python3 analysis.py

The script reads problem1.csv through problem5.csv from the same directory,
creates all figures and tables, and writes a machine-readable results.json.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
FIGURE_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
    }
)


def aicc(log_likelihood: float, n: int, k: int) -> float:
    """Return the small-sample corrected Akaike information criterion."""
    aic = 2 * k - 2 * log_likelihood
    return aic + (2 * k * k + 2 * k) / (n - k - 1)


def first_four_moments(x: np.ndarray) -> dict[str, float]:
    """Compute the Week 1 moment convention using denominator n.

    The fourth number is excess kurtosis (ordinary kurtosis minus three).
    This matches the convention shown in the course notes and avoids the
    package-to-package ambiguity in bias corrections.
    """
    centered = x - x.mean()
    variance = np.mean(centered**2)
    return {
        "mean": float(x.mean()),
        "variance": float(variance),
        "standard_deviation": float(np.sqrt(variance)),
        "skewness": float(np.mean(centered**3) / variance**1.5),
        "excess_kurtosis": float(np.mean(centered**4) / variance**2 - 3.0),
    }


def fit_student_t_regression(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Fit y = alpha + beta*x + Student-t error by maximum likelihood.

    The optimized error parameters are the scipy Student-t scale and degrees
    of freedom. We constrain nu > 2 so that the fitted error variance exists.
    Several initial values are tried because heavy-tailed likelihoods can have
    shallow local optima.
    """
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    ols_alpha, ols_beta = ols.params
    ols_scale = max(float(np.sqrt(np.mean(ols.resid**2))), 1e-8)
    robust_slope, robust_intercept, _, _ = stats.theilslopes(y, x)

    def unpack(theta: np.ndarray) -> tuple[float, float, float, float]:
        alpha, beta, log_scale, log_df_minus_two = theta
        return alpha, beta, np.exp(log_scale), 2.0 + np.exp(log_df_minus_two)

    def negative_log_likelihood(theta: np.ndarray) -> float:
        alpha, beta, scale, df = unpack(theta)
        residual = y - alpha - beta * x
        value = -np.sum(stats.t.logpdf(residual, df=df, loc=0.0, scale=scale))
        return float(value) if np.isfinite(value) else 1e100

    starts: list[np.ndarray] = []
    for alpha, beta in [(ols_alpha, ols_beta), (robust_intercept, robust_slope)]:
        residual = y - alpha - beta * x
        robust_scale = max(float(np.median(np.abs(residual)) / 0.67448975), 1e-6)
        for scale in [robust_scale, ols_scale]:
            for df in [3.0, 5.0, 10.0, 30.0]:
                starts.append(
                    np.array([alpha, beta, np.log(scale), np.log(df - 2.0)])
                )

    bounds = [(None, None), (None, None), (-20.0, 20.0), (-7.0, 12.0)]
    candidates = [
        optimize.minimize(
            negative_log_likelihood,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-8},
        )
        for start in starts
    ]
    successful = [result for result in candidates if result.success]
    if not successful:
        messages = "; ".join(str(result.message) for result in candidates)
        raise RuntimeError(f"Student-t MLE failed: {messages}")

    best = min(successful, key=lambda result: result.fun)
    alpha, beta, scale, df = unpack(best.x)
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "scale": float(scale),
        "degrees_of_freedom": float(df),
        "implied_standard_deviation": float(scale * np.sqrt(df / (df - 2.0))),
        "log_likelihood": float(-best.fun),
        "converged": bool(best.success),
    }


def problem1() -> dict:
    data = pd.read_csv(ROOT / "problem1.csv")
    x = data["x"].to_numpy(float)
    moments = first_four_moments(x)

    # Moment matching for a Normal uses the population/MLE variance (denominator n).
    normal_quantile_01 = stats.norm.ppf(
        0.01, loc=moments["mean"], scale=moments["standard_deviation"]
    )
    observed_below = int(np.sum(x < normal_quantile_01))
    expected_below = float(0.01 * len(x))

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    sns.histplot(x=x, bins=45, stat="density", color="#4C78A8", alpha=0.45, ax=ax)
    grid = np.linspace(x.min(), x.max(), 600)
    ax.plot(
        grid,
        stats.norm.pdf(
            grid, loc=moments["mean"], scale=moments["standard_deviation"]
        ),
        color="#E45756",
        linewidth=2.2,
        label="Moment-matched Normal",
    )
    ax.axvline(
        normal_quantile_01,
        color="#B279A2",
        linestyle="--",
        linewidth=1.8,
        label="Fitted Normal 1% quantile",
    )
    ax.set(title="Problem 1: Sample shape and fitted Normal", xlabel="x", ylabel="Density")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "problem1_distribution.png", bbox_inches="tight")
    plt.close(fig)

    result = {
        "n": len(x),
        **moments,
        "normal_quantile_01": float(normal_quantile_01),
        "observed_below_normal_quantile_01": observed_below,
        "expected_below_normal_quantile_01": expected_below,
        "sample_minimum": float(x.min()),
        "sample_maximum": float(x.max()),
    }
    pd.DataFrame([result]).to_csv(TABLE_DIR / "problem1_results.csv", index=False)
    return result


def problem2() -> dict:
    data = pd.read_csv(ROOT / "problem2.csv")
    x = data["x"].to_numpy(float)
    y = data["y"].to_numpy(float)
    n = len(data)
    X = sm.add_constant(x)

    # OLS and Normal MLE share alpha and beta. Their sigma conventions differ:
    # OLS reports sqrt(SSE/(n-2)); Normal MLE uses sqrt(SSE/n).
    ols = sm.OLS(y, X).fit()
    alpha_ols, beta_ols = map(float, ols.params)
    residuals_ols = np.asarray(ols.resid)
    sigma_ols = float(np.sqrt(np.sum(residuals_ols**2) / (n - 2)))
    sigma_normal_mle = float(np.sqrt(np.mean(residuals_ols**2)))
    normal_log_likelihood = float(
        np.sum(stats.norm.logpdf(residuals_ols, loc=0.0, scale=sigma_normal_mle))
    )
    normal_aicc = aicc(normal_log_likelihood, n=n, k=3)

    student = fit_student_t_regression(x, y)
    student_aicc = aicc(student["log_likelihood"], n=n, k=4)

    quantile_rows = []
    for probability in [0.95, 0.995]:
        normal_q = float(stats.norm.ppf(probability, scale=sigma_normal_mle))
        student_q = float(
            stats.t.ppf(
                probability,
                df=student["degrees_of_freedom"],
                scale=student["scale"],
            )
        )
        quantile_rows.append(
            {
                "probability": probability,
                "normal_error_quantile": normal_q,
                "student_t_error_quantile": student_q,
                "student_minus_normal": student_q - normal_q,
            }
        )
    quantiles = pd.DataFrame(quantile_rows)
    quantiles.to_csv(TABLE_DIR / "problem2_error_quantiles.csv", index=False)

    model_table = pd.DataFrame(
        [
            {
                "model": "OLS",
                "alpha": alpha_ols,
                "beta": beta_ols,
                "alpha_standard_error": float(ols.bse[0]),
                "beta_standard_error": float(ols.bse[1]),
                "error_scale": sigma_ols,
                "degrees_of_freedom": np.nan,
                "log_likelihood": normal_log_likelihood,
                "parameter_count": 3,
                "AICc": normal_aicc,
            },
            {
                "model": "Normal MLE",
                "alpha": alpha_ols,
                "beta": beta_ols,
                "alpha_standard_error": np.nan,
                "beta_standard_error": np.nan,
                "error_scale": sigma_normal_mle,
                "degrees_of_freedom": np.nan,
                "log_likelihood": normal_log_likelihood,
                "parameter_count": 3,
                "AICc": normal_aicc,
            },
            {
                "model": "Student-t MLE",
                "alpha": student["alpha"],
                "beta": student["beta"],
                "alpha_standard_error": np.nan,
                "beta_standard_error": np.nan,
                "error_scale": student["scale"],
                "degrees_of_freedom": student["degrees_of_freedom"],
                "log_likelihood": student["log_likelihood"],
                "parameter_count": 4,
                "AICc": student_aicc,
            },
        ]
    )
    model_table.to_csv(TABLE_DIR / "problem2_model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    axes[0].scatter(x, y, s=25, alpha=0.68, color="#4C78A8", edgecolor="none")
    x_grid = np.linspace(x.min(), x.max(), 300)
    axes[0].plot(
        x_grid,
        alpha_ols + beta_ols * x_grid,
        color="#E45756",
        linewidth=2.1,
        label="OLS / Normal MLE",
    )
    axes[0].plot(
        x_grid,
        student["alpha"] + student["beta"] * x_grid,
        color="#59A14F",
        linewidth=2.1,
        linestyle="--",
        label="Student-t MLE",
    )
    axes[0].set(title="Scatter and fitted lines", xlabel="x", ylabel="y")
    axes[0].legend(frameon=True)

    residual_grid = np.linspace(
        min(residuals_ols.min(), -6 * sigma_normal_mle),
        max(residuals_ols.max(), 6 * sigma_normal_mle),
        800,
    )
    sns.histplot(
        x=residuals_ols,
        bins=35,
        stat="density",
        color="#4C78A8",
        alpha=0.35,
        ax=axes[1],
        label="OLS residuals",
    )
    axes[1].plot(
        residual_grid,
        stats.norm.pdf(residual_grid, scale=sigma_normal_mle),
        color="#E45756",
        linewidth=2.0,
        label="Fitted Normal",
    )
    axes[1].plot(
        residual_grid,
        stats.t.pdf(
            residual_grid,
            df=student["degrees_of_freedom"],
            scale=student["scale"],
        ),
        color="#59A14F",
        linewidth=2.0,
        linestyle="--",
        label="Fitted Student-t",
    )
    axes[1].set(title="Error distribution", xlabel="Residual", ylabel="Density")
    axes[1].legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "problem2_regression.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "n": n,
        "ols": {
            "alpha": alpha_ols,
            "beta": beta_ols,
            "alpha_standard_error": float(ols.bse[0]),
            "beta_standard_error": float(ols.bse[1]),
            "residual_standard_error": sigma_ols,
        },
        "normal_mle": {
            "alpha": alpha_ols,
            "beta": beta_ols,
            "sigma": sigma_normal_mle,
            "log_likelihood": normal_log_likelihood,
            "AICc": normal_aicc,
        },
        "student_t_mle": {**student, "AICc": student_aicc},
        "AICc_difference_normal_minus_t": float(normal_aicc - student_aicc),
        "beta_difference_ols_minus_t": float(beta_ols - student["beta"]),
        "error_quantiles": quantile_rows,
    }


def problem3() -> dict:
    data = pd.read_csv(ROOT / "problem3.csv")
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    gap = (pearson - spearman).abs()

    pairs = []
    columns = list(data.columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            pairs.append(
                {
                    "pair": f"{first}, {second}",
                    "pearson": float(pearson.loc[first, second]),
                    "spearman": float(spearman.loc[first, second]),
                    "absolute_gap": float(gap.loc[first, second]),
                }
            )
    pair_table = pd.DataFrame(pairs).sort_values("absolute_gap", ascending=False)
    pair_table.to_csv(TABLE_DIR / "problem3_pair_correlations.csv", index=False)
    pearson.to_csv(TABLE_DIR / "problem3_pearson_matrix.csv")
    spearman.to_csv(TABLE_DIR / "problem3_spearman_matrix.csv")

    pair_grid = sns.pairplot(
        data,
        diag_kind="hist",
        plot_kws={"s": 16, "alpha": 0.55, "color": "#4C78A8", "edgecolor": "none"},
        diag_kws={"bins": 28, "color": "#4C78A8", "alpha": 0.7},
        height=2.05,
    )
    pair_grid.fig.suptitle("Problem 3: Pairwise relationships", y=1.01, fontweight="bold")
    pair_grid.fig.savefig(FIGURE_DIR / "problem3_pairplot.png", bbox_inches="tight")
    plt.close(pair_grid.fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
    sns.heatmap(
        pearson,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        cbar=False,
        ax=axes[0],
    )
    axes[0].set_title("Pearson")
    sns.heatmap(
        spearman,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        square=True,
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Spearman")
    sns.heatmap(
        gap,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(0.25, float(gap.to_numpy().max())),
        square=True,
        cbar=False,
        ax=axes[2],
    )
    axes[2].set_title("Absolute difference")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "problem3_correlations.png", bbox_inches="tight")
    plt.close(fig)

    largest = pair_table.iloc[0].to_dict()
    return {
        "n": len(data),
        "pearson": pearson.to_dict(),
        "spearman": spearman.to_dict(),
        "pairs_ranked_by_gap": pair_table.to_dict(orient="records"),
        "largest_gap_pair": largest,
    }


def problem4() -> dict:
    data = pd.read_csv(ROOT / "problem4.csv")
    x1 = data["x1"].to_numpy(float)
    x2 = data["x2"].to_numpy(float)
    n = len(data)
    means = data.mean()
    covariance = data.cov()  # Sample covariance, denominator n-1.

    sigma11 = float(covariance.loc["x1", "x1"])
    sigma12 = float(covariance.loc["x1", "x2"])
    sigma21 = float(covariance.loc["x2", "x1"])
    sigma22 = float(covariance.loc["x2", "x2"])

    beta = sigma21 / sigma11
    intercept = float(means["x2"] - beta * means["x1"])
    conditional_variance = sigma22 - sigma21 * (1.0 / sigma11) * sigma12
    variance_factor = conditional_variance / sigma22
    standard_deviation_factor = np.sqrt(variance_factor)
    conditional_sd = np.sqrt(conditional_variance)

    conditional_mean = means["x2"] + beta * (x1 - means["x1"])
    lower = conditional_mean - stats.norm.ppf(0.975) * conditional_sd
    upper = conditional_mean + stats.norm.ppf(0.975) * conditional_sd
    inside = (x2 >= lower) & (x2 <= upper)

    standardized_distance = np.abs(x1 - means["x1"]) / np.sqrt(sigma11)
    bucket = pd.cut(
        standardized_distance,
        bins=[-np.inf, 1.0, 2.0, np.inf],
        labels=["within_1_sd", "between_1_and_2_sd", "beyond_2_sd"],
        right=True,
    )
    residual = x2 - conditional_mean
    coverage_table = (
        pd.DataFrame({"bucket": bucket, "inside": inside, "residual": residual})
        .groupby("bucket", observed=False)
        .agg(
            observations=("inside", "size"),
            inside_band=("inside", "sum"),
            coverage=("inside", "mean"),
            residual_mean=("residual", "mean"),
            residual_standard_deviation=("residual", "std"),
        )
        .reset_index()
    )
    coverage_table.to_csv(TABLE_DIR / "problem4_bucket_coverage.csv", index=False)
    covariance.to_csv(TABLE_DIR / "problem4_covariance_matrix.csv")

    order = np.argsort(x1)
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    ax.scatter(x1, x2, s=18, alpha=0.45, color="#4C78A8", edgecolor="none", label="Data")
    ax.plot(
        x1[order],
        conditional_mean[order],
        color="#E45756",
        linewidth=2.2,
        label=r"$E[x_2\mid x_1]$",
    )
    ax.fill_between(
        x1[order],
        lower[order],
        upper[order],
        color="#E45756",
        alpha=0.17,
        label="Constant-width 95% Normal band",
    )
    ax.set(
        title="Problem 4: Conditional expectation and 95% band",
        xlabel="x1",
        ylabel="x2",
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "problem4_conditional_distribution.png", bbox_inches="tight")
    plt.close(fig)

    return {
        "n": n,
        "means": means.to_dict(),
        "sample_covariance": covariance.to_dict(),
        "conditional_mean_intercept": intercept,
        "conditional_mean_slope": beta,
        "conditional_variance": float(conditional_variance),
        "conditional_standard_deviation": float(conditional_sd),
        "conditional_to_unconditional_variance_factor": float(variance_factor),
        "conditional_to_unconditional_sd_factor": float(standard_deviation_factor),
        "overall_inside_band": int(inside.sum()),
        "overall_coverage": float(inside.mean()),
        "bucket_coverage": coverage_table.to_dict(orient="records"),
    }


def problem5() -> dict:
    data = pd.read_csv(ROOT / "problem5.csv")
    x = data["x"].to_numpy(float)
    n = len(x)
    max_lag = 20
    significance_band = 1.96 / np.sqrt(n)
    acf_values = acf(x, nlags=max_lag, fft=False)
    pacf_values = pacf(x, nlags=max_lag, method="ywmle")
    lag_table = pd.DataFrame(
        {
            "lag": np.arange(1, max_lag + 1),
            "ACF": acf_values[1:],
            "PACF": pacf_values[1:],
            "outside_fixed_95_percent_band_ACF": np.abs(acf_values[1:])
            > significance_band,
            "outside_fixed_95_percent_band_PACF": np.abs(pacf_values[1:])
            > significance_band,
        }
    )
    lag_table.to_csv(TABLE_DIR / "problem5_acf_pacf.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(8.3, 8.0))
    axes[0].plot(np.arange(1, n + 1), x, color="#4C78A8", linewidth=1.0)
    axes[0].axhline(np.mean(x), color="#E45756", linestyle="--", linewidth=1.3)
    axes[0].set(title="Series", xlabel="Observation", ylabel="x")
    lags = np.arange(1, max_lag + 1)
    for ax, values, title in [
        (axes[1], acf_values[1:], "ACF with fixed 95% significance band"),
        (axes[2], pacf_values[1:], "PACF with fixed 95% significance band"),
    ]:
        ax.axhspan(
            -significance_band,
            significance_band,
            color="#4C78A8",
            alpha=0.16,
            label=r"$\pm 1.96/\sqrt{n}$",
        )
        ax.axhline(0.0, color="#4C78A8", linewidth=1.0)
        ax.vlines(lags, 0.0, values, color="#4C78A8", linewidth=1.5)
        ax.scatter(lags, values, color="#4C78A8", s=28, zorder=3)
        ax.set(title=title, xlabel="Lag", ylabel="Correlation", xlim=(0.5, max_lag + 0.5))
        ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "problem5_series_acf_pacf.png", bbox_inches="tight")
    plt.close(fig)

    rows = []
    model_details = {}
    for model_type in ["AR", "MA"]:
        for order in [1, 2, 3]:
            arima_order = (order, 0, 0) if model_type == "AR" else (0, 0, order)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(
                    x,
                    order=arima_order,
                    trend="c",
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                ).fit()
            k = len(fitted.params)
            corrected_aic = aicc(float(fitted.llf), n=n, k=k)
            model_name = f"{model_type}({order})"
            row = {
                "model": model_name,
                "log_likelihood": float(fitted.llf),
                "parameter_count": k,
                "AIC": float(fitted.aic),
                "AICc": float(corrected_aic),
            }
            rows.append(row)
            model_details[model_name] = {
                "parameters": {
                    name: float(value)
                    for name, value in zip(fitted.param_names, fitted.params)
                },
                **row,
            }

    model_table = pd.DataFrame(rows).sort_values("AICc")
    model_table.to_csv(TABLE_DIR / "problem5_model_comparison.csv", index=False)
    selected = str(model_table.iloc[0]["model"])

    return {
        "n": n,
        "acf_pacf_significance_band": float(significance_band),
        "acf_pacf_by_lag": lag_table.to_dict(orient="records"),
        "models": model_details,
        "selected_model": selected,
        "models_ranked_by_AICc": model_table.to_dict(orient="records"),
    }


def make_json_safe(value):
    """Convert numpy/pandas scalar types recursively for JSON output."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def main() -> None:
    required = [ROOT / f"problem{i}.csv" for i in range(1, 6)]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input data files: " + ", ".join(missing) + ". "
            "Copy them from the class repository into this directory."
        )

    results = {
        "problem1": problem1(),
        "problem2": problem2(),
        "problem3": problem3(),
        "problem4": problem4(),
        "problem5": problem5(),
    }
    safe_results = make_json_safe(results)
    (ROOT / "results.json").write_text(
        json.dumps(safe_results, indent=2, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(safe_results, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()

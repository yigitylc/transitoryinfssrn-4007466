from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg


@dataclass(frozen=True)
class DecaySummary:
    window: int
    c: float
    mu: float
    rho_T: float
    decay_6m_pct: float
    decay_12m_pct: float
    t_star_months: float
    t_star_years: float
    valid_formula: bool
    warning: str | None


def summary_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Paper-style summary statistics."""

    clean = df[columns].copy()
    stats = pd.DataFrame(index=columns)
    stats["mean"] = clean.mean()
    stats["std_dev"] = clean.std()
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        stats[f"p{int(q * 100)}"] = clean.quantile(q)
    stats["n"] = clean.count()
    return stats.reset_index(names="variable")


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].corr()


def robust_ols(y: pd.Series, x: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS with HC1 robust standard errors."""

    data = pd.concat([y.rename("y"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    y_clean = data["y"]
    x_clean = sm.add_constant(data.drop(columns="y"), has_constant="add")
    return sm.OLS(y_clean, x_clean).fit(cov_type="HC1")


def ols_table(results: dict[str, sm.regression.linear_model.RegressionResultsWrapper]) -> pd.DataFrame:
    """Compact regression table with coefficients and t-statistics."""

    rows: list[dict[str, object]] = []
    for name, result in results.items():
        for param in result.params.index:
            rows.append(
                {
                    "model": name,
                    "variable": param,
                    "coef": result.params[param],
                    "t_stat": result.tvalues[param],
                    "p_value": result.pvalues[param],
                    "r_squared": result.rsquared,
                    "nobs": int(result.nobs),
                }
            )
    return pd.DataFrame(rows)


def run_paper_style_regressions(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the paper-style CPI/TINF regression table structure."""

    required = ["inflation_yoy", "tinf_4m", "tinf_8m", "tinf_12m", "tbill_3m"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    results = {
        "short_only": robust_ols(df["inflation_yoy"], df[["tinf_4m", "tbill_3m"]]),
        "medium_only": robust_ols(df["inflation_yoy"], df[["tinf_8m", "tbill_3m"]]),
        "long_only": robust_ols(df["inflation_yoy"], df[["tinf_12m", "tbill_3m"]]),
        "all_tinf": robust_ols(
            df["inflation_yoy"], df[["tinf_4m", "tinf_8m", "tinf_12m", "tbill_3m"]]
        ),
    }
    return ols_table(results)


def fit_ar1(series: pd.Series) -> AutoReg:
    """Fit AR(1) with constant to a univariate series."""

    y = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(y) < 8:
        raise ValueError("Need at least 8 observations for AR(1)")
    return AutoReg(y, lags=1, trend="c", old_names=False).fit()


def extract_l1_param(result) -> float:
    """Extract lag-1 coefficient by parameter name, not position."""

    candidates = [name for name in result.params.index if ".L1" in name or name.endswith("L1")]
    if not candidates:
        # fallback for statsmodels naming variants
        candidates = [name for name in result.params.index if "lag" in name.lower() or "ar.L1" in name]
    if not candidates:
        raise KeyError(f"Could not find lag-1 parameter in: {list(result.params.index)}")
    return float(result.params[candidates[0]])


def rolling_ar1_rho(
    df: pd.DataFrame,
    value_col: str = "tinf_4m",
    date_col: str = "date",
    window: int = 24,
) -> pd.DataFrame:
    """Estimate rolling AR(1) rho with correct end-date alignment."""

    if value_col not in df.columns:
        raise KeyError(f"Missing value column: {value_col}")
    if date_col not in df.columns:
        raise KeyError(f"Missing date column: {date_col}")

    clean = df[[date_col, value_col]].dropna().reset_index(drop=True)
    rows: list[dict[str, object]] = []

    for end in range(window, len(clean) + 1):
        sub = clean.iloc[end - window : end]
        try:
            result = fit_ar1(sub[value_col])
            rho = extract_l1_param(result)
            rows.append(
                {
                    "date": sub[date_col].iloc[-1],
                    "rho": rho,
                    "window": window,
                    "nobs": int(result.nobs),
                }
            )
        except Exception as exc:  # keep rolling window robust for dashboard use
            rows.append(
                {
                    "date": sub[date_col].iloc[-1],
                    "rho": np.nan,
                    "window": window,
                    "nobs": len(sub),
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def paper_decay_summary(rho_df: pd.DataFrame, window: int) -> DecaySummary:
    """Compute paper-style decay summary from rolling rho estimates."""

    clean = rho_df[["rho"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 8:
        return DecaySummary(window, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, "Insufficient rho observations")

    result = fit_ar1(clean["rho"])
    c = float(result.params.get("const", np.nan))
    mu = extract_l1_param(result)
    rho_T = float(clean["rho"].iloc[-1])

    warning: str | None = None
    valid = bool((rho_T > 0) and (0 < mu < 1))
    if rho_T <= 0:
        warning = "rho_T <= 0; paper convergence formula invalid."
    elif not (0 < mu < 1):
        warning = "mu outside (0,1); paper convergence formula invalid."
    elif rho_T > 1:
        warning = "rho_T > 1; latest transitory persistence is locally explosive."

    if valid:
        decay_6m = 100 * (1 - rho_T * (mu**5))
        decay_12m = 100 * (1 - rho_T * (mu**11))
        t_star_months = 1 + np.log(0.05 / rho_T) / np.log(mu)
        t_star_years = t_star_months / 12
    else:
        decay_6m = decay_12m = t_star_months = t_star_years = np.nan

    return DecaySummary(
        window=window,
        c=c,
        mu=mu,
        rho_T=rho_T,
        decay_6m_pct=float(decay_6m),
        decay_12m_pct=float(decay_12m),
        t_star_months=float(t_star_months),
        t_star_years=float(t_star_years),
        valid_formula=valid,
        warning=warning,
    )


def decay_curve(rho_T: float, mu: float, months: int = 48) -> pd.DataFrame:
    """Paper-style decay curve."""

    horizon = np.arange(1, months + 1)
    if not ((rho_T > 0) and (0 < mu < 1)):
        return pd.DataFrame({"month": horizon, "decay_pct": np.nan, "remaining_pct": np.nan})
    decay_pct = 100 * (1 - rho_T * (mu ** (horizon - 1)))
    return pd.DataFrame({"month": horizon, "decay_pct": decay_pct, "remaining_pct": 100 - decay_pct})


def decay_summaries_for_windows(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (24, 30),
    value_col: str = "tinf_4m",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return rolling rho observations and paper decay summaries for multiple windows."""

    rho_frames = []
    summaries = []
    for window in windows:
        rho = rolling_ar1_rho(df, value_col=value_col, window=window)
        rho_frames.append(rho)
        summaries.append(paper_decay_summary(rho, window).__dict__)
    return pd.concat(rho_frames, ignore_index=True), pd.DataFrame(summaries)

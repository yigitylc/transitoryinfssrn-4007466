from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VALID_BASELINES = {
    "full_sample",
    "rolling_36_unshifted",
    "rolling_36_shifted",
    "expanding_shifted",
    "fed_target",
}


@dataclass(frozen=True)
class BaselineMeta:
    method: str
    description: str
    live_safe: bool
    warning: str | None = None


BASELINE_META: dict[str, BaselineMeta] = {
    "full_sample": BaselineMeta(
        method="full_sample",
        description="Full-sample historical mean. Useful for ex-post paper-style replication.",
        live_safe=False,
        warning="Uses future data when viewed historically. Do not use as a live signal.",
    ),
    "rolling_36_unshifted": BaselineMeta(
        method="rolling_36_unshifted",
        description="36-month rolling mean including the current month.",
        live_safe=False,
        warning="Includes current observation in its own baseline. Prefer shifted version for live signals.",
    ),
    "rolling_36_shifted": BaselineMeta(
        method="rolling_36_shifted",
        description="36-month rolling mean shifted by one month. Real-time safer baseline.",
        live_safe=True,
        warning=None,
    ),
    "expanding_shifted": BaselineMeta(
        method="expanding_shifted",
        description="Expanding historical mean shifted by one month.",
        live_safe=True,
        warning=None,
    ),
    "fed_target": BaselineMeta(
        method="fed_target",
        description="Fixed 2% inflation target baseline.",
        live_safe=True,
        warning=None,
    ),
}


def compute_baseline(
    inflation: pd.Series,
    method: str = "rolling_36_shifted",
    rolling_window: int = 36,
    expanding_min_periods: int = 120,
    fed_target: float = 2.0,
) -> pd.Series:
    """Compute mean-reversion inflation baseline."""

    if method not in VALID_BASELINES:
        raise ValueError(f"Unknown baseline method: {method}. Expected one of {sorted(VALID_BASELINES)}")

    if method == "full_sample":
        return pd.Series(inflation.mean(skipna=True), index=inflation.index, name="baseline")
    if method == "rolling_36_unshifted":
        return inflation.rolling(rolling_window, min_periods=rolling_window).mean().rename("baseline")
    if method == "rolling_36_shifted":
        return inflation.rolling(rolling_window, min_periods=rolling_window).mean().shift(1).rename("baseline")
    if method == "expanding_shifted":
        return inflation.expanding(min_periods=expanding_min_periods).mean().shift(1).rename("baseline")
    if method == "fed_target":
        return pd.Series(fed_target, index=inflation.index, name="baseline")

    raise AssertionError("Unreachable baseline branch")


def consecutive_true_count(flag: pd.Series) -> pd.Series:
    """Return run length of consecutive True values at each observation."""

    values = flag.fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(values), dtype=int)
    run = 0
    for i, value in enumerate(values):
        run = run + 1 if value else 0
        out[i] = run
    return pd.Series(out, index=flag.index, name="run_length_above")


def _lineage_flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].fillna(False).astype(bool)


def _baseline_lineage(
    flag: pd.Series,
    method: str,
    rolling_window: int = 36,
) -> pd.Series:
    """Propagate a CPI-input flag through the selected baseline dependency set."""

    numeric = flag.fillna(False).astype(int)
    if method == "full_sample":
        return pd.Series(bool(numeric.any()), index=flag.index, dtype=bool)
    if method == "rolling_36_unshifted":
        return numeric.rolling(rolling_window, min_periods=1).max().astype(bool)
    if method == "rolling_36_shifted":
        return (
            numeric.rolling(rolling_window, min_periods=1)
            .max()
            .shift(1, fill_value=0)
            .astype(bool)
        )
    if method == "expanding_shifted":
        return numeric.expanding(min_periods=1).max().shift(1, fill_value=0).astype(bool)
    if method == "fed_target":
        return pd.Series(False, index=flag.index, dtype=bool)
    raise ValueError(f"Unknown baseline method: {method}")


def _rolling_lineage(flag: pd.Series, window: int) -> pd.Series:
    return flag.astype(int).rolling(window, min_periods=1).max().astype(bool)


def add_transitory_inflation_features(
    df: pd.DataFrame,
    inflation_col: str = "inflation_yoy",
    baseline_method: str = "rolling_36_shifted",
    tinf_windows: tuple[int, ...] = (4, 8, 12),
) -> pd.DataFrame:
    """Add continuous TINF and diagnostic persistence flags.

    The continuous TINF signal is magnitude-based. Binary flags are only diagnostics.
    """

    if inflation_col not in df.columns:
        raise KeyError(f"Missing inflation column: {inflation_col}")

    out = df.copy()
    out["baseline_method"] = baseline_method
    out["baseline"] = compute_baseline(out[inflation_col], baseline_method)
    out["epsilon"] = out[inflation_col] - out["baseline"]

    inflation_imputed = _lineage_flag(
        out,
        f"{inflation_col}_uses_imputed_input",
    )
    inflation_missing = _lineage_flag(
        out,
        f"{inflation_col}_uses_missing_input",
    )
    out["baseline_uses_imputed_input"] = _baseline_lineage(
        inflation_imputed,
        baseline_method,
    )
    out["baseline_uses_missing_input"] = _baseline_lineage(
        inflation_missing,
        baseline_method,
    )
    out["epsilon_uses_imputed_input"] = (
        inflation_imputed | out["baseline_uses_imputed_input"]
    )
    out["epsilon_uses_missing_input"] = (
        inflation_missing | out["baseline_uses_missing_input"]
    )

    for window in tinf_windows:
        out[f"tinf_{window}m"] = out["epsilon"].rolling(window, min_periods=window).mean()
        out[f"tinf_{window}m_uses_imputed_input"] = _rolling_lineage(
            out["epsilon_uses_imputed_input"],
            window,
        )
        out[f"tinf_{window}m_uses_missing_input"] = _rolling_lineage(
            out["epsilon_uses_missing_input"],
            window,
        )

    imputed_lineage_cols = [
        f"{inflation_col}_uses_imputed_input",
        "baseline_uses_imputed_input",
        "epsilon_uses_imputed_input",
        *(f"tinf_{window}m_uses_imputed_input" for window in tinf_windows),
    ]
    missing_lineage_cols = [
        f"{inflation_col}_uses_missing_input",
        "baseline_uses_missing_input",
        "epsilon_uses_missing_input",
        *(f"tinf_{window}m_uses_missing_input" for window in tinf_windows),
    ]
    out["signal_uses_imputed_input"] = pd.concat(
        [_lineage_flag(out, column) for column in imputed_lineage_cols],
        axis=1,
    ).any(axis=1)
    out["signal_uses_missing_input"] = pd.concat(
        [_lineage_flag(out, column) for column in missing_lineage_cols],
        axis=1,
    ).any(axis=1)
    out["signal_observed_only_eligible"] = ~(
        out["signal_uses_imputed_input"] | out["signal_uses_missing_input"]
    )

    out["above_baseline"] = out["epsilon"] > 0
    out["run_length_above"] = consecutive_true_count(out["above_baseline"])
    out["short_regime_flag"] = out["run_length_above"] >= 4
    out["medium_regime_flag"] = out["run_length_above"] >= 8
    out["long_regime_flag"] = out["run_length_above"] >= 12

    if {"tinf_4m", "tinf_8m", "tinf_12m"}.issubset(out.columns):
        out["tinf_term_structure"] = np.select(
            [
                (out["tinf_4m"] > out["tinf_8m"]) & (out["tinf_8m"] > out["tinf_12m"]),
                (out["tinf_4m"] < out["tinf_8m"]) & (out["tinf_8m"] < out["tinf_12m"]),
            ],
            ["accelerating", "decelerating"],
            default="mixed",
        )

    return out


def latest_signal_snapshot(df: pd.DataFrame) -> dict[str, object]:
    """Create a small current-signal dictionary for dashboards/reports."""

    cols = ["date", "inflation_yoy", "baseline", "epsilon", "tinf_4m", "tinf_8m", "tinf_12m"]
    clean = df.dropna(subset=[col for col in cols if col in df.columns]).copy()
    if clean.empty:
        return {"available": False, "reason": "No complete TINF observations available."}

    latest = clean.iloc[-1]
    tinf_hist = clean["tinf_4m"].dropna()
    percentile = float((tinf_hist <= latest["tinf_4m"]).mean() * 100)

    prev = clean.iloc[-2] if len(clean) >= 2 else latest
    rising = latest["tinf_4m"] > prev["tinf_4m"]

    component_imputation_lineage = {
        "inflation_yoy_uses_imputed_input": bool(
            latest.get("inflation_yoy_uses_imputed_input", False)
        ),
        "baseline_uses_imputed_input": bool(
            latest.get("baseline_uses_imputed_input", False)
        ),
        "epsilon_uses_imputed_input": bool(
            latest.get("epsilon_uses_imputed_input", False)
        ),
        "tinf_4m_uses_imputed_input": bool(
            latest.get("tinf_4m_uses_imputed_input", False)
        ),
        "tinf_8m_uses_imputed_input": bool(
            latest.get("tinf_8m_uses_imputed_input", False)
        ),
        "tinf_12m_uses_imputed_input": bool(
            latest.get("tinf_12m_uses_imputed_input", False)
        ),
    }
    # Percentiles and the regime thresholds are full-sample descriptive reads.
    # Conservatively retain lineage when any TINF observation contributing to
    # their reference distribution depends on an estimated CPI input.
    distribution_uses_imputed_input = bool(
        _lineage_flag(clean, "tinf_4m_uses_imputed_input").any()
    )
    component_imputation_lineage["percentile_uses_imputed_input"] = (
        distribution_uses_imputed_input
    )
    component_imputation_lineage["regime_uses_imputed_input"] = (
        distribution_uses_imputed_input
    )
    uses_imputed_input = any(component_imputation_lineage.values())
    uses_missing_input = bool(latest.get("signal_uses_missing_input", False))

    if latest["tinf_4m"] > tinf_hist.quantile(0.75):
        regime = "elevated rising" if rising else "elevated falling"
    elif latest["tinf_4m"] < tinf_hist.quantile(0.25):
        regime = "disinflationary"
    else:
        regime = "neutral"

    return {
        "available": True,
        "date": latest["date"] if "date" in clean.columns else clean.index[-1],
        "inflation_yoy": float(latest["inflation_yoy"]),
        "baseline": float(latest["baseline"]),
        "epsilon": float(latest["epsilon"]),
        "tinf_4m": float(latest["tinf_4m"]),
        "tinf_8m": float(latest["tinf_8m"]),
        "tinf_12m": float(latest["tinf_12m"]),
        "tinf_4m_percentile": percentile,
        "regime": regime,
        "term_structure": latest.get("tinf_term_structure", "n/a"),
        **component_imputation_lineage,
        "uses_imputed_input": uses_imputed_input,
        "uses_missing_input": uses_missing_input,
        "observed_only_eligible": not (uses_imputed_input or uses_missing_input),
        "imputation_policy": latest.get("imputation_policy", "unspecified"),
    }

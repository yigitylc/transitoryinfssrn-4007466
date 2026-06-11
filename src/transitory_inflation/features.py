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

    for window in tinf_windows:
        out[f"tinf_{window}m"] = out["epsilon"].rolling(window, min_periods=window).mean()

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
    }

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from transitory_inflation.benchmarks import benchmark_comparison_tables
from transitory_inflation.features import BASELINE_META, add_transitory_inflation_features

DEFAULT_ROBUSTNESS_HORIZONS: tuple[int, ...] = (3, 6, 12, 24, 36)
DEFAULT_ROBUSTNESS_THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
DEFAULT_ROBUSTNESS_BASELINES: tuple[str, ...] = (
    "rolling_36_shifted",
    "expanding_shifted",
    "full_sample",
)

def _baseline_label(baseline_method: str) -> str:
    if baseline_method == "full_sample":
        return "ex-post / paper-style only"
    meta = BASELINE_META.get(baseline_method)
    if meta is not None and meta.live_safe:
        return "live-safe"
    return "not live-safe"


def _baseline_live_safe(baseline_method: str) -> bool:
    meta = BASELINE_META.get(baseline_method)
    return bool(meta.live_safe) if meta is not None else False


def _nonempty_tuple(values, label: str) -> tuple:
    out = tuple(values)
    if not out:
        raise ValueError(f"At least one {label} is required")
    return out


def build_robustness_scorecard(
    raw_frames_by_sample_mode: Mapping[str, pd.DataFrame],
    horizons: tuple[int, ...] = DEFAULT_ROBUSTNESS_HORIZONS,
    thresholds: tuple[float, ...] = DEFAULT_ROBUSTNESS_THRESHOLDS,
    baseline_methods: tuple[str, ...] = DEFAULT_ROBUSTNESS_BASELINES,
    ar_min_observations: int = 24,
    bucket_min_observations: int = 8,
) -> pd.DataFrame:
    """Run Phase 3A CPI-only robustness benchmarks across reasonable settings."""

    sample_items = tuple(raw_frames_by_sample_mode.items())
    if not sample_items:
        raise ValueError("At least one sample mode frame is required")
    horizons = tuple(int(value) for value in _nonempty_tuple(horizons, "horizon"))
    thresholds = tuple(float(value) for value in _nonempty_tuple(thresholds, "threshold"))
    baseline_methods = tuple(str(value) for value in _nonempty_tuple(baseline_methods, "baseline"))

    rows: list[pd.DataFrame] = []
    for sample_mode, raw in sample_items:
        if raw.empty:
            continue
        for baseline_method in baseline_methods:
            featured = add_transitory_inflation_features(raw, baseline_method=baseline_method)
            for horizon in horizons:
                for threshold in thresholds:
                    _, metrics, _, _ = benchmark_comparison_tables(
                        featured,
                        horizon=horizon,
                        threshold_pp=threshold,
                        ar_min_observations=ar_min_observations,
                        bucket_min_observations=bucket_min_observations,
                    )
                    if metrics.empty:
                        continue
                    current = metrics.copy()
                    current.insert(0, "sample_mode", sample_mode)
                    current.insert(1, "baseline_method", baseline_method)
                    current.insert(2, "baseline_live_safe", _baseline_live_safe(baseline_method))
                    current.insert(3, "baseline_label", _baseline_label(baseline_method))
                    current.insert(4, "threshold_pp", threshold)
                    rows.append(current)

    if not rows:
        return pd.DataFrame()

    scorecard = pd.concat(rows, ignore_index=True)
    setting_cols = ["sample_mode", "baseline_method", "horizon_months", "threshold_pp"]
    scorecard["rank_by_mae"] = (
        scorecard.groupby(setting_cols)["mae"].rank(method="min", ascending=True).astype("Int64")
    )
    scorecard["rank_by_rmse"] = (
        scorecard.groupby(setting_cols)["rmse"].rank(method="min", ascending=True).astype("Int64")
    )
    return scorecard


def tinf_regime_verdict(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Return one row per setting showing whether TINF/regime beats key baselines."""

    if scorecard.empty:
        return pd.DataFrame()

    setting_cols = [
        "sample_mode",
        "baseline_method",
        "baseline_live_safe",
        "baseline_label",
        "horizon_months",
        "threshold_pp",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in scorecard.groupby(setting_cols, dropna=False, sort=False):
        tinf = group.loc[group["model"] == "tinf_regime_bucket"]
        if tinf.empty:
            continue
        tinf_row = tinf.iloc[0]
        row = dict(zip(setting_cols, keys, strict=True))
        row.update(
            {
                "count": int(tinf_row["count"]),
                "tinf_mae": float(tinf_row["mae"]),
                "tinf_rmse": float(tinf_row["rmse"]),
                "tinf_directional_accuracy": float(tinf_row["directional_accuracy"]),
                "tinf_rank_by_mae": int(tinf_row["rank_by_mae"]),
                "tinf_rank_by_rmse": int(tinf_row["rank_by_rmse"]),
                "tinf_best_by_mae": int(tinf_row["rank_by_mae"]) == 1,
                "tinf_best_by_rmse": int(tinf_row["rank_by_rmse"]) == 1,
                "mae_improvement_vs_no_change_pct": float(
                    tinf_row["mae_improvement_vs_no_change_pct"]
                ),
                "rmse_improvement_vs_no_change_pct": float(
                    tinf_row["rmse_improvement_vs_no_change_pct"]
                ),
                "mae_improvement_vs_mean_reversion_pct": float(
                    tinf_row["mae_improvement_vs_mean_reversion_pct"]
                ),
                "rmse_improvement_vs_mean_reversion_pct": float(
                    tinf_row["rmse_improvement_vs_mean_reversion_pct"]
                ),
            }
        )
        for benchmark in ("no_change", "mean_reversion", "ar1"):
            other = group.loc[group["model"] == benchmark]
            if other.empty:
                row[f"beats_{benchmark}_mae"] = pd.NA
                row[f"beats_{benchmark}_rmse"] = pd.NA
            else:
                other_row = other.iloc[0]
                row[f"beats_{benchmark}_mae"] = bool(tinf_row["mae"] < other_row["mae"])
                row[f"beats_{benchmark}_rmse"] = bool(tinf_row["rmse"] < other_row["rmse"])
        rows.append(row)

    return pd.DataFrame(rows)


def robustness_win_rate_summary(verdict: pd.DataFrame) -> pd.DataFrame:
    """Aggregate TINF/regime win rates across robustness settings."""

    if verdict.empty:
        return pd.DataFrame()

    group_cols = ["sample_mode", "baseline_method", "baseline_live_safe", "baseline_label"]
    rate_cols = [
        "tinf_best_by_mae",
        "tinf_best_by_rmse",
        "beats_no_change_mae",
        "beats_no_change_rmse",
        "beats_mean_reversion_mae",
        "beats_mean_reversion_rmse",
        "beats_ar1_mae",
        "beats_ar1_rmse",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in verdict.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys, strict=True))
        row["settings_count"] = int(len(group))
        for column in rate_cols:
            row[f"{column}_rate"] = float(group[column].dropna().astype(float).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_tables(
    raw_frames_by_sample_mode: Mapping[str, pd.DataFrame],
    horizons: tuple[int, ...] = DEFAULT_ROBUSTNESS_HORIZONS,
    thresholds: tuple[float, ...] = DEFAULT_ROBUSTNESS_THRESHOLDS,
    baseline_methods: tuple[str, ...] = DEFAULT_ROBUSTNESS_BASELINES,
    ar_min_observations: int = 24,
    bucket_min_observations: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return scorecard, TINF/regime verdict, and aggregate win-rate tables."""

    scorecard = build_robustness_scorecard(
        raw_frames_by_sample_mode,
        horizons=horizons,
        thresholds=thresholds,
        baseline_methods=baseline_methods,
        ar_min_observations=ar_min_observations,
        bucket_min_observations=bucket_min_observations,
    )
    verdict = tinf_regime_verdict(scorecard)
    wins = robustness_win_rate_summary(verdict)
    return scorecard, verdict, wins

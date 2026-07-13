from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.benchmarks import (
    BENCHMARK_MODELS,
    benchmark_comparison_tables,
    benchmark_confusion_summary,
    benchmark_metric_summary,
    build_benchmark_forecasts,
)
from transitory_inflation.features import add_transitory_inflation_features


def _feature_frame(periods: int = 90) -> pd.DataFrame:
    months = np.arange(periods, dtype=float)
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2015-01-31", periods=periods, freq="ME"),
            "inflation_yoy": 2.0
            + 0.35 * np.sin(months / 4.0)
            + 0.20 * np.cos(months / 9.0)
            + 0.01 * months,
        }
    )
    return add_transitory_inflation_features(raw, baseline_method="fed_target")


def _classification_frame(periods: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2018-01-31", periods=periods, freq="ME"),
            "inflation_yoy": 3.0,
            "baseline": 2.0,
            "epsilon": 1.0,
            "tinf_4m": np.linspace(0.5, 1.5, periods),
            "tinf_8m": np.linspace(0.4, 1.4, periods),
            "tinf_12m": np.linspace(0.3, 1.3, periods),
            "tinf_term_structure": "mixed",
        }
    )


def test_benchmark_outputs_include_required_models_and_tables() -> None:
    df = _feature_frame()

    forecasts, metrics, improvements, confusion = benchmark_comparison_tables(
        df,
        horizon=3,
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    assert set(forecasts["model"]) == set(BENCHMARK_MODELS)
    assert set(metrics["model"]) == set(BENCHMARK_MODELS)
    assert set(confusion["model"]) == set(BENCHMARK_MODELS)
    assert set(improvements["comparison_baseline"]) == {"no_change", "mean_reversion"}
    assert {"mae", "rmse", "directional_accuracy", "hit_rate"}.issubset(metrics.columns)
    assert {
        "false_positive_rate",
        "false_negative_rate",
        "mae_improvement_vs_no_change_pct",
        "rmse_improvement_vs_mean_reversion_pct",
    }.issubset(metrics.columns)


def test_benchmark_metrics_are_calculated_correctly() -> None:
    forecasts = pd.DataFrame(
        {
            "model": ["toy"] * 3,
            "horizon_months": [1, 1, 1],
            "actual_cpi_yoy": [3.0, 2.0, 5.0],
            "forecast_cpi_yoy": [2.0, 2.0, 6.0],
            "current_cpi_yoy": [1.0, 3.0, 4.0],
            "actual_cpi_yoy_change": [2.0, -1.0, 1.0],
            "forecast_cpi_yoy_change": [1.0, -1.0, 2.0],
            "actual_persistent_high_inflation": [True, False, True],
            "forecast_persistent_high_inflation": [False, False, True],
        }
    )

    summary = benchmark_metric_summary(forecasts)
    row = summary.iloc[0]

    assert row["mae"] == pytest.approx(2 / 3)
    assert row["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert row["directional_accuracy"] == pytest.approx(1.0)
    assert row["hit_rate"] == pytest.approx(2 / 3)
    assert row["false_positive_rate"] == pytest.approx(0.0)
    assert row["false_negative_rate"] == pytest.approx(0.5)
    assert row["true_positive"] == 1
    assert row["false_positive"] == 0
    assert row["true_negative"] == 1
    assert row["false_negative"] == 1


def test_confusion_summary_counts_positive_shock_persistence() -> None:
    forecasts = pd.DataFrame(
        {
            "model": ["a", "a", "a", "a"],
            "actual_persistent_high_inflation": [True, False, False, True],
            "forecast_persistent_high_inflation": [True, True, False, False],
        }
    )

    confusion = benchmark_confusion_summary(forecasts)
    row = confusion.iloc[0]

    assert row["true_positive"] == 1
    assert row["false_positive"] == 1
    assert row["true_negative"] == 1
    assert row["false_negative"] == 1


def test_ineligible_origin_has_nullable_predicted_and_realized_labels() -> None:
    frame = _classification_frame()
    target_pos = 40
    frame.loc[target_pos, ["inflation_yoy", "epsilon"]] = [2.49, 0.49]
    frame.loc[target_pos + 1, ["inflation_yoy", "baseline", "epsilon"]] = [4.0, 3.0, 1.0]

    forecasts = build_benchmark_forecasts(
        frame,
        horizon=1,
        ar_min_observations=8,
        bucket_min_observations=1,
    )
    row = forecasts.loc[
        (forecasts["model"] == "no_change")
        & (forecasts["date"] == frame.loc[target_pos, "date"])
    ].iloc[0]

    assert not row["eligible_positive_shock"]
    assert pd.isna(row["forecast_persistent_high_inflation"])
    assert pd.isna(row["actual_persistent_high_inflation"])

    model_rows = forecasts.loc[forecasts["model"] == "no_change"]
    classified = model_rows[
        ["forecast_persistent_high_inflation", "actual_persistent_high_inflation"]
    ].dropna()
    metrics = benchmark_metric_summary(forecasts)
    metric_row = metrics.loc[metrics["model"] == "no_change"].iloc[0]
    assert metric_row["classification_count"] == len(classified)
    assert len(classified) == int(model_rows["eligible_positive_shock"].fillna(False).sum())


def test_predicted_and_realized_persistence_share_anchor_and_strict_threshold() -> None:
    frame = _classification_frame()
    boundary_pos = 40
    above_pos = 42
    frame.loc[boundary_pos : boundary_pos + 1, ["inflation_yoy", "epsilon"]] = [2.5, 0.5]
    frame.loc[above_pos : above_pos + 1, ["inflation_yoy", "epsilon"]] = [2.5001, 0.5001]
    frame.loc[[boundary_pos + 1, above_pos + 1], "baseline"] = 3.0
    frame.loc[[boundary_pos + 1, above_pos + 1], "epsilon"] = [-0.5, -0.4999]

    forecasts = build_benchmark_forecasts(
        frame,
        horizon=1,
        ar_min_observations=8,
        bucket_min_observations=1,
    )
    rows = forecasts.loc[
        (forecasts["model"] == "no_change")
        & forecasts["date"].isin(frame.loc[[boundary_pos, above_pos], "date"])
    ].sort_values("date")

    assert rows["eligible_positive_shock"].tolist() == [True, True]
    assert rows["forecast_persistent_high_inflation"].tolist() == [False, True]
    assert rows["actual_persistent_high_inflation"].tolist() == [False, True]
    assert rows["actual_gap_from_origin_baseline"].tolist() == pytest.approx([0.5, 0.5001])


def test_forecasts_do_not_change_when_future_rows_after_t_are_perturbed() -> None:
    df = _feature_frame(periods=100)
    target_date = df.loc[60, "date"]

    base = build_benchmark_forecasts(
        df,
        horizon=3,
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    raw_perturbed = df[["date", "inflation_yoy"]].copy()
    raw_perturbed.loc[raw_perturbed.index > 60, "inflation_yoy"] += 100.0
    perturbed = add_transitory_inflation_features(raw_perturbed, baseline_method="fed_target")
    changed = build_benchmark_forecasts(
        perturbed,
        horizon=3,
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    base_forecasts = (
        base.loc[base["date"] == target_date]
        .set_index("model")["forecast_cpi_yoy"]
        .sort_index()
    )
    changed_forecasts = (
        changed.loc[changed["date"] == target_date]
        .set_index("model")["forecast_cpi_yoy"]
        .sort_index()
    )

    assert set(base_forecasts.index) == set(BENCHMARK_MODELS)
    assert changed_forecasts.index.tolist() == base_forecasts.index.tolist()
    assert changed_forecasts.to_numpy() == pytest.approx(base_forecasts.to_numpy())


def test_alternative_measure_forecasts_do_not_use_future_outcomes() -> None:
    periods = 100
    months = np.arange(periods, dtype=float)
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2015-01-31", periods=periods, freq="ME"),
            "inflation_yoy": 2.0 + 0.1 * np.sin(months / 3.0),
            "core_cpi_yoy": 2.0 + 0.35 * np.sin(months / 4.0) + 0.01 * months,
        }
    )
    target_date = raw.loc[60, "date"]
    featured = add_transitory_inflation_features(
        raw,
        inflation_col="core_cpi_yoy",
        baseline_method="fed_target",
    )

    base = build_benchmark_forecasts(
        featured,
        horizon=3,
        inflation_col="core_cpi_yoy",
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    raw_perturbed = raw.copy()
    raw_perturbed.loc[raw_perturbed.index > 60, "core_cpi_yoy"] += 100.0
    perturbed = add_transitory_inflation_features(
        raw_perturbed,
        inflation_col="core_cpi_yoy",
        baseline_method="fed_target",
    )
    changed = build_benchmark_forecasts(
        perturbed,
        horizon=3,
        inflation_col="core_cpi_yoy",
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    base_forecasts = (
        base.loc[base["date"] == target_date]
        .set_index("model")["forecast_cpi_yoy"]
        .sort_index()
    )
    changed_forecasts = (
        changed.loc[changed["date"] == target_date]
        .set_index("model")["forecast_cpi_yoy"]
        .sort_index()
    )

    assert set(base_forecasts.index) == set(BENCHMARK_MODELS)
    assert changed_forecasts.index.tolist() == base_forecasts.index.tolist()
    assert changed_forecasts.to_numpy() == pytest.approx(base_forecasts.to_numpy())

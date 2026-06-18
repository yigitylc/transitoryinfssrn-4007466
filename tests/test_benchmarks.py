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


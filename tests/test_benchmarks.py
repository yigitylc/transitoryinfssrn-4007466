from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.benchmarks import (
    BENCHMARK_MODELS,
    EmptyCommonOriginPanelError,
    benchmark_comparison_tables,
    benchmark_confusion_summary,
    benchmark_metric_summary,
    build_benchmark_forecasts,
    build_native_benchmark_forecasts,
    restrict_to_common_origins,
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

    forecasts, metrics, improvements, confusion, coverage = benchmark_comparison_tables(
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
    assert set(coverage["model"]) == set(BENCHMARK_MODELS)


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


def test_benchmarks_reject_authoritative_estimate_only_origins_and_targets() -> None:
    frame = _feature_frame(periods=100)
    frame["imputation_policy"] = "observed_only"
    frame["uses_estimated_input"] = False
    frame["estimated_input_months"] = [()] * len(frame)
    frame["signal_uses_imputed_input"] = False
    frame["signal_uses_missing_input"] = False
    frame["signal_observed_only_eligible"] = True
    contaminated_pos = 60
    contaminated_date = frame.loc[contaminated_pos, "date"]
    affected_origin_date = frame.loc[contaminated_pos - 2, "date"]
    frame.loc[contaminated_pos, "uses_estimated_input"] = True
    changed = frame.copy()
    changed.loc[contaminated_pos, ["inflation_yoy", "baseline", "epsilon"]] += 100.0

    tables = benchmark_comparison_tables(
        frame,
        horizon=2,
        ar_min_observations=8,
        bucket_min_observations=1,
    )
    changed_tables = benchmark_comparison_tables(
        changed,
        horizon=2,
        ar_min_observations=8,
        bucket_min_observations=1,
    )
    forecasts = tables[0]

    assert contaminated_date not in set(forecasts["date"])
    assert affected_origin_date not in set(forecasts["date"])
    assert set(forecasts["model"]) == set(BENCHMARK_MODELS)
    for base_table, changed_table in zip(tables, changed_tables, strict=True):
        pd.testing.assert_frame_equal(base_table, changed_table)


def test_common_origin_panel_equalizes_every_metric_denominator() -> None:
    """H2: point-loss, directional, and classification denominators must match."""

    frame = _feature_frame(120)
    forecasts, metrics, _, _, _ = benchmark_comparison_tables(
        frame,
        horizon=3,
        ar_min_observations=24,
        bucket_min_observations=8,
    )

    origins_by_model = {
        model: frozenset(group["date"]) for model, group in forecasts.groupby("model", sort=False)
    }
    assert set(origins_by_model) == set(BENCHMARK_MODELS)
    assert len(set(origins_by_model.values())) == 1

    assert metrics["count"].nunique() == 1
    assert metrics["classification_count"].nunique() == 1
    assert metrics["common_origin_n"].nunique() == 1
    assert int(metrics["count"].iloc[0]) == int(metrics["common_origin_n"].iloc[0])

    panel = next(iter(origins_by_model.values()))
    assert int(metrics["common_origin_n"].iloc[0]) == len(panel)
    assert metrics["common_origin_start"].iloc[0] == min(panel)
    assert metrics["common_origin_end"].iloc[0] == max(panel)


def test_common_panel_only_removes_native_origins() -> None:
    """Restriction is a strict subset, so observed-only gating is untouched."""

    frame = _feature_frame(120)
    native = build_native_benchmark_forecasts(
        frame,
        horizon=3,
        ar_min_observations=24,
        bucket_min_observations=8,
    )
    common = restrict_to_common_origins(native, horizon=3)

    native_pairs = set(zip(native["model"], native["date"], strict=True))
    common_pairs = set(zip(common["model"], common["date"], strict=True))
    assert common_pairs <= native_pairs
    assert not common_pairs - native_pairs

    panel = frozenset(common["date"])
    for model in BENCHMARK_MODELS:
        model_native = frozenset(native.loc[native["model"] == model, "date"])
        assert panel <= model_native


def test_native_coverage_diagnostic_discloses_per_model_exclusions() -> None:
    frame = _feature_frame(120)
    _, metrics, _, _, coverage = benchmark_comparison_tables(
        frame,
        horizon=3,
        ar_min_observations=24,
        bucket_min_observations=8,
    )

    assert set(coverage["model"]) == set(BENCHMARK_MODELS)
    assert coverage["native_exclusion_reason"].map(bool).all()
    assert coverage["common_origin_n"].nunique() == 1
    assert int(coverage["common_origin_n"].iloc[0]) == int(metrics["common_origin_n"].iloc[0])

    by_model = coverage.set_index("model")
    # Native samples are unequal, which is exactly why a common panel is needed.
    assert by_model["native_count"].nunique() > 1
    # No-change has the widest native sample and drops origins the panel excludes.
    assert int(by_model.loc["no_change", "origins_outside_common_panel"]) > 0
    # The narrowest model defines the panel, so nothing of its own is dropped.
    narrowest = by_model["native_count"].idxmin()
    assert int(by_model.loc[narrowest, "origins_outside_common_panel"]) == 0
    for model in BENCHMARK_MODELS:
        row = by_model.loc[model]
        assert int(row["native_count"]) - int(row["origins_outside_common_panel"]) == int(
            row["common_origin_n"]
        )


def test_empty_common_origin_panel_fails_explicitly() -> None:
    """Never score silently on unequal samples when no origin is shared."""

    frame = _feature_frame(48)
    with pytest.raises(EmptyCommonOriginPanelError) as excinfo:
        build_benchmark_forecasts(
            frame,
            horizon=12,
            ar_min_observations=24,
            bucket_min_observations=8,
        )
    message = str(excinfo.value)
    assert "12m" in message
    assert "no_change=" in message

    # The failure carries the native coverage so a caller can disclose the skip
    # with per-model detail instead of reporting an unexplained blank.
    coverage = excinfo.value.coverage
    assert set(coverage["model"]) == set(BENCHMARK_MODELS)
    assert coverage["native_count"].max() > 0
    assert coverage["common_origin_n"].eq(0).all()

    with pytest.raises(EmptyCommonOriginPanelError):
        benchmark_comparison_tables(
            frame,
            horizon=12,
            ar_min_observations=24,
            bucket_min_observations=8,
        )


def test_sample_with_no_forecasts_returns_empty_instead_of_failing() -> None:
    """Nothing to score is not the same failure as an empty common panel."""

    # A horizon longer than the sample leaves no origin with a realized outcome,
    # so no model produces a forecast row and there is simply nothing to score.
    frame = _feature_frame(20)
    native = build_native_benchmark_forecasts(
        frame,
        horizon=25,
        ar_min_observations=8,
        bucket_min_observations=1,
    )
    assert native.empty

    forecasts = build_benchmark_forecasts(
        frame,
        horizon=25,
        ar_min_observations=8,
        bucket_min_observations=1,
    )
    assert forecasts.empty


def test_common_panel_scoring_can_reorder_models_versus_native_samples() -> None:
    """H2 regression: native-sample ranks are not the common-origin ranks."""

    frame = _feature_frame(150)
    native = build_native_benchmark_forecasts(
        frame,
        horizon=6,
        ar_min_observations=24,
        bucket_min_observations=8,
    )
    common = restrict_to_common_origins(native, horizon=6)

    def mae_by_model(forecasts: pd.DataFrame) -> pd.Series:
        errors = forecasts.assign(
            abs_error=(forecasts["forecast_cpi_yoy"] - forecasts["actual_cpi_yoy"]).abs()
        )
        return errors.groupby("model")["abs_error"].mean()

    native_mae = mae_by_model(native)
    common_mae = mae_by_model(common)

    # The narrowest model already scores on the panel, so only its rivals move.
    narrowest = native.groupby("model")["date"].nunique().idxmin()
    assert native_mae[narrowest] == pytest.approx(common_mae[narrowest])
    moved = [
        model
        for model in BENCHMARK_MODELS
        if model != narrowest and native_mae[model] != pytest.approx(common_mae[model])
    ]
    assert moved, "restricting to common origins must change the rivals' scores"

    # The shipped summary reports the common-origin numbers, not the native ones.
    metrics = benchmark_metric_summary(common).set_index("model")
    for model in BENCHMARK_MODELS:
        assert float(metrics.loc[model, "mae"]) == pytest.approx(float(common_mae[model]))


def test_benchmark_tables_carry_no_win_or_beat_claim_columns() -> None:
    """Point-estimate language only; significance is H10's scope, not H2's."""

    frame = _feature_frame(120)
    tables = benchmark_comparison_tables(
        frame,
        horizon=3,
        ar_min_observations=24,
        bucket_min_observations=8,
    )
    offending = [
        column
        for table in tables
        for column in table.columns
        if "beats" in str(column).lower() or "win" in str(column).lower()
    ]
    assert not offending

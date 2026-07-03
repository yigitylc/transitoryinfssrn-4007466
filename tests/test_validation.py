from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.validation import (
    add_forward_outcomes,
    add_outcome_labels,
    add_walk_forward_regime_labels,
    forward_outcome_summary_by_regime,
    forward_outcome_summary_by_regime_and_pressure,
    forward_outcome_summary_by_short_term_pressure,
    regime_transition_matrix,
    threshold_sensitivity_summary,
    validation_examples,
)


def _validation_frame(periods: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=periods, freq="ME"),
            "inflation_yoy": np.arange(periods, dtype=float),
            "epsilon": np.arange(periods, dtype=float) + 10.0,
            "tinf_4m": np.arange(periods, dtype=float) + 100.0,
        }
    )


def test_forward_horizon_alignment_assigns_t_plus_h_to_t() -> None:
    df = _validation_frame(periods=8)

    out = add_forward_outcomes(df, horizons=(3,))

    assert out.loc[0, "cpi_yoy_fwd_3m"] == 3.0
    assert out.loc[0, "cpi_yoy_change_3m"] == 3.0
    assert out.loc[0, "epsilon_fwd_3m"] == 13.0
    assert out.loc[0, "epsilon_change_3m"] == 3.0
    assert out.loc[0, "tinf_4m_fwd_3m"] == 103.0
    assert out.loc[0, "tinf_4m_change_3m"] == 3.0


def test_forward_horizon_terminal_rows_are_nan() -> None:
    df = _validation_frame(periods=8)

    out = add_forward_outcomes(df, horizons=(3,))

    assert out.loc[5:, "cpi_yoy_fwd_3m"].isna().all()
    assert out.loc[5:, "epsilon_fwd_3m"].isna().all()
    assert out.loc[5:, "tinf_4m_fwd_3m"].isna().all()


def test_normalization_labels_follow_configured_thresholds() -> None:
    df = pd.DataFrame(
        {
            "inflation_yoy": [4.0, 2.4],
            "epsilon": [2.0, 0.4],
            "tinf_4m": [2.0, 0.4],
        }
    )

    out = add_forward_outcomes(df, horizons=(1,))
    out = add_outcome_labels(out, horizons=(1,), epsilon_threshold_pp=0.50)

    assert out.loc[0, "baseline_normalized_1m"]
    assert out.loc[0, "fed_target_normalized_1m"]
    assert out.loc[0, "partial_decay_50_1m"]
    assert out.loc[0, "partial_decay_80_1m"]
    assert not out.loc[0, "persistent_1m"]
    assert not out.loc[0, "reaccelerated_1m"]
    assert pd.isna(out.loc[1, "baseline_normalized_1m"])


def test_positive_shock_downside_overshoot_is_resolved_not_persistent() -> None:
    df = pd.DataFrame(
        {
            "inflation_yoy": [4.0, 0.75],
            "epsilon": [2.0, -1.25],
            "tinf_4m": [2.0, -1.25],
        }
    )

    out = add_forward_outcomes(df, horizons=(1,))
    out = add_outcome_labels(out, horizons=(1,), epsilon_threshold_pp=0.50)

    assert out.loc[0, "positive_shock_resolved_1m"]
    assert out.loc[0, "positive_shock_downside_overshoot_1m"]
    assert not out.loc[0, "positive_shock_persistent_1m"]
    assert not out.loc[0, "persistent_1m"]
    assert out.loc[0, "absolute_gap_persistent_1m"]


def test_positive_shock_still_above_threshold_is_persistent() -> None:
    df = pd.DataFrame(
        {
            "inflation_yoy": [4.0, 3.25],
            "epsilon": [2.0, 1.25],
            "tinf_4m": [2.0, 1.25],
        }
    )

    out = add_forward_outcomes(df, horizons=(1,))
    out = add_outcome_labels(out, horizons=(1,), epsilon_threshold_pp=0.50)

    assert not out.loc[0, "positive_shock_resolved_1m"]
    assert not out.loc[0, "positive_shock_downside_overshoot_1m"]
    assert out.loc[0, "positive_shock_persistent_1m"]
    assert out.loc[0, "persistent_1m"]


def test_non_positive_start_does_not_create_positive_shock_labels() -> None:
    df = pd.DataFrame(
        {
            "inflation_yoy": [1.0, 4.0],
            "epsilon": [-1.0, 2.0],
            "tinf_4m": [-1.0, 2.0],
        }
    )

    out = add_forward_outcomes(df, horizons=(1,))
    out = add_outcome_labels(out, horizons=(1,), epsilon_threshold_pp=0.50)

    assert pd.isna(out.loc[0, "positive_shock_resolved_1m"])
    assert pd.isna(out.loc[0, "positive_shock_downside_overshoot_1m"])
    assert pd.isna(out.loc[0, "positive_shock_persistent_1m"])
    assert pd.isna(out.loc[0, "persistent_1m"])


def test_decay_ratio_uses_absolute_positive_and_negative_gaps() -> None:
    df = pd.DataFrame(
        {
            "inflation_yoy": [4.0, 1.0, 2.25],
            "epsilon": [2.0, -1.0, 0.25],
            "tinf_4m": [2.0, -1.0, 0.25],
        }
    )

    out = add_forward_outcomes(df, horizons=(1,), min_initial_gap_pp=0.50)

    assert out.loc[0, "gap_decay_ratio_1m"] == pytest.approx(0.50)
    assert out.loc[1, "gap_decay_ratio_1m"] == pytest.approx(0.25)


def test_near_zero_epsilon_does_not_create_misleading_decay_ratio() -> None:
    df = pd.DataFrame(
        {
            "inflation_yoy": [2.01, 4.0],
            "epsilon": [0.01, 2.0],
            "tinf_4m": [0.01, 2.0],
        }
    )

    out = add_forward_outcomes(df, horizons=(1,), min_initial_gap_pp=0.50)
    out = add_outcome_labels(out, horizons=(1,), epsilon_threshold_pp=0.50)

    assert pd.isna(out.loc[0, "gap_decay_ratio_1m"])
    assert pd.isna(out.loc[0, "partial_decay_50_1m"])
    assert pd.isna(out.loc[0, "partial_decay_80_1m"])
    assert pd.isna(out.loc[0, "persistent_1m"])


def test_transition_matrix_rows_sum_to_one_where_data_exists() -> None:
    df = pd.DataFrame(
        {"historical_regime": ["neutral", "elevated rising", "neutral", "disinflationary"]}
    )

    matrix = regime_transition_matrix(df, horizon=1)

    assert not matrix.empty
    assert matrix.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(matrix)))


def test_summary_excludes_rows_without_future_data_for_horizon() -> None:
    df = _validation_frame(periods=6)
    df["historical_regime"] = "neutral"
    out = add_forward_outcomes(df, horizons=(2,))
    out = add_outcome_labels(out, horizons=(2,))

    summary = forward_outcome_summary_by_regime(out, horizons=(2,))

    assert summary.loc[0, "count"] == 4
    assert "positive_shock_resolution_rate" in summary.columns
    assert "absolute_gap_persistent_rate" in summary.columns


def test_single_key_summary_labels_are_plain_strings() -> None:
    # Regression: pandas 3 groupby with a length-1 list yields 1-tuple keys; the
    # summary must still emit plain string labels so string-equality filters and
    # chart category orders keep working.
    df = _validation_frame(periods=6)
    df["historical_regime"] = "neutral"
    df["historical_short_term_pressure"] = "mixed"
    out = add_forward_outcomes(df, horizons=(2,))
    out = add_outcome_labels(out, horizons=(2,))

    regime_summary = forward_outcome_summary_by_regime(out, horizons=(2,))
    pressure_summary = forward_outcome_summary_by_short_term_pressure(out, horizons=(2,))

    assert regime_summary["historical_regime"].map(type).eq(str).all()
    assert pressure_summary["historical_short_term_pressure"].map(type).eq(str).all()
    assert not regime_summary.loc[regime_summary["historical_regime"] == "neutral"].empty
    assert not pressure_summary.loc[
        pressure_summary["historical_short_term_pressure"] == "mixed"
    ].empty


def test_combined_regime_pressure_summary_uses_historical_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=5, freq="ME"),
            "historical_regime": [
                "elevated falling",
                "elevated falling",
                "elevated rising",
                "neutral",
                "neutral",
            ],
            "historical_short_term_pressure": ["cooling", "firming", "firming", "mixed", "mixed"],
            "inflation_yoy": [4.0, 3.0, 4.5, 2.25, 2.0],
            "epsilon": [2.0, 1.0, 2.5, 0.25, 0.0],
            "tinf_4m": [2.0, 1.0, 2.5, 0.25, 0.0],
        }
    )

    out = add_forward_outcomes(df, horizons=(1,))
    out = add_outcome_labels(out, horizons=(1,), epsilon_threshold_pp=0.50)
    summary = forward_outcome_summary_by_regime_and_pressure(out, horizons=(1,))

    assert "historical_regime" in summary.columns
    assert "historical_short_term_pressure" in summary.columns
    assert "regime" not in summary.columns
    assert "short_term_pressure" not in summary.columns
    assert summary["count"].sum() == 4
    assert "positive_shock_resolution_rate" in summary.columns
    assert "positive_shock_downside_overshoot_rate" in summary.columns


def test_threshold_sensitivity_calculates_all_phase_one_thresholds() -> None:
    df = _validation_frame(periods=8)

    summary = threshold_sensitivity_summary(df, horizon=1)

    assert summary["threshold_pp"].tolist() == [0.25, 0.50, 0.75, 1.00]
    assert summary["count"].tolist() == [7, 7, 7, 7]
    assert "positive_shock_resolution_rate" in summary.columns
    phase_two_columns = {"mae", "rmse", "forecast", "confusion_matrix"}
    assert summary.columns.intersection(phase_two_columns).empty


def test_validation_examples_separate_downside_overshoot_from_false_transitory() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=10, freq="ME"),
            "historical_regime": [
                "elevated falling",
                "neutral",
                "elevated falling",
                "neutral",
                "elevated falling",
                "neutral",
                "elevated rising",
                "neutral",
                "elevated rising",
                "neutral",
            ],
            "historical_short_term_pressure": [
                "cooling",
                "mixed",
                "cooling",
                "mixed",
                "cooling",
                "mixed",
                "firming",
                "mixed",
                "firming",
                "mixed",
            ],
            "epsilon": [2.0, -1.25, 2.0, 0.25, 2.0, 1.25, 2.0, 1.25, 2.0, 0.25],
            "tinf_4m": [2.0, -1.25, 2.0, 0.25, 2.0, 1.25, 2.0, 1.25, 2.0, 0.25],
        }
    )
    df["inflation_yoy"] = 2.0 + df["epsilon"]

    out = add_forward_outcomes(df, horizons=(1,))
    out = add_outcome_labels(out, horizons=(1,), epsilon_threshold_pp=0.50)
    examples = validation_examples(out, horizon=1, max_examples=20)

    assert len(examples["false_transitory"]) == 1
    assert len(examples["successful_transitory"]) == 1
    assert len(examples["successful_transitory_downside_overshoot"]) == 1
    assert len(examples["successful_persistent"]) == 1
    assert len(examples["false_persistent"]) == 1

    plain_success = examples["successful_transitory"].iloc[0]
    downside_success = examples["successful_transitory_downside_overshoot"].iloc[0]
    false_transitory = examples["false_transitory"].iloc[0]

    assert plain_success["positive_shock_resolved_1m"]
    assert not plain_success["positive_shock_downside_overshoot_1m"]
    assert downside_success["positive_shock_resolved_1m"]
    assert downside_success["positive_shock_downside_overshoot_1m"]
    assert not false_transitory["positive_shock_downside_overshoot_1m"]
    assert false_transitory["positive_shock_persistent_1m"]
    assert false_transitory["persistent_1m"]


def test_walk_forward_regime_thresholds_use_only_prior_tinf_history() -> None:
    df = pd.DataFrame(
        {
            "tinf_4m": [0.0] * 36 + [1.0] + [100.0] * 10,
        }
    )

    out = add_walk_forward_regime_labels(df)

    assert pd.isna(out.loc[35, "historical_regime"])
    assert out.loc[36, "historical_regime_upper_threshold"] == 0.0
    assert out.loc[36, "historical_regime"] == "elevated rising"

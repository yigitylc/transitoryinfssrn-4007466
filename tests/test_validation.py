from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.validation import (
    add_forward_outcomes,
    add_outcome_labels,
    add_walk_forward_regime_labels,
    forward_outcome_summary_by_regime,
    regime_transition_matrix,
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


def test_validation_examples_use_positive_shock_persistence_not_absolute_gap() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=4, freq="ME"),
            "historical_regime": ["elevated falling", "elevated falling", "elevated rising", "neutral"],
            "historical_short_term_pressure": ["cooling", "cooling", "firming", "mixed"],
            "inflation_yoy": [4.0, 0.75, 4.5, 4.0],
            "epsilon": [2.0, -1.25, 2.5, 2.0],
            "tinf_4m": [2.0, -1.25, 2.5, 2.0],
        }
    )

    out = add_forward_outcomes(df, horizons=(1,))
    out = add_outcome_labels(out, horizons=(1,), epsilon_threshold_pp=0.50)
    examples = validation_examples(out, horizon=1)

    assert examples["false_transitory"].empty
    assert len(examples["successful_transitory"]) == 1
    row = examples["successful_transitory"].iloc[0]
    assert row["positive_shock_resolved_1m"]
    assert row["positive_shock_downside_overshoot_1m"]
    assert row["absolute_gap_persistent_1m"]


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

from __future__ import annotations

import numpy as np
import pandas as pd

from transitory_inflation.data import build_base_frame
from transitory_inflation.features import add_transitory_inflation_features, consecutive_true_count


def test_consecutive_true_count() -> None:
    flag = pd.Series([True, True, False, True, True, True])
    result = consecutive_true_count(flag)
    assert result.tolist() == [1, 2, 0, 1, 2, 3]


def test_tinf_features_use_percentage_points() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=60, freq="ME"),
            "inflation_yoy": [2.0] * 40 + [3.0] * 20,
        }
    )
    out = add_transitory_inflation_features(df, baseline_method="fed_target")
    assert abs(out["epsilon"].iloc[-1] - 1.0) < 1e-9
    assert abs(out["tinf_4m"].iloc[-1] - 1.0) < 1e-9
    assert out["short_regime_flag"].iloc[-1]


def test_shifted_rolling_baseline_has_initial_nans() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=50, freq="ME"),
            "inflation_yoy": [2.0] * 50,
        }
    )
    out = add_transitory_inflation_features(df, baseline_method="rolling_36_shifted")
    assert out["baseline"].iloc[:36].isna().all()
    assert out["baseline"].iloc[36] == 2.0


def test_observed_only_feature_history_is_invariant_to_future_gap_neighbor() -> None:
    dates = pd.date_range("2015-01-31", periods=80, freq="ME")
    levels = 100.0 + np.arange(80, dtype=float)
    gap_pos = 50
    levels[gap_pos] = np.nan
    raw = pd.DataFrame({"date": dates, "CPIAUCSL": levels, "TB3MS": 1.0})
    changed_raw = raw.copy()
    changed_raw.loc[gap_pos + 1, "CPIAUCSL"] *= 1.25

    base = build_base_frame(raw, imputation_policy="observed_only")
    changed = build_base_frame(changed_raw, imputation_policy="observed_only")
    featured = add_transitory_inflation_features(
        base,
        baseline_method="rolling_36_shifted",
    )
    changed_featured = add_transitory_inflation_features(
        changed,
        baseline_method="rolling_36_shifted",
    )
    through_gap = featured["date"] <= dates[gap_pos]
    columns = ["cpi_level", "inflation_yoy", "baseline", "epsilon", "tinf_4m"]

    pd.testing.assert_frame_equal(
        featured.loc[through_gap, columns].reset_index(drop=True),
        changed_featured.loc[through_gap, columns].reset_index(drop=True),
    )
    gap = featured.iloc[gap_pos]
    assert gap["inflation_yoy_uses_missing_input"]
    assert gap["epsilon_uses_missing_input"]
    assert gap["tinf_4m_uses_missing_input"]
    assert not gap["signal_observed_only_eligible"]


def test_ex_post_imputation_lineage_propagates_through_baseline_epsilon_and_tinf() -> None:
    dates = pd.date_range("2015-01-31", periods=90, freq="ME")
    levels = 100.0 + np.arange(90, dtype=float)
    gap_pos = 50
    levels[gap_pos] = np.nan
    base = build_base_frame(
        pd.DataFrame({"date": dates, "CPIAUCSL": levels, "TB3MS": 1.0}),
        imputation_policy="ex_post_continuity",
    )

    featured = add_transitory_inflation_features(
        base,
        baseline_method="rolling_36_shifted",
    )

    assert featured.loc[gap_pos, "inflation_yoy_uses_imputed_input"]
    assert featured.loc[gap_pos, "epsilon_uses_imputed_input"]
    assert featured.loc[gap_pos, "tinf_4m_uses_imputed_input"]
    assert featured.loc[gap_pos + 1, "baseline_uses_imputed_input"]
    assert featured.loc[gap_pos + 1, "epsilon_uses_imputed_input"]
    assert featured.loc[gap_pos + 1, "signal_uses_imputed_input"]
    assert not featured.loc[gap_pos + 1, "signal_observed_only_eligible"]

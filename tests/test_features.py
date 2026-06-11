from __future__ import annotations

import pandas as pd

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

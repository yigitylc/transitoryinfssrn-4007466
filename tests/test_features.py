from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.data import (
    RELEASE_TIMESTAMP_PROVENANCE_ACTUAL,
    build_base_frame,
)
from transitory_inflation.features import (
    add_transitory_inflation_features,
    consecutive_true_count,
    latest_signal_snapshot,
)


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


def test_derived_information_timestamp_uses_latest_dependency_availability() -> None:
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    releases = pd.Series(
        (dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC")
    )
    releases.iloc[-2] = releases.iloc[-1] + pd.offsets.Day(5)
    raw = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": 100.0 + np.arange(60, dtype=float),
            "TB3MS": 1.0,
            "release_timestamp": releases,
            "release_timestamp_provenance": RELEASE_TIMESTAMP_PROVENANCE_ACTUAL,
            "timing_status": "release_aligned",
        }
    )

    featured = add_transitory_inflation_features(
        build_base_frame(raw),
        baseline_method="fed_target",
    )
    latest = featured.iloc[-1]

    assert latest["inflation_yoy_information_timestamp"] == releases.iloc[-1]
    assert latest["tinf_4m_information_timestamp"] == releases.iloc[-2]
    assert latest["information_timestamp"] == releases.iloc[-2]
    assert latest["timing_status"] == "release_aligned"


def test_reference_month_only_dependency_cannot_be_laundered_to_release_aligned() -> None:
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    information_timestamps = pd.Series(
        (dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC")
    )
    timing_status = pd.Series("release_aligned", index=dates, dtype="string")
    timing_status.iloc[-2] = "reference_month_only"
    frame = pd.DataFrame(
        {
            "date": dates,
            "inflation_yoy": 2.0 + np.arange(len(dates), dtype=float) / 100.0,
            "inflation_yoy_information_timestamp": information_timestamps,
            "inflation_yoy_information_timestamp_provenance": (
                "derived_from_actual_release_metadata"
            ),
            "timing_status": timing_status.reset_index(drop=True),
        }
    )

    featured = add_transitory_inflation_features(frame, baseline_method="fed_target")
    latest = featured.iloc[-1]

    assert pd.isna(latest["tinf_4m_information_timestamp"])
    assert pd.isna(latest["information_timestamp"])
    assert latest["timing_status"] == "reference_month_only"


def test_release_and_information_timestamp_preserve_time_of_day() -> None:
    dates = pd.date_range("2022-01-31", periods=15, freq="ME")
    releases = pd.Series(
        (dates + pd.offsets.Day(13) + pd.offsets.Hour(17)).tz_localize("UTC")
    )
    base = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": 100.0 + np.arange(15, dtype=float),
                "TB3MS": 1.0,
                "release_timestamp": releases,
                "release_timestamp_provenance": (
                    RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
                ),
                "timing_status": "release_aligned",
            }
        )
    )

    latest = base.iloc[-1]
    assert latest["release_timestamp"] == releases.iloc[-1]
    assert latest["release_timestamp"].hour == 17
    assert latest["information_timestamp"] == releases.iloc[-1]
    assert latest["information_timestamp"].hour == 17


def test_missing_release_metadata_fails_closed_to_reference_month_only() -> None:
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    base = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": 100.0 + np.arange(30, dtype=float),
                "TB3MS": 1.0,
            }
        )
    )
    featured = add_transitory_inflation_features(base, baseline_method="fed_target")
    latest = featured.iloc[-1]

    assert pd.isna(latest["release_timestamp"])
    assert pd.isna(latest["information_timestamp"])
    assert latest["timing_status"] == "reference_month_only"
    assert latest["data_vintage_status"] == "latest_revised_non_vintage"


def test_ex_post_estimate_waits_for_following_cpi_release() -> None:
    dates = pd.date_range("2015-01-31", periods=72, freq="ME")
    levels = 100.0 + np.arange(72, dtype=float)
    gap_pos = 50
    levels[gap_pos] = np.nan
    releases = pd.Series(
        (dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC")
    )
    base = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": levels,
                "TB3MS": 1.0,
                "release_timestamp": releases,
                "release_timestamp_provenance": (
                    RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
                ),
                "timing_status": "release_aligned",
            }
        ),
        imputation_policy="ex_post_continuity",
    )
    featured = add_transitory_inflation_features(base, baseline_method="fed_target")

    following_release = releases.iloc[gap_pos + 1]
    assert base.loc[gap_pos, "imputation_available_at"] == following_release
    assert base.loc[gap_pos, "information_timestamp"] >= following_release
    assert featured.loc[gap_pos, "epsilon_information_timestamp"] >= following_release


@pytest.mark.parametrize(
    "baseline_method",
    ["rolling_36_shifted", "expanding_shifted"],
)
def test_baseline_information_uses_latest_dependency_timestamp(
    baseline_method: str,
) -> None:
    dates = pd.date_range("2010-01-31", periods=150, freq="ME")
    releases = pd.Series(
        (dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC")
    )
    delayed_dependency = len(dates) - 10
    releases.iloc[delayed_dependency] = releases.iloc[-1] + pd.offsets.Day(7)
    featured = add_transitory_inflation_features(
        build_base_frame(
            pd.DataFrame(
                {
                    "date": dates,
                    "CPIAUCSL": 100.0 + np.arange(len(dates), dtype=float),
                    "TB3MS": 1.0,
                    "release_timestamp": releases,
                    "release_timestamp_provenance": (
                        RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
                    ),
                    "timing_status": "release_aligned",
                }
            )
        ),
        baseline_method=baseline_method,
    )

    latest = featured.iloc[-1]
    assert latest["baseline_information_timestamp"] == releases.iloc[
        delayed_dependency
    ]
    assert latest["epsilon_information_timestamp"] == releases.iloc[
        delayed_dependency
    ]


def test_missing_inflation_value_clears_component_and_signal_timing() -> None:
    periods = 24
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    information_timestamps = pd.Series(
        (dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC")
    )
    inflation = pd.Series(2.5, index=range(periods), dtype=float)
    inflation.iloc[-1] = np.nan
    frame = pd.DataFrame(
        {
            "date": dates,
            "inflation_yoy": inflation,
            "inflation_yoy_information_timestamp": information_timestamps,
            "inflation_yoy_information_timestamp_provenance": (
                "derived_from_actual_release_metadata"
            ),
            "inflation_yoy_timing_status": "release_aligned",
        }
    )

    latest = add_transitory_inflation_features(
        frame,
        baseline_method="fed_target",
    ).iloc[-1]

    assert pd.isna(latest["inflation_yoy_information_timestamp"])
    assert latest["inflation_yoy_timing_status"] == "derived_value_unavailable"
    assert pd.isna(latest["epsilon_information_timestamp"])
    assert latest["epsilon_timing_status"] == "derived_value_unavailable"
    for window in (4, 8, 12):
        assert pd.isna(latest[f"tinf_{window}m_information_timestamp"])
        assert latest[f"tinf_{window}m_timing_status"] == "derived_value_unavailable"
    assert pd.isna(latest["information_timestamp"])
    assert latest["timing_status"] == "derived_value_unavailable"
    assert pd.isna(latest["above_baseline"])
    assert pd.isna(latest["run_length_above"])
    assert pd.isna(latest["short_regime_flag"])
    assert pd.isna(latest["medium_regime_flag"])
    assert pd.isna(latest["long_regime_flag"])
    assert pd.isna(latest["tinf_term_structure"])


def test_non_headline_measure_does_not_borrow_generic_headline_timing() -> None:
    periods = 24
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    frame = pd.DataFrame(
        {
            "date": dates,
            "core_cpi_yoy": np.linspace(2.0, 3.0, periods),
            "information_timestamp": pd.Series(
                (dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC")
            ),
            "information_timestamp_provenance": (
                "derived_from_actual_release_metadata"
            ),
            "timing_status": "release_aligned",
        }
    )

    latest = add_transitory_inflation_features(
        frame,
        inflation_col="core_cpi_yoy",
        baseline_method="fed_target",
    ).iloc[-1]

    assert pd.isna(latest["core_cpi_yoy_information_timestamp"])
    assert latest["core_cpi_yoy_timing_status"] == "reference_month_only"
    assert pd.isna(latest["information_timestamp"])
    assert latest["timing_status"] == "reference_month_only"


@pytest.mark.parametrize(
    ("baseline_method", "periods"),
    [
        ("rolling_36_shifted", 37),
        ("expanding_shifted", 121),
    ],
)
def test_dependency_timestamp_maximum_preserves_one_nanosecond(
    baseline_method: str,
    periods: int,
) -> None:
    base = pd.Timestamp("2025-01-01T00:00:00Z").as_unit("ns")
    timestamp_ns = np.full(periods, base.value, dtype=np.int64)
    delayed_dependency = 10
    timestamp_ns[delayed_dependency] += 1
    expected = pd.to_datetime(
        timestamp_ns[delayed_dependency],
        unit="ns",
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-31", periods=periods, freq="ME"),
            "inflation_yoy": np.linspace(2.0, 3.0, periods),
            "inflation_yoy_information_timestamp": pd.Series(
                pd.to_datetime(timestamp_ns, unit="ns", utc=True)
            ),
            "inflation_yoy_information_timestamp_provenance": (
                "derived_from_actual_release_metadata"
            ),
            "inflation_yoy_timing_status": "release_aligned",
        }
    )

    latest = add_transitory_inflation_features(
        frame,
        baseline_method=baseline_method,
    ).iloc[-1]

    assert latest["baseline_information_timestamp"] == expected
    assert latest["baseline_information_timestamp"].value == expected.value
    assert latest["epsilon_information_timestamp"] == expected


def test_snapshot_percentile_and_regime_use_full_distribution_availability() -> None:
    periods = 60
    dates = pd.date_range("2015-01-31", periods=periods, freq="ME")
    ordinary_timestamp = pd.Timestamp("2025-01-01T13:00:00Z")
    delayed_timestamp = ordinary_timestamp + pd.Timedelta(days=5)
    information_timestamps = pd.Series(ordinary_timestamp, index=range(periods))
    delayed_dependency = 25
    information_timestamps.iloc[delayed_dependency] = delayed_timestamp
    frame = pd.DataFrame(
        {
            "date": dates,
            "inflation_yoy": 2.0 + np.arange(periods, dtype=float) / 100.0,
            "inflation_yoy_information_timestamp": information_timestamps,
            "inflation_yoy_information_timestamp_provenance": (
                "derived_from_actual_release_metadata"
            ),
            "inflation_yoy_timing_status": "release_aligned",
        }
    )
    featured = add_transitory_inflation_features(
        frame,
        baseline_method="fed_target",
    )

    assert featured.iloc[-1]["information_timestamp"] == ordinary_timestamp
    snapshot = latest_signal_snapshot(featured)

    assert snapshot["percentile_information_timestamp"] == delayed_timestamp
    assert snapshot["percentile_timing_status"] == "release_aligned"
    assert snapshot["regime_information_timestamp"] == delayed_timestamp
    assert snapshot["regime_timing_status"] == "release_aligned"
    assert snapshot["information_timestamp"] == delayed_timestamp
    assert snapshot["timing_status"] == "release_aligned"

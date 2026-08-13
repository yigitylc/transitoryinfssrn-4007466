from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.data import (
    CURRENT_MONITORING_SCENARIOS,
    RELEASE_TIMESTAMP_PROVENANCE_ACTUAL,
    build_base_frame,
    build_current_monitoring_scenario_frame,
)
from transitory_inflation.features import (
    add_transitory_inflation_features,
    consecutive_true_count,
    latest_signal_snapshot,
    observed_only_historical_eligibility,
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


@pytest.mark.parametrize("scenario_id", CURRENT_MONITORING_SCENARIOS)
def test_current_scenario_timing_uses_following_release_without_weakening_h5(
    scenario_id: str,
) -> None:
    dates = pd.date_range("2020-01-31", "2026-06-30", freq="ME")
    gap_date = pd.Timestamp("2025-10-31")
    gap_pos = int(np.flatnonzero(dates == gap_date)[0])
    levels = 250.0 * (1.002 ** np.arange(len(dates), dtype=float))
    levels[gap_pos] = np.nan
    core_levels = 245.0 * (1.0018 ** np.arange(len(dates), dtype=float))
    core_levels[gap_pos] = np.nan
    releases = pd.Series(
        (dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC")
    )
    core_releases = releases + pd.offsets.Hour(1)
    observed = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": levels,
                "CPILFESL": core_levels,
                "TB3MS": 1.0,
                "release_timestamp": releases,
                "release_timestamp_provenance": RELEASE_TIMESTAMP_PROVENANCE_ACTUAL,
                "release_timing_status": "release_aligned",
                "core_cpi_release_timestamp": core_releases,
                "core_cpi_release_timestamp_provenance": (
                    RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
                ),
                "core_cpi_release_timing_status": "release_aligned",
            }
        ),
        imputation_policy="observed_only",
    )

    scenario = build_current_monitoring_scenario_frame(
        observed,
        scenario_id=scenario_id,
    )
    gap = scenario.loc[scenario["date"] == gap_date].iloc[0]
    following_release = releases.iloc[gap_pos + 1]
    core_following_release = core_releases.iloc[gap_pos + 1]

    assert gap["imputation_available_at"] == following_release
    assert gap["imputation_availability_basis"] == "following_release_timestamp"
    assert gap["cpi_level_information_timestamp"] == following_release
    assert gap["information_timestamp"] >= following_release
    assert gap["core_cpi_imputation_available_at"] == core_following_release
    assert (
        gap["core_cpi_imputation_availability_basis"]
        == "following_release_timestamp"
    )
    assert gap["core_cpi_level_information_timestamp"] == core_following_release


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


def test_current_snapshot_thresholds_use_only_prior_observed_eligible_history() -> None:
    periods = 120
    months = np.arange(periods, dtype=float)
    featured = add_transitory_inflation_features(
        pd.DataFrame(
            {
                "date": pd.date_range("2016-01-31", periods=periods, freq="ME"),
                "inflation_yoy": 2.0 + 0.7 * np.sin(months / 6.0) + 0.004 * months,
            }
        ),
        baseline_method="fed_target",
    )
    featured["imputation_policy"] = "observed_only"
    featured["signal_observed_only_eligible"] = True
    excluded_index = 70
    featured.loc[excluded_index, "tinf_4m"] = 10_000.0
    featured.loc[excluded_index, "signal_observed_only_eligible"] = False

    candidate = featured.copy()
    candidate["scenario_id"] = "base"
    candidate["calibration_policy"] = "observed_only_eligible_history"
    snapshot = latest_signal_snapshot(candidate, calibration_df=featured)

    latest_month = pd.Timestamp(candidate["date"].iloc[-1])
    eligible = (
        featured["signal_observed_only_eligible"]
        & featured["tinf_4m"].notna()
        & pd.to_datetime(featured["date"]).lt(latest_month)
    )
    expected_history = featured.loc[eligible, "tinf_4m"]
    assert snapshot["available"]
    assert snapshot["regime_lower_threshold"] == pytest.approx(
        expected_history.quantile(0.25)
    )
    assert snapshot["regime_upper_threshold"] == pytest.approx(
        expected_history.quantile(0.75)
    )
    assert snapshot["calibration_observation_count"] == len(expected_history)
    assert snapshot["calibration_end_month"] < snapshot["reference_month"]
    assert snapshot["calibration_policy"] == "observed_only_eligible_history"
    assert (
        snapshot["calibration_cutoff_policy"]
        == "strictly_prior_to_current_reference_month"
    )

    perturbed_candidate = candidate.copy()
    perturbed_candidate.loc[40:80, "tinf_4m"] += 1_000.0
    perturbed_snapshot = latest_signal_snapshot(
        perturbed_candidate,
        calibration_df=featured,
    )
    changed_excluded_calibration = featured.copy()
    changed_excluded_calibration.loc[excluded_index, "tinf_4m"] = -10_000.0
    excluded_snapshot = latest_signal_snapshot(
        candidate,
        calibration_df=changed_excluded_calibration,
    )

    for other in (perturbed_snapshot, excluded_snapshot):
        assert other["regime_lower_threshold"] == pytest.approx(
            snapshot["regime_lower_threshold"]
        )
        assert other["regime_upper_threshold"] == pytest.approx(
            snapshot["regime_upper_threshold"]
        )
        assert other["tinf_4m_percentile"] == pytest.approx(
            snapshot["tinf_4m_percentile"]
        )


def test_canonical_historical_eligibility_rejects_every_contamination_marker() -> None:
    frame = pd.DataFrame(
        {
            "uses_estimated_input": [False] * 9,
            "uses_imputed_input": [False] * 9,
            "estimated_input_months": [()] * 9,
            "signal_uses_imputed_input": [False] * 9,
            "signal_uses_missing_input": [False] * 9,
            "signal_observed_only_eligible": [True] * 9,
            "cpi_imputed": [False] * 9,
            "imputation_policy": ["observed_only"] * 9,
            "historical_population_policy": ["observed_only"] * 9,
            "data_policy": ["observed_only"] * 9,
        }
    )
    frame.loc[1, "uses_estimated_input"] = True
    frame.loc[2, "uses_imputed_input"] = True
    frame.at[3, "estimated_input_months"] = ("2025-10-31",)
    frame.loc[4, "imputation_policy"] = "ex_post_continuity"
    frame.loc[5, "signal_observed_only_eligible"] = False
    frame.loc[6, "cpi_imputed"] = True
    frame.loc[7, "historical_population_policy"] = "scenario_conditioned"
    frame.loc[8, "data_policy"] = "ex_post_continuity"

    assert observed_only_historical_eligibility(frame).tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_canonical_signal_gate_leaves_target_lineage_to_each_horizon() -> None:
    frame = pd.DataFrame(
        {
            "uses_estimated_input": [False],
            "uses_missing_input": [False],
            "imputation_policy": ["observed_only"],
            "outcome_12m_uses_imputed_input": [True],
            "outcome_12m_uses_missing_input": [True],
            "cpi_yoy_fwd_12m_uses_imputed_input": [True],
            "cpi_yoy_fwd_12m_uses_missing_input": [True],
        }
    )

    assert observed_only_historical_eligibility(frame).iloc[0]


def test_calibration_rejects_authoritative_estimate_marker_without_legacy_flags() -> None:
    periods = 120
    months = np.arange(periods, dtype=float)
    calibration = add_transitory_inflation_features(
        pd.DataFrame(
            {
                "date": pd.date_range("2016-01-31", periods=periods, freq="ME"),
                "inflation_yoy": 2.0 + 0.6 * np.sin(months / 7.0),
            }
        ),
        baseline_method="fed_target",
    )
    calibration["imputation_policy"] = "observed_only"
    contaminated_pos = 70
    calibration.loc[contaminated_pos, "tinf_4m"] = 10_000.0
    calibration.loc[contaminated_pos, "uses_estimated_input"] = True
    calibration.loc[contaminated_pos, "signal_uses_imputed_input"] = False
    calibration.loc[contaminated_pos, "signal_observed_only_eligible"] = True
    calibration.at[contaminated_pos, "estimated_input_months"] = ()

    candidate = calibration.copy()
    candidate.loc[contaminated_pos, "tinf_4m"] = 0.0
    snapshot = latest_signal_snapshot(candidate, calibration_df=calibration)
    eligible = (
        observed_only_historical_eligibility(calibration)
        & calibration["tinf_4m"].notna()
        & calibration["date"].lt(candidate["date"].iloc[-1])
    )
    expected = calibration.loc[eligible, "tinf_4m"]

    assert snapshot["available"]
    assert snapshot["calibration_observation_count"] == len(expected)
    assert snapshot["regime_upper_threshold"] == pytest.approx(
        expected.quantile(0.75)
    )
    assert snapshot["regime_upper_threshold"] < 10_000.0


@pytest.mark.parametrize(
    ("marker", "marker_value", "baseline_lineage"),
    [
        ("uses_estimated_input", True, "baseline_uses_imputed_input"),
        ("estimated_input_months", ("2025-10-31",), "baseline_uses_imputed_input"),
        ("uses_missing_input", True, "baseline_uses_missing_input"),
    ],
)
def test_generic_authoritative_marker_propagates_through_feature_dependencies(
    marker: str,
    marker_value: object,
    baseline_lineage: str,
) -> None:
    periods = 180
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-31", periods=periods, freq="ME"),
            "inflation_yoy": 2.0 + np.sin(np.arange(periods, dtype=float) / 8.0),
            "imputation_policy": "observed_only",
            "uses_estimated_input": False,
            "estimated_input_months": [()] * periods,
            "uses_missing_input": False,
        }
    )
    contaminated_pos = 100
    frame.at[contaminated_pos, marker] = marker_value
    changed = frame.copy()
    changed.loc[contaminated_pos, "inflation_yoy"] += 100.0

    featured = add_transitory_inflation_features(
        frame,
        baseline_method="rolling_36_shifted",
    )
    changed_featured = add_transitory_inflation_features(
        changed,
        baseline_method="rolling_36_shifted",
    )
    eligible = observed_only_historical_eligibility(featured)
    changed_baseline = featured["baseline"].sub(changed_featured["baseline"]).abs().gt(1e-12)
    changed_tinf = featured["tinf_4m"].sub(changed_featured["tinf_4m"]).abs().gt(1e-12)

    assert changed_baseline.sum() == 36
    assert featured.loc[changed_baseline, baseline_lineage].all()
    assert not eligible.loc[changed_baseline | changed_tinf].any()
    pd.testing.assert_series_equal(
        featured.loc[eligible, "baseline"],
        changed_featured.loc[eligible, "baseline"],
    )
    pd.testing.assert_series_equal(
        featured.loc[eligible, "tinf_4m"],
        changed_featured.loc[eligible, "tinf_4m"],
    )

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from transitory_inflation.data import INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
from transitory_inflation.features import (
    add_transitory_inflation_features,
    observed_only_historical_eligibility,
)
from transitory_inflation.market_data import (
    MARKET_TIMESTAMP_COLUMN,
    MARKET_TIMESTAMP_PROVENANCE_ACTUAL,
    MARKET_TIMESTAMP_PROVENANCE_COLUMN,
    MARKET_TIMESTAMP_STATUS_COLUMN,
    MARKET_TIMESTAMP_STATUS_EXACT,
    MARKET_VALUE_COLUMNS,
    build_market_close_frame,
)
from transitory_inflation.market_linkage import (
    MARKET_ORIGIN_INFORMATION_TIMESTAMP,
    MARKET_ORIGIN_UNAVAILABLE,
    MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED,
    MARKET_TIMING_UNAVAILABLE,
    build_market_linkage_tables,
)
from transitory_inflation.trader_research import (
    available_regimes,
    build_trader_research_view,
    conditional_forward_distribution,
    current_bucket_from_snapshot,
    latest_walk_forward_bucket,
    regime_analog_months,
)
from transitory_inflation.validation import (
    REGIME_ORDER,
    add_short_term_pressure_labels,
    add_walk_forward_regime_labels,
)


def _signal_frame(periods: int = 120) -> pd.DataFrame:
    months = np.arange(periods, dtype=float)
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-31", periods=periods, freq="ME"),
            "inflation_yoy": 2.0 + 0.8 * np.sin(months / 5.0) + 0.01 * months,
        }
    )
    return add_transitory_inflation_features(raw, baseline_method="fed_target")


def _market_frame(signal: pd.DataFrame) -> pd.DataFrame:
    n = len(signal)
    return pd.DataFrame(
        {
            "date": signal["date"],
            "yield_2y": np.linspace(1.0, 3.0, n),
            "yield_10y": np.linspace(2.0, 4.0, n),
            "breakeven_5y": np.linspace(2.1, 2.6, n),
            "breakeven_10y": np.linspace(2.0, 2.5, n),
            "real_yield_5y": np.linspace(0.1, 0.6, n),
            "real_yield_10y": np.linspace(0.2, 0.7, n),
        }
    )


def _explicit_neutral_signal(periods: int = 8) -> pd.DataFrame:
    ramp = np.linspace(0.0, 0.7, periods)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=periods, freq="ME"),
            "epsilon": ramp,
            "tinf_4m": ramp,
            "tinf_8m": ramp,
            "tinf_12m": ramp,
            "historical_regime": ["neutral"] * periods,
            "historical_short_term_pressure": ["mixed"] * periods,
        }
    )


def test_bucket_uses_latest_row_lookahead_safe_walk_forward_label() -> None:
    signal = _signal_frame(120)
    bucket = latest_walk_forward_bucket(signal)

    labeled = add_short_term_pressure_labels(add_walk_forward_regime_labels(signal))
    valid = labeled["historical_regime"].notna() & labeled["historical_short_term_pressure"].notna()
    last = labeled.loc[valid].sort_values("date").iloc[-1]

    assert bucket.available
    assert bucket.regime == str(last["historical_regime"])
    assert bucket.pressure == str(last["historical_short_term_pressure"])
    assert bucket.reference_month == pd.Timestamp(last["date"])
    assert bucket.information_timestamp is None
    assert bucket.timing_status == "reference_month_only"
    assert bucket.as_of == bucket.reference_month
    assert "not a signal availability" in bucket.as_of_semantics
    assert bucket.regime_count > 0
    assert 0 < bucket.regime_pressure_count <= bucket.regime_count


def test_bucket_exposes_reference_month_and_information_timestamp_separately() -> None:
    signal = _signal_frame(120)
    signal["reference_month"] = signal["date"]
    signal["information_timestamp"] = (
        signal["date"] + pd.offsets.Day(13) + pd.offsets.Hour(13)
    ).dt.tz_localize("UTC")
    signal["information_timestamp_provenance"] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    signal["timing_status"] = "release_aligned"
    signal["tinf_4m_information_timestamp"] = signal["information_timestamp"]
    signal["tinf_4m_information_timestamp_provenance"] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    signal["tinf_4m_timing_status"] = "release_aligned"

    bucket = latest_walk_forward_bucket(signal)

    assert bucket.available
    assert bucket.reference_month == pd.Timestamp(signal["date"].iloc[-1])
    assert bucket.information_timestamp == signal["information_timestamp"].iloc[-1]
    assert bucket.reference_month != bucket.information_timestamp
    assert bucket.timing_status == "release_aligned"


def test_bucket_timing_waits_for_regime_and_pressure_dependencies() -> None:
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [pd.Timestamp("2024-02-13 17:00:00+00:00")],
            "information_timestamp_provenance": [INFORMATION_TIMESTAMP_PROVENANCE_RELEASES],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_regime_information_timestamp": [pd.Timestamp("2024-02-13 19:00:00+00:00")],
            "historical_regime_information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "historical_regime_timing_status": ["release_aligned"],
            "historical_short_term_pressure": ["mixed"],
            "historical_short_term_pressure_information_timestamp": [
                pd.Timestamp("2024-02-13 18:00:00+00:00")
            ],
            "historical_short_term_pressure_information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "historical_short_term_pressure_timing_status": ["release_aligned"],
            "tinf_4m": [1.0],
        }
    )

    bucket = latest_walk_forward_bucket(signal)

    assert bucket.available
    assert bucket.information_timestamp == pd.Timestamp("2024-02-13 19:00:00+00:00")
    assert bucket.timing_status == "release_aligned"


def test_bucket_unavailable_without_enough_prior_history() -> None:
    bucket = latest_walk_forward_bucket(_signal_frame(20))

    assert not bucket.available
    assert bucket.regime is None
    assert "history" in (bucket.reason or "").lower()


def test_current_scenario_bucket_uses_strict_observed_historical_population() -> None:
    observed = _signal_frame(120)
    observed["imputation_policy"] = "observed_only"
    labeled = add_short_term_pressure_labels(add_walk_forward_regime_labels(observed))
    valid = (
        labeled["signal_observed_only_eligible"].fillna(False).astype(bool)
        & labeled["historical_regime"].notna()
        & labeled["historical_short_term_pressure"].notna()
    )
    query_row = labeled.loc[valid].iloc[-1]
    current_reference_month = pd.Timestamp(observed["date"].iloc[-1]) + pd.offsets.MonthEnd(1)
    snapshot = {
        "available": True,
        "scenario_id": "base",
        "regime": str(query_row["historical_regime"]),
        "pressure": str(query_row["historical_short_term_pressure"]),
        "reference_month": current_reference_month,
        "information_timestamp": pd.NaT,
        "timing_status": "reference_month_only",
        "estimate_method": "geometric_midpoint_adjacent_official_levels",
        "estimated_reference_month": pd.Timestamp("2025-10-31"),
        "estimate_value": 324.65,
        "uses_estimated_input": True,
        "estimated_input_months": ("2025-10-31",),
        "calibration_policy": "observed_only_eligible_history",
        "data_vintage_status": "latest_revised_non_vintage",
        "sample_mode": "live_dashboard",
        "baseline_method": "rolling_36_shifted",
        "ex_post_status": "ex_post_missing_cpi_sensitivity",
        "lookahead_status": "uses_official_november_endpoint",
        "retrieved_at": "2026-07-16T00:00:00+00:00",
        "series_lineage": {
            "headline_cpi": {
                "series_key": "headline_cpi",
                "missing_reference_month": "2025-10-31T00:00:00",
                "september_endpoint": {
                    "month": "2025-09-30T00:00:00",
                    "value": 324.245,
                },
                "november_endpoint": {
                    "month": "2025-11-30T00:00:00",
                    "value": 325.063,
                },
                "estimate_value": 324.65,
                "available": True,
            },
            "core_cpi": {
                "series_key": "core_cpi",
                "missing_reference_month": "2025-10-31T00:00:00",
                "september_endpoint": {
                    "month": "2025-09-30T00:00:00",
                    "value": 330.418,
                },
                "november_endpoint": {
                    "month": "2025-11-30T00:00:00",
                    "value": 331.043,
                },
                "estimate_value": 330.73,
                "available": True,
            },
        },
    }

    bucket = current_bucket_from_snapshot(snapshot, observed)

    expected_regime = valid & labeled["historical_regime"].eq(bucket.regime)
    expected_regime_pressure = expected_regime & labeled["historical_short_term_pressure"].eq(
        bucket.pressure
    )
    assert bucket.available
    assert bucket.reference_month == current_reference_month
    assert bucket.reference_month > pd.Timestamp(observed["date"].max())
    assert bucket.scenario_id == "base"
    assert bucket.uses_estimated_input
    assert bucket.estimated_input_months == ("2025-10-31",)
    assert bucket.historical_population_policy == "observed_only"
    assert bucket.regime_count == int(expected_regime.sum())
    assert bucket.regime_pressure_count == int(expected_regime_pressure.sum())
    assert bucket.historical_evidence_endpoint == pd.Timestamp(labeled.loc[valid, "date"].max())
    assert set(bucket.series_lineage) == {"headline_cpi", "core_cpi"}
    assert bucket.sample_mode == "live_dashboard"
    assert bucket.baseline_method == "rolling_36_shifted"
    assert bucket.ex_post_status == "ex_post_missing_cpi_sensitivity"
    assert bucket.lookahead_status == "uses_official_november_endpoint"
    payload = json.loads(json.dumps(bucket.to_dict()))
    assert payload["series_lineage"] == snapshot["series_lineage"]
    assert payload["reference_month"] == current_reference_month.isoformat()


def test_unavailable_current_bucket_preserves_both_series_lineage_round_trip() -> None:
    snapshot = {
        "available": False,
        "reason": "core_cpi exact November endpoint is missing",
        "classification_status": "unavailable_required_series",
        "reference_month": pd.Timestamp("2025-10-31"),
        "information_timestamp": pd.NaT,
        "timing_status": "derived_value_unavailable",
        "scenario_id": "base",
        "uses_estimated_input": True,
        "uses_imputed_input": True,
        "estimated_input_months": ["2025-10-31"],
        "sample_mode": "live_dashboard",
        "baseline_method": "rolling_36_shifted",
        "ex_post_status": "ex_post_missing_cpi_sensitivity",
        "lookahead_status": "unavailable_required_endpoint",
        "series_lineage": {
            "headline_cpi": {
                "series_key": "headline_cpi",
                "available": True,
                "estimate_attempted": True,
                "estimate_value": 324.65,
            },
            "core_cpi": {
                "series_key": "core_cpi",
                "available": False,
                "failure_code": "missing_november_endpoint",
                "estimate_attempted": False,
                "estimate_value": None,
            },
        },
    }

    bucket = current_bucket_from_snapshot(snapshot, _signal_frame(120))

    assert not bucket.available
    assert bucket.regime is None
    assert bucket.pressure is None
    assert bucket.reference_month == pd.Timestamp("2025-10-31")
    assert bucket.information_timestamp is None
    assert bucket.timing_status == "derived_value_unavailable"
    assert bucket.as_of == pd.Timestamp("2025-10-31")
    assert bucket.classification_status == "unavailable_required_series"
    assert bucket.uses_estimated_input
    assert bucket.uses_imputed_input
    payload = json.loads(json.dumps(bucket.to_dict()))
    assert payload["series_lineage"] == snapshot["series_lineage"]
    assert payload["reason"] == snapshot["reason"]
    assert payload["reference_month"] == "2025-10-31T00:00:00"
    assert payload["information_timestamp"] is None
    assert payload["timing_status"] == "derived_value_unavailable"


def test_current_scenario_bucket_excludes_uses_estimated_input_only_row() -> None:
    observed = add_short_term_pressure_labels(add_walk_forward_regime_labels(_signal_frame(120)))
    observed["imputation_policy"] = "observed_only"
    valid = (
        observed["historical_regime"].notna() & observed["historical_short_term_pressure"].notna()
    )
    contaminated_index = observed.index[valid][-1]
    contaminated_row = observed.loc[contaminated_index]
    snapshot = {
        "available": True,
        "scenario_id": "base",
        "regime": str(contaminated_row["historical_regime"]),
        "pressure": str(contaminated_row["historical_short_term_pressure"]),
        "reference_month": pd.Timestamp("2026-06-30"),
    }

    clean = current_bucket_from_snapshot(snapshot, observed)
    contaminated = observed.copy()
    contaminated["uses_estimated_input"] = False
    contaminated.loc[contaminated_index, "uses_estimated_input"] = True
    filtered = current_bucket_from_snapshot(snapshot, contaminated)

    assert clean.available and filtered.available
    assert filtered.regime_count == clean.regime_count - 1
    assert filtered.regime_pressure_count == clean.regime_pressure_count - 1
    assert filtered.historical_evidence_endpoint < clean.historical_evidence_endpoint


def test_latest_bucket_rejects_prelabelled_estimated_only_row_and_keeps_clean_rows() -> None:
    labeled = add_short_term_pressure_labels(add_walk_forward_regime_labels(_signal_frame(120)))
    labeled["imputation_policy"] = "observed_only"
    valid = labeled["historical_regime"].notna() & labeled["historical_short_term_pressure"].notna()
    latest_valid_index = labeled.index[valid][-1]

    clean = latest_walk_forward_bucket(labeled)
    contaminated = labeled.copy()
    contaminated["uses_estimated_input"] = False
    contaminated.loc[latest_valid_index, "uses_estimated_input"] = True
    filtered = latest_walk_forward_bucket(contaminated)
    eligible = observed_only_historical_eligibility(contaminated)
    eligible_labels = eligible & valid
    expected_latest = contaminated.loc[eligible_labels].sort_values("date").iloc[-1]
    expected_regime = str(expected_latest["historical_regime"])
    expected_pressure = str(expected_latest["historical_short_term_pressure"])
    expected_regime_mask = eligible & contaminated["historical_regime"].eq(expected_regime)

    assert clean.available and filtered.available
    assert clean.reference_month == pd.Timestamp(labeled.loc[latest_valid_index, "date"])
    assert filtered.reference_month == pd.Timestamp(expected_latest["date"])
    assert filtered.reference_month < clean.reference_month
    assert filtered.regime == expected_regime
    assert filtered.pressure == expected_pressure
    assert filtered.regime_count == int(expected_regime_mask.sum())
    assert filtered.regime_pressure_count == int(
        (
            expected_regime_mask
            & contaminated["historical_short_term_pressure"].eq(expected_pressure)
        ).sum()
    )


def test_prelabelled_middle_contamination_forces_trader_label_recalculation() -> None:
    prelabelled = add_short_term_pressure_labels(add_walk_forward_regime_labels(_signal_frame(160)))
    prelabelled["imputation_policy"] = "observed_only"
    prelabelled["uses_estimated_input"] = False
    prelabelled.loc[50, "uses_estimated_input"] = True
    label_columns = [
        column
        for column in prelabelled
        if column.startswith("historical_regime")
        or column.startswith("historical_short_term_pressure")
    ]

    routed = latest_walk_forward_bucket(prelabelled)
    recomputed = latest_walk_forward_bucket(prelabelled.drop(columns=label_columns))

    assert routed.available and recomputed.available
    assert routed.reference_month == recomputed.reference_month
    assert routed.regime == recomputed.regime
    assert routed.pressure == recomputed.pressure
    assert routed.regime_count == recomputed.regime_count
    assert routed.regime_pressure_count == recomputed.regime_pressure_count


def test_conditional_distribution_equals_regime_filtered_summary() -> None:
    signal = _signal_frame(120)
    tables = build_market_linkage_tables(signal, _market_frame(signal), horizons=(3, 6))
    bucket = latest_walk_forward_bucket(signal)

    distribution = conditional_forward_distribution(tables, bucket.regime, horizons=(3,))
    expected = tables.summary_by_regime.loc[
        (tables.summary_by_regime["historical_regime"] == bucket.regime)
        & (tables.summary_by_regime["horizon_months"] == 3)
    ].reset_index(drop=True)

    assert not distribution.empty
    pd.testing.assert_frame_equal(distribution, expected)


def test_conditional_distribution_regime_and_pressure_match_labels() -> None:
    signal = _signal_frame(120)
    tables = build_market_linkage_tables(signal, _market_frame(signal), horizons=(3,))
    combined = tables.summary_by_regime_and_pressure
    assert not combined.empty

    first = combined.iloc[0]
    regime = first["historical_regime"]
    pressure = first["historical_short_term_pressure"]

    distribution = conditional_forward_distribution(tables, regime, pressure)
    expected = combined.loc[
        (combined["historical_regime"] == regime)
        & (combined["historical_short_term_pressure"] == pressure)
    ]

    assert len(distribution) == len(expected)
    assert (distribution["historical_regime"] == regime).all()
    assert (distribution["historical_short_term_pressure"] == pressure).all()


def test_analog_months_are_in_bucket_with_changes_and_sorted() -> None:
    signal = _signal_frame(120)
    tables = build_market_linkage_tables(signal, _market_frame(signal), horizons=(3,))
    bucket = latest_walk_forward_bucket(signal)
    panel = tables.panel

    analog = regime_analog_months(panel, bucket.regime, horizons=(3,))

    change_cols = [
        f"{variable}_change_3m_bp"
        for variable in MARKET_VALUE_COLUMNS
        if f"{variable}_change_3m_bp" in panel.columns
    ]
    expected_mask = (panel["historical_regime"] == bucket.regime) & panel[change_cols].notna().any(
        axis=1
    )

    assert not analog.empty
    assert (analog["historical_regime"] == bucket.regime).all()
    assert analog["date"].is_monotonic_increasing
    assert len(analog) == int(expected_mask.sum())
    assert analog[change_cols].notna().any(axis=1).all()


def test_trader_view_preserves_per_series_timing_for_partial_rows() -> None:
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [pd.Timestamp("2024-02-13 17:00:00+00:00")],
            "information_timestamp_provenance": [INFORMATION_TIMESTAMP_PROVENANCE_RELEASES],
            "timing_status": ["release_aligned"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
        }
    )
    market = build_market_close_frame(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-02-14", "2024-03-14"]),
                "DGS2": [4.2, 4.5],
                "DGS10": [np.nan, np.nan],
                MARKET_TIMESTAMP_COLUMN: pd.to_datetime(
                    [
                        "2024-02-14 18:00:00+00:00",
                        "2024-03-14 18:00:00+00:00",
                    ]
                ),
                MARKET_TIMESTAMP_PROVENANCE_COLUMN: [MARKET_TIMESTAMP_PROVENANCE_ACTUAL] * 2,
                MARKET_TIMESTAMP_STATUS_COLUMN: [MARKET_TIMESTAMP_STATUS_EXACT] * 2,
            }
        )
    )
    tables = build_market_linkage_tables(signal, market, horizons=(1,))

    view = build_trader_research_view(
        tables,
        "neutral",
        "mixed",
        horizons=(1,),
    )

    assert view.available
    analog = view.analog_months.iloc[0]
    assert analog["yield_2y_origin_basis"] == MARKET_ORIGIN_INFORMATION_TIMESTAMP
    assert analog["yield_2y_timing_status"] == MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED
    assert analog["yield_10y_origin_basis"] == MARKET_ORIGIN_UNAVAILABLE
    assert analog["yield_10y_timing_status"] == MARKET_TIMING_UNAVAILABLE
    summary = view.series_timing_summary.set_index("market_variable")
    assert summary.loc["yield_2y", "market_origin_basis"] == (MARKET_ORIGIN_INFORMATION_TIMESTAMP)
    assert summary.loc["yield_10y", "market_origin_basis"] == (MARKET_ORIGIN_UNAVAILABLE)


def test_weak_evidence_flag_set_for_small_bucket() -> None:
    signal = _explicit_neutral_signal(8)
    market = _market_frame(signal)
    tables = build_market_linkage_tables(signal, market, horizons=(2,))

    view = build_trader_research_view(tables, "neutral", "mixed", horizons=(2,))

    assert view.available
    assert view.weak_evidence is True
    assert not view.distribution.empty


def test_view_unavailable_without_market_variables() -> None:
    signal = _signal_frame(120)
    market_dateonly = pd.DataFrame({"date": signal["date"]})
    tables = build_market_linkage_tables(signal, market_dateonly, horizons=(3,))

    view = build_trader_research_view(tables, "neutral", horizons=(3,))

    assert not view.available
    assert "market" in (view.reason or "").lower()


def test_view_unavailable_for_empty_bucket() -> None:
    signal = _signal_frame(120)
    tables = build_market_linkage_tables(signal, _market_frame(signal), horizons=(3,))

    view = build_trader_research_view(tables, "no-such-regime", horizons=(3,))

    assert not view.available
    assert view.distribution.empty
    assert view.analog_months.empty


def test_available_regimes_ordered_subset() -> None:
    signal = _signal_frame(120)
    tables = build_market_linkage_tables(signal, _market_frame(signal), horizons=(3,))

    regimes = available_regimes(tables)

    assert regimes
    assert set(regimes).issubset(set(REGIME_ORDER))
    ordered_reference = [regime for regime in REGIME_ORDER if regime in set(regimes)]
    assert list(regimes) == ordered_reference

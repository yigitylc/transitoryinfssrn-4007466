from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.data import INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
from transitory_inflation.features import add_transitory_inflation_features
from transitory_inflation.market_data import (
    MARKET_TIMESTAMP_COLUMN,
    MARKET_TIMESTAMP_PROVENANCE_ACTUAL,
    MARKET_TIMESTAMP_PROVENANCE_COLUMN,
    MARKET_TIMESTAMP_STATUS_COLUMN,
    MARKET_TIMESTAMP_STATUS_EXACT,
    build_market_close_frame,
)
from transitory_inflation.market_linkage import (
    MARKET_AVAILABILITY_FULL,
    MARKET_AVAILABILITY_PARTIAL,
    MARKET_AVAILABILITY_UNAVAILABLE,
    MARKET_CHANNELS,
    MARKET_ORIGIN_CONSERVATIVE_PROXY,
    MARKET_ORIGIN_INFORMATION_TIMESTAMP,
    MARKET_ORIGIN_MIXED,
    MARKET_ORIGIN_NEXT_OBSERVATION_PROXY,
    MARKET_ORIGIN_PARTIAL,
    MARKET_ORIGIN_UNAVAILABLE,
    MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED,
    MARKET_TIMING_MIXED,
    MARKET_TIMING_PARTIAL,
    MARKET_TIMING_UNAVAILABLE,
    add_forward_market_changes,
    build_market_linkage_panel,
    build_market_linkage_tables,
    channel_regime_summary,
    forward_market_change_summary_by_regime,
    market_signal_correlations,
    rank_regime_pressure_market_changes,
)


def _signal_frame(periods: int = 60) -> pd.DataFrame:
    months = np.arange(periods, dtype=float)
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=periods, freq="ME"),
            "inflation_yoy": 2.0 + 0.4 * np.sin(months / 4.0) + 0.01 * months,
        }
    )
    return add_transitory_inflation_features(raw, baseline_method="fed_target")


def test_forward_market_changes_assign_t_plus_h_to_t_and_terminal_nan() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=5, freq="ME"),
            "yield_2y": [1.0, 1.1, 1.3, 1.6, 2.0],
        }
    )

    out = add_forward_market_changes(panel, horizons=(2,))

    assert out.loc[0, "yield_2y_fwd_2m"] == 1.3
    assert out.loc[0, "yield_2y_change_2m_bp"] == pytest.approx(30.0)
    assert out.loc[2, "yield_2y_change_2m_bp"] == pytest.approx(70.0)
    assert out.loc[3:, "yield_2y_change_2m_bp"].isna().all()


def test_market_linkage_requires_completed_signal_features() -> None:
    raw_only = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=5, freq="ME"),
            "inflation_yoy": [2.0, 2.1, 2.2, 2.3, 2.4],
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=5, freq="ME"),
            "yield_2y": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )

    with pytest.raises(KeyError, match="epsilon"):
        build_market_linkage_panel(raw_only, market)


def test_release_aligned_market_origin_never_precedes_publication() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-02-13", "2024-02-13", "2024-02-14", "2024-05-13"]
            ),
            "market_timestamp": pd.to_datetime(
                [
                    "2024-02-13 16:00:00+00:00",
                    "2024-02-13 18:00:00+00:00",
                    "2024-02-14 16:00:00+00:00",
                    "2024-05-13 18:00:00+00:00",
                ]
            ),
            "market_timestamp_provenance": [
                MARKET_TIMESTAMP_PROVENANCE_ACTUAL
            ]
            * 4,
            "market_timestamp_status": [MARKET_TIMESTAMP_STATUS_EXACT] * 4,
            "yield_2y": [1.0, 1.1, 1.2, 1.5],
        }
    )

    panel = build_market_linkage_panel(signal, market, horizons=(3,))
    row = panel.iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_INFORMATION_TIMESTAMP
    assert row["yield_2y_origin_timestamp"] == pd.Timestamp(
        "2024-02-13 18:00:00+00:00"
    )
    assert row["yield_2y_origin_timestamp"] >= publication
    assert row["yield_2y_origin_timestamp"] != pd.Timestamp(
        "2024-02-13 16:00:00+00:00"
    )
    assert row["yield_2y_fwd_3m_timestamp"] == pd.Timestamp(
        "2024-05-13 18:00:00+00:00"
    )
    assert row["yield_2y_change_3m_bp"] == pytest.approx(40.0)


def test_market_origin_waits_for_walk_forward_label_information() -> None:
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [
                pd.Timestamp("2024-02-13 17:00:00+00:00")
            ],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_regime_information_timestamp": [
                pd.Timestamp("2024-02-13 19:00:00+00:00")
            ],
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
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-02-13", "2024-02-13", "2024-03-13"]
            ),
            MARKET_TIMESTAMP_COLUMN: pd.to_datetime(
                [
                    "2024-02-13 18:00:00+00:00",
                    "2024-02-13 20:00:00+00:00",
                    "2024-03-13 20:00:00+00:00",
                ]
            ),
            MARKET_TIMESTAMP_PROVENANCE_COLUMN: [
                MARKET_TIMESTAMP_PROVENANCE_ACTUAL
            ]
            * 3,
            MARKET_TIMESTAMP_STATUS_COLUMN: [MARKET_TIMESTAMP_STATUS_EXACT] * 3,
            "yield_2y": [1.0, 1.2, 1.5],
        }
    )

    row = build_market_linkage_panel(signal, market, horizons=(1,)).iloc[0]

    assert row["yield_2y_origin_timestamp"] == pd.Timestamp(
        "2024-02-13 20:00:00+00:00"
    )
    assert row["yield_2y_origin_timestamp"] >= pd.Timestamp(
        "2024-02-13 19:00:00+00:00"
    )


def test_duplicate_market_rows_remain_coherent_through_normalizer_and_linkage() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    raw_market = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-02-13", "2024-02-13", "2024-02-14", "2024-03-14"]
            ),
            "DGS2": [4.0, np.nan, 4.2, 4.5],
            MARKET_TIMESTAMP_COLUMN: pd.to_datetime(
                [
                    "2024-02-13 16:00:00+00:00",
                    "2024-02-13 18:00:00+00:00",
                    "2024-02-14 21:00:00+00:00",
                    "2024-03-14 21:00:00+00:00",
                ]
            ),
            MARKET_TIMESTAMP_PROVENANCE_COLUMN: (
                [MARKET_TIMESTAMP_PROVENANCE_ACTUAL] * 4
            ),
            MARKET_TIMESTAMP_STATUS_COLUMN: [MARKET_TIMESTAMP_STATUS_EXACT] * 4,
        }
    )
    normalized = build_market_close_frame(raw_market)
    duplicate_date = normalized.loc[normalized["date"] == pd.Timestamp("2024-02-13")].iloc[0]

    assert pd.isna(duplicate_date["yield_2y"])
    assert duplicate_date[MARKET_TIMESTAMP_COLUMN] == pd.Timestamp(
        "2024-02-13 18:00:00+00:00"
    )

    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    row = build_market_linkage_panel(signal, normalized, horizons=(1,)).iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_INFORMATION_TIMESTAMP
    assert row["yield_2y_origin_timestamp"] == pd.Timestamp(
        "2024-02-14 21:00:00+00:00"
    )
    assert row["yield_2y"] == pytest.approx(4.2)


def test_exact_alignment_without_post_information_observation_is_unavailable() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    market = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-02-13")],
            MARKET_TIMESTAMP_COLUMN: [pd.Timestamp("2024-02-13 16:00:00+00:00")],
            MARKET_TIMESTAMP_PROVENANCE_COLUMN: [MARKET_TIMESTAMP_PROVENANCE_ACTUAL],
            MARKET_TIMESTAMP_STATUS_COLUMN: [MARKET_TIMESTAMP_STATUS_EXACT],
            "yield_2y": [4.0],
        }
    )

    row = build_market_linkage_panel(signal, market, horizons=(1,)).iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_UNAVAILABLE
    assert row["market_timing_status"] == MARKET_TIMING_UNAVAILABLE
    assert pd.isna(row["market_origin_timestamp"])
    assert pd.isna(row["yield_2y"])


def test_mixed_exact_series_availability_is_partial_and_preserves_usable_value() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    market = build_market_close_frame(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-02-13", "2024-02-14", "2024-03-14"]
                ),
                "DGS2": [4.0, 4.2, 4.5],
                "DGS10": [5.0, np.nan, np.nan],
                MARKET_TIMESTAMP_COLUMN: pd.to_datetime(
                    [
                        "2024-02-13 16:00:00+00:00",
                        "2024-02-14 18:00:00+00:00",
                        "2024-03-14 18:00:00+00:00",
                    ]
                ),
                MARKET_TIMESTAMP_PROVENANCE_COLUMN: [
                    MARKET_TIMESTAMP_PROVENANCE_ACTUAL
                ]
                * 3,
                MARKET_TIMESTAMP_STATUS_COLUMN: [MARKET_TIMESTAMP_STATUS_EXACT] * 3,
            }
        )
    )

    tables = build_market_linkage_tables(signal, market, horizons=(1,))
    row = tables.panel.iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_PARTIAL
    assert row["market_timing_status"] == MARKET_TIMING_PARTIAL
    assert row["market_availability_status"] == MARKET_AVAILABILITY_PARTIAL
    assert row["market_required_series_count"] == 2
    assert row["market_available_series_count"] == 1
    assert row["yield_2y_origin_basis"] == MARKET_ORIGIN_INFORMATION_TIMESTAMP
    assert row["yield_10y_origin_basis"] == MARKET_ORIGIN_UNAVAILABLE
    assert (
        row["yield_2y_timing_status"]
        == MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED
    )
    assert row["yield_10y_timing_status"] == MARKET_TIMING_UNAVAILABLE
    assert row["yield_2y_origin_timestamp"] == pd.Timestamp(
        "2024-02-14 18:00:00+00:00"
    )
    assert pd.isna(row["yield_10y_origin_timestamp"])
    assert row["yield_2y"] == pytest.approx(4.2)
    assert pd.isna(row["yield_10y"])
    summary = tables.timing_summary.iloc[0]
    assert summary["market_origin_basis"] == MARKET_ORIGIN_PARTIAL
    assert summary["exact_series_origin_count"] == 1
    assert summary["unavailable_series_origin_count"] == 1


def test_entirely_missing_series_does_not_downgrade_an_exact_series() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
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
                MARKET_TIMESTAMP_PROVENANCE_COLUMN: [
                    MARKET_TIMESTAMP_PROVENANCE_ACTUAL
                ]
                * 2,
                MARKET_TIMESTAMP_STATUS_COLUMN: [MARKET_TIMESTAMP_STATUS_EXACT] * 2,
            }
        )
    )

    tables = build_market_linkage_tables(signal, market, horizons=(1,))
    row = tables.panel.iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_PARTIAL
    assert row["market_availability_status"] == MARKET_AVAILABILITY_PARTIAL
    assert row["yield_2y_origin_basis"] == MARKET_ORIGIN_INFORMATION_TIMESTAMP
    assert (
        row["yield_2y_timing_status"]
        == MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED
    )
    assert row["yield_2y_origin_timestamp"] == pd.Timestamp(
        "2024-02-14 18:00:00+00:00"
    )
    assert row["yield_10y_origin_basis"] == MARKET_ORIGIN_UNAVAILABLE
    assert row["yield_2y"] == pytest.approx(4.2)
    assert pd.isna(row["yield_10y"])
    series_summary = tables.series_timing_summary
    exact_2y = series_summary.loc[
        (series_summary["market_variable"] == "yield_2y")
        & (
            series_summary["market_origin_basis"]
            == MARKET_ORIGIN_INFORMATION_TIMESTAMP
        )
    ]
    unavailable_10y = series_summary.loc[
        (series_summary["market_variable"] == "yield_10y")
        & (series_summary["market_origin_basis"] == MARKET_ORIGIN_UNAVAILABLE)
    ]
    assert len(exact_2y) == 1
    assert len(unavailable_10y) == 1


def test_fully_available_heterogeneous_series_are_labelled_mixed() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    market = build_market_close_frame(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-02-14", "2024-02-15", "2024-03-14", "2024-03-15"]
                ),
                "DGS2": [4.2, np.nan, 4.5, np.nan],
                "DGS10": [np.nan, 5.3, np.nan, 5.6],
                MARKET_TIMESTAMP_COLUMN: pd.Series(
                    [
                        pd.Timestamp("2024-02-14 18:00:00+00:00"),
                        pd.NaT,
                        pd.Timestamp("2024-03-14 18:00:00+00:00"),
                        pd.NaT,
                    ],
                    dtype="datetime64[ns, UTC]",
                ),
                MARKET_TIMESTAMP_PROVENANCE_COLUMN: [
                    MARKET_TIMESTAMP_PROVENANCE_ACTUAL
                ]
                * 4,
                MARKET_TIMESTAMP_STATUS_COLUMN: [MARKET_TIMESTAMP_STATUS_EXACT] * 4,
            }
        )
    )

    row = build_market_linkage_panel(signal, market, horizons=(1,)).iloc[0]

    assert row["market_availability_status"] == MARKET_AVAILABILITY_FULL
    assert row["market_origin_basis"] == MARKET_ORIGIN_MIXED
    assert row["market_timing_status"] == MARKET_TIMING_MIXED
    assert row["yield_2y_origin_basis"] == MARKET_ORIGIN_INFORMATION_TIMESTAMP
    assert row["yield_10y_origin_basis"] == MARKET_ORIGIN_NEXT_OBSERVATION_PROXY
    assert row["yield_2y_origin_timestamp"] == pd.Timestamp(
        "2024-02-14 18:00:00+00:00"
    )
    assert row["yield_10y_origin_observation_date"] == pd.Timestamp("2024-02-15")


def test_all_exact_series_can_use_different_eligible_origin_timestamps() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    market = build_market_close_frame(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-02-14", "2024-02-15", "2024-03-15"]
                ),
                "DGS2": [4.2, np.nan, 4.5],
                "DGS10": [np.nan, 5.3, 5.6],
                MARKET_TIMESTAMP_COLUMN: pd.to_datetime(
                    [
                        "2024-02-14 18:00:00+00:00",
                        "2024-02-15 19:00:00+00:00",
                        "2024-03-15 19:00:00+00:00",
                    ]
                ),
                MARKET_TIMESTAMP_PROVENANCE_COLUMN: [
                    MARKET_TIMESTAMP_PROVENANCE_ACTUAL
                ]
                * 3,
                MARKET_TIMESTAMP_STATUS_COLUMN: [MARKET_TIMESTAMP_STATUS_EXACT] * 3,
            }
        )
    )

    row = build_market_linkage_panel(signal, market, horizons=(1,)).iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_INFORMATION_TIMESTAMP
    assert row["market_availability_status"] == MARKET_AVAILABILITY_FULL
    assert row["market_available_series_count"] == 2
    assert row["yield_2y_origin_timestamp"] == pd.Timestamp(
        "2024-02-14 18:00:00+00:00"
    )
    assert row["yield_10y_origin_timestamp"] == pd.Timestamp(
        "2024-02-15 19:00:00+00:00"
    )
    assert (
        row["yield_2y_timing_status"]
        == MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED
    )
    assert (
        row["yield_10y_timing_status"]
        == MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED
    )
    assert row["market_origin_timestamp"] == pd.Timestamp(
        "2024-02-15 19:00:00+00:00"
    )
    assert row["yield_2y"] == pytest.approx(4.2)
    assert row["yield_10y"] == pytest.approx(5.3)


def test_missing_signal_timing_status_cannot_produce_exact_alignment() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    market = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-02-29")],
            "market_timestamp": [pd.Timestamp("2024-02-29 21:00:00+00:00")],
            "market_timestamp_provenance": [MARKET_TIMESTAMP_PROVENANCE_ACTUAL],
            "market_timestamp_status": [MARKET_TIMESTAMP_STATUS_EXACT],
            "yield_2y": [1.2],
        }
    )

    row = build_market_linkage_panel(signal, market, horizons=(3,)).iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_CONSERVATIVE_PROXY
    assert pd.isna(row["market_origin_timestamp"])
    assert "exact_information_timestamp" not in row["market_timing_status"]


def test_date_only_market_observation_uses_next_observation_proxy() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-13", "2024-02-14", "2024-05-14"]),
            "yield_2y": [1.0, 1.2, 1.6],
        }
    )

    row = build_market_linkage_panel(signal, market, horizons=(3,)).iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_NEXT_OBSERVATION_PROXY
    assert pd.isna(row["market_origin_timestamp"])
    assert row["market_origin_observation_date"] == pd.Timestamp("2024-02-14")
    assert "exact_information_timestamp" not in row["market_timing_status"]
    assert "next_observation_date_proxy" in row["market_timing_status"]


def test_failed_next_observation_proxy_lookup_is_explicitly_unavailable() -> None:
    publication = pd.Timestamp("2024-02-13 17:00:00+00:00")
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "reference_month": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [publication],
            "information_timestamp_provenance": [
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ],
            "timing_status": ["release_aligned"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    date_only_market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-12", "2024-02-13"]),
            "yield_2y": [4.0, 4.1],
            "yield_10y": [5.0, 5.1],
        }
    )

    row = build_market_linkage_panel(signal, date_only_market, horizons=(1,)).iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_UNAVAILABLE
    assert row["market_timing_status"] == MARKET_TIMING_UNAVAILABLE
    assert row["market_availability_status"] == MARKET_AVAILABILITY_UNAVAILABLE
    assert row["market_available_series_count"] == 0
    assert pd.isna(row["market_origin_observation_date"])
    assert pd.isna(row["yield_2y"])
    assert pd.isna(row["yield_10y"])
    assert row["yield_2y_origin_basis"] == MARKET_ORIGIN_UNAVAILABLE
    assert row["yield_10y_origin_basis"] == MARKET_ORIGIN_UNAVAILABLE


def test_missing_release_metadata_uses_labelled_month_end_t_plus_1_proxy() -> None:
    signal = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")],
            "information_timestamp": [pd.Timestamp("2024-01-31")],
            "timing_status": ["reference_month_only"],
            "historical_regime": ["neutral"],
            "historical_short_term_pressure": ["mixed"],
            "epsilon": [1.0],
            "tinf_4m": [1.0],
            "tinf_8m": [1.0],
            "tinf_12m": [1.0],
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-05-31"]),
            "yield_2y": [1.0, 1.2, 1.6],
        }
    )

    tables = build_market_linkage_tables(signal, market, horizons=(3,))
    row = tables.panel.iloc[0]

    assert row["market_origin_basis"] == MARKET_ORIGIN_CONSERVATIVE_PROXY
    assert pd.isna(row["market_origin_target_timestamp"])
    assert row["market_origin_target_observation_date"] == pd.Timestamp("2024-02-29")
    assert pd.isna(row["yield_2y_origin_timestamp"])
    assert row["yield_2y_origin_observation_date"] == pd.Timestamp("2024-02-29")
    assert "exact_information_timestamp" not in row["market_timing_status"]
    assert "latest_revised_non_vintage" in row["market_timing_status"]
    assert tables.timing_summary.iloc[0]["market_origin_basis"] == (
        MARKET_ORIGIN_CONSERVATIVE_PROXY
    )


def test_future_market_values_do_not_alter_tinf_or_regime_labels() -> None:
    signal = _signal_frame(periods=70)
    dates = signal["date"]
    market = pd.DataFrame({"date": dates, "yield_2y": np.linspace(1.0, 3.0, len(dates))})
    market_perturbed = market.copy()
    market_perturbed.loc[market_perturbed.index > 40, "yield_2y"] += 100.0

    panel = build_market_linkage_panel(signal, market, horizons=(3,))
    changed = build_market_linkage_panel(signal, market_perturbed, horizons=(3,))

    label_cols = [
        "epsilon",
        "tinf_4m",
        "tinf_8m",
        "tinf_12m",
        "historical_regime",
        "historical_short_term_pressure",
    ]
    for column in label_cols:
        assert panel[column].equals(changed[column])


def test_market_linkage_masks_imputed_signal_rows_without_compressing_horizons() -> None:
    signal = _signal_frame(periods=60)
    contaminated_pos = 40
    signal.loc[contaminated_pos, "signal_uses_imputed_input"] = True
    signal.loc[contaminated_pos, "signal_observed_only_eligible"] = False
    market = pd.DataFrame(
        {
            "date": signal["date"],
            "yield_2y": np.arange(len(signal), dtype=float),
        }
    )

    panel = build_market_linkage_panel(signal, market, horizons=(2,))

    assert pd.isna(panel.loc[contaminated_pos, "tinf_4m"])
    assert pd.isna(panel.loc[contaminated_pos, "historical_regime"])
    assert pd.isna(panel.loc[contaminated_pos, "historical_short_term_pressure"])
    assert panel.loc[contaminated_pos - 1, "yield_2y_fwd_2m"] == market.loc[
        contaminated_pos + 2,
        "yield_2y",
    ]


def test_market_summary_excludes_rows_without_full_forward_market_data() -> None:
    signal = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=5, freq="ME"),
            "epsilon": [0.0, 0.1, 0.2, 0.3, 0.4],
            "tinf_4m": [0.0, 0.1, 0.2, 0.3, 0.4],
            "tinf_8m": [0.0, 0.1, 0.2, 0.3, 0.4],
            "tinf_12m": [0.0, 0.1, 0.2, 0.3, 0.4],
            "historical_regime": ["neutral"] * 5,
            "historical_short_term_pressure": ["mixed"] * 5,
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=5, freq="ME"),
            "yield_2y": [1.0, 1.2, 1.5, 1.9, 2.4],
        }
    )

    panel = build_market_linkage_panel(signal, market, horizons=(2,))
    summary = forward_market_change_summary_by_regime(panel, horizons=(2,))

    row = summary.loc[summary["market_variable"] == "yield_2y"].iloc[0]
    assert row["historical_regime"] == "neutral"
    assert row["count"] == 2
    assert row["avg_change_bp"] == pytest.approx(((1.9 - 1.2) + (2.4 - 1.5)) * 100 / 2)
    assert row["increase_hit_rate"] == pytest.approx(1.0)
    assert row["decrease_hit_rate"] == pytest.approx(0.0)
    assert row["weak_evidence"]


def test_signal_market_correlations_use_only_complete_forward_changes() -> None:
    panel = pd.DataFrame(
        {
            "epsilon": [1.0, 2.0, 3.0, 4.0, 5.0],
            "tinf_4m": [1.0, 2.0, 3.0, 4.0, 5.0],
            "yield_2y": [1.0, 2.0, 4.0, 7.0, 11.0],
        }
    )
    panel = add_forward_market_changes(panel, horizons=(1,))

    correlations = market_signal_correlations(panel, signal_columns=("epsilon",), horizons=(1,))

    row = correlations.iloc[0]
    assert row["signal_variable"] == "epsilon"
    assert row["market_variable"] == "yield_2y"
    assert row["count"] == 4
    assert row["correlation"] == pytest.approx(1.0)


def test_market_linkage_tables_include_required_table_outputs() -> None:
    signal = _signal_frame(periods=70)
    market = pd.DataFrame(
        {
            "date": signal["date"],
            "yield_2y": np.linspace(1.0, 3.0, len(signal)),
            "yield_10y": np.linspace(2.0, 4.0, len(signal)),
            "breakeven_5y": np.linspace(2.1, 2.6, len(signal)),
            "breakeven_10y": np.linspace(2.0, 2.5, len(signal)),
        }
    )

    tables = build_market_linkage_tables(signal, market, horizons=(3,))

    assert not tables.availability.empty
    assert not tables.current_snapshot.empty
    assert not tables.summary_by_regime.empty
    assert not tables.summary_by_pressure.empty
    assert not tables.summary_by_regime_and_pressure.empty
    assert not tables.regime_pressure_rankings.empty
    assert not tables.channel_summary_by_regime.empty
    assert not tables.correlations.empty


def test_regime_pressure_rankings_sort_by_average_change_and_include_evidence() -> None:
    summary = pd.DataFrame(
        {
            "historical_regime": ["neutral", "elevated", "cooling"],
            "historical_short_term_pressure": ["mixed", "firming", "cooling"],
            "horizon_months": [12, 12, 12],
            "market_variable": ["yield_2y", "yield_2y", "yield_2y"],
            "count": [45, 25, 60],
            "avg_change_bp": [5.0, 40.0, -10.0],
            "median_change_bp": [4.0, 35.0, -8.0],
            "p25_change_bp": [1.0, 20.0, -20.0],
            "p75_change_bp": [8.0, 50.0, -2.0],
            "pct_positive_change": [0.6, 0.9, 0.3],
            "increase_hit_rate": [0.6, 0.9, 0.3],
            "decrease_hit_rate": [0.4, 0.1, 0.7],
            "weak_evidence": [False, True, False],
            "evidence_note": ["", "Fewer than 30 complete observations; interpret cautiously.", ""],
        }
    )

    rankings = rank_regime_pressure_market_changes(summary, horizons=(12,))

    assert rankings["avg_change_bp"].tolist() == [40.0, 5.0, -10.0]
    assert rankings["highest_change_rank"].tolist() == [1, 2, 3]
    assert rankings["lowest_change_rank"].tolist() == [3, 2, 1]
    required_cols = {
        "avg_change_bp",
        "median_change_bp",
        "count",
        "increase_hit_rate",
        "decrease_hit_rate",
        "weak_evidence",
    }
    assert required_cols <= set(rankings.columns)
    assert rankings.loc[rankings["count"] == 25, "weak_evidence"].iloc[0]


def test_low_count_rows_below_30_are_weak_evidence_but_30_is_not() -> None:
    dates = pd.date_range("2020-01-31", periods=35, freq="ME")
    signal = pd.DataFrame(
        {
            "date": dates,
            "epsilon": np.linspace(0.0, 1.0, len(dates)),
            "tinf_4m": np.linspace(0.0, 1.0, len(dates)),
            "tinf_8m": np.linspace(0.0, 1.0, len(dates)),
            "tinf_12m": np.linspace(0.0, 1.0, len(dates)),
            "historical_regime": ["neutral"] * len(dates),
            "historical_short_term_pressure": ["mixed"] * len(dates),
        }
    )
    market = pd.DataFrame(
        {
            "date": dates,
            "yield_2y": np.linspace(1.0, 2.0, len(dates)),
        }
    )

    weak_panel = build_market_linkage_panel(signal.iloc[:32], market.iloc[:32], horizons=(2,))
    weak_summary = forward_market_change_summary_by_regime(weak_panel, horizons=(2,))
    sufficient_panel = build_market_linkage_panel(signal, market, horizons=(4,))
    sufficient_summary = forward_market_change_summary_by_regime(
        sufficient_panel,
        horizons=(4,),
    )

    assert weak_summary.iloc[0]["count"] == 29
    assert weak_summary.iloc[0]["weak_evidence"]
    assert "Fewer than 30" in weak_summary.iloc[0]["evidence_note"]
    assert sufficient_summary.iloc[0]["count"] == 30
    assert not sufficient_summary.iloc[0]["weak_evidence"]
    assert sufficient_summary.iloc[0]["evidence_note"] == ""


def test_channel_summaries_use_only_approved_phase_4a_channel_mapping() -> None:
    assert MARKET_CHANNELS == {
        "nominal_rates": ("yield_2y", "yield_10y"),
        "breakevens": ("breakeven_5y", "breakeven_10y"),
        "real_yields": ("real_yield_5y", "real_yield_10y"),
    }

    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    signal = pd.DataFrame(
        {
            "date": dates,
            "epsilon": np.linspace(0.0, 1.0, len(dates)),
            "tinf_4m": np.linspace(0.0, 1.0, len(dates)),
            "tinf_8m": np.linspace(0.0, 1.0, len(dates)),
            "tinf_12m": np.linspace(0.0, 1.0, len(dates)),
            "historical_regime": ["neutral"] * len(dates),
            "historical_short_term_pressure": ["mixed"] * len(dates),
        }
    )
    market = pd.DataFrame(
        {
            "date": dates,
            "yield_2y": np.linspace(1.0, 2.0, len(dates)),
            "yield_10y": np.linspace(2.0, 3.0, len(dates)),
            "breakeven_5y": np.linspace(2.0, 2.5, len(dates)),
            "breakeven_10y": np.linspace(2.2, 2.7, len(dates)),
            "real_yield_5y": np.linspace(0.0, 1.0, len(dates)),
            "real_yield_10y": np.linspace(0.2, 1.2, len(dates)),
            "spy": np.linspace(100.0, 200.0, len(dates)),
        }
    )

    panel = build_market_linkage_panel(signal, market, horizons=(3,))
    summary = channel_regime_summary(panel, horizons=(3,))

    assert set(summary["market_channel"]) == {"nominal_rates", "breakevens", "real_yields"}
    assert "spy" not in set(summary["market_channel"])
    nominal = summary.loc[summary["market_channel"] == "nominal_rates"].iloc[0]
    assert nominal["avg_change_bp"] == pytest.approx(
        (
            panel["yield_2y_change_3m_bp"].dropna()
            + panel["yield_10y_change_3m_bp"].dropna()
        ).mean()
        / 2
    )


def test_channel_summaries_exclude_rows_without_both_channel_forward_changes() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    panel = pd.DataFrame(
        {
            "date": dates,
            "historical_regime": ["neutral"] * 6,
            "yield_2y": [1.0, 1.1, 1.3, 1.6, 2.0, 2.5],
            "yield_10y": [2.0, 2.2, None, 2.7, 3.1, 3.6],
        }
    )
    panel = add_forward_market_changes(panel, horizons=(2,))

    summary = channel_regime_summary(panel, horizons=(2,))

    row = summary.loc[summary["market_channel"] == "nominal_rates"].iloc[0]
    assert row["count"] == 2

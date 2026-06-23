from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.features import add_transitory_inflation_features
from transitory_inflation.market_linkage import (
    MARKET_CHANNELS,
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
    assert row["count"] == 3
    assert row["avg_change_bp"] == pytest.approx(((1.5 - 1.0) + (1.9 - 1.2) + (2.4 - 1.5)) * 100 / 3)
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
    dates = pd.date_range("2020-01-31", periods=34, freq="ME")
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

    weak_panel = build_market_linkage_panel(signal.iloc[:31], market.iloc[:31], horizons=(2,))
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

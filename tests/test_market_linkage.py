from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.features import add_transitory_inflation_features
from transitory_inflation.market_linkage import (
    add_forward_market_changes,
    build_market_linkage_panel,
    build_market_linkage_tables,
    forward_market_change_summary_by_regime,
    market_signal_correlations,
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
            "breakeven_10y": np.linspace(2.0, 2.5, len(signal)),
        }
    )

    tables = build_market_linkage_tables(signal, market, horizons=(3,))

    assert not tables.availability.empty
    assert not tables.current_snapshot.empty
    assert not tables.summary_by_regime.empty
    assert not tables.summary_by_pressure.empty
    assert not tables.summary_by_regime_and_pressure.empty
    assert not tables.correlations.empty

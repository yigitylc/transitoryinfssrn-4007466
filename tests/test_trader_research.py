from __future__ import annotations

import numpy as np
import pandas as pd

from transitory_inflation.features import add_transitory_inflation_features
from transitory_inflation.market_data import MARKET_VALUE_COLUMNS
from transitory_inflation.market_linkage import build_market_linkage_tables
from transitory_inflation.trader_research import (
    available_regimes,
    build_trader_research_view,
    conditional_forward_distribution,
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


def test_bucket_uses_latest_live_safe_walk_forward_label() -> None:
    signal = _signal_frame(120)
    bucket = latest_walk_forward_bucket(signal)

    labeled = add_short_term_pressure_labels(add_walk_forward_regime_labels(signal))
    valid = labeled["historical_regime"].notna() & labeled["historical_short_term_pressure"].notna()
    last = labeled.loc[valid].sort_values("date").iloc[-1]

    assert bucket.available
    assert bucket.regime == str(last["historical_regime"])
    assert bucket.pressure == str(last["historical_short_term_pressure"])
    assert bucket.as_of == pd.Timestamp(last["date"])
    assert bucket.regime_count > 0
    assert 0 < bucket.regime_pressure_count <= bucket.regime_count


def test_bucket_unavailable_without_enough_prior_history() -> None:
    bucket = latest_walk_forward_bucket(_signal_frame(20))

    assert not bucket.available
    assert bucket.regime is None
    assert "history" in (bucket.reason or "").lower()


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

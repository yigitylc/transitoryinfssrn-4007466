from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from transitory_inflation.data import make_demo_data
from transitory_inflation.features import add_transitory_inflation_features
from transitory_inflation.models import decay_curve, decay_summaries_for_windows
from transitory_inflation.plots import (
    apply_macro_theme,
    cpi_vs_baseline_figure,
    decay_curve_figure,
    forward_change_by_regime_channel_figure,
    forward_change_range_figure,
    heatmap_figure,
    hit_rate_bar_figure,
    rolling_rho_figure,
    threshold_sensitivity_figure,
    tinf_term_structure_figure,
)


def _demo_features() -> pd.DataFrame:
    return add_transitory_inflation_features(
        make_demo_data(), baseline_method="rolling_36_shifted"
    )


def _demo_distribution() -> pd.DataFrame:
    """Minimal trader-research distribution frame (no network/market data)."""

    return pd.DataFrame(
        {
            "market_variable": ["DGS2", "DGS10"],
            "count": [40, 35],
            "median_change_bp": [12.0, -8.0],
            "p25_change_bp": [-5.0, -20.0],
            "p75_change_bp": [30.0, 5.0],
            "avg_change_bp": [11.0, -7.0],
            "increase_hit_rate": [0.62, 0.40],
            "decrease_hit_rate": [0.38, 0.60],
            "weak_evidence": [False, False],
        }
    )


# (column, label, color) rate specs for the validation hit-rate / sensitivity charts.
_RATE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("positive_shock_resolution_rate", "Resolved", "#2471a3"),
    ("baseline_normalization_hit_rate", "Converged", "#7f8c8d"),
)


def _demo_outcome_summary() -> pd.DataFrame:
    """Minimal validation-summary frame (regime buckets x outcome rates)."""

    return pd.DataFrame(
        {
            "historical_regime": ["elevated rising", "disinflationary"],
            "horizon_months": [12, 12],
            "count": [40, 35],
            "positive_shock_resolution_rate": [0.30, 0.80],
            "baseline_normalization_hit_rate": [0.50, 0.60],
        }
    )


def _demo_sensitivity() -> pd.DataFrame:
    """Minimal threshold-sensitivity frame (one row per threshold)."""

    return pd.DataFrame(
        {
            "threshold_pp": [0.25, 0.50, 0.75, 1.00],
            "horizon_months": [12, 12, 12, 12],
            "count": [50, 45, 40, 35],
            "positive_shock_resolution_rate": [0.40, 0.50, 0.60, 0.70],
            "baseline_normalization_hit_rate": [0.55, 0.50, 0.45, 0.40],
        }
    )


def _demo_transition() -> pd.DataFrame:
    """Minimal row-normalized regime-transition matrix."""

    return pd.DataFrame(
        [[0.60, 0.40], [0.30, 0.70]],
        index=["elevated rising", "disinflationary"],
        columns=["elevated rising", "disinflationary"],
    )


def _demo_channel_summary() -> pd.DataFrame:
    """Minimal channel forward-change frame (two channels x two regimes)."""

    return pd.DataFrame(
        {
            "horizon_months": [12, 12, 12, 12],
            "market_channel": ["nominal_rates", "real_yields", "nominal_rates", "real_yields"],
            "historical_regime": [
                "elevated rising",
                "elevated rising",
                "disinflationary",
                "disinflationary",
            ],
            "avg_change_bp": [15.0, 8.0, -10.0, -4.0],
            "median_change_bp": [12.0, 7.0, -9.0, -3.0],
            "count": [40, 40, 35, 35],
        }
    )


def test_apply_macro_theme_sets_shared_layout() -> None:
    fig = apply_macro_theme(go.Figure(), title="Title", yaxis_title="y")
    assert fig.layout.hovermode == "x unified"
    assert fig.layout.template is not None
    assert fig.layout.title.text == "Title"


def test_cpi_vs_baseline_figure_returns_figure() -> None:
    fig = cpi_vs_baseline_figure(_demo_features())
    assert isinstance(fig, go.Figure)
    # Epsilon shading adds 4 helper traces (2 baseline anchors + hot/cold fills)
    # on top of the visible Baseline and CPI lines, plus a current-point marker.
    assert len(fig.data) == 7
    names = {trace.name for trace in fig.data}
    assert {"CPI YoY", "Baseline", "Current"} <= names
    assert fig.layout.hovermode == "x unified"


def test_tinf_term_structure_figure_returns_figure() -> None:
    fig = tinf_term_structure_figure(_demo_features())
    assert isinstance(fig, go.Figure)
    # One trace per TINF horizon (4M/8M/12M) plus a current-value marker trace;
    # the above/below-zero zone tints are layout shapes, not data traces.
    assert len(fig.data) == 4
    names = {trace.name for trace in fig.data}
    assert {"TINF 4M", "TINF 8M", "TINF 12M", "Current"} <= names


def test_forward_change_range_figure_returns_figure() -> None:
    fig = forward_change_range_figure(_demo_distribution())
    assert isinstance(fig, go.Figure)
    # Single range/dot trace (median markers with p25-p75 error bars).
    assert len(fig.data) == 1
    assert fig.layout.hovermode == "closest"


def test_forward_change_range_figure_empty_branch() -> None:
    fig = forward_change_range_figure(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert "no data" in (fig.layout.title.text or "").lower()


def test_rolling_rho_figure_with_data() -> None:
    rho_df, _ = decay_summaries_for_windows(
        _demo_features(), windows=(24, 36), value_col="tinf_4m"
    )
    fig = rolling_rho_figure(rho_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_rolling_rho_figure_empty_branch() -> None:
    fig = rolling_rho_figure(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert "no data" in (fig.layout.title.text or "").lower()


def test_decay_curve_figure_returns_figure() -> None:
    curve = decay_curve(rho_T=1.10, mu=0.92, months=24)
    fig = decay_curve_figure(curve)
    assert isinstance(fig, go.Figure)
    # Decay % and remaining % traces.
    assert len(fig.data) == 2


def test_hit_rate_bar_figure_returns_figure() -> None:
    fig = hit_rate_bar_figure(_demo_outcome_summary(), "historical_regime", _RATE_SPECS)
    assert isinstance(fig, go.Figure)
    # One grouped bar series per (present) rate spec.
    assert len(fig.data) == 2
    assert fig.layout.barmode == "group"
    assert {"Resolved", "Converged"} <= {trace.name for trace in fig.data}


def test_hit_rate_bar_figure_empty_branch() -> None:
    fig = hit_rate_bar_figure(pd.DataFrame(), "historical_regime", _RATE_SPECS)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert "no data" in (fig.layout.title.text or "").lower()


def test_threshold_sensitivity_figure_returns_figure() -> None:
    fig = threshold_sensitivity_figure(_demo_sensitivity(), _RATE_SPECS)
    assert isinstance(fig, go.Figure)
    # One line per (present) rate spec.
    assert len(fig.data) == 2
    assert "Resolved" in {trace.name for trace in fig.data}


def test_threshold_sensitivity_figure_empty_branch() -> None:
    fig = threshold_sensitivity_figure(pd.DataFrame(), _RATE_SPECS)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert "no data" in (fig.layout.title.text or "").lower()


def test_heatmap_figure_returns_figure() -> None:
    fig = heatmap_figure(_demo_transition(), title="Transitions", value_fmt=".0%")
    assert isinstance(fig, go.Figure)
    # A single heatmap trace.
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_heatmap_figure_empty_branch() -> None:
    fig = heatmap_figure(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert "no data" in (fig.layout.title.text or "").lower()


def test_forward_change_by_regime_channel_figure_returns_figure() -> None:
    fig = forward_change_by_regime_channel_figure(
        _demo_channel_summary(),
        channel_labels={"nominal_rates": "Nominal rates", "real_yields": "Real yields"},
    )
    assert isinstance(fig, go.Figure)
    # One grouped bar series per channel.
    assert len(fig.data) == 2
    assert fig.layout.barmode == "group"
    assert {"Nominal rates", "Real yields"} <= {trace.name for trace in fig.data}


def test_forward_change_by_regime_channel_figure_empty_branch() -> None:
    fig = forward_change_by_regime_channel_figure(pd.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert "no data" in (fig.layout.title.text or "").lower()

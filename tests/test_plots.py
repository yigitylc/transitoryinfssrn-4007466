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
    forward_change_range_figure,
    rolling_rho_figure,
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

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from transitory_inflation.data import make_demo_data
from transitory_inflation.features import add_transitory_inflation_features
from transitory_inflation.models import decay_curve, decay_summaries_for_windows
from transitory_inflation.plots import (
    cpi_vs_baseline_figure,
    decay_curve_figure,
    rolling_rho_figure,
    tinf_term_structure_figure,
)


def _demo_features() -> pd.DataFrame:
    return add_transitory_inflation_features(
        make_demo_data(), baseline_method="rolling_36_shifted"
    )


def test_cpi_vs_baseline_figure_returns_figure() -> None:
    fig = cpi_vs_baseline_figure(_demo_features())
    assert isinstance(fig, go.Figure)
    # CPI line plus baseline line.
    assert len(fig.data) == 2


def test_tinf_term_structure_figure_returns_figure() -> None:
    fig = tinf_term_structure_figure(_demo_features())
    assert isinstance(fig, go.Figure)
    # One trace per TINF horizon (4M/8M/12M).
    assert len(fig.data) == 3


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

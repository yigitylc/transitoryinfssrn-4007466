from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def cpi_vs_baseline_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["inflation_yoy"], name="CPI YoY"))
    if "baseline" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["baseline"], name="Baseline"))
    fig.add_hline(y=2.0, line_dash="dot", annotation_text="2% reference")
    fig.update_layout(title="CPI YoY vs Mean-Reversion Baseline", yaxis_title="Percent / percentage points")
    return fig


def tinf_term_structure_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, label in [("tinf_4m", "TINF 4M"), ("tinf_8m", "TINF 8M"), ("tinf_12m", "TINF 12M")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=label))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(title="Transitory Inflation Horizon Comparison", yaxis_title="Deviation, percentage points")
    return fig


def rolling_rho_figure(rho_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if rho_df.empty:
        fig.update_layout(title="Rolling AR(1) Persistence — no data")
        return fig
    for window, sub in rho_df.groupby("window"):
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["rho"], name=f"{window}M window"))
    fig.add_hline(y=1.0, line_dash="dot", annotation_text="rho = 1")
    fig.add_hline(y=0.0, line_dash="dot")
    fig.update_layout(title="Rolling AR(1) Persistence of TINF 4M", yaxis_title="rho")
    return fig


def decay_curve_figure(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["month"], y=curve["decay_pct"], name="Decay %"))
    fig.add_trace(go.Scatter(x=curve["month"], y=curve["remaining_pct"], name="Remaining %"))
    fig.add_hline(y=95, line_dash="dot", annotation_text="95% convergence threshold")
    fig.update_layout(title="Paper-Style Transitory Inflation Decay Curve", xaxis_title="Months ahead", yaxis_title="Percent")
    return fig

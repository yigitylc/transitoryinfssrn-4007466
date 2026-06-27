from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Hot/cold semantic palette (presentation only — never changes any value):
#   hot  = above baseline / inflationary   -> red
#   cold = below baseline / disinflationary -> blue
#   neutral                                  -> gray
# This is a *regime* palette, not a good/bad scheme: above-baseline inflation is
# a state to read, not a failure to flag.
HOT = "#c0392b"
COLD = "#2471a3"
NEUTRAL = "#7f8c8d"
INK = "#1a1a1a"
HOT_FILL = "rgba(192, 57, 43, 0.18)"
COLD_FILL = "rgba(36, 113, 163, 0.18)"
HOT_TINT = "rgba(192, 57, 43, 0.06)"
COLD_TINT = "rgba(36, 113, 163, 0.06)"

_GRID = "rgba(0,0,0,0.07)"
_BASE_FONT = dict(family="Inter, Segoe UI, system-ui, sans-serif", size=13)


def apply_macro_theme(
    fig: go.Figure,
    *,
    title: str | None = None,
    yaxis_title: str | None = None,
    xaxis_title: str | None = None,
    hovermode: str | bool = "x unified",
) -> go.Figure:
    """Apply the shared macro chart theme so every figure speaks one visual
    language: consistent font, light gridlines, unified hover, tight margins, and
    a horizontal legend on top.

    Presentation only — this never touches the plotted data, only how it looks.
    Route every figure in this module through it.
    """

    layout: dict[str, object] = dict(
        template="plotly_white",
        font=_BASE_FONT,
        hovermode=hovermode,
        margin=dict(l=64, r=24, t=56, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    if title is not None:
        layout["title"] = dict(text=title, x=0.0, xanchor="left")
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False)
    if yaxis_title is not None:
        fig.update_yaxes(title_text=yaxis_title)
    if xaxis_title is not None:
        fig.update_xaxes(title_text=xaxis_title)
    return fig


def cpi_vs_baseline_figure(df: pd.DataFrame) -> go.Figure:
    """CPI YoY against the mean-reversion baseline, with the epsilon gap shaded
    (hot where inflation runs above baseline, cold below) and the current point
    marked. The shaded vertical gap *is* epsilon — the quantity every TINF
    measure averages — so readers see it instead of inferring it."""

    fig = go.Figure()
    x = df["date"]
    cpi = df["inflation_yoy"]
    has_baseline = "baseline" in df.columns

    if has_baseline:
        baseline = df["baseline"]
        upper = np.maximum(cpi, baseline)  # caps the hot (above-baseline) band
        lower = np.minimum(cpi, baseline)  # floors the cold (below-baseline) band
        # Hot fill: anchor on baseline, then fill up to max(cpi, baseline).
        fig.add_trace(
            go.Scatter(x=x, y=baseline, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=upper, mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor=HOT_FILL, hoverinfo="skip", showlegend=False,
            )
        )
        # Cold fill: anchor on baseline again, then fill down to min(cpi, baseline).
        fig.add_trace(
            go.Scatter(x=x, y=baseline, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                fillcolor=COLD_FILL, hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=baseline, name="Baseline",
                line=dict(color=NEUTRAL, width=2, dash="dash"),
                hovertemplate="Baseline: %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x, y=cpi, name="CPI YoY", line=dict(color=INK, width=2.2),
            hovertemplate="CPI YoY: %{y:.2f}%<extra></extra>",
        )
    )

    # Current point, colored by the sign of epsilon (hot above / cold below).
    if has_baseline:
        valid = df.dropna(subset=["inflation_yoy", "baseline"])
        if not valid.empty:
            last = valid.iloc[-1]
            eps = float(last["inflation_yoy"]) - float(last["baseline"])
            color = HOT if eps > 0 else COLD if eps < 0 else NEUTRAL
            fig.add_trace(
                go.Scatter(
                    x=[last["date"]], y=[float(last["inflation_yoy"])], mode="markers",
                    name="Current", marker=dict(color=color, size=11, line=dict(color="white", width=1.5)),
                    hovertemplate=f"Current CPI YoY: %{{y:.2f}}%<br>epsilon: {eps:+.2f}pp<extra></extra>",
                )
            )

    fig.add_hline(
        y=2.0, line_dash="dot", line_color=NEUTRAL,
        annotation_text="2% reference", annotation_position="top left",
    )
    return apply_macro_theme(
        fig,
        title="CPI YoY vs Mean-Reversion Baseline (epsilon shaded)",
        yaxis_title="Percent / percentage points",
    )


def tinf_term_structure_figure(df: pd.DataFrame) -> go.Figure:
    """The 4M/8M/12M transitory-inflation horizons around zero. The zone above
    zero is tinted hot (inflationary pressure), below zero cold (disinflationary),
    and each line's current value is marked, so 'firming vs cooling' reads at a
    glance."""

    fig = go.Figure()
    x = df["date"]
    series = [("tinf_4m", "TINF 4M"), ("tinf_8m", "TINF 8M"), ("tinf_12m", "TINF 12M")]
    line_colors = {"tinf_4m": HOT, "tinf_8m": "#8e44ad", "tinf_12m": COLD}

    marker_x: list[object] = []
    marker_y: list[float] = []
    marker_colors: list[str] = []
    present: list[pd.Series] = []
    for col, label in series:
        if col in df.columns:
            present.append(df[col])
            fig.add_trace(
                go.Scatter(
                    x=x, y=df[col], name=label, line=dict(color=line_colors[col], width=2),
                    hovertemplate=f"{label}: %{{y:.2f}}pp<extra></extra>",
                )
            )
            valid = df[["date", col]].dropna()
            if not valid.empty:
                last = valid.iloc[-1]
                value = float(last[col])
                marker_x.append(last["date"])
                marker_y.append(value)
                marker_colors.append(HOT if value > 0 else COLD if value < 0 else NEUTRAL)

    if marker_x:
        fig.add_trace(
            go.Scatter(
                x=marker_x, y=marker_y, mode="markers", name="Current",
                marker=dict(color=marker_colors, size=10, line=dict(color="white", width=1.5)),
                hovertemplate="Current: %{y:.2f}pp<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)

    # Tint the inflationary (above-zero) and disinflationary (below-zero) zones.
    # Pin the y-range to the data so the tints cover exactly the plotted band.
    if present:
        all_vals = pd.concat(present).dropna()
        if not all_vals.empty:
            ymin = float(all_vals.min())
            ymax = float(all_vals.max())
            pad = max((ymax - ymin) * 0.08, 0.05)
            lo, hi = ymin - pad, ymax + pad
            if hi > 0:
                fig.add_hrect(y0=0, y1=hi, fillcolor=HOT_TINT, line_width=0, layer="below")
            if lo < 0:
                fig.add_hrect(y0=lo, y1=0, fillcolor=COLD_TINT, line_width=0, layer="below")
            fig.update_yaxes(range=[lo, hi])

    return apply_macro_theme(
        fig,
        title="Transitory Inflation Horizon Comparison",
        yaxis_title="Deviation, percentage points",
    )


def forward_change_range_figure(distribution: pd.DataFrame) -> go.Figure:
    """Horizontal range/dot plot of the forward rate-change distribution per
    approved FRED instrument: the p25-p75 range as a whisker, the median as a
    marker colored hot/cold by sign, and a zero reference line.

    Plots ``trader_research`` ``view.distribution`` only — descriptive, no new
    series, no forecast.
    """

    fig = go.Figure()
    required = {"market_variable", "median_change_bp", "p25_change_bp", "p75_change_bp"}
    empty_title = "Forward rate-change distribution — no data"
    if distribution is None or distribution.empty or not required.issubset(distribution.columns):
        return apply_macro_theme(
            fig, title=empty_title, xaxis_title="Forward change, basis points", hovermode="closest"
        )

    data = distribution.dropna(subset=["median_change_bp", "p25_change_bp", "p75_change_bp"])
    if data.empty:
        return apply_macro_theme(
            fig, title=empty_title, xaxis_title="Forward change, basis points", hovermode="closest"
        )

    median = data["median_change_bp"].astype(float)
    p25 = data["p25_change_bp"].astype(float)
    p75 = data["p75_change_bp"].astype(float)
    instrument = data["market_variable"].astype(str)
    colors = [HOT if m > 0 else COLD if m < 0 else NEUTRAL for m in median]

    n = len(data)
    avg = data["avg_change_bp"].astype(float) if "avg_change_bp" in data.columns else median
    count = data["count"].astype(float) if "count" in data.columns else pd.Series([np.nan] * n)
    inc = data["increase_hit_rate"].astype(float) if "increase_hit_rate" in data.columns else pd.Series([np.nan] * n)
    customdata = np.column_stack([p25.to_numpy(), p75.to_numpy(), avg.to_numpy(), count.to_numpy(), inc.to_numpy()])

    fig.add_trace(
        go.Scatter(
            x=median, y=instrument, mode="markers", name="Median (p25-p75)",
            marker=dict(color=colors, size=13, symbol="diamond", line=dict(color="white", width=1)),
            error_x=dict(
                type="data", symmetric=False,
                array=(p75 - median).to_numpy(), arrayminus=(median - p25).to_numpy(),
                color=NEUTRAL, thickness=1.5, width=6,
            ),
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>"
                "Median: %{x:.1f} bp<br>"
                "p25-p75: %{customdata[0]:.1f} to %{customdata[1]:.1f} bp<br>"
                "Average: %{customdata[2]:.1f} bp<br>"
                "n=%{customdata[3]:.0f} | increase rate %{customdata[4]:.0%}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color=NEUTRAL)
    return apply_macro_theme(
        fig,
        title="Forward rate-change distribution by instrument",
        xaxis_title="Forward change, basis points",
        yaxis_title="",
        hovermode="closest",
    )


def rolling_rho_figure(rho_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if rho_df.empty:
        return apply_macro_theme(fig, title="Rolling AR(1) Persistence — no data")
    for window, sub in rho_df.groupby("window"):
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["rho"], name=f"{window}M window"))
    fig.add_hline(y=1.0, line_dash="dot", line_color=NEUTRAL, annotation_text="rho = 1")
    fig.add_hline(y=0.0, line_dash="dot", line_color=NEUTRAL)
    return apply_macro_theme(fig, title="Rolling AR(1) Persistence of TINF 4M", yaxis_title="rho")


def decay_curve_figure(curve: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["month"], y=curve["decay_pct"], name="Decay %"))
    fig.add_trace(go.Scatter(x=curve["month"], y=curve["remaining_pct"], name="Remaining %"))
    fig.add_hline(y=95, line_dash="dot", line_color=NEUTRAL, annotation_text="95% convergence threshold")
    return apply_macro_theme(
        fig,
        title="Paper-Style Transitory Inflation Decay Curve",
        xaxis_title="Months ahead",
        yaxis_title="Percent",
        hovermode="x unified",
    )


# Batch-2 "table-heavy tab" figures (Validation + Market Linkage). Each plots
# values the app already computes (validation-summary rates, threshold-sensitivity
# rates, the regime-transition matrix, channel forward changes, and signal/market
# correlations) — presentation only, no recomputation of any number or series.
#
# Channel categorical colors mirror the TINF-horizon figure's three-way use of the
# palette (color identifies the channel; it is not a hot/cold sign read here).
_CHANNEL_COLORS: dict[str, str] = {
    "nominal_rates": HOT,
    "breakevens": "#8e44ad",
    "real_yields": COLD,
}


def hit_rate_bar_figure(
    summary: pd.DataFrame,
    group_col: str,
    rate_specs: Sequence[tuple[str, str, str]],
    *,
    title: str | None = None,
    group_order: Sequence[str] | None = None,
) -> go.Figure:
    """Grouped bar chart of historical outcome rates per bucket (regime or
    short-term pressure). Each ``rate_specs`` entry is ``(column, label, color)``
    and is drawn as one bar series of the 0-1 rates already in ``summary``, shown
    as percentages.

    Presentation only — plots existing validation-summary rates, no recomputation.
    """

    fig = go.Figure()
    empty_title = (title or "Historical outcome rates") + " — no data"
    usable = summary is not None and not summary.empty and group_col in summary.columns
    present_specs = [spec for spec in rate_specs if spec[0] in summary.columns] if usable else []
    if not present_specs:
        return apply_macro_theme(fig, title=empty_title, yaxis_title="Historical rate", hovermode="closest")

    data = summary.dropna(subset=[group_col]).copy()
    data[group_col] = data[group_col].astype(str)
    for column, label, color in present_specs:
        fig.add_trace(
            go.Bar(
                x=data[group_col],
                y=pd.to_numeric(data[column], errors="coerce"),
                name=label,
                marker_color=color,
                hovertemplate=f"%{{x}}<br>{label}: %{{y:.0%}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    if group_order is not None:
        present = set(data[group_col])
        order = [str(value) for value in group_order if str(value) in present]
        if order:
            fig.update_xaxes(categoryorder="array", categoryarray=order)
    return apply_macro_theme(
        fig,
        title=title or "Historical outcome rates by bucket",
        yaxis_title="Historical rate",
        hovermode="closest",
    )


def threshold_sensitivity_figure(
    sensitivity: pd.DataFrame,
    rate_specs: Sequence[tuple[str, str, str]],
    *,
    title: str | None = None,
) -> go.Figure:
    """Line chart of historical outcome rates across the fixed sensitivity
    thresholds (one line per ``rate_specs`` ``(column, label, color)`` entry).

    Presentation only — plots existing threshold-sensitivity rates.
    """

    fig = go.Figure()
    empty_title = (title or "Threshold sensitivity") + " — no data"
    usable = (
        sensitivity is not None and not sensitivity.empty and "threshold_pp" in sensitivity.columns
    )
    present_specs = [spec for spec in rate_specs if spec[0] in sensitivity.columns] if usable else []
    if not present_specs:
        return apply_macro_theme(
            fig, title=empty_title, xaxis_title="Outcome threshold (pp)", yaxis_title="Historical rate"
        )

    data = sensitivity.sort_values("threshold_pp")
    x = pd.to_numeric(data["threshold_pp"], errors="coerce")
    for column, label, color in present_specs:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=pd.to_numeric(data[column], errors="coerce"),
                name=label,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(color=color, size=7),
                hovertemplate=f"threshold %{{x:.2f}}pp<br>{label}: %{{y:.0%}}<extra></extra>",
            )
        )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    return apply_macro_theme(
        fig,
        title=title or "Outcome rates vs threshold",
        xaxis_title="Outcome threshold (pp)",
        yaxis_title="Historical rate",
        hovermode="x unified",
    )


def heatmap_figure(
    matrix: pd.DataFrame,
    *,
    title: str | None = None,
    colorscale: str = "RdBu",
    reversescale: bool = False,
    zmid: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    value_fmt: str = ".2f",
    colorbar_title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    hover_value_label: str = "value",
) -> go.Figure:
    """Generic labeled heatmap of a wide ``rows x columns -> z`` matrix. Backs both
    the regime-transition matrix and the signal-vs-market correlation grid.

    Presentation only — renders an already-computed matrix, never recomputes it.
    """

    fig = go.Figure()
    empty_title = (title or "Heatmap") + " — no data"
    if matrix is None or matrix.empty:
        return apply_macro_theme(fig, title=empty_title, hovermode="closest")

    fig.add_trace(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=[str(column) for column in matrix.columns],
            y=[str(index) for index in matrix.index],
            colorscale=colorscale,
            reversescale=reversescale,
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            texttemplate=f"%{{z:{value_fmt}}}",
            textfont=dict(size=11),
            colorbar=dict(title=dict(text=colorbar_title)) if colorbar_title else None,
            hovertemplate=(
                f"%{{y}} → %{{x}}<br>{hover_value_label}: %{{z:{value_fmt}}}<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(autorange="reversed")
    return apply_macro_theme(
        fig, title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title, hovermode="closest"
    )


def forward_change_by_regime_channel_figure(
    channel_summary: pd.DataFrame,
    *,
    value_col: str = "avg_change_bp",
    channel_labels: dict[str, str] | None = None,
    regime_order: Sequence[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Grouped bar chart of forward rate change (bp) by historical regime, one bar
    series per market channel (nominal / breakevens / real yields).

    Presentation only — plots existing channel-summary forward changes.
    """

    fig = go.Figure()
    empty_title = (title or "Forward change by regime per channel") + " — no data"
    required = {"market_channel", "historical_regime", value_col}
    usable = (
        channel_summary is not None
        and not channel_summary.empty
        and required.issubset(channel_summary.columns)
    )
    data = channel_summary.dropna(subset=["historical_regime"]).copy() if usable else pd.DataFrame()
    if data.empty:
        return apply_macro_theme(
            fig, title=empty_title, yaxis_title="Forward change, basis points", hovermode="closest"
        )

    data["historical_regime"] = data["historical_regime"].astype(str)
    labels = channel_labels or {}
    has_median = "median_change_bp" in data.columns
    has_count = "count" in data.columns
    # Stable channel order: nominal, breakevens, real yields, then any extras.
    channels = set(data["market_channel"])
    channel_keys = [key for key in _CHANNEL_COLORS if key in channels]
    channel_keys += [key for key in data["market_channel"].unique() if key not in channel_keys]

    for channel in channel_keys:
        sub = data.loc[data["market_channel"] == channel]
        if sub.empty:
            continue
        label = labels.get(channel, str(channel).replace("_", " ").title())
        median = sub["median_change_bp"] if has_median else sub[value_col]
        count = sub["count"] if has_count else pd.Series([np.nan] * len(sub), index=sub.index)
        customdata = np.column_stack(
            [
                pd.to_numeric(median, errors="coerce").to_numpy(),
                pd.to_numeric(count, errors="coerce").to_numpy(),
            ]
        )
        fig.add_trace(
            go.Bar(
                x=sub["historical_regime"],
                y=pd.to_numeric(sub[value_col], errors="coerce"),
                name=label,
                marker_color=_CHANNEL_COLORS.get(channel, NEUTRAL),
                customdata=customdata,
                hovertemplate=(
                    f"%{{x}} · {label}<br>"
                    "Avg: %{y:.1f} bp<br>"
                    "Median: %{customdata[0]:.1f} bp<br>"
                    "n=%{customdata[1]:.0f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(barmode="group")
    fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
    if regime_order is not None:
        order = [str(regime) for regime in regime_order if str(regime) in set(data["historical_regime"])]
        if order:
            fig.update_xaxes(categoryorder="array", categoryarray=order)
    return apply_macro_theme(
        fig,
        title=title or "Forward change by regime per channel",
        yaxis_title="Forward change, basis points",
        hovermode="closest",
    )

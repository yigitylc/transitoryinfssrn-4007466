from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow running from project root without installing as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from transitory_inflation import benchmarks as benchmark_mod
from transitory_inflation import data as macro_data
from transitory_inflation import market_data as market_data_mod
from transitory_inflation import market_linkage as market_linkage_mod
from transitory_inflation import plots as plots_mod
from transitory_inflation import report as report_mod
from transitory_inflation import robustness as robustness_mod
from transitory_inflation import trader_research as trader_research_mod
from transitory_inflation import validation as validation_mod
from transitory_inflation.config import DEFAULT_SAMPLE_MODE, SAMPLE_MODES
from transitory_inflation.diagnostics import ljung_box_table, stationarity_diagnostics
from transitory_inflation.features import (
    BASELINE_META,
    add_transitory_inflation_features,
    latest_signal_snapshot,
)
from transitory_inflation.models import (
    correlation_matrix,
    decay_curve,
    decay_summaries_for_windows,
    run_paper_style_regressions,
    summary_stats,
)

if (
    not hasattr(macro_data, "load_macro_data_for_mode_with_status")
    or not hasattr(macro_data, "latest_valid_observation_date")
    or not hasattr(macro_data, "date_label")
    or not hasattr(macro_data, "INFLATION_MEASURES")
):
    macro_data = importlib.reload(macro_data)
if not hasattr(benchmark_mod, "benchmark_comparison_tables") or not hasattr(
    benchmark_mod,
    "BENCHMARK_VALIDATION_SIGNATURE_GUARD",
):
    benchmark_mod = importlib.reload(benchmark_mod)
if not hasattr(market_data_mod, "load_market_data_for_mode_with_status") or not hasattr(
    market_data_mod,
    "MARKET_FRED_SERIES",
):
    market_data_mod = importlib.reload(market_data_mod)
if not hasattr(market_linkage_mod, "build_market_linkage_tables") or not hasattr(
    market_linkage_mod,
    "channel_regime_summary",
):
    market_linkage_mod = importlib.reload(market_linkage_mod)
_REQUIRED_PLOT_ATTRS = (
    "HOT",
    "COLD",
    "NEUTRAL",
    "cpi_vs_baseline_figure",
    "tinf_term_structure_figure",
    "hit_rate_bar_figure",
    "threshold_sensitivity_figure",
    "heatmap_figure",
    "forward_change_by_regime_channel_figure",
    "forward_change_range_figure",
    "rolling_rho_figure",
    "decay_curve_figure",
)
if not all(hasattr(plots_mod, attr) for attr in _REQUIRED_PLOT_ATTRS):
    plots_mod = importlib.reload(plots_mod)
if not hasattr(report_mod, "build_macro_research_report") or not hasattr(
    report_mod,
    "MacroResearchReport",
):
    report_mod = importlib.reload(report_mod)
if (
    not hasattr(robustness_mod, "robustness_tables")
    or not hasattr(robustness_mod, "inflation_measure_availability")
    or not hasattr(robustness_mod, "ROBUSTNESS_BENCHMARK_SIGNATURE_GUARD")
):
    robustness_mod = importlib.reload(robustness_mod)
if not hasattr(trader_research_mod, "build_trader_research_view") or not hasattr(
    trader_research_mod,
    "TraderResearchView",
):
    trader_research_mod = importlib.reload(trader_research_mod)
if not hasattr(validation_mod, "pressure_label") or not hasattr(
    validation_mod,
    "forward_outcome_summary_by_regime_and_pressure",
):
    validation_mod = importlib.reload(validation_mod)

st.set_page_config(page_title="Transitory Inflation Macro Research", layout="wide")
st.title("Transitory Inflation Macro Research")
st.caption(
    "Reference: SSRN 4007466 — Peron & Bonaparte, "
    "Transitory Inflation and Projection of Future Inflation"
)


def section_notes(answers: str, interpretation: str) -> None:
    """Per-section explainer: what the artifact answers, then how to read it."""

    st.markdown(f"**➜** {answers}")
    st.markdown(f"**↳** {interpretation}")


# Plain-language concept definitions (descriptive only; consistent with
# docs/01_RESEARCH_SPEC.md and the UI prose). Backs the reusable glossary.
GLOSSARY_ITEMS: tuple[tuple[str, str], ...] = (
    (
        "Baseline",
        "the mean-reversion anchor inflation is compared against (e.g. the prior 36-month "
        "rolling mean). Shifted baselines use only data through the prior month (live-safe); "
        "the full-sample baseline is flat and ex-post.",
    ),
    (
        "epsilon (ε)",
        "the raw deviation `inflation_yoy − baseline`, in percentage points. Positive = inflation "
        "above baseline; negative = below.",
    ),
    (
        "TINF 4M / 8M / 12M",
        "the n-month rolling average of epsilon. 4M is recent transitory pressure; 8M/12M are "
        "slower-moving. Units: percentage points.",
    ),
    (
        "Regime",
        "a label combining level (vs the sample's 25th/75th-percentile band) with direction (vs the "
        "prior month): elevated rising, elevated falling, neutral, or disinflationary.",
    ),
    (
        "Short-term pressure",
        "summarizes the 4M vs 8M vs 12M TINF ordering: firming (4M highest), cooling (4M lowest), "
        "or mixed.",
    ),
    (
        "weak_evidence",
        "a flag that a bucket has fewer than 30 complete observations; its rates are unstable and "
        "should be read cautiously.",
    ),
    (
        "live-safe vs ex-post",
        "live-safe = no full-sample lookahead (signal usable in real time); ex-post = uses "
        "information not available at the time (paper-style only). 'live-safe' means no full-sample "
        "lookahead, not a real-time data-vintage backtest.",
    ),
)


def render_glossary(*, expanded: bool = False) -> None:
    """Reusable Concepts / glossary expander (sidebar and inline)."""

    with st.expander("Concepts / glossary", expanded=expanded):
        for term, definition in GLOSSARY_ITEMS:
            st.markdown(f"**{term}** — {definition}")


def scope_caveats(one_liner: str, caveats: tuple[str, ...]) -> None:
    """Keep one visible caption; tuck the full caveat wall into an expander.

    All guardrail text is preserved — only relocated, never dropped.
    """

    st.caption(one_liner)
    with st.expander("Scope & caveats"):
        for caveat in caveats:
            st.markdown(f"- {caveat}")


# Hot/cold regime badge (matches plots.py: red = above baseline / inflationary,
# blue = below baseline / disinflationary, gray = neutral). A regime read, not a
# good/bad verdict.
_REGIME_BADGES = {
    "elevated rising": "🔴 Elevated · rising",
    "elevated falling": "🔴 Elevated · falling",
    "neutral": "⚪ Neutral",
    "disinflationary": "🔵 Disinflationary",
}


def regime_badge(regime: object) -> str:
    """Emoji-coded regime label for the executive read (no HTML dependency)."""

    return _REGIME_BADGES.get(str(regime), f"⚪ {regime}")


def signal_headline(snapshot: dict[str, object]) -> str:
    """One-line plain-English current read: epsilon vs baseline + pressure."""

    epsilon = float(snapshot.get("epsilon", 0.0))
    pressure = validation_mod.pressure_label(snapshot.get("term_structure", "mixed"))
    if abs(epsilon) < 0.05:
        gap = "at its mean-reversion baseline"
    else:
        side = "above" if epsilon > 0 else "below"
        gap = f"{abs(epsilon):.2f}pp {side} baseline"
    return f"Inflation is {gap} · short-term pressure {pressure}"


def signal_conclusion(snapshot: dict[str, object]) -> tuple[str, ...]:
    """Concise current-signal interpretation for the report tab."""

    regime = str(snapshot.get("regime", "neutral"))
    pressure = validation_mod.pressure_label(snapshot.get("term_structure", "mixed"))
    tinf = float(snapshot.get("tinf_4m", 0.0))
    tinf_8m = float(snapshot.get("tinf_8m", 0.0))
    tinf_12m = float(snapshot.get("tinf_12m", 0.0))
    epsilon = float(snapshot.get("epsilon", 0.0))
    percentile = float(snapshot.get("tinf_4m_percentile", 50.0))

    if regime == "elevated rising":
        state = "more consistent with persistent inflation pressure"
    elif regime == "elevated falling":
        state = "more consistent with transitory inflation normalizing, though pressure remains elevated"
    elif regime == "disinflationary":
        state = "more consistent with normalization below the selected baseline"
    else:
        state = "more consistent with neutral or mixed conditions"

    if abs(epsilon) < 0.05:
        baseline_read = "running close to the selected mean-reversion baseline"
    else:
        side = "above" if epsilon > 0 else "below"
        baseline_read = f"running {abs(epsilon):.2f}pp {side} the selected mean-reversion baseline"

    if percentile >= 75:
        distribution_read = "elevated relative to the historical distribution"
    elif percentile <= 25:
        distribution_read = "low relative to the historical distribution"
    else:
        distribution_read = "not extreme relative to the historical distribution"

    if pressure == "firming":
        horizon_read = (
            "the 4-month TINF reading is above the longer TINF horizons, so recent inflation "
            "deviations are firmer than the medium-term average"
        )
    elif pressure == "cooling":
        horizon_read = (
            "the 4-month TINF reading is below the longer TINF horizons, so recent inflation "
            "deviations are cooling versus the medium-term average"
        )
    else:
        horizon_read = (
            "the 4M, 8M, and 12M TINF readings are not aligned, so recent inflation pressure "
            "is mixed across horizons"
        )

    return (
        f"Current inflation is {baseline_read}, and the TINF 4M percentile is {percentile:.1f}%, "
        f"which is {distribution_read}.",
        f"TINF 4M is {tinf:+.2f}pp versus TINF 8M {tinf_8m:+.2f}pp and TINF 12M "
        f"{tinf_12m:+.2f}pp: {horizon_read}.",
        f"Under this baseline, the paper framework is {state}.",
        "This is a descriptive inflation-regime signal from the paper framework, not a portfolio "
        "instruction.",
    )


MODE_LABELS = {
    "live_dashboard": "Live dashboard (1982 → latest)",
    "paper_replication": "Paper window (1982-01 → 2021-07)",
    "max_history": "Max history (earliest FRED → latest)",
}

BASELINE_DESCRIPTION_OVERRIDES = {
    "full_sample": "Full-sample historical mean. Useful for ex-post paper-framework implementation.",
}

INFLATION_MEASURE_LABELS = {
    key: measure.label for key, measure in macro_data.INFLATION_MEASURES.items()
}

MARKET_LINKAGE_HORIZON_OPTIONS = {"3M": 3, "6M": 6, "12M": 12, "24M": 24, "36M": 36}
MARKET_CHANNEL_LABELS = {
    "nominal_rates": "Nominal rates",
    "breakevens": "Breakevens",
    "real_yields": "Real yields",
}
MARKET_CHANNEL_INTERPRETATION_COLUMNS = [
    "historical_regime",
    "historical_direction",
    "avg_change_bp",
    "median_change_bp",
    "count",
    "increase_hit_rate",
    "decrease_hit_rate",
    "weak_evidence",
    "evidence_note",
]
MARKET_RANKING_COLUMNS = [
    "market_variable",
    "historical_regime",
    "historical_short_term_pressure",
    "avg_change_bp",
    "median_change_bp",
    "count",
    "increase_hit_rate",
    "decrease_hit_rate",
    "weak_evidence",
]
# Validation outcome rates to chart, as (column, label, hot/cold color). These are
# existing forward-outcome rates: resolved (faded -> cold), baseline convergence
# (neutral), persisted (still elevated -> hot). Presentation only.
VALIDATION_RATE_SPECS = (
    ("positive_shock_resolution_rate", "Positive-shock resolved", plots_mod.COLD),
    ("baseline_normalization_hit_rate", "Baseline convergence", plots_mod.NEUTRAL),
    ("positive_shock_persistent_rate", "Positive-shock persisted", plots_mod.HOT),
)

with st.sidebar:
    st.header("Configuration")
    mode_names = list(MODE_LABELS)
    sample_mode = st.radio(
        "Sample mode",
        options=mode_names,
        index=mode_names.index(DEFAULT_SAMPLE_MODE),
        format_func=lambda name: MODE_LABELS[name],
    )
    st.caption(
        "**Live dashboard**: current macro signal on the latest available FRED data (default). "
        "**Paper window**: fixed 1982-01 to 2021-07 sample for the reference framework. "
        "**Max history**: earliest available FRED history, for robustness — the longer sample "
        "shifts percentiles and regime stats, so it is not necessarily the default current signal."
    )
    baseline_method = st.selectbox(
        "Mean-reversion baseline",
        options=list(BASELINE_META.keys()),
        index=list(BASELINE_META.keys()).index("rolling_36_shifted"),
    )
    st.divider()
    render_glossary()

meta = BASELINE_META[baseline_method]
mode_meta = SAMPLE_MODES[sample_mode]


@st.cache_data(show_spinner=True)
def get_data(sample_mode: str):
    return macro_data.load_macro_data_for_mode_with_status(sample_mode)


@st.cache_data(show_spinner=True)
def get_market_data(sample_mode: str):
    return market_data_mod.load_market_data_for_mode_with_status(sample_mode)


# Heavy table builders are wrapped in @st.cache_data so they recompute only when
# their inputs change, not on every Streamlit rerun. DataFrame/scalar/tuple args
# hash by value; the non-hashable status dataclasses are excluded from the hash
# via Streamlit's leading-underscore parameter convention (they are derived from
# the already-hashed sample_mode/data, so excluding them is safe).
@st.cache_data(show_spinner=False)
def get_benchmark_tables(featured: pd.DataFrame, horizon: int, threshold_pp: float):
    return benchmark_mod.benchmark_comparison_tables(
        featured, horizon=horizon, threshold_pp=threshold_pp
    )


@st.cache_data(show_spinner=False)
def get_validation_frame(
    featured: pd.DataFrame,
    forward_horizons: tuple[int, ...],
    label_horizons: tuple[int, ...],
    threshold_pp: float,
):
    return validation_mod.build_historical_validation_frame(
        featured,
        forward_horizons=forward_horizons,
        label_horizons=label_horizons,
        epsilon_threshold_pp=threshold_pp,
        fed_target_threshold_pp=threshold_pp,
    )


@st.cache_data(show_spinner=False)
def get_threshold_sensitivity(featured: pd.DataFrame, horizon: int):
    return validation_mod.threshold_sensitivity_summary(
        featured, horizon=horizon, thresholds=(0.25, 0.50, 0.75, 1.00)
    )


@st.cache_data(show_spinner=False)
def get_market_linkage_tables(featured: pd.DataFrame, market_monthly: pd.DataFrame):
    return market_linkage_mod.build_market_linkage_tables(featured, market_monthly)


@st.cache_data(show_spinner=True)
def get_robustness_tables(
    robustness_raw: dict[str, pd.DataFrame],
    baseline_methods: tuple[str, ...],
    inflation_measures: tuple[str, ...],
):
    scorecard, verdict, win_rates = robustness_mod.robustness_tables(
        robustness_raw,
        baseline_methods=baseline_methods,
        inflation_measures=inflation_measures,
    )
    availability = robustness_mod.inflation_measure_availability(
        robustness_raw,
        inflation_measures=inflation_measures,
    )
    return scorecard, verdict, win_rates, availability


@st.cache_data(show_spinner=True)
def get_macro_research_report(
    raw: pd.DataFrame,
    featured: pd.DataFrame,
    baseline_method: str,
    sample_mode: str,
    market_monthly: pd.DataFrame | None,
    _macro_status: object,
    _market_status: object,
):
    return report_mod.build_macro_research_report(
        raw,
        featured,
        baseline_method=baseline_method,
        sample_mode=sample_mode,
        macro_status=_macro_status,
        market_monthly=market_monthly,
        market_status=_market_status,
    )


load_result = get_data(sample_mode)
raw = load_result.data

raw_cache_end = pd.to_datetime(raw["date"].max()).date() if not raw.empty else "unknown"
latest_cpi_yoy = macro_data.latest_valid_observation_date(raw, "inflation_yoy")
if load_result.data_source_used == "fred_csv":
    st.warning(
        "Official FRED API was unavailable or not configured, so the dashboard is using "
        "public FRED CSV data."
    )
elif load_result.data_source_used == "cached_fred":
    st.warning(
        f"Live FRED fetch failed, so the dashboard is using cached data from "
        f"`{load_result.cache_file_used}` with CPI YoY through "
        f"{macro_data.date_label(latest_cpi_yoy)} "
        f"(raw cache rows through {raw_cache_end}). Refresh the cache when network access returns."
    )
elif load_result.data_source_used == "demo":
    st.error(
        "Emergency demo data is being shown because official FRED API, public FRED CSV, "
        "and local cached FRED data were all unavailable. Do not use these values for "
        "research conclusions."
    )

if raw.empty:
    st.error("No data rows inside the selected sample mode window.")
    st.stop()

span_start = pd.to_datetime(raw["date"].min()).date()
span_end = macro_data.date_label(
    macro_data.latest_valid_observation_date(raw, "inflation_yoy")
)
raw_span_end = pd.to_datetime(raw["date"].max()).date()
span_text = f"{span_start} -> {span_end}"
if str(raw_span_end) != span_end:
    span_text += f"; raw rows through {raw_span_end}"
baseline_description = BASELINE_DESCRIPTION_OVERRIDES.get(baseline_method, meta.description)
st.info(
    f"Sample: **{MODE_LABELS[sample_mode]}** ({span_text}) — {mode_meta.purpose}  \n"
    f"Baseline: **{baseline_method}** — {baseline_description}"
)
if meta.warning:
    st.warning(meta.warning)

df = add_transitory_inflation_features(raw, baseline_method=baseline_method)
status_snapshot = latest_signal_snapshot(df)
latest_signal_date = (
    macro_data.date_label(pd.to_datetime(status_snapshot["date"]))
    if status_snapshot.get("available")
    else "unavailable"
)

with st.sidebar:
    st.divider()
    st.markdown("**Current reading**")
    if status_snapshot.get("available"):
        st.markdown(regime_badge(status_snapshot["regime"]))
        st.caption(
            f"{signal_headline(status_snapshot)}  ·  as of {latest_signal_date}. "
            "See the Current Macro Signal tab for the full read."
        )
    else:
        st.caption("No complete signal yet for this mode/baseline.")
latest_cpi_observation = macro_data.latest_valid_observation_date(raw, "cpi_level")
imputation_applied = bool(
    "cpi_imputed" in raw.columns and raw["cpi_imputed"].fillna(False).astype(bool).any()
)
st.caption(
    "Data status: "
    f"data_source_used={load_result.data_source_used}; "
    f"live_fetch_status={load_result.live_fetch_status}; "
    f"cache_file_used={load_result.cache_file_used or 'n/a'}; "
    f"raw_data_end={raw_span_end}; "
    f"latest_cpi_observation_date={macro_data.date_label(latest_cpi_observation)}; "
    f"latest_valid_cpi_yoy_date={span_end}; "
    f"latest_valid_signal_date={latest_signal_date}; "
    f"cpi_imputation_applied={imputation_applied}."
)

if "cpi_imputed" in raw.columns and raw["cpi_imputed"].any():
    imputed_months = ", ".join(
        d.strftime("%Y-%m") for d in pd.to_datetime(raw.loc[raw["cpi_imputed"], "date"])
    )
    st.caption(
        f"Data status: CPI level log-linearly imputed for {imputed_months} "
        "(no CPI was published for these months). "
        "YoY and TINF values touching them are partly estimates."
    )

# Benchmarks (Phase 2) precede Market Linkage (Phase 4) to match the roadmap's
# "no linkage until Phase 2 confirms usefulness" narrative. The `with tab_*:`
# content blocks below are routed by handle name, so their source order can differ.
(
    tab_signal,
    tab_validation,
    tab_benchmarks,
    tab_market_linkage,
    tab_trader_research,
    tab_framework,
    tab_decay,
    tab_robustness,
    tab_report,
) = st.tabs(
    [
        "Current Macro Signal",
        "Historical Signal Validation",
        "Benchmark Comparison",
        "Market Linkage",
        "Trader Research",
        "Paper Framework",
        "Decay / Convergence",
        "Robustness",
        "Macro Research Report",
    ]
)

with tab_signal:
    snapshot = latest_signal_snapshot(df)
    if not snapshot.get("available"):
        st.warning(snapshot.get("reason", "No signal available."))
    else:
        epsilon = float(snapshot["epsilon"])
        percentile = float(snapshot["tinf_4m_percentile"])
        st.markdown(f"### {signal_headline(snapshot)}")
        st.markdown(
            f"**Regime:** {regime_badge(snapshot['regime'])}  |  "
            f"**Short-term pressure:** {validation_mod.pressure_label(snapshot['term_structure'])}  |  "
            f"**As of:** {pd.to_datetime(snapshot['date']).date()}"
        )

        col1, col2, col3, col4 = st.columns(4)
        # delta_color="off" keeps the arrows gray: these are regime reads (hot/cold),
        # not good/bad outcomes, so the default green-up/red-down scheme is suppressed.
        col1.metric(
            "CPI YoY", f"{snapshot['inflation_yoy']:.2f}%",
            delta=f"{epsilon:+.2f}pp vs baseline", delta_color="off",
        )
        col2.metric("Baseline", f"{snapshot['baseline']:.2f}%")
        col3.metric("TINF 4M", f"{snapshot['tinf_4m']:.2f} pp")
        col4.metric(
            "TINF 4M Percentile", f"{percentile:.1f}%",
            delta=f"{percentile - 50:+.0f} vs median", delta_color="off",
        )

        section_notes(
            "What the latest complete macro reading is under the selected sample mode and baseline: "
            "current CPI YoY inflation, the baseline it is compared against, the 4-month transitory "
            "component (TINF 4M = average of the last four monthly deviations, in percentage points), "
            "and where that reading sits in this sample's history (percentile). The regime label "
            "combines level (versus the sample's 25th/75th percentile band) with direction (versus the "
            "prior month); the short-term pressure label summarizes the TINF 4M vs 8M vs 12M ordering.",
            "TINF 4M above zero means inflation has been running above the baseline recently "
            "(inflationary pressure); below zero means below baseline (disinflationary). A percentile "
            "near 100 marks one of the most inflationary readings in this sample, near 0 one of the "
            "most disinflationary. 'Elevated rising' means pressure is still building; 'elevated "
            "falling' means elevated but easing. The date is the last month where every input is "
            "available. Percentile and regime depend on the sample mode: max_history includes pre-1982 "
            "regimes and will shift both.",
        )

    st.plotly_chart(plots_mod.cpi_vs_baseline_figure(df), width="stretch")
    section_notes(
        "How CPI YoY inflation has tracked the selected mean-reversion baseline over the whole loaded "
        "sample: when inflation crossed above or below it, how long those episodes lasted, and how "
        "today's gap compares with history. The dotted 2% line is a fixed policy reference, "
        "independent of the baseline choice.",
        "The vertical gap between the two lines is epsilon, the raw deviation in percentage points "
        "that every TINF measure averages. Persistent one-sided gaps are exactly the episodes TINF "
        "quantifies. Rolling and expanding baselines adapt slowly by design; that lag is what makes "
        "deviations measurable. Shifted baselines use only data through the prior month (live-safe), "
        "while the full-sample baseline is flat and ex-post only.",
    )

    st.plotly_chart(plots_mod.tinf_term_structure_figure(df), width="stretch")
    section_notes(
        "Whether the transitory component is building or fading across horizons: the same deviation "
        "series averaged over 4, 8, and 12 months, so recent pressure (4M) can be compared against "
        "the slower-moving 8M and 12M readings around the zero line.",
        "4M above 8M above 12M means short-term pressure is firming: recent deviations exceed older ones. "
        "The reverse ordering means short-term pressure is cooling. All three hugging zero means inflation is tracking the "
        "baseline. Because the lines average the same epsilon over nested windows they co-move; the "
        "signal is how fast they separate and when they cross, which usually precedes a change in "
        "the regime label above.",
    )

with tab_validation:
    st.subheader("Historical Signal Validation")
    section_notes(
        "Did the current-month signal historically contain forward information? Future CPI "
        "outcomes are used only for validation here, never for signal construction.",
        "Read the bars as historical outcome rates per bucket, the line as their sensitivity to "
        "the outcome threshold, and the heatmap as regime persistence. Small buckets give "
        "unstable rates, and Phase 2 benchmark comparison is still needed before treating these "
        "as forecast skill.",
    )
    with st.expander("How to read this tab"):
        st.markdown(
            "This tab tests whether the current-month signal historically contained forward "
            "information. Future CPI outcomes are used only for validation, not signal construction."
        )
        st.markdown(
            "Positive-shock resolution asks whether above-baseline inflation pressure faded. "
            "If epsilon moves from positive to negative, the high-inflation shock resolved and "
            "overshot lower."
        )
        st.markdown(
            "Absolute baseline convergence asks whether inflation ended close to baseline, "
            "regardless of direction. A negative overshoot may fail this test because inflation "
            "is still far from baseline, but it is not persistent high inflation."
        )
        st.markdown(
            "For trading interpretation, positive-shock resolution is usually more actionable. "
            "For equilibrium or policy-stability interpretation, absolute baseline convergence "
            "remains useful. Counts matter: small groups can produce unstable hit rates, and "
            "Phase 2 benchmark comparison is still needed before treating hit rates as signal skill."
        )
    scope_caveats(
        "Historical validation is live-like only under shifted baselines; full_sample is ex-post.",
        (
            "Historical validation is live-like only under rolling_36_shifted or expanding_shifted "
            "baselines.",
            "full_sample is ex-post and should not be used to judge live signal success.",
        ),
    )

    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        validation_horizon = st.selectbox(
            "Horizon",
            options=[3, 6, 12, 24, 36],
            index=2,
            format_func=lambda months: f"{months} months",
        )
    with control_col2:
        outcome_threshold = st.number_input(
            "Outcome threshold (pp)",
            min_value=0.01,
            value=0.50,
            step=0.05,
            format="%.2f",
        )
    with control_col3:
        validation_group = st.selectbox(
            "Group by",
            options=["regime", "short-term pressure"],
        )
    st.caption(
        "Outcome threshold (pp) defines a material distance from baseline in percentage points. "
        "Positive-shock labels ask whether above-baseline inflation pressure faded; absolute "
        "baseline convergence asks whether inflation ended close to baseline regardless of "
        "direction. Regime captures signal level and direction, while short-term pressure captures "
        "the 4M/8M/12M TINF ordering. Their combination is usually more actionable than either "
        "group alone."
    )

    # Ensure forward + label columns exist for whatever horizon is selected
    # (DEFAULT_LABEL_HORIZONS omits 3M), so a 3-month selection is not empty.
    validation_forward_horizons = tuple(
        sorted(set(validation_mod.DEFAULT_FORWARD_HORIZONS) | {validation_horizon})
    )
    validation_label_horizons = tuple(
        sorted(set(validation_mod.DEFAULT_LABEL_HORIZONS) | {validation_horizon})
    )
    validation_df = get_validation_frame(
        df,
        validation_forward_horizons,
        validation_label_horizons,
        float(outcome_threshold),
    )
    combined_summary = validation_mod.forward_outcome_summary_by_regime_and_pressure(
        validation_df, horizons=(validation_horizon,)
    )
    regime_summary = validation_mod.forward_outcome_summary_by_regime(
        validation_df, horizons=(validation_horizon,)
    )
    pressure_summary = validation_mod.forward_outcome_summary_by_short_term_pressure(
        validation_df, horizons=(validation_horizon,)
    )
    sensitivity_summary = get_threshold_sensitivity(df, validation_horizon)

    st.markdown("#### Combined regime x short-term pressure summary")
    st.caption(
        "This table crosses the historical regime with short-term pressure. It is often the "
        "most useful Phase 1 cut because it separates elevated-and-rising inflation pressure "
        "from elevated-but-cooling pressure. Counts matter: small groups can produce unstable "
        "rates."
    )
    st.dataframe(combined_summary, width="stretch")

    st.markdown(f"#### Outcome rates by bucket ({validation_horizon} months)")
    st.caption(
        "Bars show three historical outcome rates per bucket at the selected horizon: "
        "positive-shock resolved (cold), baseline convergence (gray), and positive-shock "
        "persisted (hot). Same rates as the tables below; small buckets give unstable rates."
    )
    rate_chart_col1, rate_chart_col2 = st.columns(2)
    with rate_chart_col1:
        st.plotly_chart(
            plots_mod.hit_rate_bar_figure(
                regime_summary,
                "historical_regime",
                VALIDATION_RATE_SPECS,
                title="By regime",
                group_order=validation_mod.REGIME_ORDER,
            ),
            width="stretch",
        )
    with rate_chart_col2:
        st.plotly_chart(
            plots_mod.hit_rate_bar_figure(
                pressure_summary,
                "historical_short_term_pressure",
                VALIDATION_RATE_SPECS,
                title="By short-term pressure",
                group_order=validation_mod.PRESSURE_ORDER,
            ),
            width="stretch",
        )
    with st.expander("Outcome rate tables (selected, by regime, by short-term pressure)"):
        st.markdown("##### Selected summary")
        if validation_group == "regime":
            st.dataframe(regime_summary, width="stretch")
        else:
            st.dataframe(pressure_summary, width="stretch")

        st.markdown("##### Forward outcome summary by regime")
        st.dataframe(regime_summary, width="stretch")

        st.markdown("##### Forward outcome summary by short-term pressure")
        st.dataframe(pressure_summary, width="stretch")

    st.markdown(f"#### Threshold sensitivity ({validation_horizon} months)")
    st.caption(
        "This recomputes outcome labels at fixed thresholds of 0.25, 0.50, 0.75, "
        "and 1.00 pp. It is sensitivity analysis only, not threshold optimization. Phase 2 "
        "benchmark comparison is still required before treating hit rates as forecast skill."
    )
    if sensitivity_summary.empty:
        st.info("No threshold-sensitivity rows are available for the selected horizon.")
    else:
        st.plotly_chart(
            plots_mod.threshold_sensitivity_figure(
                sensitivity_summary,
                VALIDATION_RATE_SPECS,
                title=f"Outcome rates vs threshold ({validation_horizon} months)",
            ),
            width="stretch",
        )
        with st.expander("Threshold sensitivity table"):
            st.dataframe(sensitivity_summary, width="stretch")

    st.markdown(f"#### Regime transition matrix ({validation_horizon} months)")
    transition = validation_mod.regime_transition_matrix(validation_df, horizon=validation_horizon)
    if transition.empty:
        st.info("No valid regime transitions are available for the selected horizon.")
    else:
        st.caption(
            "Each row is the current regime; each column is the regime h months later. Cells are "
            "row-normalized transition probabilities (each row sums to 1)."
        )
        st.plotly_chart(
            plots_mod.heatmap_figure(
                transition,
                title=f"Regime transitions ({validation_horizon} months)",
                colorscale="Blues",
                zmin=0.0,
                zmax=1.0,
                value_fmt=".0%",
                colorbar_title="P",
                xaxis_title="Regime at t+h",
                yaxis_title="Regime at t",
                hover_value_label="probability",
            ),
            width="stretch",
        )
        with st.expander("Regime transition table"):
            st.dataframe(transition, width="stretch")

    with st.expander("Worked examples — false / successful positives & negatives"):
        st.caption(
            "Audit trail: representative months behind the rates above, capped per category."
        )
        examples = validation_mod.validation_examples(validation_df, horizon=validation_horizon)
        example_titles = {
            "false_transitory": (
                "False transitory: signal suggested fading pressure, but positive inflation shock persisted"
            ),
            "false_persistent": (
                "False persistent: signal suggested persistent pressure, but positive shock resolved"
            ),
            "successful_transitory": (
                "Successful transitory calls: positive shock resolved without downside overshoot"
            ),
            "successful_transitory_downside_overshoot": (
                "Successful transitory with downside overshoot: positive shock resolved below baseline"
            ),
            "successful_persistent": "Successful persistent calls: positive shock stayed above threshold",
        }
        for key, title in example_titles.items():
            st.markdown(f"##### {title}")
            table = examples[key]
            if table.empty:
                st.info("No examples found under the current settings.")
            else:
                st.dataframe(table, width="stretch")

with tab_market_linkage:
    st.subheader("Phase 4A Market Linkage")
    section_notes(
        "How did FRED Treasury yields, breakevens, and real yields historically move after each "
        "inflation-signal state? This links the already-built signal to market history as "
        "descriptive evidence only.",
        "Read the grouped bars as the average forward change (bp) by regime per channel and the "
        "heatmap as signal-vs-future-change correlation. A positive change means the rate rose "
        "after the signal date — descriptive history, not a forecast or trade.",
    )
    scope_caveats(
        "Descriptive historical market linkage only — not a trading signal or forecast.",
        (
            "This tab links the already-built inflation signal to FRED Treasury yield, breakeven, "
            "and real-yield history. It is descriptive historical evidence only.",
            "This is not a trading signal, not a forecast model, and not a model-generated trade "
            "recommendation. It summarizes how selected FRED rates changed after past TINF/regime "
            "states.",
            "This is descriptive market linkage, not a trading signal.",
            "A positive forward change means the market variable rose after the signal date.",
            "Nominal yields, breakevens, and real yields measure different channels.",
            "Market linkage may be useful even if TINF/regime is not the best CPI point-forecast model.",
        ),
    )
    market_linkage_horizon_label = st.selectbox(
        "Market linkage horizon",
        options=list(MARKET_LINKAGE_HORIZON_OPTIONS),
        index=list(MARKET_LINKAGE_HORIZON_OPTIONS).index("12M"),
    )
    market_linkage_horizon = MARKET_LINKAGE_HORIZON_OPTIONS[market_linkage_horizon_label]

    market_result = get_market_data(sample_mode)
    latest_market_dates = market_result.latest_valid_date_by_variable or {}
    latest_market_dates_text = "; ".join(
        f"{variable}={macro_data.date_label(date)}"
        for variable, date in latest_market_dates.items()
        if variable in market_result.available_market_variables
    )
    if not latest_market_dates_text:
        latest_market_dates_text = "none"
    available_market_variables = ", ".join(market_result.available_market_variables) or "none"

    if market_result.market_data_source_used == "fred_csv":
        st.warning(
            "Official FRED API market fetch was unavailable or not configured, so this tab is "
            "using public FRED CSV market data."
        )
    elif market_result.market_data_source_used == "cached_fred_market":
        st.warning(
            f"Live FRED market fetch failed, so this tab is using cached market data from "
            f"`{market_result.market_cache_file_used}`."
        )
    elif market_result.market_data_source_used == "unavailable":
        st.warning(
            "Market linkage is unavailable because official FRED API, public FRED CSV, and "
            "local cached market data all failed. No demo market data are used."
        )

    st.caption(
        "Market data status: "
        f"market_data_source_used={market_result.market_data_source_used}; "
        f"market_live_fetch_status={market_result.market_live_fetch_status}; "
        f"market_cache_file_used={market_result.market_cache_file_used or 'n/a'}; "
        f"available_market_variables={available_market_variables}; "
        f"latest_valid_date_by_variable={latest_market_dates_text}."
    )

    availability = market_data_mod.market_data_availability(market_result.data)
    with st.expander("Market data availability"):
        st.caption(
            "FRED real-yield and breakeven histories start later than CPI history. Counts are "
            "reported by variable rather than forced to match the inflation sample."
        )
        st.dataframe(availability, width="stretch")

    if not market_result.available_market_variables:
        st.info("No approved market variables are available for the selected sample.")
    else:
        market_tables = get_market_linkage_tables(df, market_result.data)

        st.markdown("#### Current market snapshot")
        st.caption("Latest available FRED observation by approved market variable.")
        st.dataframe(market_tables.current_snapshot, width="stretch")

        selected_channel_summary = market_tables.channel_summary_by_regime.loc[
            market_tables.channel_summary_by_regime["horizon_months"] == market_linkage_horizon
        ]
        st.markdown(f"#### Forward change by regime per channel ({market_linkage_horizon_label})")
        st.caption(
            "Average forward change in basis points by historical regime, grouped by channel "
            "(nominal 2Y/10Y, 5Y/10Y breakevens, 5Y/10Y real yields). A positive bar means the "
            "channel's rates rose after that regime; hover for the median and count."
        )
        if selected_channel_summary.empty:
            st.info("No channel summary is available for the selected horizon.")
        else:
            if selected_channel_summary["weak_evidence"].fillna(False).astype(bool).any():
                st.warning(
                    "Rows marked weak_evidence=True have fewer than 30 complete observations."
                )
            st.plotly_chart(
                plots_mod.forward_change_by_regime_channel_figure(
                    selected_channel_summary,
                    value_col="avg_change_bp",
                    channel_labels=MARKET_CHANNEL_LABELS,
                    regime_order=validation_mod.REGIME_ORDER,
                    title=f"Avg forward change by regime per channel ({market_linkage_horizon_label})",
                ),
                width="stretch",
            )

            with st.expander(f"Channel interpretation tables ({market_linkage_horizon_label})"):
                st.caption(
                    "Channel rows use row-wise averages across the two approved FRED variables in "
                    "each channel: 2Y/10Y nominal yields, 5Y/10Y breakevens, and 5Y/10Y real yields."
                )
                for channel, channel_label in MARKET_CHANNEL_LABELS.items():
                    channel_rows = selected_channel_summary.loc[
                        selected_channel_summary["market_channel"] == channel
                    ].copy()
                    st.markdown(f"##### {channel_label}")
                    if channel_rows.empty:
                        st.info(f"No {channel_label.lower()} summary is available.")
                    else:
                        st.dataframe(
                            channel_rows.loc[
                                :,
                                [
                                    column
                                    for column in MARKET_CHANNEL_INTERPRETATION_COLUMNS
                                    if column in channel_rows.columns
                                ],
                            ],
                            width="stretch",
                        )

                st.markdown("##### What historically happened?")
                what_happened = selected_channel_summary.copy()
                what_happened["market_channel"] = (
                    what_happened["market_channel"]
                    .map(MARKET_CHANNEL_LABELS)
                    .fillna(what_happened["market_channel"])
                )
                what_happened_cols = [
                    "historical_regime",
                    "market_channel",
                    "historical_direction",
                    "avg_change_bp",
                    "median_change_bp",
                    "count",
                    "weak_evidence",
                    "evidence_note",
                ]
                st.dataframe(
                    what_happened.loc[
                        :,
                        [column for column in what_happened_cols if column in what_happened.columns],
                    ].sort_values(["market_channel", "historical_regime"]),
                    width="stretch",
                )

        selected_rankings = market_tables.regime_pressure_rankings.loc[
            market_tables.regime_pressure_rankings["horizon_months"] == market_linkage_horizon
        ]
        with st.expander(f"Regime x pressure rankings ({market_linkage_horizon_label})"):
            st.caption(
                "Rankings compare historical_regime x historical_short_term_pressure groups "
                "within each market variable at the selected horizon."
            )
            if selected_rankings.empty:
                st.info("No regime x pressure rankings are available for the selected horizon.")
            else:
                rank_col1, rank_col2 = st.columns(2)
                with rank_col1:
                    st.markdown("##### Largest average increases")
                    highest_cols = [
                        "highest_change_rank",
                        *MARKET_RANKING_COLUMNS,
                    ]
                    st.dataframe(
                        selected_rankings.sort_values(["highest_change_rank", "market_variable"])
                        .head(12)
                        .loc[
                            :,
                            [column for column in highest_cols if column in selected_rankings.columns],
                        ],
                        width="stretch",
                    )
                with rank_col2:
                    st.markdown("##### Largest average decreases")
                    lowest_cols = [
                        "lowest_change_rank",
                        *MARKET_RANKING_COLUMNS,
                    ]
                    st.dataframe(
                        selected_rankings.sort_values(["lowest_change_rank", "market_variable"])
                        .head(12)
                        .loc[
                            :,
                            [column for column in lowest_cols if column in selected_rankings.columns],
                        ],
                        width="stretch",
                    )

        with st.expander("Forward market-change summary tables (by regime / pressure / combined)"):
            st.markdown("##### Forward market-change summary by historical regime")
            st.caption(
                "Changes are t to t+h in basis points. Rows without full future market data "
                "for that variable and horizon are excluded."
            )
            if market_tables.summary_by_regime.empty:
                st.info("No regime summary is available with the current market data.")
            else:
                st.dataframe(market_tables.summary_by_regime, width="stretch")

            st.markdown("##### Forward market-change summary by short-term pressure")
            if market_tables.summary_by_pressure.empty:
                st.info("No short-term pressure summary is available with the current market data.")
            else:
                st.dataframe(market_tables.summary_by_pressure, width="stretch")

            st.markdown("##### Combined regime x short-term pressure")
            if market_tables.summary_by_regime_and_pressure.empty:
                st.info("No combined regime x pressure summary is available.")
            else:
                st.dataframe(
                    market_tables.summary_by_regime_and_pressure,
                    width="stretch",
                )

        st.markdown("#### Signal-to-future-market-change correlations")
        st.caption(
            "Simple Pearson correlations between current epsilon/TINF readings and future "
            "FRED market changes. These are descriptive associations, not a trading model."
        )
        if market_tables.correlations.empty:
            st.info("No correlations are available with the current market data.")
        else:
            # Display reshape only: pivot the already-computed correlations for the
            # selected horizon into a signal x market grid (values unchanged).
            selected_corr = market_tables.correlations.loc[
                market_tables.correlations["horizon_months"] == market_linkage_horizon
            ]
            if selected_corr.empty:
                st.info("No correlations are available for the selected horizon.")
            else:
                corr_wide = selected_corr.pivot(
                    index="signal_variable", columns="market_variable", values="correlation"
                )
                signal_order = [
                    column
                    for column in market_linkage_mod.DEFAULT_SIGNAL_COLUMNS
                    if column in corr_wide.index
                ]
                market_order = [
                    column
                    for column in market_data_mod.MARKET_VALUE_COLUMNS
                    if column in corr_wide.columns
                ]
                corr_wide = corr_wide.reindex(
                    index=signal_order or None, columns=market_order or None
                )
                st.plotly_chart(
                    plots_mod.heatmap_figure(
                        corr_wide,
                        title=(
                            "Signal vs future market-change correlation "
                            f"({market_linkage_horizon_label})"
                        ),
                        colorscale="RdBu",
                        reversescale=True,
                        zmid=0.0,
                        zmin=-1.0,
                        zmax=1.0,
                        value_fmt=".2f",
                        colorbar_title="r",
                        xaxis_title="Market variable",
                        yaxis_title="Signal",
                        hover_value_label="correlation",
                    ),
                    width="stretch",
                )
            with st.expander("Correlation table (all horizons)"):
                st.dataframe(market_tables.correlations, width="stretch")


with tab_trader_research:
    st.subheader("Trader Research (descriptive, rates-only)")
    section_notes(
        "Given the macro state we are in today, what did the six approved FRED rate instruments "
        "historically do over the next 3/6/12/24/36 months? This collapses the Market Linkage "
        "history to the current live-safe regime bucket and shows the forward-change distribution "
        "plus the analog months behind it.",
        "Read the range plot per instrument: the marker is the median forward change and the "
        "whisker is the p25-p75 spread, in basis points, with hot/cold by the sign of the median. "
        "These are descriptive historical base rates conditioned on today's bucket, never a "
        "forecast or trade.",
    )
    scope_caveats(
        "Descriptive historical base rates only — not a forecast or a trade.",
        (
            "Descriptive historical base rates only. This is not a forecast, not a trading signal, "
            "and not a model-generated trade recommendation. No sizing, timing, or instruments are "
            "implied.",
            "The bucket uses live-safe walk-forward labels (no full-sample lookahead); rate changes "
            "are in basis points; small buckets (weak_evidence) and the baseline/sample choice "
            "shift these readings; forward changes describe history only and never construct the "
            "signal.",
        ),
    )

    bucket = trader_research_mod.latest_walk_forward_bucket(df)
    if not bucket.available:
        st.info(bucket.reason or "No live-safe regime bucket is available yet.")
    else:
        st.markdown(
            "#### Current bucket "
            f"(live-safe walk-forward, as of {macro_data.date_label(bucket.as_of)})"
        )
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("Regime", regime_badge(bucket.regime))
        bcol2.metric("Short-term pressure", str(bucket.pressure))
        bcol3.metric(
            "Regime analogs", f"{bucket.regime_count}",
            delta=f"{bucket.regime_pressure_count} also share pressure", delta_color="off",
        )
        bcol4.metric("As of", macro_data.date_label(bucket.as_of))
        trader_snapshot = latest_signal_snapshot(df)
        if trader_snapshot.get("available"):
            st.caption(
                "Cross-check only: the ex-post full-sample snapshot labels this month "
                f"'{trader_snapshot['regime']}'. Bucket matching uses the live-safe walk-forward "
                "label above, never this full-sample one."
            )

        market_result = get_market_data(sample_mode)
        if market_result.market_data_source_used == "fred_csv":
            st.warning(
                "Official FRED API market fetch was unavailable or not configured, so this tab "
                "is using public FRED CSV market data."
            )
        elif market_result.market_data_source_used == "cached_fred_market":
            st.warning(
                "Live FRED market fetch failed, so this tab is using cached market data from "
                f"`{market_result.market_cache_file_used}`."
            )
        elif market_result.market_data_source_used == "unavailable":
            st.warning(
                "Trader Research is unavailable because official FRED API, public FRED CSV, and "
                "local cached market data all failed. No demo market data are used."
            )

        if not market_result.available_market_variables:
            st.info("No approved market variables are available for the selected sample.")
        else:
            market_tables = get_market_linkage_tables(df, market_result.data)

            trader_horizon_label = st.selectbox(
                "Forward horizon",
                options=list(MARKET_LINKAGE_HORIZON_OPTIONS),
                index=list(MARKET_LINKAGE_HORIZON_OPTIONS).index("12M"),
                key="trader_research_horizon",
            )
            trader_horizon = MARKET_LINKAGE_HORIZON_OPTIONS[trader_horizon_label]

            trader_regimes = trader_research_mod.available_regimes(market_tables) or (
                (bucket.regime,) if bucket.regime else ()
            )
            if trader_regimes:
                default_regime_index = (
                    trader_regimes.index(bucket.regime)
                    if bucket.regime in trader_regimes
                    else 0
                )
                chosen_regime = st.selectbox(
                    "Condition on regime (defaults to today's bucket)",
                    options=list(trader_regimes),
                    index=default_regime_index,
                    key="trader_research_regime",
                )
            else:
                chosen_regime = bucket.regime

            condition_on_pressure = st.checkbox(
                "Also condition on short-term pressure",
                value=False,
                key="trader_research_pressure_toggle",
            )
            chosen_pressure = None
            if condition_on_pressure:
                trader_pressures = trader_research_mod.available_pressures(market_tables) or (
                    (bucket.pressure,) if bucket.pressure else ()
                )
                if trader_pressures:
                    default_pressure_index = (
                        trader_pressures.index(bucket.pressure)
                        if bucket.pressure in trader_pressures
                        else 0
                    )
                    chosen_pressure = st.selectbox(
                        "Condition on pressure",
                        options=list(trader_pressures),
                        index=default_pressure_index,
                        key="trader_research_pressure",
                    )

            view = trader_research_mod.build_trader_research_view(
                market_tables,
                chosen_regime,
                chosen_pressure,
                horizons=(trader_horizon,),
            )

            if not view.available:
                st.info(view.reason or "No historical analogs for this bucket.")
            else:
                conditioning = f"regime '{view.regime}'" + (
                    f" x pressure '{view.pressure}'" if view.pressure else ""
                )
                st.caption(f"Conditioned on {conditioning}. Forward changes are t to t+h in basis points.")
                if view.weak_evidence:
                    st.warning(
                        "Some rows have fewer than 30 complete observations "
                        "(weak_evidence=True) - interpret cautiously."
                    )

                st.markdown(f"#### Forward rate-change distribution ({trader_horizon_label})")
                st.caption(
                    "Per approved FRED instrument: the median (marker) and p25-p75 range (whisker) "
                    "of the forward change in basis points, colored hot/cold by the sign of the "
                    "median. A positive change means the instrument rose after the signal month."
                )
                trader_dist_cols = [
                    column
                    for column in (
                        "market_variable",
                        "count",
                        "median_change_bp",
                        "p25_change_bp",
                        "p75_change_bp",
                        "avg_change_bp",
                        "increase_hit_rate",
                        "decrease_hit_rate",
                        "weak_evidence",
                    )
                    if column in view.distribution.columns
                ]
                if view.distribution.empty:
                    st.info("No distribution rows for this bucket and horizon.")
                else:
                    st.plotly_chart(
                        plots_mod.forward_change_range_figure(view.distribution),
                        width="stretch",
                    )
                    with st.expander("Distribution table"):
                        st.dataframe(view.distribution.loc[:, trader_dist_cols], width="stretch")

                st.markdown(f"#### Channel roll-up ({trader_horizon_label})")
                st.caption(
                    "Row-wise channel averages: nominal (2Y/10Y), breakevens (5Y/10Y), and real "
                    "yields (5Y/10Y), for the selected regime."
                )
                trader_channel = view.channel_rollup
                if not trader_channel.empty and "market_channel" in trader_channel.columns:
                    trader_channel = trader_channel.copy()
                    trader_channel["market_channel"] = (
                        trader_channel["market_channel"]
                        .map(MARKET_CHANNEL_LABELS)
                        .fillna(trader_channel["market_channel"])
                    )
                trader_channel_cols = [
                    column
                    for column in (
                        "market_channel",
                        "historical_direction",
                        "avg_change_bp",
                        "median_change_bp",
                        "count",
                        "weak_evidence",
                    )
                    if column in trader_channel.columns
                ]
                if trader_channel.empty:
                    st.info("No channel roll-up for this regime and horizon.")
                else:
                    st.dataframe(trader_channel.loc[:, trader_channel_cols], width="stretch")

                with st.expander("Analog months (audit trail)"):
                    st.caption(
                        "The historical months in this bucket and their forward rate changes (bp) - "
                        "the observations behind the distribution above."
                    )
                    if view.analog_months.empty:
                        st.info("No analog months with forward market data for this bucket.")
                    else:
                        st.dataframe(view.analog_months, width="stretch")


with tab_benchmarks:
    st.subheader("Phase 2 Benchmark Comparison")
    st.markdown(
        "This tab tests whether the TINF/regime signal adds useful forward inflation "
        "information beyond simple baselines. Phase 1 hit rates are not enough unless "
        "they beat naive alternatives."
    )
    st.markdown(
        "Forecasts use only information available at month t. Future CPI outcomes are "
        "used only for evaluation. Results are historical validation, not a guaranteed "
        "trading signal."
    )
    st.warning(
        "No market variables are used by these benchmark forecasts. Market linkage is reported "
        "only in its separate descriptive tab."
    )
    if not meta.live_safe:
        st.warning(
            "The selected baseline is not live-safe. Benchmark results under this baseline are "
            "descriptive, not live-like."
        )

    bench_col1, bench_col2 = st.columns(2)
    with bench_col1:
        benchmark_horizon = st.selectbox(
            "Benchmark horizon",
            options=[3, 6, 12, 24, 36],
            index=2,
            format_func=lambda months: f"{months} months",
        )
    with bench_col2:
        benchmark_threshold = st.number_input(
            "Benchmark outcome threshold (pp)",
            min_value=0.01,
            value=0.50,
            step=0.05,
            format="%.2f",
        )

    forecasts, benchmark_metrics, benchmark_improvements, benchmark_confusion = (
        get_benchmark_tables(df, benchmark_horizon, float(benchmark_threshold))
    )

    if benchmark_metrics.empty:
        st.info("No benchmark comparison is available for the selected horizon and sample.")
    else:
        tinf_rows = benchmark_metrics.loc[benchmark_metrics["model"] == "tinf_regime_bucket"]
        if not tinf_rows.empty:
            tinf_row = tinf_rows.iloc[0]
            beats_no_change = (
                tinf_row["mae_improvement_vs_no_change_pct"] > 0
                or tinf_row["rmse_improvement_vs_no_change_pct"] > 0
            )
            beats_mean_reversion = (
                tinf_row["mae_improvement_vs_mean_reversion_pct"] > 0
                or tinf_row["rmse_improvement_vs_mean_reversion_pct"] > 0
            )
            st.info(
                "TINF/regime bucket improvement: "
                f"{tinf_row['mae_improvement_vs_no_change_pct']:.1f}% MAE vs no-change, "
                f"{tinf_row['rmse_improvement_vs_no_change_pct']:.1f}% RMSE vs no-change; "
                f"{tinf_row['mae_improvement_vs_mean_reversion_pct']:.1f}% MAE vs mean "
                f"reversion, {tinf_row['rmse_improvement_vs_mean_reversion_pct']:.1f}% "
                "RMSE vs mean reversion."
            )
            if not beats_no_change and not beats_mean_reversion:
                st.warning(
                    "Under the current settings, the TINF/regime bucket does not improve on "
                    "the simple no-change or mean-reversion baselines by MAE/RMSE."
                )

        st.markdown("#### Benchmark metric summary")
        st.caption(
            "MAE and RMSE score CPI YoY forecast errors. Directional accuracy scores whether "
            "the forecast got the direction of the CPI YoY change right. Hit, false-positive, "
            "and false-negative rates classify persistent high-inflation outcomes for current "
            "positive-shock rows. Note: no-change forecasts zero CPI change, so its directional "
            "accuracy is ~0 by construction rather than a skill signal."
        )
        st.dataframe(benchmark_metrics, width="stretch")

        st.markdown("#### Benchmark-relative improvement")
        st.caption(
            "Positive values mean the model reduced MAE or RMSE versus the comparison baseline. "
            "These are historical validation statistics, not optimized thresholds."
        )
        st.dataframe(benchmark_improvements, width="stretch")

        st.markdown("#### Classification summary")
        st.caption(
            "Positive class: persistent high inflation after a current positive inflation shock. "
            "Forecast classifications compare each model forecast with the current-month baseline "
            "and the selected threshold. Note: mean reversion forecasts the baseline, so by "
            "construction it can never be classified persistent (its confusion row is all-negative) "
            "- read it as a structural floor, not a failure to detect."
        )
        st.dataframe(benchmark_confusion, width="stretch")

        st.markdown("#### Forecast audit sample")
        sample_cols = [
            "date",
            "horizon_months",
            "model",
            "historical_regime",
            "current_cpi_yoy",
            "forecast_cpi_yoy",
            "actual_cpi_yoy",
            "forecast_error",
            "forecast_persistent_high_inflation",
            "actual_persistent_high_inflation",
        ]
        st.dataframe(
            forecasts.loc[:, sample_cols].sort_values("date", ascending=False).head(50),
            width="stretch",
        )

with tab_report:
    st.subheader("Phase 5 Macro Research Report")
    report_market_result = get_market_data(sample_mode)
    report = get_macro_research_report(
        raw,
        df,
        baseline_method,
        sample_mode,
        report_market_result.data,
        load_result,
        report_market_result,
    )
    if not report.available:
        st.warning(report.reason or "Report unavailable.")
    else:
        if not meta.live_safe:
            st.warning(
                "Selected baseline is ex-post. This report describes history, not a live signal — "
                "switch to a live-safe baseline for current-signal use."
            )
        st.markdown(f"**{report.headline}**")

        st.markdown("#### 1. Current Regime")
        st.markdown("\n".join(f"- {line}" for line in report.current_regime_lines))
        if not report.current_regime_table.empty:
            st.dataframe(report.current_regime_table, width="stretch")

        st.markdown("#### 2. Signal Confidence")
        st.markdown("\n".join(f"- {line}" for line in report.signal_confidence_lines))
        if not report.benchmark_comparisons.empty:
            st.dataframe(report.benchmark_comparisons, width="stretch")
        if not report.benchmark_metrics.empty:
            with st.expander("Benchmark metric detail"):
                st.dataframe(report.benchmark_metrics, width="stretch")

        st.markdown("#### 3. Robustness Summary")
        st.markdown("\n".join(f"- {line}" for line in report.robustness_lines))
        if not report.inflation_measure_availability.empty:
            st.markdown("##### Inflation measure availability")
            st.dataframe(report.inflation_measure_availability, width="stretch")
        if not report.robustness_win_rates.empty:
            st.markdown("##### Aggregate TINF/regime win rates")
            st.dataframe(report.robustness_win_rates, width="stretch")
        if not report.robustness_verdict.empty:
            with st.expander("Robustness verdict detail"):
                st.dataframe(report.robustness_verdict, width="stretch")

        st.markdown("#### 4. Historical Analogs")
        st.markdown("\n".join(f"- {line}" for line in report.historical_analog_lines))
        if report.historical_analogs.empty:
            st.info("No analog rows are available for the current signal state.")
        else:
            st.dataframe(report.historical_analogs, width="stretch")

        st.markdown("#### 5. Market Linkage Summary")
        st.markdown("\n".join(f"- {line}" for line in report.market_linkage_lines))
        if report.market_channel_summary.empty:
            st.info("No market channel summary is available for this report run.")
        else:
            st.dataframe(report.market_channel_summary, width="stretch")

        st.markdown("#### 6. Caveats / Model Risk")
        st.markdown("\n".join(f"- {line}" for line in report.caveats))

        st.markdown("#### 7. Watchlist / What to Monitor Next")
        st.markdown("\n".join(f"- {line}" for line in report.watchlist))

        section_notes(
            "A single research-report layer that synthesizes current regime, benchmark evidence, "
            "robustness, analog history, market linkage, caveats, and freshness from the validated "
            "dashboard tables.",
            "Read it as decision support. Signal interpretation, CPI point-forecast accuracy, and "
            "descriptive market linkage are separate evidence layers; none of them is a trading "
            "recommendation or PnL backtest.",
        )

with tab_framework:
    st.subheader("Paper-style descriptive tables")
    table_cols = ["inflation_yoy", "tinf_4m", "tinf_8m", "tinf_12m", "tbill_3m"]
    st.dataframe(summary_stats(df, table_cols), width="stretch")
    section_notes(
        "Paper-style descriptive moments for the selected sample: mean, standard deviation, and "
        "decile/quartile cuts of CPI YoY, the three TINF horizons, and the 3-month T-bill control, "
        "plus the observation count each column actually has.",
        "TINF means sit near zero by construction (they are deviations from a mean-reversion "
        "baseline), and standard deviation shrinks as the window lengthens because averaging "
        "smooths. Compare the p10-p90 spread across horizons to gauge how much variation is "
        "short-lived. Units are percentage points. Lower n for TINF columns reflects rolling-window "
        "warm-up, not missing data. Only in the fixed paper window do these correspond to the "
        "paper's table; in other modes they are descriptive.",
    )

    st.subheader("Correlation matrix")
    st.dataframe(correlation_matrix(df, table_cols), width="stretch")
    section_notes(
        "How strongly the current YoY inflation level co-moves with each TINF horizon and the "
        "T-bill control, and how the TINF horizons co-move with each other, over the selected "
        "sample.",
        "High correlation between inflation and TINF 4M is partly mechanical, since TINF is built "
        "from inflation minus a slow baseline. Correlation falling as the window lengthens means "
        "the current inflation level is dominated by recent deviations. The very high TINF-to-TINF "
        "correlations come from overlapping windows: the three horizons are not independent "
        "signals, which is why coefficients turn unstable when all three enter the regression "
        "below jointly.",
    )

    st.subheader("OLS table with robust standard errors")
    try:
        st.dataframe(run_paper_style_regressions(df), width="stretch")
        section_notes(
            "The paper's core contemporaneous regressions: how much of the current CPI YoY level "
            "is explained by each transitory-inflation horizon, controlling for the 3-month "
            "T-bill, with HC1 robust t-statistics, R-squared, and per-spec sample sizes. One "
            "specification per horizon, plus all horizons jointly.",
            "A tinf_4m coefficient near 1 says a 1pp rise in the 4-month average deviation maps "
            "roughly one-for-one into YoY inflation. In the joint specification, overlapping "
            "windows create multicollinearity, so individual signs can flip (a negative tinf_8m, "
            "for example); judge the joint model by its R-squared, not by per-coefficient stories. "
            "These are same-month descriptive fits on an ex-post sample, not forecasts. nobs "
            "differs across specs because each drops rows missing its own inputs.",
        )
    except Exception as exc:
        st.error(f"Regression failed: {exc}")

    st.subheader("White-noise / autocorrelation diagnostic")
    try:
        st.dataframe(ljung_box_table(df["tinf_4m"], lags=40), width="stretch")
        section_notes(
            "Whether TINF 4M is white noise. The Ljung-Box statistic jointly tests for "
            "autocorrelation up to the shown lag (40 months when the sample allows; fewer "
            "otherwise), and the table reports the test statistic and p-value at that lag.",
            "A p-value near zero rejects white noise: TINF is serially correlated, i.e. "
            "persistent, which is the property the decay/convergence analysis depends on. If the "
            "p-value were large, deviations would carry no exploitable structure and the AR(1) "
            "persistence and decay estimates would be meaningless. Rejection says structure "
            "exists; it does not by itself validate the AR(1) specification.",
        )
    except Exception as exc:
        st.error(f"Ljung-Box diagnostic failed: {exc}")

with tab_decay:
    st.subheader("Rolling AR(1) persistence and decay")
    windows = st.multiselect(
        "Rolling windows", options=[24, 30, 36, 48, 60, 84, 120], default=[24, 30, 36, 60]
    )
    if windows:
        rho_df, decay_df = decay_summaries_for_windows(
            df, windows=tuple(windows), value_col="tinf_4m"
        )
        st.plotly_chart(plots_mod.rolling_rho_figure(rho_df), width="stretch")
        section_notes(
            "How persistent the transitory component has been through time: the AR(1) coefficient "
            "(rho) of TINF 4M, re-estimated in rolling windows ending at each date, one line per "
            "selected window length, with reference lines at rho = 0 and rho = 1.",
            "Rho near 0 means deviations die out quickly; near 1 means they linger (slow mean "
            "reversion); above 1 means deviations were locally compounding inside that window, "
            "which is typical around regime shifts and is flagged as explosive in the table below. "
            "Shorter windows respond faster but are noisier. Agreement across window lengths makes "
            "a persistence reading robust; divergence means the conclusion depends on the "
            "estimation window.",
        )

        st.dataframe(decay_df, width="stretch")
        section_notes(
            "The paper's convergence arithmetic for each selected window: the latest rho (rho_T), "
            "an AR(1) fit of the rolling-rho series itself (mu and c). The published paper decay "
            "formula uses rho_T and mu only — the intercept c is estimated but intentionally "
            "unused, a disclosed deviation from the paper. The table also shows the implied share "
            "of a current deviation gone after 6 and 12 months, the time to 95% convergence (t* in "
            "months and years), and whether the formula's validity conditions hold.",
            "Only read decay_6m_pct, decay_12m_pct, and t* when valid_formula is True (requires "
            "rho_T > 0 and 0 < mu < 1). rho_T above 1 with a valid mu describes persistence that "
            "is currently explosive but expected to mean-revert as rho itself decays; the warning "
            "column is part of the result, not an error. Treat everything here as a model estimate "
            "with sampling error: compare across windows before concluding, and disclose which "
            "window you quote.",
        )

        valid_decay = decay_df.loc[decay_df["valid_formula"]].copy()
        if not valid_decay.empty:
            selected_row = valid_decay.iloc[0]
            curve = decay_curve(float(selected_row["rho_T"]), float(selected_row["mu"]), months=48)
            st.plotly_chart(plots_mod.decay_curve_figure(curve), width="stretch")
            section_notes(
                "The implied forward path of the current deviation, using rho_T and mu from the "
                "first valid row of the table above: what percentage has decayed, and what "
                "remains, after each month ahead, against the dotted 95% convergence threshold.",
                "A steep early slope means fast normalization toward the baseline; a long flat "
                "tail means lingering pressure. The curve mechanically extrapolates two estimated "
                "parameters, so read it as what the fitted persistence implies, not as a forecast "
                "with confidence bands. When rho_T or mu sits near a validity bound, small "
                "re-estimates move this curve a lot; cross-check it against the other windows in "
                "the table.",
            )
        else:
            st.warning("No valid decay curve available for selected windows.")

with tab_robustness:
    st.subheader("Phase 3 Benchmark Robustness")
    st.markdown(
        "Robustness asks whether the benchmark conclusion survives reasonable choices. "
        "A signal that only works under one horizon or threshold is weaker than one that "
        "works across settings."
    )
    st.markdown(
        "Phase 3B adds headline CPI, core CPI, PCE, and core PCE comparisons. Headline CPI "
        "remains the paper/default measure; core and PCE measures are robustness checks, not "
        "paper-exact replication. Future inflation outcomes are evaluation only, thresholds "
        "are reported rather than optimized, and no market variables are included."
    )
    st.warning("`full_sample` is ex-post / paper-style only and is not a live-safe baseline.")

    robustness_col1, robustness_col2, robustness_col3 = st.columns(3)
    with robustness_col1:
        robustness_sample_modes = st.multiselect(
            "Robustness sample modes",
            options=list(SAMPLE_MODES),
            default=[sample_mode],
            format_func=lambda name: MODE_LABELS[name],
        )
    with robustness_col2:
        robustness_baselines = st.multiselect(
            "Robustness baselines",
            options=list(BASELINE_META),
            default=list(robustness_mod.DEFAULT_ROBUSTNESS_BASELINES),
        )
    with robustness_col3:
        robustness_inflation_measures = st.multiselect(
            "Inflation measures",
            options=list(macro_data.INFLATION_MEASURES),
            default=list(robustness_mod.DEFAULT_ROBUSTNESS_INFLATION_MEASURES),
            format_func=lambda key: INFLATION_MEASURE_LABELS[key],
        )

    st.caption(
        "Fixed Phase 3 grid: horizons 3M, 6M, 12M, 24M, and 36M; thresholds "
        "0.25, 0.50, 0.75, and 1.00 pp. These settings are shown together; the "
        "dashboard does not choose a best threshold. Select additional sample modes "
        "or inflation measures above to compare paper, live, max-history, headline, "
        "core, and PCE robustness settings."
    )

    if not robustness_sample_modes or not robustness_baselines or not robustness_inflation_measures:
        st.info(
            "Select at least one sample mode, one baseline, and one inflation measure "
            "to run robustness tables."
        )
    else:
        robustness_raw = {}
        status_rows = []
        for mode_name in robustness_sample_modes:
            mode_result = get_data(mode_name)
            robustness_raw[mode_name] = mode_result.data
            status_rows.append(
                {
                    "sample_mode": mode_name,
                    "data_source_used": mode_result.data_source_used,
                    "live_fetch_status": mode_result.live_fetch_status,
                    "rows": len(mode_result.data),
                    "available_measures": ", ".join(
                        INFLATION_MEASURE_LABELS[key]
                        for key in macro_data.available_inflation_measures(mode_result.data)
                    ),
                }
            )

        scorecard, verdict, win_rates, availability = get_robustness_tables(
            robustness_raw,
            tuple(robustness_baselines),
            tuple(robustness_inflation_measures),
        )

        st.markdown("#### Robustness data status")
        st.dataframe(pd.DataFrame(status_rows), width="stretch")
        st.markdown("#### Inflation measure availability")
        st.caption(
            "Unavailable measures are skipped rather than filled with demo data. If live FRED "
            "and cache files lack a selected core/PCE series, the headline CPI rows still run "
            "and the missing measure is disclosed here."
        )
        st.dataframe(availability, width="stretch")

        if scorecard.empty:
            st.info("No robustness scorecard is available for the selected settings.")
        else:
            st.markdown("#### Robustness scorecard")
            scorecard_cols = [
                "sample_mode",
                "inflation_measure_label",
                "fred_series_id",
                "paper_exact",
                "baseline_method",
                "baseline_live_safe",
                "baseline_label",
                "model",
                "horizon_months",
                "threshold_pp",
                "count",
                "mae",
                "rmse",
                "directional_accuracy",
                "mae_improvement_vs_no_change_pct",
                "rmse_improvement_vs_no_change_pct",
                "mae_improvement_vs_mean_reversion_pct",
                "rmse_improvement_vs_mean_reversion_pct",
                "rank_by_mae",
                "rank_by_rmse",
            ]
            st.dataframe(scorecard.loc[:, scorecard_cols], width="stretch")

            st.markdown("#### TINF/regime verdict across settings")
            verdict_cols = [
                "sample_mode",
                "inflation_measure_label",
                "fred_series_id",
                "paper_exact",
                "baseline_method",
                "baseline_live_safe",
                "baseline_label",
                "horizon_months",
                "threshold_pp",
                "count",
                "tinf_mae",
                "tinf_rmse",
                "tinf_directional_accuracy",
                "tinf_rank_by_mae",
                "tinf_rank_by_rmse",
                "beats_no_change_mae",
                "beats_no_change_rmse",
                "beats_mean_reversion_mae",
                "beats_mean_reversion_rmse",
                "beats_ar1_mae",
                "beats_ar1_rmse",
            ]
            st.dataframe(verdict.loc[:, verdict_cols], width="stretch")

            st.markdown("#### Aggregate TINF/regime win rates")
            st.caption(
                "Win rates summarize how often TINF/regime beats each benchmark across the "
                "visible horizon and threshold grid. They are diagnostics, not a setting "
                "selection rule."
            )
            st.dataframe(win_rates, width="stretch")

    st.subheader("Baseline robustness quick comparison")
    rows = []
    for method in BASELINE_META:
        temp = add_transitory_inflation_features(raw, baseline_method=method)
        snap = latest_signal_snapshot(temp)
        if snap.get("available"):
            rows.append(
                {
                    "baseline_method": method,
                    "live_safe": BASELINE_META[method].live_safe,
                    "date": snap["date"],
                    "tinf_4m": snap["tinf_4m"],
                    "percentile": snap["tinf_4m_percentile"],
                    "regime": snap["regime"],
                    "short_term_pressure": validation_mod.pressure_label(
                        snap["term_structure"]
                    ),
                }
            )
    st.dataframe(pd.DataFrame(rows), width="stretch")
    section_notes(
        "Whether today's signal survives changing the baseline definition: the latest snapshot "
        "(date, TINF 4M, percentile, regime, short-term pressure) recomputed under every baseline, "
        "alongside each baseline's live-safe flag.",
        "A regime call that agrees across the live-safe rows (rolling_36_shifted, "
        "expanding_shifted, fed_target) is robust; if it flips between baselines, the conclusion "
        "is baseline-dependent and must be quoted together with its baseline. The full_sample and "
        "rolling_36_unshifted rows are ex-post references only, never live signals. Dates can "
        "differ by row because each baseline has different warm-up requirements.",
    )

    st.subheader("Stationarity diagnostics for selected TINF 4M")
    st.dataframe(stationarity_diagnostics(df["tinf_4m"]), width="stretch")
    section_notes(
        "Whether the selected TINF 4M series is statistically mean-reverting over this sample, "
        "using two complementary tests: ADF (null hypothesis: unit root) and KPSS (null "
        "hypothesis: stationary).",
        "The clean supporting case is an ADF p-value below 0.05 (reject unit root) together with "
        "a KPSS p-value above 0.05 (fail to reject stationarity); then the AR(1)/decay machinery "
        "rests on solid ground. Mixed outcomes are common for persistent series and mean the "
        "evidence is ambiguous. KPSS p-values are table-bounded, so extreme results are reported "
        "at the bound. Non-stationarity would undermine the convergence estimates far more than "
        "the descriptive tables.",
    )

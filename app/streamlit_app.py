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

from transitory_inflation import data as macro_data
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
from transitory_inflation.plots import (
    cpi_vs_baseline_figure,
    decay_curve_figure,
    rolling_rho_figure,
    tinf_term_structure_figure,
)
from transitory_inflation.report import build_trader_report

if not hasattr(macro_data, "load_macro_data_for_mode_with_status"):
    macro_data = importlib.reload(macro_data)
if not hasattr(validation_mod, "forward_outcome_summary_by_regime_and_pressure") or not hasattr(
    validation_mod, "threshold_sensitivity_summary"
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


def pressure_label(term_structure: object) -> str:
    """Map internal TINF horizon ordering to clearer UI wording."""

    labels = {
        "accelerating": "firming",
        "decelerating": "cooling",
        "mixed": "mixed",
    }
    return labels.get(str(term_structure), "mixed")


def date_label(date: pd.Timestamp | None) -> str:
    """Format optional dates for dashboard status text."""

    if date is None:
        return "unknown"
    return str(pd.to_datetime(date).date())


def latest_valid_date(
    df: pd.DataFrame,
    value_col: str = "inflation_yoy",
    date_col: str = "date",
) -> pd.Timestamp | None:
    """Return the latest date where a value column is actually available."""

    dates = pd.to_datetime(df.loc[df[value_col].notna(), date_col])
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())


def signal_conclusion(snapshot: dict[str, object]) -> tuple[str, ...]:
    """Concise current-signal interpretation for the report tab."""

    regime = str(snapshot.get("regime", "neutral"))
    pressure = pressure_label(snapshot.get("term_structure", "mixed"))
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
        baseline_read = (
            f"running {abs(epsilon):.2f}pp {side} the selected mean-reversion baseline"
        )

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

meta = BASELINE_META[baseline_method]
mode_meta = SAMPLE_MODES[sample_mode]

@st.cache_data(show_spinner=True)
def get_data(sample_mode: str):
    return macro_data.load_macro_data_for_mode_with_status(sample_mode)

load_result = get_data(sample_mode)
raw = load_result.data

raw_cache_end = pd.to_datetime(raw["date"].max()).date() if not raw.empty else "unknown"
latest_cpi_yoy = latest_valid_date(raw, "inflation_yoy")
if load_result.data_source_used == "fred_csv":
    st.warning(
        "Official FRED API was unavailable or not configured, so the dashboard is using "
        "public FRED CSV data."
    )
elif load_result.data_source_used == "cached_fred":
    st.warning(
        f"Live FRED fetch failed, so the dashboard is using cached data from "
        f"`{load_result.cache_file_used}` with CPI YoY through {date_label(latest_cpi_yoy)} "
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
span_end = date_label(latest_valid_date(raw, "inflation_yoy"))
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
    date_label(pd.to_datetime(status_snapshot["date"]))
    if status_snapshot.get("available")
    else "unavailable"
)
latest_cpi_observation = latest_valid_date(raw, "cpi_level")
imputation_applied = bool(
    "cpi_imputed" in raw.columns and raw["cpi_imputed"].fillna(False).astype(bool).any()
)
st.caption(
    "Data status: "
    f"data_source_used={load_result.data_source_used}; "
    f"live_fetch_status={load_result.live_fetch_status}; "
    f"cache_file_used={load_result.cache_file_used or 'n/a'}; "
    f"raw_data_end={raw_span_end}; "
    f"latest_cpi_observation_date={date_label(latest_cpi_observation)}; "
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

tab_signal, tab_validation, tab_framework, tab_decay, tab_robustness, tab_report = st.tabs(
    [
        "Current Macro Signal",
        "Historical Signal Validation",
        "Paper Framework",
        "Decay / Convergence",
        "Robustness",
        "Report",
    ]
)

with tab_signal:
    snapshot = latest_signal_snapshot(df)
    if not snapshot.get("available"):
        st.warning(snapshot.get("reason", "No signal available."))
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CPI YoY", f"{snapshot['inflation_yoy']:.2f}%")
        col2.metric("Baseline", f"{snapshot['baseline']:.2f}%")
        col3.metric("TINF 4M", f"{snapshot['tinf_4m']:.2f} pp")
        col4.metric("TINF 4M Percentile", f"{snapshot['tinf_4m_percentile']:.1f}%")

        st.write(
            f"**Regime:** {snapshot['regime']} | "
            f"**Short-term pressure:** {pressure_label(snapshot['term_structure'])} | "
            f"**Date:** {pd.to_datetime(snapshot['date']).date()}"
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

    st.plotly_chart(cpi_vs_baseline_figure(df), use_container_width=True)
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

    st.plotly_chart(tinf_term_structure_figure(df), use_container_width=True)
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
    st.markdown(
        "This tab tests whether the current-month signal historically contained forward information. "
        "Future CPI outcomes are used only for validation, not signal construction."
    )
    st.warning(
        "Historical validation is live-like only under rolling_36_shifted or expanding_shifted baselines."
    )
    st.warning("full_sample is ex-post and should not be used to judge live signal success.")
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

    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        validation_horizon = st.selectbox(
            "Horizon",
            options=[6, 12, 24, 36],
            index=1,
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

    validation_df = validation_mod.build_historical_validation_frame(
        df,
        epsilon_threshold_pp=float(outcome_threshold),
        fed_target_threshold_pp=float(outcome_threshold),
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
    sensitivity_summary = validation_mod.threshold_sensitivity_summary(
        df,
        horizon=validation_horizon,
        thresholds=(0.25, 0.50, 0.75, 1.00),
    )

    st.markdown("#### Combined regime x short-term pressure summary")
    st.caption(
        "This table crosses the historical regime with short-term pressure. It is often the "
        "most useful Phase 1 cut because it separates elevated-and-rising inflation pressure "
        "from elevated-but-cooling pressure. Counts matter: small groups can produce unstable "
        "rates."
    )
    st.dataframe(combined_summary, use_container_width=True)

    st.markdown("#### Selected summary")
    if validation_group == "regime":
        st.dataframe(regime_summary, use_container_width=True)
    else:
        st.dataframe(pressure_summary, use_container_width=True)

    st.markdown("#### Forward outcome summary by regime")
    st.dataframe(regime_summary, use_container_width=True)

    st.markdown("#### Forward outcome summary by short-term pressure")
    st.dataframe(pressure_summary, use_container_width=True)

    st.markdown(f"#### Threshold sensitivity ({validation_horizon} months)")
    st.caption(
        "This table recomputes outcome labels at fixed thresholds of 0.25, 0.50, 0.75, "
        "and 1.00 pp. It is sensitivity analysis only, not threshold optimization. Phase 2 "
        "benchmark comparison is still required before treating hit rates as forecast skill."
    )
    st.dataframe(sensitivity_summary, use_container_width=True)

    st.markdown(f"#### Regime transition matrix ({validation_horizon} months)")
    transition = validation_mod.regime_transition_matrix(
        validation_df, horizon=validation_horizon
    )
    if transition.empty:
        st.info("No valid regime transitions are available for the selected horizon.")
    else:
        st.dataframe(transition, use_container_width=True)

    st.markdown("#### False positive / false negative examples")
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
            st.dataframe(table, use_container_width=True)

with tab_report:
    st.subheader("Report")
    report = build_trader_report(raw, df, baseline_method=baseline_method, sample_mode=sample_mode)
    if not report.available:
        st.warning(report.reason or "Report unavailable.")
    else:
        if not meta.live_safe:
            st.warning(
                "Selected baseline is ex-post. This report describes history, not a live signal — "
                "switch to a live-safe baseline for current-signal use."
            )
        st.markdown(f"**{report.headline}**")

        st.markdown("#### 1. Where inflation stands")
        st.markdown("\n".join(f"- {line}" for line in report.state_lines))

        st.markdown("#### 2. Persistence and model-implied normalization")
        st.markdown("\n".join(f"- {line}" for line in report.persistence_lines))

        st.markdown("#### 3. Robustness across live-safe baselines")
        st.markdown("\n".join(f"- {line}" for line in report.robustness_lines))

        st.markdown("#### 4. What to watch")
        st.markdown("\n".join(f"- {line}" for line in report.watch_lines))

        report_snapshot = latest_signal_snapshot(df)
        st.markdown("#### Concluding remarks")
        st.markdown("\n".join(f"- {line}" for line in signal_conclusion(report_snapshot)))

        section_notes(
            "How to read today's transitory-inflation signal: where inflation sits against its "
            "baseline, how persistent the framework estimates the deviation to be, whether the "
            "call survives live-safe baseline changes, and what upcoming data would confirm or "
            "invalidate it. Every number is generated from the currently selected sample mode and "
            "baseline.",
            "Read it top-down: sections 1-3 are computed facts from this dashboard; section 4 is a "
            "mechanical watch list and approximate flip level for the next CPI print; the conclusion "
            "summarizes the current regime read without turning it into a portfolio instruction. If "
            "the headline regime conflicts with the robustness section, trust the disagreement: it "
            "means the signal is baseline-dependent.",
        )

with tab_framework:
    st.subheader("Paper-style descriptive tables")
    table_cols = ["inflation_yoy", "tinf_4m", "tinf_8m", "tinf_12m", "tbill_3m"]
    st.dataframe(summary_stats(df, table_cols), use_container_width=True)
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
    st.dataframe(correlation_matrix(df, table_cols), use_container_width=True)
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
        st.dataframe(run_paper_style_regressions(df), use_container_width=True)
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
        st.dataframe(ljung_box_table(df["tinf_4m"], lags=40), use_container_width=True)
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
    windows = st.multiselect("Rolling windows", options=[24, 30, 36, 48, 60, 84, 120], default=[24, 30, 36, 60])
    if windows:
        rho_df, decay_df = decay_summaries_for_windows(df, windows=tuple(windows), value_col="tinf_4m")
        st.plotly_chart(rolling_rho_figure(rho_df), use_container_width=True)
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

        st.dataframe(decay_df, use_container_width=True)
        section_notes(
            "The paper's convergence arithmetic for each selected window: the latest rho (rho_T), "
            "an AR(1) fit of the rolling-rho series itself (mu and c), the implied share of a "
            "current deviation gone after 6 and 12 months, the time to 95% convergence (t* in "
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
            st.plotly_chart(decay_curve_figure(curve), use_container_width=True)
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
                    "short_term_pressure": pressure_label(snap["term_structure"]),
                }
            )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
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
    st.dataframe(stationarity_diagnostics(df["tinf_4m"]), use_container_width=True)
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

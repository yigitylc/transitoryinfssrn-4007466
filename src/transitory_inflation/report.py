from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from . import benchmarks as benchmark_mod
from . import market_linkage as market_linkage_mod
from . import robustness as robustness_mod
from . import validation as validation_mod
from .dashboard import current_signal_imputation_notice
from .data import (
    HEADLINE_INFLATION_MEASURE,
    INFLATION_MEASURES,
    date_label,
    latest_valid_observation_date,
)
from .features import BASELINE_META, add_transitory_inflation_features, latest_signal_snapshot
from .market_data import available_market_variables
from .models import decay_summaries_for_windows

LIVE_SAFE_BASELINES: tuple[str, ...] = tuple(
    name for name, meta in BASELINE_META.items() if meta.live_safe
)
DEFAULT_REPORT_HORIZONS: tuple[int, ...] = (3, 6, 12, 24, 36)
DEFAULT_REPORT_THRESHOLD_PP = 0.50
DEFAULT_REPORT_INFLATION_MEASURES: tuple[str, ...] = tuple(INFLATION_MEASURES)
WEAK_EVIDENCE_MIN_COUNT = 30

MODEL_LABELS: dict[str, str] = {
    "no_change": "no-change CPI forecast",
    "cpi_persistence": "CPI persistence",
    "mean_reversion": "mean reversion to baseline",
    "ar1": "AR(1)",
    "tinf_regime_bucket": "TINF/regime bucket",
}

# Stylized macro-trader regime priors. Descriptive interpretation only: these
# are not validated inside this project yet and must never be rendered as
# trade recommendations (see docs/01_RESEARCH_SPEC.md, trader research mode).
# Future/experimental: the trader layer (REGIME_PLAYBOOK + build_trader_report)
# is intentionally NOT wired into the Streamlit app, which ships only the Macro
# Research Report (build_macro_research_report).
REGIME_PLAYBOOK: dict[str, tuple[tuple[str, str], ...]] = {
    "elevated rising": (
        (
            "Macro read",
            "Inflation is running persistently above baseline and the pressure is still "
            "building. Historically this is the hawkish-repricing regime: policy-tightening "
            "expectations get pulled forward and CPI prints carry the most event risk.",
        ),
        (
            "Rates & curve",
            "Front-end yields typically lead moves higher; curves flatten while policy "
            "credibility holds and bear-steepen when it slips. Long-duration exposure carries "
            "the most regime risk.",
        ),
        (
            "Inflation markets",
            "Breakevens and inflation swaps tend to stay bid; TIPS historically outperform "
            "matched-maturity nominals while the deviation keeps widening.",
        ),
        (
            "Equities, FX & vol",
            "Long-duration equity (unprofitable growth) usually de-rates as discount rates "
            "reprice; value, energy and real-asset exposure historically cope better. Hawkish "
            "CPI surprises tend to support the currency, and CPI-day volatility trades rich.",
        ),
    ),
    "elevated falling": (
        (
            "Macro read",
            "Inflation is still well above baseline but the impulse is fading - the classic "
            "peak-inflation regime, where disinflation dynamics start to matter before levels "
            "normalize.",
        ),
        (
            "Rates & curve",
            "Duration stabilizes and rallies on downside CPI surprises; the front end prices "
            "cuts only gradually. Steepening pressure historically builds later in this phase.",
        ),
        (
            "Inflation markets",
            "Breakevens tend to compress from the front end first; long TIPS lose their carry "
            "advantage as the deviation narrows.",
        ),
        (
            "Equities, FX & vol",
            "Equity leadership often rotates back toward duration-sensitive growth; FX impact "
            "flips as hike premia get priced out, and CPI-day volatility fades from its highs.",
        ),
    ),
    "neutral": (
        (
            "Macro read",
            "Inflation is tracking its baseline - deviations are small or short-lived, so "
            "inflation stops being the dominant macro driver; growth and positioning data "
            "take over.",
        ),
        (
            "Rates & curve",
            "Rates tend to trade ranges; carry and roll-down dominate directional inflation "
            "views.",
        ),
        (
            "Inflation markets",
            "Breakevens hover near policy-consistent levels; inflation-market alpha is thin.",
        ),
        (
            "Equities, FX & vol",
            "Equity factor leadership decouples from inflation; CPI releases are second-tier "
            "events unless a print is large enough to threaten the regime itself.",
        ),
    ),
    "disinflationary": (
        (
            "Macro read",
            "Inflation is running persistently below baseline. Historically the "
            "dovish-repricing regime: markets price easier policy, and deflation tails fatten "
            "if the undershoot deepens.",
        ),
        (
            "Rates & curve",
            "Duration historically performs and the front end leads as cuts get priced. The "
            "main risk to that pattern is a growth re-acceleration, not inflation.",
        ),
        (
            "Inflation markets",
            "Breakevens compress and can undershoot; TIPS underperform nominals, and deflation "
            "floors in inflation options gain attention if the undershoot extends.",
        ),
        (
            "Equities, FX & vol",
            "Falling discount rates support long-duration assets only while growth holds up - "
            "the regime splits into soft-landing and demand-weakness variants, so growth data "
            "decide the equity read. Gold trades off real yields.",
        ),
    ),
}

TERM_STRUCTURE_NOTES: dict[str, str] = {
    "accelerating": (
        "Term structure is accelerating (4M > 8M > 12M): the newest deviations are the "
        "largest, so the regime is young rather than fading - treat normalization estimates "
        "as early."
    ),
    "decelerating": (
        "Term structure is decelerating (4M < 8M < 12M): the freshest pressure sits below the "
        "longer averages - momentum favors convergence toward baseline."
    ),
    "mixed": (
        "Term structure is mixed: horizons disagree, which usually marks transition phases - "
        "wait for alignment before leaning hard on the regime label."
    ),
}


def _ordinal(n: float) -> str:
    """Format a number as an English ordinal, e.g. 51 -> '51st'."""

    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


@dataclass(frozen=True)
class TraderReport:
    """Structured macro-trader briefing built from computed signals."""

    available: bool
    reason: str | None = None
    as_of: str | None = None
    headline: str = ""
    state_lines: tuple[str, ...] = field(default_factory=tuple)
    persistence_lines: tuple[str, ...] = field(default_factory=tuple)
    robustness_lines: tuple[str, ...] = field(default_factory=tuple)
    playbook: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    watch_lines: tuple[str, ...] = field(default_factory=tuple)
    caveats: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MacroResearchReport:
    """Phase 5 research report assembled from validated dashboard layers."""

    available: bool
    reason: str | None = None
    as_of: str | None = None
    headline: str = ""
    current_signal_notice: str | None = None
    current_regime_lines: tuple[str, ...] = field(default_factory=tuple)
    signal_confidence_lines: tuple[str, ...] = field(default_factory=tuple)
    robustness_lines: tuple[str, ...] = field(default_factory=tuple)
    historical_analog_lines: tuple[str, ...] = field(default_factory=tuple)
    market_linkage_lines: tuple[str, ...] = field(default_factory=tuple)
    caveats: tuple[str, ...] = field(default_factory=tuple)
    watchlist: tuple[str, ...] = field(default_factory=tuple)
    current_regime_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_comparisons: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness_verdict: pd.DataFrame = field(default_factory=pd.DataFrame)
    robustness_win_rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    inflation_measure_availability: pd.DataFrame = field(default_factory=pd.DataFrame)
    historical_analogs: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_channel_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def _status_get(status: object | None, key: str, default: object = None) -> object:
    if status is None:
        return default
    if isinstance(status, Mapping):
        return status.get(key, default)
    return getattr(status, key, default)


def _tinf_state(value: object, near_zero_threshold: float = 0.05) -> str:
    if pd.isna(value):
        return "unavailable"
    numeric = float(value)
    if numeric > near_zero_threshold:
        return "positive"
    if numeric < -near_zero_threshold:
        return "negative"
    return "near zero"


def _weak_evidence_note(count: int) -> str:
    return "Fewer than 30 complete observations; interpret cautiously." if count < 30 else ""


def _benchmark_tables(
    featured: pd.DataFrame,
    horizons: tuple[int, ...],
    threshold_pp: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, object]] = []

    for horizon in horizons:
        _, metrics, _, _ = benchmark_mod.benchmark_comparison_tables(
            featured,
            horizon=horizon,
            threshold_pp=threshold_pp,
        )
        if metrics.empty:
            continue
        current_metrics = metrics.copy()
        current_metrics.insert(0, "threshold_pp", threshold_pp)
        metrics_frames.append(current_metrics)

        by_model = current_metrics.set_index("model", drop=False)
        if "tinf_regime_bucket" not in by_model.index:
            continue
        tinf = by_model.loc["tinf_regime_bucket"]
        for model in ("no_change", "cpi_persistence", "mean_reversion", "ar1"):
            if model not in by_model.index:
                continue
            other = by_model.loc[model]
            comparison_rows.append(
                {
                    "horizon_months": horizon,
                    "threshold_pp": threshold_pp,
                    "comparison_model": model,
                    "comparison_label": MODEL_LABELS.get(model, model),
                    "tinf_count": int(tinf["count"]),
                    "comparison_count": int(other["count"]),
                    "tinf_mae": float(tinf["mae"]),
                    "comparison_mae": float(other["mae"]),
                    "beats_mae": bool(tinf["mae"] < other["mae"]),
                    "tinf_rmse": float(tinf["rmse"]),
                    "comparison_rmse": float(other["rmse"]),
                    "beats_rmse": bool(tinf["rmse"] < other["rmse"]),
                }
            )

    metrics_all = pd.concat(metrics_frames, ignore_index=True) if metrics_frames else pd.DataFrame()
    return metrics_all, pd.DataFrame(comparison_rows)


def _benchmark_lines(comparisons: pd.DataFrame) -> tuple[str, ...]:
    if comparisons.empty:
        return ("No benchmark comparison is available for the selected report settings.",)

    lines: list[str] = []
    for model in ("no_change", "cpi_persistence", "mean_reversion", "ar1"):
        group = comparisons.loc[comparisons["comparison_model"] == model]
        if group.empty:
            continue
        mae_wins = int(group["beats_mae"].sum())
        rmse_wins = int(group["beats_rmse"].sum())
        total = int(len(group))
        label = MODEL_LABELS.get(model, model)
        lines.append(
            f"TINF/regime beats {label} in {mae_wins}/{total} tested horizons by MAE "
            f"and {rmse_wins}/{total} by RMSE."
        )

    ar1 = comparisons.loc[comparisons["comparison_model"] == "ar1"]
    if not ar1.empty:
        ar1_beats_mae = int((~ar1["beats_mae"]).sum())
        ar1_beats_rmse = int((~ar1["beats_rmse"]).sum())
        if ar1_beats_mae or ar1_beats_rmse:
            lines.append(
                "AR(1) has lower CPI point-forecast error than TINF/regime in "
                f"{ar1_beats_mae}/{len(ar1)} horizons by MAE and "
                f"{ar1_beats_rmse}/{len(ar1)} by RMSE; do not overstate TINF/regime "
                "as a pure CPI forecast model."
            )
    lines.append(
        "Best framing: TINF/regime is an interpretable inflation-regime diagnostic; "
        "it does not automatically dominate simple point-forecast benchmarks."
    )
    return tuple(lines)


def _robustness_tables(
    raw_frames_by_sample_mode: Mapping[str, pd.DataFrame],
    baselines: tuple[str, ...],
    inflation_measures: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not raw_frames_by_sample_mode:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    availability = robustness_mod.inflation_measure_availability(
        raw_frames_by_sample_mode,
        inflation_measures=inflation_measures,
    )
    scorecard, verdict, win_rates = robustness_mod.robustness_tables(
        raw_frames_by_sample_mode,
        baseline_methods=baselines,
        inflation_measures=inflation_measures,
    )
    return availability, verdict, win_rates


def _robustness_lines(
    verdict: pd.DataFrame,
    win_rates: pd.DataFrame,
    availability: pd.DataFrame,
) -> tuple[str, ...]:
    lines: list[str] = []
    if not availability.empty:
        missing = availability.loc[~availability["available"], "inflation_measure_label"].unique()
        if len(missing):
            lines.append(
                "Missing inflation measures are disclosed and skipped, not backfilled: "
                + ", ".join(str(value) for value in missing)
                + "."
            )

    if verdict.empty:
        lines.append("No robustness verdict is available for the selected report settings.")
        return tuple(lines)

    settings_count = int(len(verdict))
    best_mae = float(verdict["tinf_best_by_mae"].dropna().astype(float).mean())
    best_rmse = float(verdict["tinf_best_by_rmse"].dropna().astype(float).mean())
    beats_ar1_mae = float(verdict["beats_ar1_mae"].dropna().astype(float).mean())
    beats_ar1_rmse = float(verdict["beats_ar1_rmse"].dropna().astype(float).mean())
    lines.append(
        f"Across {settings_count} robustness settings, TINF/regime is best by MAE "
        f"{best_mae:.0%} of the time and best by RMSE {best_rmse:.0%} of the time."
    )
    lines.append(
        f"Against AR(1), TINF/regime wins {beats_ar1_mae:.0%} of settings by MAE "
        f"and {beats_ar1_rmse:.0%} by RMSE."
    )

    measures = ", ".join(str(value) for value in verdict["inflation_measure_label"].dropna().unique())
    baselines = ", ".join(str(value) for value in verdict["baseline_method"].dropna().unique())
    lines.append(f"Covered inflation measures: {measures or 'none'}; baselines: {baselines or 'none'}.")
    if not verdict["baseline_live_safe"].dropna().astype(bool).all():
        lines.append(
            "Rows using full_sample or other non-live-safe baselines are ex-post / paper-style "
            "checks, not live-signal evidence."
        )
    if not win_rates.empty:
        lines.append(
            "Aggregate win-rate rows are diagnostics across the fixed grid; they are not a "
            "threshold-optimization rule."
        )
    return tuple(lines)


def _historical_analog_table(
    featured: pd.DataFrame,
    snapshot: dict[str, object],
    market_monthly: pd.DataFrame | None,
    horizons: tuple[int, ...],
    threshold_pp: float,
) -> pd.DataFrame:
    if not snapshot.get("available"):
        return pd.DataFrame()

    validation_df = validation_mod.build_historical_validation_frame(
        featured,
        forward_horizons=horizons,
        label_horizons=horizons,
        epsilon_threshold_pp=threshold_pp,
        fed_target_threshold_pp=threshold_pp,
    )
    validation_df["historical_tinf_state"] = validation_df["tinf_4m"].map(_tinf_state)

    current_regime = str(snapshot["regime"])
    current_pressure = validation_mod.pressure_label(str(snapshot.get("term_structure", "mixed")))
    current_tinf_state = _tinf_state(snapshot["tinf_4m"])
    analog_group = f"{current_regime} / {current_pressure} / {current_tinf_state} TINF"
    analog_mask = (
        (validation_df["historical_regime"] == current_regime)
        & (validation_df["historical_short_term_pressure"] == current_pressure)
        & (validation_df["historical_tinf_state"] == current_tinf_state)
    )

    market_panel = pd.DataFrame()
    market_variables: tuple[str, ...] = ()
    if market_monthly is not None and not market_monthly.empty:
        market_variables = available_market_variables(market_monthly)
        if market_variables:
            market_panel = market_linkage_mod.build_market_linkage_panel(
                validation_df,
                market_monthly,
                horizons=horizons,
            )
            market_panel["historical_tinf_state"] = market_panel["tinf_4m"].map(_tinf_state)

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        suffix = f"{horizon}m"
        cpi_col = f"cpi_yoy_change_{suffix}"
        epsilon_col = f"epsilon_change_{suffix}"
        if cpi_col not in validation_df.columns or epsilon_col not in validation_df.columns:
            continue
        inflation_rows = validation_df.loc[analog_mask & validation_df[cpi_col].notna()]
        inflation_count = int(len(inflation_rows))
        inflation_summary = {
            "analog_group": analog_group,
            "conditioning_signal_date": date_label(snapshot["date"]),
            "historical_regime": current_regime,
            "historical_short_term_pressure": current_pressure,
            "historical_tinf_state": current_tinf_state,
            "horizon_months": horizon,
            "future_inflation_count": inflation_count,
            "avg_future_cpi_yoy_change": float(inflation_rows[cpi_col].mean())
            if inflation_count
            else float("nan"),
            "avg_future_epsilon_change": float(inflation_rows[epsilon_col].mean())
            if inflation_count
            else float("nan"),
        }

        if market_panel.empty:
            count = inflation_count
            rows.append(
                {
                    **inflation_summary,
                    "market_channel": "none",
                    "market_variable": "inflation_only",
                    "count": count,
                    "avg_market_change_bp": float("nan"),
                    "median_market_change_bp": float("nan"),
                    "weak_evidence": count < WEAK_EVIDENCE_MIN_COUNT,
                    "evidence_note": _weak_evidence_note(count),
                }
            )
            continue

        market_mask = (
            (market_panel["historical_regime"] == current_regime)
            & (market_panel["historical_short_term_pressure"] == current_pressure)
            & (market_panel["historical_tinf_state"] == current_tinf_state)
        )
        for channel, variables in market_linkage_mod.MARKET_CHANNELS.items():
            for variable in variables:
                if variable not in market_variables:
                    continue
                change_col = f"{variable}_change_{suffix}_bp"
                if change_col not in market_panel.columns:
                    continue
                changes = pd.to_numeric(
                    market_panel.loc[market_mask, change_col],
                    errors="coerce",
                ).dropna()
                count = int(len(changes))
                rows.append(
                    {
                        **inflation_summary,
                        "market_channel": channel,
                        "market_variable": variable,
                        "count": count,
                        "avg_market_change_bp": float(changes.mean()) if count else float("nan"),
                        "median_market_change_bp": float(changes.median()) if count else float("nan"),
                        "weak_evidence": count < WEAK_EVIDENCE_MIN_COUNT,
                        "evidence_note": _weak_evidence_note(count),
                    }
                )

    return pd.DataFrame(rows)


def _historical_analog_lines(analogs: pd.DataFrame) -> tuple[str, ...]:
    if analogs.empty:
        return ("No historical analog table is available for the current regime/pressure state.",)
    group_name = str(analogs["analog_group"].iloc[0])
    conditioning_date = str(analogs["conditioning_signal_date"].iloc[0])
    counts = pd.to_numeric(analogs["future_inflation_count"], errors="coerce").dropna()
    weak_rows = int(analogs["weak_evidence"].fillna(False).astype(bool).sum())
    lines = [
        f"Observed-only analog conditioning signal as of {conditioning_date}: {group_name}.",
        f"Future CPI and epsilon changes are summarized across {int(counts.max()) if len(counts) else 0} "
        "matching historical signal dates at the richest horizon sample.",
    ]
    if weak_rows:
        lines.append(f"{weak_rows} analog-market rows are weak evidence with fewer than 30 observations.")
    return tuple(lines)


def _market_summary(
    featured: pd.DataFrame,
    market_monthly: pd.DataFrame | None,
    horizons: tuple[int, ...],
) -> tuple[tuple[str, ...], pd.DataFrame]:
    base_lines = [
        "A positive forward change means the market variable rose after the signal date.",
        "Nominal yields, breakevens, and real yields represent different macro channels.",
        "Market linkage is descriptive history; it is not a live trading signal.",
    ]
    if market_monthly is None or market_monthly.empty or not available_market_variables(market_monthly):
        return tuple([*base_lines, "No approved market-rate data are available for this report run."]), pd.DataFrame()

    tables = market_linkage_mod.build_market_linkage_tables(
        featured,
        market_monthly,
        horizons=horizons,
    )
    channel_summary = tables.channel_summary_by_regime
    if channel_summary.empty:
        return tuple([*base_lines, "No channel summary is available for the current market data."]), channel_summary
    weak = int(channel_summary["weak_evidence"].fillna(False).astype(bool).sum())
    lines = [
        *base_lines,
        f"Approved channels included: {', '.join(market_linkage_mod.MARKET_CHANNELS)}.",
    ]
    if weak:
        lines.append(f"{weak} market-linkage rows are weak evidence with fewer than 30 observations.")
    return tuple(lines), channel_summary


def _current_regime_section(
    raw: pd.DataFrame,
    featured: pd.DataFrame,
    snapshot: dict[str, object],
    baseline_method: str,
    sample_mode: str,
    macro_status: object | None,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    if not snapshot.get("available"):
        return (snapshot.get("reason", "No complete current signal is available."),), pd.DataFrame()

    measure = INFLATION_MEASURES[HEADLINE_INFLATION_MEASURE]
    official_level_col = (
        measure.observed_level_col if measure.observed_level_col in raw.columns else measure.level_col
    )
    latest_cpi_observation = latest_valid_observation_date(raw, official_level_col)
    latest_cpi_yoy = latest_valid_observation_date(raw, "inflation_yoy")
    raw_end = pd.to_datetime(raw["date"].max()).date() if "date" in raw.columns and not raw.empty else "unknown"
    imputation_applied = bool(snapshot.get("uses_imputed_input", False))
    pressure = validation_mod.pressure_label(str(snapshot.get("term_structure", "mixed")))
    table = pd.DataFrame(
        [
            {
                "latest_valid_signal_date": date_label(snapshot["date"]),
                "data_source_used": _status_get(macro_status, "data_source_used", "unknown"),
                "inflation_measure": measure.label,
                "fred_series_id": measure.series_id,
                "sample_mode": sample_mode,
                "baseline_method": baseline_method,
                "current_cpi_yoy": float(snapshot["inflation_yoy"]),
                "baseline": float(snapshot["baseline"]),
                "epsilon": float(snapshot["epsilon"]),
                "tinf_4m": float(snapshot["tinf_4m"]),
                "tinf_8m": float(snapshot["tinf_8m"]),
                "tinf_12m": float(snapshot["tinf_12m"]),
                "current_regime": snapshot["regime"],
                "current_short_term_pressure": pressure,
                "raw_data_end": raw_end,
                "latest_cpi_observation_date": date_label(latest_cpi_observation),
                "latest_valid_cpi_yoy_date": date_label(latest_cpi_yoy),
                "cpi_imputation_applied": imputation_applied,
                "imputation_policy": snapshot.get("imputation_policy", "unspecified"),
                "current_signal_uses_imputed_input": imputation_applied,
                "current_signal_observed_only_eligible": bool(
                    snapshot.get("observed_only_eligible", True)
                ),
            }
        ]
    )
    lines = (
        f"As of {date_label(snapshot['date'])}, headline CPI YoY is "
        f"{float(snapshot['inflation_yoy']):.2f}% versus a "
        f"{float(snapshot['baseline']):.2f}% {baseline_method} baseline.",
        f"TINF 4M is {float(snapshot['tinf_4m']):+.2f}pp, with TINF 8M "
        f"{float(snapshot['tinf_8m']):+.2f}pp and TINF 12M "
        f"{float(snapshot['tinf_12m']):+.2f}pp.",
        f"Current regime is '{snapshot['regime']}' and short-term pressure is '{pressure}'.",
        f"Data freshness: source={_status_get(macro_status, 'data_source_used', 'unknown')}; "
        f"raw data through {raw_end}; latest CPI observation {date_label(latest_cpi_observation)}; "
        f"latest CPI YoY {date_label(latest_cpi_yoy)}; imputation applied={imputation_applied}.",
    )
    return lines, table


def _watchlist(raw: pd.DataFrame, featured: pd.DataFrame, baseline_method: str) -> tuple[str, ...]:
    snapshot = latest_signal_snapshot(featured)
    latest_cpi = latest_valid_observation_date(raw, "inflation_yoy")
    latest_pce = latest_valid_observation_date(raw, "pce_yoy")
    threshold = next_print_flip_threshold(featured, baseline_method)

    lines = [
        f"Next CPI update: confirm whether FRED CPI YoY advances beyond {date_label(latest_cpi)}.",
        "Monitor whether TINF 4M rises or falls versus TINF 8M and TINF 12M.",
        "Monitor whether elevated positive inflation pressure is resolving, persisting, or overshooting below baseline.",
        "Watch 2Y and 10Y Treasury yield reactions around new inflation data.",
        "Watch 5Y/10Y breakevens and 5Y/10Y real yields separately; they represent different channels.",
        "Re-check benchmark confidence when new observations change the out-of-sample scoring window.",
    ]
    if latest_pce is not None:
        lines.insert(
            1,
            f"Next PCE update: confirm whether FRED PCE inflation advances beyond {date_label(latest_pce)}.",
        )
    else:
        lines.insert(1, "Next PCE update: monitor availability; selected data do not currently expose PCE YoY.")
    if threshold is not None and snapshot.get("available"):
        direction = "below" if float(snapshot["tinf_4m"]) > 0 else "above"
        lines.append(
            f"Approximate TINF flip watch: a next CPI YoY print {direction} "
            f"{threshold:.2f}% would move TINF 4M across zero under {baseline_method}."
        )
    return tuple(lines)


def _caveats(
    baseline_method: str,
    sample_mode: str,
    macro_status: object | None,
    market_status: object | None,
    signal_notice: str | None,
) -> tuple[str, ...]:
    meta = BASELINE_META[baseline_method]
    caveats = [
        "This dashboard is historical macro research, not a trading signal, investment advice, or a buy/sell recommendation.",
        "TINF/regime is an interpretable inflation-regime diagnostic and should not be treated as robustly beating AR(1) as a CPI point-forecast model unless the benchmark tables show that across settings.",
        "Market linkage is descriptive and may not hold out of sample.",
        "Some regime and market buckets may have small sample sizes; rows below 30 observations are flagged as weak evidence.",
        "Live-snapshot regime labels and historical walk-forward validation labels are related "
        "but not identical: historical validation uses walk-forward thresholds while the live "
        "snapshot uses the current configured signal context, so historical analogs are "
        "historical comparisons, not exact regime-identity matches.",
        "Full-sample baselines are ex-post / paper-style and are not live-safe.",
        "Data freshness matters; FRED publication lags can affect the latest usable signal date.",
        "The signal is built on latest-revised FRED data and treats each month's CPI as known "
        "within that month; 'live-safe' here means no full-sample lookahead, not a real-time "
        "data-vintage backtest.",
        "Future market changes are used only for evaluation and descriptive linkage, never for signal construction.",
        f"Computed under sample mode '{sample_mode}' and baseline '{baseline_method}' "
        f"({'live-safe' if meta.live_safe else 'EX-POST / not live-safe'}).",
        f"Macro data source used: {_status_get(macro_status, 'data_source_used', 'unknown')}; "
        f"market data source used: {_status_get(market_status, 'market_data_source_used', 'unknown')}.",
    ]
    if signal_notice:
        caveats.append(signal_notice)
    else:
        caveats.append("No CPI imputation is flagged in the current loaded sample.")
    return tuple(caveats)


def build_macro_research_report(
    raw: pd.DataFrame,
    featured: pd.DataFrame,
    baseline_method: str,
    sample_mode: str,
    macro_status: object | None = None,
    market_monthly: pd.DataFrame | None = None,
    market_status: object | None = None,
    robustness_sample_frames: Mapping[str, pd.DataFrame] | None = None,
    benchmark_horizons: tuple[int, ...] = DEFAULT_REPORT_HORIZONS,
    market_horizons: tuple[int, ...] = DEFAULT_REPORT_HORIZONS,
    threshold_pp: float = DEFAULT_REPORT_THRESHOLD_PP,
    robustness_baselines: tuple[str, ...] = robustness_mod.DEFAULT_ROBUSTNESS_BASELINES,
    robustness_inflation_measures: tuple[str, ...] = DEFAULT_REPORT_INFLATION_MEASURES,
    current_raw: pd.DataFrame | None = None,
    current_featured: pd.DataFrame | None = None,
) -> MacroResearchReport:
    """Build the report with separate current-monitoring and research authorities."""

    if "imputation_policy" in raw.columns:
        research_policies = set(raw["imputation_policy"].dropna().astype(str))
        if research_policies and research_policies != {"observed_only"}:
            raise ValueError("Macro report research inputs must use observed_only data")

    current_raw = raw if current_raw is None else current_raw
    current_featured = featured if current_featured is None else current_featured
    current_snapshot = latest_signal_snapshot(current_featured)
    research_snapshot = latest_signal_snapshot(featured)
    if not current_snapshot.get("available"):
        return MacroResearchReport(
            available=False,
            reason=current_snapshot.get("reason", "No complete current signal is available."),
        )

    current_lines, current_table = _current_regime_section(
        current_raw,
        current_featured,
        current_snapshot,
        baseline_method,
        sample_mode,
        macro_status,
    )
    benchmark_metrics, benchmark_comparisons = _benchmark_tables(
        featured,
        horizons=benchmark_horizons,
        threshold_pp=threshold_pp,
    )
    signal_confidence_lines = _benchmark_lines(benchmark_comparisons)

    robust_frames = robustness_sample_frames or {sample_mode: raw}
    availability, robustness_verdict, robustness_win_rates = _robustness_tables(
        robust_frames,
        baselines=robustness_baselines,
        inflation_measures=robustness_inflation_measures,
    )
    robustness_lines = _robustness_lines(
        robustness_verdict,
        robustness_win_rates,
        availability,
    )

    analogs = _historical_analog_table(
        featured,
        research_snapshot,
        market_monthly=market_monthly,
        horizons=market_horizons,
        threshold_pp=threshold_pp,
    )
    analog_lines = _historical_analog_lines(analogs)
    market_lines, market_summary = _market_summary(
        featured,
        market_monthly=market_monthly,
        horizons=market_horizons,
    )
    signal_notice = current_signal_imputation_notice(current_snapshot)
    as_of = date_label(current_snapshot["date"])
    headline = (
        f"As of {as_of}: {current_snapshot['regime']} inflation regime, "
        f"short-term pressure "
        f"{validation_mod.pressure_label(str(current_snapshot.get('term_structure', 'mixed')))}, "
        f"TINF 4M {float(current_snapshot['tinf_4m']):+.2f}pp."
    )

    return MacroResearchReport(
        available=True,
        as_of=as_of,
        headline=headline,
        current_signal_notice=signal_notice,
        current_regime_lines=current_lines,
        signal_confidence_lines=signal_confidence_lines,
        robustness_lines=robustness_lines,
        historical_analog_lines=analog_lines,
        market_linkage_lines=market_lines,
        caveats=_caveats(
            baseline_method,
            sample_mode,
            macro_status=macro_status,
            market_status=market_status,
            signal_notice=signal_notice,
        ),
        watchlist=_watchlist(current_raw, current_featured, baseline_method),
        current_regime_table=current_table,
        benchmark_metrics=benchmark_metrics,
        benchmark_comparisons=benchmark_comparisons,
        robustness_verdict=robustness_verdict,
        robustness_win_rates=robustness_win_rates,
        inflation_measure_availability=availability,
        historical_analogs=analogs,
        market_channel_summary=market_summary,
    )


def next_print_flip_threshold(
    df: pd.DataFrame,
    baseline_method: str,
    window: int = 4,
    rolling_window: int = 36,
    fed_target: float = 2.0,
) -> float | None:
    """Approximate next-month CPI YoY print that would flip the TINF sign.

    Solves mean(eps[t-2], eps[t-1], eps[t], eps[t+1]) = 0 for the next YoY
    print, projecting the baseline one month ahead. Only defined for live-safe
    baselines whose next value is mechanical; returns None otherwise.
    """

    if baseline_method not in LIVE_SAFE_BASELINES:
        return None

    eps = df["epsilon"].dropna()
    yoy = df["inflation_yoy"].dropna()
    if len(eps) < window - 1:
        return None

    eps_needed = -float(eps.iloc[-(window - 1) :].sum())

    if baseline_method == "fed_target":
        baseline_next = fed_target
    elif baseline_method == "rolling_36_shifted":
        if len(yoy) < rolling_window:
            return None
        baseline_next = float(yoy.iloc[-rolling_window:].mean())
    elif baseline_method == "expanding_shifted":
        baseline_next = float(yoy.mean())
    else:
        return None

    return baseline_next + eps_needed


def build_trader_report(
    raw: pd.DataFrame,
    df: pd.DataFrame,
    baseline_method: str,
    sample_mode: str,
    decay_windows: tuple[int, ...] = (24, 30),
) -> TraderReport:
    """Build the trader briefing from the raw frame and the feature frame.

    Future/experimental scope: not surfaced in the Streamlit dashboard (the app
    ships only ``build_macro_research_report``); retained for the planned trader
    research mode and kept descriptive, never a trade recommendation.

    ``raw`` is needed to recompute the snapshot under the other live-safe
    baselines for the robustness section; ``df`` is the feature frame under
    the user-selected baseline.
    """

    snapshot = latest_signal_snapshot(df)
    if not snapshot.get("available"):
        return TraderReport(
            available=False,
            reason=snapshot.get("reason", "No complete TINF observation available."),
        )

    as_of = str(pd.to_datetime(snapshot["date"]).date())
    tinf = float(snapshot["tinf_4m"])
    pct = float(snapshot["tinf_4m_percentile"])
    regime = str(snapshot["regime"])
    term = str(snapshot["term_structure"])
    pressure = validation_mod.pressure_label(term)
    meta = BASELINE_META[baseline_method]

    # --- 1. Where the tape is -------------------------------------------------
    row = df.loc[df["date"] == snapshot["date"]]
    run_above = int(row["run_length_above"].iloc[0]) if len(row) else 0
    side = "above" if snapshot["epsilon"] > 0 else "below"
    state_lines = (
        f"CPI YoY {snapshot['inflation_yoy']:.2f}% vs baseline {snapshot['baseline']:.2f}% "
        f"({baseline_method}): inflation is {abs(float(snapshot['epsilon'])):.2f}pp {side} "
        f"its mean-reversion anchor this month.",
        f"TINF 4M {tinf:+.2f}pp at the {_ordinal(pct)} percentile of this sample - "
        f"TINF 8M {float(snapshot['tinf_8m']):+.2f}pp, TINF 12M {float(snapshot['tinf_12m']):+.2f}pp.",
        f"Run length: {run_above} consecutive month(s) above baseline "
        f"(the paper's diagnostic flags trip at 4/8/12).",
        f"Regime label: '{regime}'; short-term pressure: '{pressure}'.",
    )

    # --- 2. Persistence / normalization ---------------------------------------
    persistence_lines: list[str] = []
    valid_t_stars: list[float] = []
    try:
        _, decay_df = decay_summaries_for_windows(df, windows=decay_windows, value_col="tinf_4m")
        for _, drow in decay_df.iterrows():
            window = int(drow["window"])
            if bool(drow["valid_formula"]):
                persistence_lines.append(
                    f"Window {window}M: rho_T {drow['rho_T']:.2f}, mu {drow['mu']:.2f} -> "
                    f"{drow['decay_6m_pct']:.0f}% of the current deviation gone in 6 months, "
                    f"{drow['decay_12m_pct']:.0f}% in 12, 95% convergence in "
                    f"~{drow['t_star_months']:.0f} months ({drow['t_star_years']:.1f}y)."
                    + (f" Note: {drow['warning']}" if drow["warning"] else "")
                )
                valid_t_stars.append(float(drow["t_star_months"]))
            else:
                persistence_lines.append(
                    f"Window {window}M: convergence formula invalid ({drow['warning']}). "
                    "Do not quote a normalization horizon from this window."
                )
    except Exception as exc:  # keep the report renderable on short samples
        persistence_lines.append(f"Persistence estimation unavailable: {exc}")

    # --- 3. Robustness across live-safe baselines ------------------------------
    robustness_lines: list[str] = []
    signs: set[str] = set()
    for method in LIVE_SAFE_BASELINES:
        temp = add_transitory_inflation_features(raw, baseline_method=method)
        snap = latest_signal_snapshot(temp)
        if not snap.get("available"):
            robustness_lines.append(f"{method}: no complete observation.")
            continue
        v = float(snap["tinf_4m"])
        signs.add("positive" if v > 0 else "negative")
        robustness_lines.append(
            f"{method}: TINF 4M {v:+.2f}pp ({_ordinal(float(snap['tinf_4m_percentile']))} pct), "
            f"regime '{snap['regime']}', as of {pd.to_datetime(snap['date']).date()}."
        )
    if len(signs) == 1:
        robustness_lines.append(
            f"All live-safe baselines agree the deviation is {signs.pop()} - "
            "the direction of the signal is not a baseline artifact."
        )
    elif signs:
        robustness_lines.append(
            "Live-safe baselines DISAGREE on the sign of the deviation - the call is "
            "baseline-dependent. Quote the baseline with any conclusion."
        )

    # --- 4. Stylized regime read ------------------------------------------------
    playbook = REGIME_PLAYBOOK.get(regime, REGIME_PLAYBOOK["neutral"])
    playbook = playbook + (
        ("Term-structure modifier", TERM_STRUCTURE_NOTES.get(term, TERM_STRUCTURE_NOTES["mixed"])),
    )

    # --- 5. What to watch --------------------------------------------------------
    watch_lines = [
        "Each 1pp CPI YoY surprise moves TINF 4M by 0.25pp on the print (it is a 4-month "
        "average), so single releases bend the signal but rarely flip a large deviation.",
    ]
    threshold = next_print_flip_threshold(df, baseline_method)
    if threshold is not None and tinf != 0:
        flip_dir = "below" if tinf > 0 else "above"
        flip_sign = "negative" if tinf > 0 else "positive"
        watch_lines.append(
            f"A next-month CPI YoY print {flip_dir} ~{threshold:.2f}% would flip TINF 4M "
            f"{flip_sign} (approximate: assumes the {baseline_method} baseline updates "
            "mechanically)."
        )
    watch_lines.append(
        "Confirmation: a 4M cross of the 8M line on the short-term pressure chart historically "
        "marks regime turns; alignment of all three horizons strengthens the read."
    )

    # --- Caveats ------------------------------------------------------------------
    caveats = [
        "Descriptive regime interpretation for research only - not investment advice and not "
        "a trading system; no sizing, timing or instruments are implied.",
        "The market linkages above are stylized priors; they have not yet been validated "
        "empirically inside this project (trader research layer is on the backlog).",
        f"Computed under baseline '{baseline_method}' "
        f"({'live-safe' if meta.live_safe else 'EX-POST: uses information unavailable in real time'}) "
        f"and sample mode '{sample_mode}'. Percentiles and regime cutoffs shift with the sample.",
    ]
    if "cpi_imputed" in raw.columns and raw["cpi_imputed"].any():
        months = ", ".join(
            d.strftime("%Y-%m") for d in pd.to_datetime(raw.loc[raw["cpi_imputed"], "date"])
        )
        caveats.append(
            f"CPI level was imputed for: {months}. Readings touching these months are partly "
            "estimates."
        )

    headline = (
        f"As of {as_of}: TINF 4M {tinf:+.2f}pp ({_ordinal(pct)} percentile) - regime '{regime}', "
        f"short-term pressure {pressure}."
    )
    if valid_t_stars:
        headline += (
            f" Model-implied 95% normalization in ~{min(valid_t_stars):.0f}-"
            f"{max(valid_t_stars):.0f} months."
        )

    return TraderReport(
        available=True,
        as_of=as_of,
        headline=headline,
        state_lines=state_lines,
        persistence_lines=tuple(persistence_lines),
        robustness_lines=tuple(robustness_lines),
        playbook=playbook,
        watch_lines=tuple(watch_lines),
        caveats=tuple(caveats),
    )

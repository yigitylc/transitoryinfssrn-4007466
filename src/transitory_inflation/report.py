from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .features import BASELINE_META, add_transitory_inflation_features, latest_signal_snapshot
from .models import decay_summaries_for_windows

LIVE_SAFE_BASELINES: tuple[str, ...] = tuple(
    name for name, meta in BASELINE_META.items() if meta.live_safe
)

# Stylized macro-trader regime priors. Descriptive interpretation only: these
# are not validated inside this project yet and must never be rendered as
# trade recommendations (see docs/01_RESEARCH_SPEC.md, trader research mode).
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


def _pressure_label(term_structure: str) -> str:
    """Convert internal TINF ordering labels to clearer report wording."""

    return {
        "accelerating": "firming",
        "decelerating": "cooling",
        "mixed": "mixed",
    }.get(term_structure, "mixed")


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
    pressure = _pressure_label(term)
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

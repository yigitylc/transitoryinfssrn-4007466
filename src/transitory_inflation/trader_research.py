"""Trader Research mode: a descriptive, rates-only reading of market linkage.

Scope (decided 2026-06-24): descriptive only, rates-only (the six approved FRED
series), live-safe. This module does NOT add forecasts, PnL, sizing, timing,
instruments, or trade recommendations, and it does not touch the shelved
``report.build_trader_report`` / ``REGIME_PLAYBOOK`` layer.

It answers one trader-first question -- "given the state we are in today, what
did the approved FRED rate instruments historically do over the next
3/6/12/24/36 months?" -- by collapsing the Phase 4 market-linkage cross-tab to
the current bucket and exposing the forward-change distribution plus the analog
months behind it.

The current bucket is taken from the live-safe walk-forward labels
(``historical_regime`` / ``historical_short_term_pressure``), NOT from
``features.latest_signal_snapshot``'s full-sample (ex-post) regime, so the
current state is matched against history in the same label space and
methodology.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from .market_data import MARKET_VALUE_COLUMNS
from .market_linkage import DEFAULT_MARKET_LINKAGE_HORIZONS, MarketLinkageTables
from .validation import (
    PRESSURE_ORDER,
    REGIME_ORDER,
    add_short_term_pressure_labels,
    add_walk_forward_regime_labels,
)

REGIME_COL = "historical_regime"
PRESSURE_COL = "historical_short_term_pressure"


@dataclass(frozen=True)
class CurrentBucket:
    """The latest live-safe walk-forward regime/pressure state."""

    available: bool
    reason: str | None = None
    regime: str | None = None
    pressure: str | None = None
    as_of: pd.Timestamp | None = None
    regime_count: int = 0
    regime_pressure_count: int = 0


@dataclass(frozen=True)
class TraderResearchView:
    """A current-state-conditioned, descriptive market-linkage reading."""

    available: bool
    reason: str | None = None
    regime: str | None = None
    pressure: str | None = None
    distribution: pd.DataFrame = field(default_factory=pd.DataFrame)
    channel_rollup: pd.DataFrame = field(default_factory=pd.DataFrame)
    analog_months: pd.DataFrame = field(default_factory=pd.DataFrame)
    weak_evidence: bool = False


def _ensure_walk_forward_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add the live-safe walk-forward labels if they are not already present."""

    out = df
    if REGIME_COL not in out.columns:
        out = add_walk_forward_regime_labels(out)
    if PRESSURE_COL not in out.columns:
        out = add_short_term_pressure_labels(out)
    return out


def latest_walk_forward_bucket(df: pd.DataFrame) -> CurrentBucket:
    """Return today's live-safe walk-forward (regime, pressure) bucket.

    The bucket is the latest row with both walk-forward labels defined. Counts
    report how many historical months share the current regime (and the current
    regime crossed with pressure), so the dashboard can flag thin samples.
    """

    if df is None or df.empty or "tinf_4m" not in df.columns:
        return CurrentBucket(available=False, reason="No signal frame is available.")

    labeled = _ensure_walk_forward_labels(df)
    valid = labeled[REGIME_COL].notna() & labeled[PRESSURE_COL].notna()
    if not valid.any():
        return CurrentBucket(
            available=False,
            reason=(
                "No live-safe walk-forward regime label is available yet "
                "(needs enough prior history)."
            ),
        )

    rows = labeled.loc[valid]
    if "date" in rows.columns:
        rows = rows.sort_values("date")
    latest = rows.iloc[-1]
    regime = str(latest[REGIME_COL])
    pressure = str(latest[PRESSURE_COL])
    regime_mask = labeled[REGIME_COL] == regime
    return CurrentBucket(
        available=True,
        regime=regime,
        pressure=pressure,
        as_of=pd.Timestamp(latest["date"]) if "date" in rows.columns else None,
        regime_count=int(regime_mask.sum()),
        regime_pressure_count=int((regime_mask & (labeled[PRESSURE_COL] == pressure)).sum()),
    )


def _filter_summary(
    summary: pd.DataFrame | None,
    keys: Mapping[str, str],
    horizons: Iterable[int] | None,
) -> pd.DataFrame:
    """Filter a precomputed linkage summary to a bucket, preserving columns."""

    if summary is None:
        return pd.DataFrame()
    selected = summary
    for column, value in keys.items():
        if column not in selected.columns:
            return selected.iloc[0:0].copy()
        selected = selected.loc[selected[column] == value]
    if horizons is not None and "horizon_months" in selected.columns:
        wanted = {int(horizon) for horizon in horizons}
        selected = selected.loc[selected["horizon_months"].isin(wanted)]
    return selected.reset_index(drop=True)


def conditional_forward_distribution(
    tables: MarketLinkageTables,
    regime: str,
    pressure: str | None = None,
    horizons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Per-instrument forward-change distribution for the selected bucket.

    Reads the precomputed ``summary_by_regime`` (regime only) or
    ``summary_by_regime_and_pressure`` (regime x pressure). Values are forward
    changes in basis points with median / p25 / p75 / hit rates / weak-evidence
    already computed upstream; nothing is recomputed here.
    """

    if pressure is None:
        return _filter_summary(tables.summary_by_regime, {REGIME_COL: regime}, horizons)
    return _filter_summary(
        tables.summary_by_regime_and_pressure,
        {REGIME_COL: regime, PRESSURE_COL: pressure},
        horizons,
    )


def conditional_channel_rollup(
    tables: MarketLinkageTables,
    regime: str,
    horizons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Channel roll-up (nominal / breakeven / real) for the selected regime."""

    return _filter_summary(tables.channel_summary_by_regime, {REGIME_COL: regime}, horizons)


def regime_analog_months(
    panel: pd.DataFrame,
    regime: str,
    pressure: str | None = None,
    horizons: Iterable[int] | None = None,
    market_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return the in-bucket months and their forward market changes (bp).

    This is the audit trail behind the distribution: the historical months that
    share the bucket and carry at least one forward market observation.
    """

    if panel is None or panel.empty or REGIME_COL not in panel.columns:
        return pd.DataFrame()

    mask = panel[REGIME_COL] == regime
    if pressure is not None and PRESSURE_COL in panel.columns:
        mask = mask & (panel[PRESSURE_COL] == pressure)
    selected = panel.loc[mask]
    if selected.empty:
        return pd.DataFrame()

    horizons = (
        tuple(int(horizon) for horizon in horizons)
        if horizons is not None
        else DEFAULT_MARKET_LINKAGE_HORIZONS
    )
    market_columns = (
        tuple(market_columns)
        if market_columns is not None
        else tuple(column for column in MARKET_VALUE_COLUMNS if column in selected.columns)
    )
    change_cols = [
        f"{variable}_change_{horizon}m_bp"
        for horizon in horizons
        for variable in market_columns
        if f"{variable}_change_{horizon}m_bp" in selected.columns
    ]
    base_cols = [
        column
        for column in ("date", REGIME_COL, PRESSURE_COL, "epsilon", "tinf_4m")
        if column in selected.columns
    ]
    out = selected.loc[:, [*base_cols, *change_cols]].copy()
    if change_cols:
        out = out.loc[out[change_cols].notna().any(axis=1)]
    if "date" in out.columns:
        out = out.sort_values("date")
    return out.reset_index(drop=True)


def available_regimes(tables: MarketLinkageTables) -> tuple[str, ...]:
    """Regimes present in the linkage summary, ordered for display."""

    return _ordered_present(tables.summary_by_regime, REGIME_COL, REGIME_ORDER)


def available_pressures(tables: MarketLinkageTables) -> tuple[str, ...]:
    """Pressure labels present in the linkage summary, ordered for display."""

    return _ordered_present(tables.summary_by_regime_and_pressure, PRESSURE_COL, PRESSURE_ORDER)


def _ordered_present(
    summary: pd.DataFrame | None,
    column: str,
    preferred_order: tuple[str, ...],
) -> tuple[str, ...]:
    if summary is None or summary.empty or column not in summary.columns:
        return ()
    present = {str(value) for value in summary[column].dropna()}
    ordered = [label for label in preferred_order if label in present]
    extras = sorted(present - set(preferred_order))
    return tuple(ordered + extras)


def build_trader_research_view(
    tables: MarketLinkageTables,
    regime: str | None,
    pressure: str | None = None,
    horizons: Iterable[int] | None = None,
) -> TraderResearchView:
    """Assemble the descriptive Trader Research view for one bucket.

    ``regime`` is the bucket to view (default in the app is today's walk-forward
    regime; the dashboard may override it to explore other states). ``pressure``
    is ``None`` to condition on regime only, or a pressure label to condition on
    regime x pressure.
    """

    if regime is None:
        return TraderResearchView(available=False, reason="No regime is available to view.")

    panel = tables.panel
    market_columns = tuple(
        column
        for column in MARKET_VALUE_COLUMNS
        if panel is not None and column in panel.columns
    )
    if not market_columns:
        return TraderResearchView(
            available=False,
            reason="No approved market variables are available for the selected sample.",
            regime=regime,
            pressure=pressure,
        )

    distribution = conditional_forward_distribution(tables, regime, pressure, horizons)
    channel_rollup = conditional_channel_rollup(tables, regime, horizons)
    analog_months = regime_analog_months(panel, regime, pressure, horizons, market_columns)
    weak_evidence = bool(
        "weak_evidence" in distribution.columns
        and distribution["weak_evidence"].fillna(False).astype(bool).any()
    )
    available = not (distribution.empty and analog_months.empty)
    return TraderResearchView(
        available=available,
        reason=None if available else "No historical analogs for this bucket.",
        regime=regime,
        pressure=pressure,
        distribution=distribution,
        channel_rollup=channel_rollup,
        analog_months=analog_months,
        weak_evidence=weak_evidence,
    )

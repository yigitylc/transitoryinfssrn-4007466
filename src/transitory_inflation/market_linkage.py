from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from .market_data import (
    MARKET_VALUE_COLUMNS,
    current_market_snapshot,
    market_data_availability,
)
from .validation import add_short_term_pressure_labels, add_walk_forward_regime_labels

DEFAULT_MARKET_LINKAGE_HORIZONS: tuple[int, ...] = (3, 6, 12, 24, 36)
DEFAULT_SIGNAL_COLUMNS: tuple[str, ...] = ("epsilon", "tinf_4m", "tinf_8m", "tinf_12m")
MARKET_CHANNELS: dict[str, tuple[str, str]] = {
    "nominal_rates": ("yield_2y", "yield_10y"),
    "breakevens": ("breakeven_5y", "breakeven_10y"),
    "real_yields": ("real_yield_5y", "real_yield_10y"),
}
WEAK_EVIDENCE_MIN_COUNT = 30
WEAK_EVIDENCE_NOTE = "Fewer than 30 complete observations; interpret cautiously."
MARKET_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon_months",
    "market_variable",
    "count",
    "avg_change_bp",
    "median_change_bp",
    "p25_change_bp",
    "p75_change_bp",
    "pct_positive_change",
    "increase_hit_rate",
    "decrease_hit_rate",
    "weak_evidence",
    "evidence_note",
)
REGIME_PRESSURE_RANKING_COLUMNS: tuple[str, ...] = (
    "historical_regime",
    "historical_short_term_pressure",
    *MARKET_SUMMARY_COLUMNS,
    "highest_change_rank",
    "lowest_change_rank",
)
CHANNEL_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon_months",
    "market_channel",
    "historical_regime",
    "historical_direction",
    "count",
    "avg_change_bp",
    "median_change_bp",
    "p25_change_bp",
    "p75_change_bp",
    "increase_hit_rate",
    "decrease_hit_rate",
    "weak_evidence",
    "evidence_note",
)


@dataclass(frozen=True)
class MarketLinkageTables:
    panel: pd.DataFrame
    availability: pd.DataFrame
    current_snapshot: pd.DataFrame
    summary_by_regime: pd.DataFrame
    summary_by_pressure: pd.DataFrame
    summary_by_regime_and_pressure: pd.DataFrame
    regime_pressure_rankings: pd.DataFrame
    channel_summary_by_regime: pd.DataFrame
    correlations: pd.DataFrame


def _horizons(values: Iterable[int]) -> tuple[int, ...]:
    horizons = tuple(int(value) for value in values)
    if not horizons:
        raise ValueError("At least one horizon is required")
    invalid = [value for value in horizons if value <= 0]
    if invalid:
        raise ValueError(f"Horizons must be positive month counts: {invalid}")
    return horizons


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _month_end_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp("M")


def _available_market_columns(df: pd.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in MARKET_VALUE_COLUMNS if column in df.columns)


def _change_col(variable: str, horizon: int) -> str:
    return f"{variable}_change_{horizon}m_bp"


def _summary_metrics(changes: pd.Series) -> dict[str, object]:
    changes = pd.to_numeric(changes, errors="coerce").dropna()
    count = int(len(changes))
    weak_evidence = count < WEAK_EVIDENCE_MIN_COUNT
    return {
        "count": count,
        "avg_change_bp": float(changes.mean()),
        "median_change_bp": float(changes.median()),
        "p25_change_bp": float(changes.quantile(0.25)),
        "p75_change_bp": float(changes.quantile(0.75)),
        "pct_positive_change": float((changes > 0).mean()),
        "increase_hit_rate": float((changes > 0).mean()),
        "decrease_hit_rate": float((changes < 0).mean()),
        "weak_evidence": weak_evidence,
        "evidence_note": WEAK_EVIDENCE_NOTE if weak_evidence else "",
    }


def _historical_direction(avg_change_bp: float) -> str:
    if avg_change_bp > 0:
        return "rose"
    if avg_change_bp < 0:
        return "fell"
    return "flat"


def add_forward_market_changes(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_MARKET_LINKAGE_HORIZONS,
    market_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Add t+h market-rate changes to row t in basis points."""

    horizons = _horizons(horizons)
    out = df.copy()
    selected_market_columns = (
        tuple(market_columns) if market_columns is not None else _available_market_columns(out)
    )

    for variable in selected_market_columns:
        if variable not in out.columns:
            continue
        current = pd.to_numeric(out[variable], errors="coerce")
        for horizon in horizons:
            future = current.shift(-horizon)
            out[f"{variable}_fwd_{horizon}m"] = future
            out[_change_col(variable, horizon)] = (future - current) * 100.0
    return out


def build_market_linkage_panel(
    signal_features: pd.DataFrame,
    market_monthly: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_MARKET_LINKAGE_HORIZONS,
) -> pd.DataFrame:
    """Join completed inflation-signal features to market data, then add future changes."""

    _require_columns(signal_features, ("date", *DEFAULT_SIGNAL_COLUMNS))
    signal = signal_features.copy()
    signal["date"] = _month_end_dates(signal["date"])

    has_lineage = bool(
        {
            "signal_observed_only_eligible",
            "signal_uses_imputed_input",
            "signal_uses_missing_input",
        }
        & set(signal.columns)
    )
    if "signal_observed_only_eligible" in signal.columns:
        eligible = signal["signal_observed_only_eligible"].fillna(False).astype(bool)
    else:
        imputed = signal.get(
            "signal_uses_imputed_input",
            pd.Series(False, index=signal.index, dtype=bool),
        ).fillna(False).astype(bool)
        missing = signal.get(
            "signal_uses_missing_input",
            pd.Series(False, index=signal.index, dtype=bool),
        ).fillna(False).astype(bool)
        eligible = ~(imputed | missing)

    invalid = ~eligible
    signal.loc[invalid, list(DEFAULT_SIGNAL_COLUMNS)] = float("nan")
    for label_col in ("historical_regime", "historical_short_term_pressure"):
        if label_col in signal.columns:
            signal.loc[invalid, label_col] = pd.NA

    if has_lineage or "historical_short_term_pressure" not in signal.columns:
        signal = add_short_term_pressure_labels(signal)
    if has_lineage or "historical_regime" not in signal.columns:
        signal = add_walk_forward_regime_labels(signal)
    if has_lineage:
        signal.loc[invalid, ["historical_regime", "historical_short_term_pressure"]] = pd.NA

    if "date" not in market_monthly.columns:
        return add_forward_market_changes(signal, horizons=horizons, market_columns=())

    market_columns = _available_market_columns(market_monthly)
    market = market_monthly.loc[:, ["date", *market_columns]].copy()
    market["date"] = _month_end_dates(market["date"])
    panel = signal.merge(market, on="date", how="left", validate="one_to_one")
    return add_forward_market_changes(panel, horizons=horizons, market_columns=market_columns)


def _forward_market_change_summary_by_groups(
    panel: pd.DataFrame,
    group_cols: Iterable[str],
    horizons: Iterable[int] = DEFAULT_MARKET_LINKAGE_HORIZONS,
    market_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Summarize future market changes by current-month signal groups."""

    horizons = _horizons(horizons)
    group_cols = tuple(group_cols)
    if not group_cols:
        raise ValueError("At least one grouping column is required")
    _require_columns(panel, group_cols)

    selected_market_columns = (
        tuple(market_columns) if market_columns is not None else _available_market_columns(panel)
    )
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        for variable in selected_market_columns:
            change_col = _change_col(variable, horizon)
            if change_col not in panel.columns:
                continue
            valid = panel.loc[:, list(group_cols)].notna().all(axis=1) & panel[change_col].notna()
            current = panel.loc[valid, [*group_cols, change_col]].copy()
            for group_values, group in current.groupby(list(group_cols), dropna=True):
                if len(group_cols) == 1 and not isinstance(group_values, tuple):
                    group_values = (group_values,)
                changes = pd.to_numeric(group[change_col], errors="coerce").dropna()
                if changes.empty:
                    continue
                row = dict(zip(group_cols, group_values, strict=True))
                row.update(
                    {
                        "horizon_months": horizon,
                        "market_variable": variable,
                    }
                )
                row.update(_summary_metrics(changes))
                rows.append(row)

    return pd.DataFrame(rows, columns=(*group_cols, *MARKET_SUMMARY_COLUMNS))


def forward_market_change_summary_by_regime(
    panel: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_MARKET_LINKAGE_HORIZONS,
    regime_col: str = "historical_regime",
) -> pd.DataFrame:
    return _forward_market_change_summary_by_groups(
        panel,
        group_cols=(regime_col,),
        horizons=horizons,
    )


def forward_market_change_summary_by_short_term_pressure(
    panel: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_MARKET_LINKAGE_HORIZONS,
    pressure_col: str = "historical_short_term_pressure",
) -> pd.DataFrame:
    return _forward_market_change_summary_by_groups(
        panel,
        group_cols=(pressure_col,),
        horizons=horizons,
    )


def forward_market_change_summary_by_regime_and_pressure(
    panel: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_MARKET_LINKAGE_HORIZONS,
    regime_col: str = "historical_regime",
    pressure_col: str = "historical_short_term_pressure",
) -> pd.DataFrame:
    return _forward_market_change_summary_by_groups(
        panel,
        group_cols=(regime_col, pressure_col),
        horizons=horizons,
    )


def rank_regime_pressure_market_changes(
    summary: pd.DataFrame,
    horizons: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Rank regime x pressure groups by average forward market change."""

    if summary.empty:
        return pd.DataFrame(columns=REGIME_PRESSURE_RANKING_COLUMNS)

    _require_columns(
        summary,
        (
            "historical_regime",
            "historical_short_term_pressure",
            "horizon_months",
            "market_variable",
            "avg_change_bp",
        ),
    )
    ranked = summary.copy()
    if horizons is not None:
        selected_horizons = set(_horizons(horizons))
        ranked = ranked.loc[ranked["horizon_months"].isin(selected_horizons)].copy()
    if ranked.empty:
        return pd.DataFrame(columns=REGIME_PRESSURE_RANKING_COLUMNS)

    ranked["avg_change_bp"] = pd.to_numeric(ranked["avg_change_bp"], errors="coerce")
    valid = ranked["avg_change_bp"].notna()
    ranked = ranked.loc[valid].copy()
    if ranked.empty:
        return pd.DataFrame(columns=REGIME_PRESSURE_RANKING_COLUMNS)

    ranking_groups = ["horizon_months", "market_variable"]
    ranked["highest_change_rank"] = (
        ranked.groupby(ranking_groups)["avg_change_bp"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    ranked["lowest_change_rank"] = (
        ranked.groupby(ranking_groups)["avg_change_bp"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return ranked.loc[:, REGIME_PRESSURE_RANKING_COLUMNS].sort_values(
        ["horizon_months", "market_variable", "highest_change_rank"]
    )


def channel_regime_summary(
    panel: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_MARKET_LINKAGE_HORIZONS,
    regime_col: str = "historical_regime",
) -> pd.DataFrame:
    """Summarize row-wise channel averages by historical regime."""

    horizons = _horizons(horizons)
    _require_columns(panel, (regime_col,))

    rows: list[dict[str, object]] = []
    for horizon in horizons:
        for channel, variables in MARKET_CHANNELS.items():
            change_cols = tuple(_change_col(variable, horizon) for variable in variables)
            if not all(change_col in panel.columns for change_col in change_cols):
                continue
            changes = panel.loc[:, change_cols].apply(pd.to_numeric, errors="coerce")
            valid = panel[regime_col].notna() & changes.notna().all(axis=1)
            current = pd.DataFrame(
                {
                    regime_col: panel.loc[valid, regime_col],
                    "channel_change_bp": changes.loc[valid].mean(axis=1),
                }
            )
            for regime, group in current.groupby(regime_col, dropna=True):
                channel_changes = pd.to_numeric(
                    group["channel_change_bp"], errors="coerce"
                ).dropna()
                if channel_changes.empty:
                    continue
                metrics = _summary_metrics(channel_changes)
                avg_change = float(metrics["avg_change_bp"])
                rows.append(
                    {
                        "horizon_months": horizon,
                        "market_channel": channel,
                        regime_col: regime,
                        "historical_direction": _historical_direction(avg_change),
                        "count": metrics["count"],
                        "avg_change_bp": metrics["avg_change_bp"],
                        "median_change_bp": metrics["median_change_bp"],
                        "p25_change_bp": metrics["p25_change_bp"],
                        "p75_change_bp": metrics["p75_change_bp"],
                        "increase_hit_rate": metrics["increase_hit_rate"],
                        "decrease_hit_rate": metrics["decrease_hit_rate"],
                        "weak_evidence": metrics["weak_evidence"],
                        "evidence_note": metrics["evidence_note"],
                    }
                )

    return pd.DataFrame(rows, columns=CHANNEL_SUMMARY_COLUMNS)


def market_signal_correlations(
    panel: pd.DataFrame,
    signal_columns: Iterable[str] = DEFAULT_SIGNAL_COLUMNS,
    horizons: Iterable[int] = DEFAULT_MARKET_LINKAGE_HORIZONS,
    market_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate simple contemporaneous-signal vs future-market-change correlations."""

    horizons = _horizons(horizons)
    selected_signal_columns = tuple(column for column in signal_columns if column in panel.columns)
    selected_market_columns = (
        tuple(market_columns) if market_columns is not None else _available_market_columns(panel)
    )

    rows: list[dict[str, object]] = []
    for signal_col in selected_signal_columns:
        signal = pd.to_numeric(panel[signal_col], errors="coerce")
        for variable in selected_market_columns:
            for horizon in horizons:
                change_col = _change_col(variable, horizon)
                if change_col not in panel.columns:
                    continue
                change = pd.to_numeric(panel[change_col], errors="coerce")
                valid = signal.notna() & change.notna()
                current = pd.DataFrame(
                    {"signal": signal.loc[valid], "change": change.loc[valid]}
                )
                if (
                    len(current) >= 2
                    and current["signal"].nunique(dropna=True) > 1
                    and current["change"].nunique(dropna=True) > 1
                ):
                    correlation = float(current["signal"].corr(current["change"]))
                else:
                    correlation = float("nan")
                rows.append(
                    {
                        "signal_variable": signal_col,
                        "market_variable": variable,
                        "horizon_months": horizon,
                        "count": int(len(current)),
                        "correlation": correlation,
                    }
                )

    return pd.DataFrame(
        rows,
        columns=(
            "signal_variable",
            "market_variable",
            "horizon_months",
            "count",
            "correlation",
        ),
    )


def build_market_linkage_tables(
    signal_features: pd.DataFrame,
    market_monthly: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_MARKET_LINKAGE_HORIZONS,
) -> MarketLinkageTables:
    """Build all Phase 4A table-first market-linkage outputs."""

    horizons = _horizons(horizons)
    panel = build_market_linkage_panel(signal_features, market_monthly, horizons=horizons)
    summary_by_regime_and_pressure = forward_market_change_summary_by_regime_and_pressure(
        panel,
        horizons=horizons,
    )
    return MarketLinkageTables(
        panel=panel,
        availability=market_data_availability(market_monthly),
        current_snapshot=current_market_snapshot(market_monthly),
        summary_by_regime=forward_market_change_summary_by_regime(
            panel,
            horizons=horizons,
        ),
        summary_by_pressure=forward_market_change_summary_by_short_term_pressure(
            panel,
            horizons=horizons,
        ),
        summary_by_regime_and_pressure=summary_by_regime_and_pressure,
        regime_pressure_rankings=rank_regime_pressure_market_changes(
            summary_by_regime_and_pressure,
            horizons=horizons,
        ),
        channel_summary_by_regime=channel_regime_summary(panel, horizons=horizons),
        correlations=market_signal_correlations(panel, horizons=horizons),
    )

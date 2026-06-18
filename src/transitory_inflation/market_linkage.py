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
MARKET_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon_months",
    "market_variable",
    "count",
    "avg_change_bp",
    "median_change_bp",
    "p25_change_bp",
    "p75_change_bp",
    "pct_positive_change",
)


@dataclass(frozen=True)
class MarketLinkageTables:
    panel: pd.DataFrame
    availability: pd.DataFrame
    current_snapshot: pd.DataFrame
    summary_by_regime: pd.DataFrame
    summary_by_pressure: pd.DataFrame
    summary_by_regime_and_pressure: pd.DataFrame
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

    if "historical_short_term_pressure" not in signal.columns:
        signal = add_short_term_pressure_labels(signal)
    if "historical_regime" not in signal.columns:
        signal = add_walk_forward_regime_labels(signal)

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
                        "count": int(len(changes)),
                        "avg_change_bp": float(changes.mean()),
                        "median_change_bp": float(changes.median()),
                        "p25_change_bp": float(changes.quantile(0.25)),
                        "p75_change_bp": float(changes.quantile(0.75)),
                        "pct_positive_change": float((changes > 0).mean()),
                    }
                )
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
        summary_by_regime_and_pressure=forward_market_change_summary_by_regime_and_pressure(
            panel,
            horizons=horizons,
        ),
        correlations=market_signal_correlations(panel, horizons=horizons),
    )

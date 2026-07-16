from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import (
    INFORMATION_TIMESTAMP_PROVENANCE_RELEASES,
    TIMING_STATUS_RELEASE_ALIGNED,
    _has_explicit_timezone,
)
from .market_data import (
    MARKET_TIMESTAMP_COLUMN,
    MARKET_TIMESTAMP_PROVENANCE_ACTUAL,
    MARKET_TIMESTAMP_PROVENANCE_COLUMN,
    MARKET_TIMESTAMP_STATUS_COLUMN,
    MARKET_TIMESTAMP_STATUS_EXACT,
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
MARKET_ORIGIN_INFORMATION_TIMESTAMP = (
    "exact_information_timestamp_aligned_market_close"
)
# Compatibility aliases for callers that imported the original H5 constant names.
MARKET_ORIGIN_PUBLICATION = MARKET_ORIGIN_INFORMATION_TIMESTAMP
MARKET_ORIGIN_NEXT_OBSERVATION_PROXY = "conservative_next_observation_date_proxy"
MARKET_ORIGIN_CONSERVATIVE_PROXY = "conservative_month_end_t_plus_1_proxy"
MARKET_ORIGIN_MIXED = "mixed_market_origin_basis"
MARKET_ORIGIN_PARTIAL = "partial_market_origin_availability"
MARKET_ORIGIN_UNAVAILABLE = "unavailable_no_trustworthy_post_information_observation"
MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED = (
    "exact_information_timestamp_aligned_market_close_latest_revised_non_vintage"
)
# Compatibility alias matching the pre-correction public symbol.
MARKET_TIMING_RELEASE_ALIGNED = MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED
MARKET_TIMING_NEXT_OBSERVATION_PROXY = (
    "conservative_next_observation_date_proxy_latest_revised_non_vintage"
)
MARKET_TIMING_CONSERVATIVE_PROXY = (
    "conservative_month_end_t_plus_1_proxy_latest_revised_non_vintage"
)
MARKET_TIMING_MIXED = "mixed_market_origin_basis_latest_revised_non_vintage"
MARKET_TIMING_PARTIAL = (
    "partial_market_origin_availability_latest_revised_non_vintage"
)
MARKET_TIMING_UNAVAILABLE = (
    "unavailable_no_trustworthy_market_observation_at_or_after_information_timestamp"
)
MARKET_AVAILABILITY_FULL = "fully_available"
MARKET_AVAILABILITY_PARTIAL = "partially_available"
MARKET_AVAILABILITY_UNAVAILABLE = "unavailable"
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
    timing_summary: pd.DataFrame
    series_timing_summary: pd.DataFrame


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


def _observation_dates(values: pd.Series) -> pd.Series:
    """Normalize source observation dates only for conservative proxy matching."""

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None).dt.normalize()


def _trusted_signal_information(
    signal: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    supplied = signal.get(
        "information_timestamp",
        pd.Series(pd.NaT, index=signal.index),
    )
    timing_status = signal.get(
        "timing_status",
        pd.Series(pd.NA, index=signal.index, dtype="string"),
    ).astype("string")
    provenance = signal.get(
        "information_timestamp_provenance",
        pd.Series(pd.NA, index=signal.index, dtype="string"),
    ).astype("string")
    trustworthy = (
        supplied.map(_has_explicit_timezone).astype(bool)
        & timing_status.eq(TIMING_STATUS_RELEASE_ALIGNED).fillna(False)
        & provenance.eq(INFORMATION_TIMESTAMP_PROVENANCE_RELEASES).fillna(False)
    )
    parsed = pd.to_datetime(supplied, errors="coerce", utc=True)
    combined = parsed.copy()

    # Walk-forward regime and pressure labels can become available later than
    # the row's underlying TINF values. When validation supplies their timing
    # metadata, market routing must wait for every label dependency actually
    # used by the row and must fail closed if any such dependency is non-exact.
    for label_column in (
        "historical_regime",
        "historical_short_term_pressure",
    ):
        timestamp_column = f"{label_column}_information_timestamp"
        status_column = f"{label_column}_timing_status"
        provenance_column = f"{label_column}_information_timestamp_provenance"
        if not {
            timestamp_column,
            status_column,
            provenance_column,
        }.intersection(signal.columns):
            continue

        dependency_used = signal.get(
            label_column,
            pd.Series(pd.NA, index=signal.index, dtype="string"),
        ).notna()
        dependency_timestamp = signal.get(
            timestamp_column,
            pd.Series(pd.NaT, index=signal.index),
        )
        dependency_status = signal.get(
            status_column,
            pd.Series(pd.NA, index=signal.index, dtype="string"),
        ).astype("string")
        dependency_provenance = signal.get(
            provenance_column,
            pd.Series(pd.NA, index=signal.index, dtype="string"),
        ).astype("string")
        dependency_trustworthy = (
            dependency_timestamp.map(_has_explicit_timezone).astype(bool)
            & dependency_status.eq(TIMING_STATUS_RELEASE_ALIGNED).fillna(False)
            & dependency_provenance.eq(
                INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
            ).fillna(False)
        )
        trustworthy &= ~dependency_used | dependency_trustworthy
        dependency_parsed = pd.to_datetime(
            dependency_timestamp,
            errors="coerce",
            utc=True,
        )
        combined = pd.concat(
            [combined, dependency_parsed.where(dependency_used)],
            axis=1,
        ).max(axis=1)

    return combined.where(trustworthy), trustworthy


def _trusted_market_timestamps(
    market: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    supplied = market.get(
        MARKET_TIMESTAMP_COLUMN,
        pd.Series(pd.NaT, index=market.index),
    )
    status = market.get(
        MARKET_TIMESTAMP_STATUS_COLUMN,
        pd.Series(pd.NA, index=market.index, dtype="string"),
    ).astype("string")
    provenance = market.get(
        MARKET_TIMESTAMP_PROVENANCE_COLUMN,
        pd.Series(pd.NA, index=market.index, dtype="string"),
    ).astype("string")
    trustworthy = (
        supplied.map(_has_explicit_timezone).astype(bool)
        & status.eq(MARKET_TIMESTAMP_STATUS_EXACT).fillna(False)
        & provenance.eq(MARKET_TIMESTAMP_PROVENANCE_ACTUAL).fillna(False)
    )
    parsed = pd.to_datetime(supplied, errors="coerce", utc=True)
    return parsed.where(trustworthy), trustworthy


def _market_origin_targets(
    signal: pd.DataFrame,
    *,
    exact_market_available: bool,
) -> pd.DataFrame:
    if "reference_month" in signal.columns:
        reference_month = _month_end_dates(signal["reference_month"])
    else:
        reference_month = _month_end_dates(signal["date"])

    information_timestamp, trustworthy_information = _trusted_signal_information(signal)
    exact = trustworthy_information & exact_market_available
    next_observation_proxy = trustworthy_information & (not exact_market_available)
    conservative_proxy = reference_month + pd.offsets.MonthEnd(1)
    next_observation_date = (
        information_timestamp.dt.tz_convert(None).dt.normalize() + pd.offsets.Day(1)
    )

    targets = pd.DataFrame(index=signal.index)
    targets["reference_month"] = reference_month
    targets["market_origin_target_timestamp"] = information_timestamp.where(exact)
    targets["market_origin_target_observation_date"] = conservative_proxy
    targets.loc[
        next_observation_proxy,
        "market_origin_target_observation_date",
    ] = next_observation_date.loc[next_observation_proxy]
    targets.loc[exact, "market_origin_target_observation_date"] = pd.NaT
    targets["market_origin_basis"] = pd.Series(
        MARKET_ORIGIN_CONSERVATIVE_PROXY,
        index=signal.index,
        dtype="string",
    )
    targets.loc[next_observation_proxy, "market_origin_basis"] = (
        MARKET_ORIGIN_NEXT_OBSERVATION_PROXY
    )
    targets.loc[exact, "market_origin_basis"] = (
        MARKET_ORIGIN_INFORMATION_TIMESTAMP
    )
    targets["market_timing_status"] = pd.Series(
        MARKET_TIMING_CONSERVATIVE_PROXY,
        index=signal.index,
        dtype="string",
    )
    targets.loc[next_observation_proxy, "market_timing_status"] = (
        MARKET_TIMING_NEXT_OBSERVATION_PROXY
    )
    targets.loc[exact, "market_timing_status"] = (
        MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED
    )
    targets["market_data_vintage_status"] = "latest_revised_non_vintage"
    return targets


def _first_market_value_on_or_after_timestamp(
    market_timestamps: pd.Series,
    market_values: pd.Series,
    targets: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    eligible = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                market_timestamps,
                errors="coerce",
                utc=True,
            ).astype("datetime64[ns, UTC]"),
            "value": pd.to_numeric(market_values, errors="coerce"),
        }
    ).dropna()
    eligible = eligible.sort_values("timestamp")
    selected_timestamps = pd.Series(
        pd.NaT,
        index=targets.index,
        dtype="datetime64[ns, UTC]",
    )
    selected_values = pd.Series(float("nan"), index=targets.index, dtype=float)
    if eligible.empty:
        return selected_timestamps, selected_values

    target_timestamps = pd.to_datetime(
        targets,
        errors="coerce",
        utc=True,
    ).astype("datetime64[ns, UTC]")
    eligible_ns = eligible["timestamp"].astype("int64").to_numpy()
    target_ns = target_timestamps.astype("int64").to_numpy()
    positions = np.searchsorted(eligible_ns, target_ns, side="left")
    usable = target_timestamps.notna().to_numpy() & (positions < len(eligible))
    if usable.any():
        eligible_timestamps = eligible["timestamp"].array
        eligible_values = eligible["value"].to_numpy()
        selected_timestamps.iloc[np.flatnonzero(usable)] = eligible_timestamps[
            positions[usable]
        ]
        selected_values.iloc[np.flatnonzero(usable)] = eligible_values[positions[usable]]
    return selected_timestamps, selected_values


def _first_market_value_on_or_after_date(
    market_dates: pd.Series,
    market_values: pd.Series,
    targets: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    eligible = pd.DataFrame(
        {
            "date": _observation_dates(market_dates),
            "value": pd.to_numeric(market_values, errors="coerce"),
        }
    ).dropna()
    eligible = eligible.sort_values("date")
    selected_dates = pd.Series(pd.NaT, index=targets.index, dtype="datetime64[ns]")
    selected_values = pd.Series(float("nan"), index=targets.index, dtype=float)
    if eligible.empty:
        return selected_dates, selected_values

    target_dates = _observation_dates(targets)
    eligible_dates = eligible["date"].to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(
        eligible_dates,
        target_dates.to_numpy(dtype="datetime64[ns]"),
        side="left",
    )
    usable = target_dates.notna().to_numpy() & (positions < len(eligible))
    if usable.any():
        eligible_values = eligible["value"].to_numpy()
        selected_dates.iloc[np.flatnonzero(usable)] = eligible_dates[positions[usable]]
        selected_values.iloc[np.flatnonzero(usable)] = eligible_values[positions[usable]]
    return selected_dates, selected_values


def _align_market_outcomes(
    signal: pd.DataFrame,
    market_observations: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    market_columns: tuple[str, ...],
) -> pd.DataFrame:
    out = signal.copy()
    if "date" not in market_observations.columns or not market_columns:
        targets = _market_origin_targets(out, exact_market_available=False)
        for column in targets.columns:
            out[column] = targets[column]
        out["market_origin_timestamp"] = pd.NaT
        out["market_origin_observation_date"] = pd.NaT
        out["market_required_series_count"] = len(market_columns)
        out["market_available_series_count"] = 0
        out["market_availability_status"] = MARKET_AVAILABILITY_UNAVAILABLE
        out["market_origin_basis"] = MARKET_ORIGIN_UNAVAILABLE
        out["market_timing_status"] = MARKET_TIMING_UNAVAILABLE
        for variable in market_columns:
            out[variable] = float("nan")
            out[f"{variable}_origin_timestamp"] = pd.NaT
            out[f"{variable}_origin_observation_date"] = pd.NaT
            out[f"{variable}_origin_basis"] = MARKET_ORIGIN_UNAVAILABLE
            out[f"{variable}_timing_status"] = MARKET_TIMING_UNAVAILABLE
        return out

    metadata_columns = [
        column
        for column in (
            MARKET_TIMESTAMP_COLUMN,
            MARKET_TIMESTAMP_PROVENANCE_COLUMN,
            MARKET_TIMESTAMP_STATUS_COLUMN,
        )
        if column in market_observations.columns
    ]
    market = market_observations.loc[
        :,
        ["date", *market_columns, *metadata_columns],
    ].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market = market.dropna(subset=["date"]).sort_values("date")
    trusted_market_timestamps, trusted_market_rows = _trusted_market_timestamps(market)
    out["market_data_vintage_status"] = "latest_revised_non_vintage"
    variable_origin_timestamp_columns: list[str] = []
    variable_origin_date_columns: list[str] = []
    variable_target_timestamp_columns: list[str] = []
    variable_target_date_columns: list[str] = []
    variable_origin_basis_columns: list[str] = []
    variable_available: list[pd.Series] = []

    for variable in market_columns:
        market_values = pd.to_numeric(market[variable], errors="coerce")
        series_has_trusted_exact_value = bool(
            (trusted_market_rows & market_values.notna()).any()
        )
        targets = _market_origin_targets(
            out,
            exact_market_available=series_has_trusted_exact_value,
        )
        out["reference_month"] = targets["reference_month"]
        exact_rows = targets["market_origin_basis"].eq(
            MARKET_ORIGIN_INFORMATION_TIMESTAMP
        )
        next_observation_proxy_rows = targets["market_origin_basis"].eq(
            MARKET_ORIGIN_NEXT_OBSERVATION_PROXY
        )
        conservative_proxy_rows = targets["market_origin_basis"].eq(
            MARKET_ORIGIN_CONSERVATIVE_PROXY
        )
        target_timestamp_col = f"{variable}_origin_target_timestamp"
        target_date_col = f"{variable}_origin_target_observation_date"
        out[target_timestamp_col] = targets["market_origin_target_timestamp"]
        out[target_date_col] = targets["market_origin_target_observation_date"]
        variable_target_timestamp_columns.append(target_timestamp_col)
        variable_target_date_columns.append(target_date_col)

        exact_origin_timestamp, exact_origin_value = (
            _first_market_value_on_or_after_timestamp(
                trusted_market_timestamps,
                market_values,
                targets["market_origin_target_timestamp"],
            )
        )
        proxy_origin_date, proxy_origin_value = _first_market_value_on_or_after_date(
            market["date"],
            market_values,
            targets["market_origin_target_observation_date"],
        )
        origin_timestamp_col = f"{variable}_origin_timestamp"
        origin_date_col = f"{variable}_origin_observation_date"
        out[origin_timestamp_col] = exact_origin_timestamp.where(exact_rows)
        out[origin_date_col] = proxy_origin_date.where(~exact_rows)
        out[variable] = proxy_origin_value.where(~exact_rows, exact_origin_value)
        variable_origin_timestamp_columns.append(origin_timestamp_col)
        variable_origin_date_columns.append(origin_date_col)

        exact_available = exact_rows & out[origin_timestamp_col].notna()
        next_proxy_available = (
            next_observation_proxy_rows & out[origin_date_col].notna()
        )
        conservative_proxy_available = (
            conservative_proxy_rows & out[origin_date_col].notna()
        )
        available = (
            exact_available | next_proxy_available | conservative_proxy_available
        )
        variable_available.append(available)
        origin_basis_col = f"{variable}_origin_basis"
        timing_status_col = f"{variable}_timing_status"
        variable_origin_basis_columns.append(origin_basis_col)
        out[origin_basis_col] = pd.Series(
            MARKET_ORIGIN_UNAVAILABLE,
            index=out.index,
            dtype="string",
        )
        out[timing_status_col] = pd.Series(
            MARKET_TIMING_UNAVAILABLE,
            index=out.index,
            dtype="string",
        )
        out.loc[exact_available, origin_basis_col] = (
            MARKET_ORIGIN_INFORMATION_TIMESTAMP
        )
        out.loc[exact_available, timing_status_col] = (
            MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED
        )
        out.loc[next_proxy_available, origin_basis_col] = (
            MARKET_ORIGIN_NEXT_OBSERVATION_PROXY
        )
        out.loc[next_proxy_available, timing_status_col] = (
            MARKET_TIMING_NEXT_OBSERVATION_PROXY
        )
        out.loc[conservative_proxy_available, origin_basis_col] = (
            MARKET_ORIGIN_CONSERVATIVE_PROXY
        )
        out.loc[conservative_proxy_available, timing_status_col] = (
            MARKET_TIMING_CONSERVATIVE_PROXY
        )

        for horizon in horizons:
            exact_endpoint_targets = exact_origin_timestamp + pd.DateOffset(months=horizon)
            exact_endpoint_timestamp, exact_endpoint_value = (
                _first_market_value_on_or_after_timestamp(
                    trusted_market_timestamps,
                    market_values,
                    exact_endpoint_targets,
                )
            )
            proxy_endpoint_targets = proxy_origin_date + pd.DateOffset(months=horizon)
            proxy_endpoint_date, proxy_endpoint_value = _first_market_value_on_or_after_date(
                market["date"],
                market_values,
                proxy_endpoint_targets,
            )
            out[f"{variable}_fwd_{horizon}m_timestamp"] = (
                exact_endpoint_timestamp.where(exact_rows)
            )
            endpoint_date_col = f"{variable}_fwd_{horizon}m_observation_date"
            out[endpoint_date_col] = proxy_endpoint_date.where(~exact_rows)
            endpoint_value = proxy_endpoint_value.where(
                ~exact_rows,
                exact_endpoint_value,
            )
            out[f"{variable}_fwd_{horizon}m"] = endpoint_value
            out[_change_col(variable, horizon)] = (endpoint_value - out[variable]) * 100.0

    out["market_origin_target_timestamp"] = pd.concat(
        [out[column] for column in variable_target_timestamp_columns],
        axis=1,
    ).max(axis=1)
    out["market_origin_target_observation_date"] = pd.concat(
        [out[column] for column in variable_target_date_columns],
        axis=1,
    ).max(axis=1)
    out["market_origin_timestamp"] = pd.concat(
        [out[column] for column in variable_origin_timestamp_columns],
        axis=1,
    ).max(axis=1)
    out["market_origin_observation_date"] = pd.concat(
        [out[column] for column in variable_origin_date_columns],
        axis=1,
    ).max(axis=1)
    available_count = pd.concat(variable_available, axis=1).sum(axis=1).astype(int)
    required_count = len(market_columns)
    fully_available = available_count.eq(required_count)
    partially_available = available_count.gt(0) & ~fully_available
    unavailable = available_count.eq(0)
    origin_bases = pd.concat(
        [out[column] for column in variable_origin_basis_columns],
        axis=1,
    )
    fully_exact = fully_available & origin_bases.eq(
        MARKET_ORIGIN_INFORMATION_TIMESTAMP
    ).all(axis=1)
    fully_next_observation_proxy = fully_available & origin_bases.eq(
        MARKET_ORIGIN_NEXT_OBSERVATION_PROXY
    ).all(axis=1)
    fully_conservative_proxy = fully_available & origin_bases.eq(
        MARKET_ORIGIN_CONSERVATIVE_PROXY
    ).all(axis=1)
    fully_mixed = fully_available & ~(
        fully_exact
        | fully_next_observation_proxy
        | fully_conservative_proxy
    )
    out["market_required_series_count"] = required_count
    out["market_available_series_count"] = available_count
    out["market_availability_status"] = pd.Series(
        MARKET_AVAILABILITY_UNAVAILABLE,
        index=out.index,
        dtype="string",
    )
    out.loc[fully_available, "market_availability_status"] = (
        MARKET_AVAILABILITY_FULL
    )
    out.loc[partially_available, "market_availability_status"] = (
        MARKET_AVAILABILITY_PARTIAL
    )
    out["market_origin_basis"] = pd.Series(
        MARKET_ORIGIN_UNAVAILABLE,
        index=out.index,
        dtype="string",
    )
    out["market_timing_status"] = pd.Series(
        MARKET_TIMING_UNAVAILABLE,
        index=out.index,
        dtype="string",
    )
    out.loc[fully_exact, "market_origin_basis"] = (
        MARKET_ORIGIN_INFORMATION_TIMESTAMP
    )
    out.loc[fully_exact, "market_timing_status"] = (
        MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED
    )
    out.loc[fully_next_observation_proxy, "market_origin_basis"] = (
        MARKET_ORIGIN_NEXT_OBSERVATION_PROXY
    )
    out.loc[fully_next_observation_proxy, "market_timing_status"] = (
        MARKET_TIMING_NEXT_OBSERVATION_PROXY
    )
    out.loc[fully_conservative_proxy, "market_origin_basis"] = (
        MARKET_ORIGIN_CONSERVATIVE_PROXY
    )
    out.loc[fully_conservative_proxy, "market_timing_status"] = (
        MARKET_TIMING_CONSERVATIVE_PROXY
    )
    out.loc[fully_mixed, "market_origin_basis"] = MARKET_ORIGIN_MIXED
    out.loc[fully_mixed, "market_timing_status"] = MARKET_TIMING_MIXED
    out.loc[partially_available, "market_origin_basis"] = MARKET_ORIGIN_PARTIAL
    out.loc[partially_available, "market_timing_status"] = MARKET_TIMING_PARTIAL
    out.loc[unavailable, "market_origin_timestamp"] = pd.NaT
    out.loc[unavailable, "market_origin_observation_date"] = pd.NaT
    return out


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
    """Align signal rows to eligible market observations, then add future changes.

    ``market_monthly`` is retained as the public argument name for compatibility;
    callers may pass timestamped observations for exact alignment at or after the
    full signal-information timestamp, or date-only observations for an explicitly
    conservative proxy alignment.
    """

    horizons = _horizons(horizons)
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
        return _align_market_outcomes(
            signal,
            market_monthly,
            horizons=horizons,
            market_columns=(),
        )

    market_columns = _available_market_columns(market_monthly)
    return _align_market_outcomes(
        signal,
        market_monthly,
        horizons=horizons,
        market_columns=market_columns,
    )


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


def market_timing_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize exact, proxy, mixed, partial, and unavailable origins."""

    columns = (
        "market_origin_basis",
        "market_timing_status",
        "market_availability_status",
        "row_count",
        "required_series_count",
        "minimum_available_series_count",
        "maximum_available_series_count",
        "exact_series_origin_count",
        "next_observation_proxy_series_origin_count",
        "month_end_proxy_series_origin_count",
        "unavailable_series_origin_count",
        "first_reference_month",
        "latest_reference_month",
        "first_market_origin_timestamp",
        "latest_market_origin_timestamp",
        "first_market_origin_observation_date",
        "latest_market_origin_observation_date",
    )
    required = {
        "market_origin_basis",
        "market_timing_status",
        "market_availability_status",
    }
    if panel.empty or not required.issubset(panel.columns):
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    origin_basis_columns = [
        column
        for column in panel.columns
        if column.endswith("_origin_basis") and column != "market_origin_basis"
    ]
    for (basis, status, availability), group in panel.groupby(
        [
            "market_origin_basis",
            "market_timing_status",
            "market_availability_status",
        ],
        dropna=False,
        sort=False,
    ):
        reference_months = pd.to_datetime(
            group.get("reference_month", group["date"]),
            errors="coerce",
        )
        origin_timestamps = pd.to_datetime(
            group.get("market_origin_timestamp"),
            errors="coerce",
            utc=True,
        )
        origin_observation_dates = pd.to_datetime(
            group.get("market_origin_observation_date"),
            errors="coerce",
        )
        required_series_count = pd.to_numeric(
            group.get("market_required_series_count"),
            errors="coerce",
        )
        available_series_count = pd.to_numeric(
            group.get("market_available_series_count"),
            errors="coerce",
        )
        series_origins = (
            group.loc[:, origin_basis_columns].stack(future_stack=True)
            if origin_basis_columns
            else pd.Series(dtype="string")
        )
        rows.append(
            {
                "market_origin_basis": basis,
                "market_timing_status": status,
                "market_availability_status": availability,
                "row_count": int(len(group)),
                "required_series_count": int(required_series_count.max()),
                "minimum_available_series_count": int(available_series_count.min()),
                "maximum_available_series_count": int(available_series_count.max()),
                "exact_series_origin_count": int(
                    series_origins.eq(MARKET_ORIGIN_INFORMATION_TIMESTAMP).sum()
                ),
                "next_observation_proxy_series_origin_count": int(
                    series_origins.eq(MARKET_ORIGIN_NEXT_OBSERVATION_PROXY).sum()
                ),
                "month_end_proxy_series_origin_count": int(
                    series_origins.eq(MARKET_ORIGIN_CONSERVATIVE_PROXY).sum()
                ),
                "unavailable_series_origin_count": int(
                    series_origins.eq(MARKET_ORIGIN_UNAVAILABLE).sum()
                ),
                "first_reference_month": reference_months.min(),
                "latest_reference_month": reference_months.max(),
                "first_market_origin_timestamp": origin_timestamps.min(),
                "latest_market_origin_timestamp": origin_timestamps.max(),
                "first_market_origin_observation_date": origin_observation_dates.min(),
                "latest_market_origin_observation_date": origin_observation_dates.max(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def market_series_timing_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Return authoritative per-series origin basis and timing-status counts."""

    columns = (
        "market_variable",
        "market_origin_basis",
        "market_timing_status",
        "row_count",
        "first_reference_month",
        "latest_reference_month",
        "first_market_origin_timestamp",
        "latest_market_origin_timestamp",
        "first_market_origin_observation_date",
        "latest_market_origin_observation_date",
    )
    if panel.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for variable in MARKET_VALUE_COLUMNS:
        basis_column = f"{variable}_origin_basis"
        status_column = f"{variable}_timing_status"
        if basis_column not in panel.columns or status_column not in panel.columns:
            continue

        for (basis, status), group in panel.groupby(
            [basis_column, status_column],
            dropna=False,
            sort=False,
        ):
            reference_months = pd.to_datetime(
                group.get("reference_month", group["date"]),
                errors="coerce",
            )
            origin_timestamps = pd.to_datetime(
                group.get(f"{variable}_origin_timestamp"),
                errors="coerce",
                utc=True,
            )
            origin_observation_dates = pd.to_datetime(
                group.get(f"{variable}_origin_observation_date"),
                errors="coerce",
            )
            rows.append(
                {
                    "market_variable": variable,
                    "market_origin_basis": basis,
                    "market_timing_status": status,
                    "row_count": int(len(group)),
                    "first_reference_month": reference_months.min(),
                    "latest_reference_month": reference_months.max(),
                    "first_market_origin_timestamp": origin_timestamps.min(),
                    "latest_market_origin_timestamp": origin_timestamps.max(),
                    "first_market_origin_observation_date": (
                        origin_observation_dates.min()
                    ),
                    "latest_market_origin_observation_date": (
                        origin_observation_dates.max()
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


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
        timing_summary=market_timing_summary(panel),
        series_timing_summary=market_series_timing_summary(panel),
    )

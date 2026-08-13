from __future__ import annotations

import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from .config import DEFAULT_SAMPLE_MODE, PROJECT_ROOT, RAW_DATA_DIR, SampleMode, resolve_sample_mode

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
CACHED_BASE_MACRO_STEM = "fred_base_macro"
BASE_MACRO_COLUMNS = ("date", "cpi_level", "tbill_3m", "inflation_yoy")
MACRO_CACHE_SCHEMA_COLUMN = "macro_cache_schema_version"
MACRO_CACHE_SCHEMA_VERSION = 2
SOURCE_MONTH_OBSERVATION_COUNT_COLUMN = "source_month_observation_count"

TIMING_STATUS_RELEASE_ALIGNED = "release_aligned"
TIMING_STATUS_REFERENCE_MONTH_ONLY = "reference_month_only"
TIMING_STATUS_UNAVAILABLE = "derived_value_unavailable"
DATA_VINTAGE_STATUS_LATEST_REVISED = "latest_revised_non_vintage"
RELEASE_TIMESTAMP_PROVENANCE_ACTUAL = "actual_release_metadata"
RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED = "release_metadata_unavailable_or_unverified"
VALUE_PROVENANCE_OFFICIAL_FRED = "official_fred_observation"
VALUE_PROVENANCE_UNVERIFIED = "value_provenance_unavailable_or_unverified"
INFORMATION_TIMESTAMP_PROVENANCE_RELEASES = "derived_from_actual_release_metadata"
INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED = "information_timing_unavailable_or_unverified"
REFERENCE_MONTH_COMPATIBILITY_SEMANTICS = (
    "compatibility alias for reference_month; not a signal availability or information timestamp"
)

ImputationPolicy = Literal["observed_only", "ex_post_continuity"]
VALID_IMPUTATION_POLICIES: tuple[ImputationPolicy, ...] = (
    "observed_only",
    "ex_post_continuity",
)
DEFAULT_IMPUTATION_POLICY: ImputationPolicy = "observed_only"

CurrentMonitoringScenarioId = Literal["low", "base", "high"]
CURRENT_MONITORING_SCENARIOS: tuple[CurrentMonitoringScenarioId, ...] = (
    "low",
    "base",
    "high",
)
CURRENT_MONITORING_CALIBRATION_POLICY = "observed_only_eligible_history"
CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH = "2025-10-31"
CURRENT_MONITORING_PREVIOUS_ENDPOINT_MONTH = "2025-09-30"
CURRENT_MONITORING_FOLLOWING_ENDPOINT_MONTH = "2025-11-30"
CURRENT_MONITORING_MEASURE_KEYS = ("headline_cpi", "core_cpi")
CURRENT_MONITORING_ESTIMATE_METHODS: dict[CurrentMonitoringScenarioId, str] = {
    "low": "previous_official_level",
    "base": "geometric_midpoint_adjacent_official_levels",
    "high": "following_official_level",
}

# Months fetched before a sample's start_date so 12-month YoY inflation is
# defined from the first sample row instead of 12 months later.
YOY_WARMUP_MONTHS = 12


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    name: str
    frequency: str = "monthly"


@dataclass(frozen=True)
class InflationMeasure:
    """Inflation index metadata for headline defaults and robustness checks."""

    key: str
    label: str
    series_id: str
    level_col: str
    yoy_col: str
    imputed_col: str
    paper_exact: bool = False

    @property
    def lineage_prefix(self) -> str:
        return self.imputed_col.removesuffix("_imputed")

    @property
    def observed_level_col(self) -> str:
        return f"{self.lineage_prefix}_observed_level"

    @property
    def originally_missing_col(self) -> str:
        return f"{self.lineage_prefix}_originally_missing"

    @property
    def source_unavailable_col(self) -> str:
        return f"{self.lineage_prefix}_source_unavailable"

    @property
    def source_present_col(self) -> str:
        return f"{self.lineage_prefix}_source_present"

    @property
    def imputation_method_col(self) -> str:
        if self.key == "headline_cpi":
            return "imputation_method"
        return f"{self.lineage_prefix}_imputation_method"

    @property
    def imputation_available_at_col(self) -> str:
        if self.key == "headline_cpi":
            return "imputation_available_at"
        return f"{self.lineage_prefix}_imputation_available_at"

    @property
    def imputation_availability_basis_col(self) -> str:
        if self.key == "headline_cpi":
            return "imputation_availability_basis"
        return f"{self.lineage_prefix}_imputation_availability_basis"

    @property
    def estimate_method_col(self) -> str:
        if self.key == "headline_cpi":
            return "estimate_method"
        return f"{self.lineage_prefix}_estimate_method"

    @property
    def estimated_reference_month_col(self) -> str:
        if self.key == "headline_cpi":
            return "estimated_reference_month"
        return f"{self.lineage_prefix}_estimated_reference_month"

    @property
    def estimate_value_col(self) -> str:
        if self.key == "headline_cpi":
            return "estimate_value"
        return f"{self.lineage_prefix}_estimate_value"

    @property
    def yoy_uses_imputed_input_col(self) -> str:
        return f"{self.yoy_col}_uses_imputed_input"

    @property
    def yoy_uses_missing_input_col(self) -> str:
        return f"{self.yoy_col}_uses_missing_input"

    @property
    def release_timestamp_col(self) -> str:
        if self.key == HEADLINE_INFLATION_MEASURE:
            return "release_timestamp"
        return f"{self.lineage_prefix}_release_timestamp"

    @property
    def value_provenance_col(self) -> str:
        return f"{self.lineage_prefix}_value_provenance"

    @property
    def release_timestamp_provenance_col(self) -> str:
        if self.key == HEADLINE_INFLATION_MEASURE:
            return "release_timestamp_provenance"
        return f"{self.lineage_prefix}_release_timestamp_provenance"

    @property
    def release_timing_status_col(self) -> str:
        if self.key == HEADLINE_INFLATION_MEASURE:
            return "release_timing_status"
        return f"{self.lineage_prefix}_release_timing_status"

    @property
    def level_information_timestamp_col(self) -> str:
        return f"{self.lineage_prefix}_level_information_timestamp"

    @property
    def yoy_information_timestamp_col(self) -> str:
        return f"{self.yoy_col}_information_timestamp"

    @property
    def yoy_information_timestamp_provenance_col(self) -> str:
        return f"{self.yoy_col}_information_timestamp_provenance"

    @property
    def yoy_timing_status_col(self) -> str:
        return f"{self.yoy_col}_timing_status"


HEADLINE_INFLATION_MEASURE = "headline_cpi"
INFLATION_MEASURES: dict[str, InflationMeasure] = {
    HEADLINE_INFLATION_MEASURE: InflationMeasure(
        key=HEADLINE_INFLATION_MEASURE,
        label="Headline CPI",
        series_id="CPIAUCSL",
        level_col="cpi_level",
        yoy_col="inflation_yoy",
        imputed_col="cpi_imputed",
        paper_exact=True,
    ),
    "core_cpi": InflationMeasure(
        key="core_cpi",
        label="Core CPI",
        series_id="CPILFESL",
        level_col="core_cpi_level",
        yoy_col="core_cpi_yoy",
        imputed_col="core_cpi_imputed",
    ),
    "pce": InflationMeasure(
        key="pce",
        label="PCE",
        series_id="PCEPI",
        level_col="pce_level",
        yoy_col="pce_yoy",
        imputed_col="pce_imputed",
    ),
    "core_pce": InflationMeasure(
        key="core_pce",
        label="Core PCE",
        series_id="PCEPILFE",
        level_col="core_pce_level",
        yoy_col="core_pce_yoy",
        imputed_col="core_pce_imputed",
    ),
}
INFLATION_MEASURE_ORDER: tuple[str, ...] = tuple(INFLATION_MEASURES)
BASE_FRED_SERIES = tuple(
    dict.fromkeys([measure.series_id for measure in INFLATION_MEASURES.values()] + ["TB3MS"])
)
CACHED_INPUT_COLUMN_SETS = (
    ("date", "CPIAUCSL", "TB3MS"),
    ("date", "cpi_level", "tbill_3m"),
)


@dataclass(frozen=True)
class MacroDataLoadResult:
    """Macro data plus source metadata for dashboard disclosure."""

    data: pd.DataFrame
    data_source_used: str
    live_fetch_status: str
    cache_file_used: str | None = None
    api_key_configured: bool = False
    imputation_policy: ImputationPolicy = DEFAULT_IMPUTATION_POLICY
    warmup_data: pd.DataFrame | None = None


def _json_scalar(value: object) -> object:
    """Return a scalar that can be passed directly to ``json.dumps``."""

    if value is None or value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class CurrentMonitoringSeriesLineage:
    """JSON-compatible authority and estimate lineage for one required CPI series."""

    series_key: str
    series_label: str
    series_id: str
    missing_reference_month: pd.Timestamp
    september_endpoint_month: pd.Timestamp
    september_endpoint_value: float | None
    september_value_provenance: str | None
    september_release_timestamp: object | None
    september_release_timestamp_provenance: str | None
    september_timing_status: str | None
    november_endpoint_month: pd.Timestamp
    november_endpoint_value: float | None
    november_value_provenance: str | None
    november_release_timestamp: object | None
    november_release_timestamp_provenance: str | None
    november_timing_status: str | None
    estimate_method: str
    estimate_value: float | None
    estimate_attempted: bool
    available: bool
    failure_code: str | None
    failure_reason: str | None
    uses_estimated_input: bool
    estimated_input_months: tuple[str, ...]
    ex_post_status: str
    data_vintage_status: str
    retrieved_at: object | None

    def to_dict(self) -> dict[str, object]:
        """Serialize without pandas timestamps, NA sentinels, or non-finite floats."""

        return {
            "series_key": self.series_key,
            "series_label": self.series_label,
            "series_id": self.series_id,
            "missing_reference_month": _json_scalar(self.missing_reference_month),
            "september_endpoint": {
                "month": _json_scalar(self.september_endpoint_month),
                "value": _json_scalar(self.september_endpoint_value),
                "value_provenance": _json_scalar(self.september_value_provenance),
                "release_timestamp": _json_scalar(self.september_release_timestamp),
                "release_timestamp_provenance": _json_scalar(
                    self.september_release_timestamp_provenance
                ),
                "timing_status": _json_scalar(self.september_timing_status),
            },
            "november_endpoint": {
                "month": _json_scalar(self.november_endpoint_month),
                "value": _json_scalar(self.november_endpoint_value),
                "value_provenance": _json_scalar(self.november_value_provenance),
                "release_timestamp": _json_scalar(self.november_release_timestamp),
                "release_timestamp_provenance": _json_scalar(
                    self.november_release_timestamp_provenance
                ),
                "timing_status": _json_scalar(self.november_timing_status),
            },
            "estimate_method": self.estimate_method,
            "estimate_value": _json_scalar(self.estimate_value),
            "estimate_attempted": self.estimate_attempted,
            "available": self.available,
            "failure_code": _json_scalar(self.failure_code),
            "failure_reason": _json_scalar(self.failure_reason),
            "uses_estimated_input": self.uses_estimated_input,
            "estimated_input_months": list(self.estimated_input_months),
            "ex_post_status": self.ex_post_status,
            "data_vintage_status": self.data_vintage_status,
            "retrieved_at": _json_scalar(self.retrieved_at),
        }


def resolve_imputation_policy(value: str) -> ImputationPolicy:
    """Validate and normalize the CPI missing-data policy."""

    policy = str(value)
    if policy not in VALID_IMPUTATION_POLICIES:
        raise ValueError(
            f"Unknown imputation policy: {value}. Expected one of {list(VALID_IMPUTATION_POLICIES)}"
        )
    return cast(ImputationPolicy, policy)


def resolve_current_monitoring_scenario(
    value: str,
) -> CurrentMonitoringScenarioId:
    """Validate a current-monitoring missing-CPI scenario identifier."""

    scenario_id = str(value)
    if scenario_id not in CURRENT_MONITORING_SCENARIOS:
        raise ValueError(
            f"Unknown current-monitoring scenario: {value}. "
            f"Expected one of {list(CURRENT_MONITORING_SCENARIOS)}"
        )
    return cast(CurrentMonitoringScenarioId, scenario_id)


def _boolean_values(values: pd.Series) -> pd.Series:
    """Coerce cache/source flag values without treating non-empty strings as true."""

    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _missingness_flags(
    observed_level: pd.Series,
    explicit_originally_missing: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Separate within-coverage missing observations from pre-series unavailability."""

    explicit = (
        pd.Series(False, index=observed_level.index, dtype=bool)
        if explicit_originally_missing is None
        else explicit_originally_missing.fillna(False).astype(bool)
    )
    raw_observed = observed_level.mask(explicit)
    coverage_started = raw_observed.notna().cummax()
    originally_missing = explicit | (raw_observed.isna() & coverage_started)
    source_unavailable = raw_observed.isna() & ~coverage_started & ~explicit
    return raw_observed, originally_missing.astype(bool), source_unavailable.astype(bool)


def fred_api_key() -> str | None:
    """Load the optional FRED API key from the environment or project .env."""

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    value = os.getenv("FRED_API_KEY", "").strip()
    return value or None


def _safe_error(exc: Exception, secret: str | None = None) -> str:
    """Return a diagnostic string that cannot leak configured secrets."""

    message = str(exc)
    if secret:
        message = message.replace(secret, "[redacted]")
    return f"{exc.__class__.__name__}: {message}"


def _fred_observations_to_frame(payload: dict[str, object], series_id: str) -> pd.DataFrame:
    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise ValueError(f"Unexpected FRED API response for {series_id}")

    rows = [
        {
            "date": item.get("date"),
            series_id: item.get("value"),
        }
        for item in observations
        if isinstance(item, dict)
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), series_id: []})

    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
    return df[["date", series_id]].dropna().reset_index(drop=True)


def _merge_series_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for current in frames:
        merged = current if merged is None else merged.merge(current, on="date", how="outer")

    if merged is None:
        raise ValueError("No series IDs provided")

    return merged.sort_values("date").reset_index(drop=True)


def fetch_fred_series_api(
    series_id: str,
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch one series from the official FRED observations API."""

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start_date is not None:
        params["observation_start"] = start_date
    if end_date is not None:
        params["observation_end"] = end_date

    response = requests.get(FRED_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "error_code" in payload:
        raise ValueError(f"FRED API error for {series_id}: {payload.get('error_code')}")

    return _fred_observations_to_frame(payload, series_id)


def merge_fred_series_api(
    series_ids: Iterable[str],
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch and outer-join official FRED API series by date."""

    return _merge_series_frames(
        fetch_fred_series_api(series_id, api_key, start_date=start_date, end_date=end_date)
        for series_id in series_ids
    )


def fetch_fred_series_csv(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch a public FRED series through the CSV endpoint.

    This endpoint does not require a FRED API key. It is suitable for a research
    scaffold and avoids hardcoding keys in notebooks. ``start_date`` and
    ``end_date`` are inclusive bounds; ``None`` means unbounded.
    """

    url = FRED_CSV_URL.format(series_id=series_id)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    if "observation_date" not in df.columns or series_id not in df.columns:
        raise ValueError(f"Unexpected FRED CSV format for {series_id}")

    df = df.rename(columns={"observation_date": "date", series_id: series_id})
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")

    df = slice_date_range(df, start_date=start_date, end_date=end_date)
    return df[["date", series_id]].dropna().reset_index(drop=True)


def fetch_fred_series(
    series_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for the public FRED CSV fetcher."""

    return fetch_fred_series_csv(series_id, start_date=start_date, end_date=end_date)


def merge_fred_series(
    series_ids: Iterable[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch and outer-join multiple FRED series by date."""

    return _merge_series_frames(
        fetch_fred_series_csv(series_id, start_date=start_date, end_date=end_date)
        for series_id in series_ids
    )


def slice_date_range(
    df: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Slice a frame to the inclusive ``[start_date, end_date]`` window.

    ``None`` bounds are open ends, so passing both as ``None`` is a no-op copy.
    """

    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    mask = pd.Series(True, index=out.index)
    if start_date is not None:
        mask &= dates >= pd.to_datetime(start_date)
    if end_date is not None:
        mask &= dates <= pd.to_datetime(end_date)
    return out.loc[mask].reset_index(drop=True)


def apply_sample_mode(
    df: pd.DataFrame, mode: SampleMode | str, date_col: str = "date"
) -> pd.DataFrame:
    """Slice a frame to a named sample mode's inclusive date window."""

    resolved = resolve_sample_mode(mode)
    return slice_date_range(
        df, start_date=resolved.start_date, end_date=resolved.end_date, date_col=date_col
    )


def latest_valid_observation_date(
    df: pd.DataFrame,
    value_col: str = "inflation_yoy",
    date_col: str = "date",
) -> pd.Timestamp | None:
    """Return the latest date where a value column is actually available.

    Returns None when the requested columns are absent (for example an optional
    inflation measure that was not loaded) rather than raising, so callers can
    probe possibly-missing series safely.
    """

    if value_col not in df.columns or date_col not in df.columns:
        return None

    dates = pd.to_datetime(df.loc[df[value_col].notna(), date_col])
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())


def date_label(date: object) -> str:
    """Format an optional date for dashboard/report status text."""

    if date is None or pd.isna(date):
        return "unknown"
    return str(pd.to_datetime(date).date())


def timestamp_label(timestamp: object) -> str:
    """Format an optional timestamp without dropping its time or UTC offset."""

    if timestamp is None or pd.isna(timestamp):
        return "unknown"
    return pd.Timestamp(timestamp).isoformat()


def available_inflation_measures(df: pd.DataFrame) -> tuple[str, ...]:
    """Return inflation measure keys with at least one usable YoY observation."""

    available: list[str] = []
    for key, measure in INFLATION_MEASURES.items():
        if measure.yoy_col in df.columns and df[measure.yoy_col].notna().any():
            available.append(key)
    return tuple(available)


def monthly_last(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Convert daily/mixed-frequency data to monthly last observations."""

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.set_index(date_col).sort_index().resample("ME").last().reset_index()
    out["date"] = out["date"].dt.to_period("M").dt.to_timestamp("M")
    return out


def _monthly_macro_physical_rows(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Select one coherent physical macro row per month.

    The latest dated row wins. Stable input order breaks ties for duplicate
    dates, with the last physical row selected. Unlike ``resample().last()``,
    this preserves every value and null from that one source row.
    """

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col, kind="stable")
    reference_month = out[date_col].dt.to_period("M")
    physical_counts = reference_month.map(reference_month.value_counts()).astype(int)
    if SOURCE_MONTH_OBSERVATION_COUNT_COLUMN in out.columns:
        incoming_counts = pd.to_numeric(
            out[SOURCE_MONTH_OBSERVATION_COUNT_COLUMN],
            errors="coerce",
        ).fillna(1)
        physical_counts = pd.concat(
            [physical_counts, incoming_counts],
            axis=1,
        ).max(axis=1)
    out[SOURCE_MONTH_OBSERVATION_COUNT_COLUMN] = physical_counts.astype("Int64")
    out = out.loc[~reference_month.duplicated(keep="last")].copy()
    out[date_col] = out[date_col].dt.to_period("M").dt.to_timestamp("M")
    return out.reset_index(drop=True)


def _source_month_counts_by_period(
    frame: pd.DataFrame,
    periods: pd.Series,
) -> dict[pd.Period, int]:
    """Return durable physical-row counts for each visible reference month."""

    physical = periods.value_counts().astype(int)
    counts = {period: int(count) for period, count in physical.items()}
    if SOURCE_MONTH_OBSERVATION_COUNT_COLUMN not in frame.columns:
        return counts

    incoming = pd.to_numeric(
        frame[SOURCE_MONTH_OBSERVATION_COUNT_COLUMN],
        errors="coerce",
    )
    for period, values in incoming.groupby(periods):
        known = values.dropna()
        if not known.empty:
            counts[period] = max(counts.get(period, 0), int(known.max()))
    return counts


def _current_monitoring_required_periods() -> frozenset[pd.Period]:
    return frozenset(
        pd.Timestamp(month).to_period("M")
        for month in (
            CURRENT_MONITORING_PREVIOUS_ENDPOINT_MONTH,
            CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH,
            CURRENT_MONITORING_FOLLOWING_ENDPOINT_MONTH,
        )
    )


def validate_monthly_warmup_authority(
    frame: pd.DataFrame,
    *,
    sample_start_date: str | None,
    date_col: str = "date",
    authority_label: str = "Monthly macro authority",
    allow_full_history_origin: bool = False,
) -> None:
    """Require coherent calendar months and exact pre-sample YoY warm-up.

    September-November 2025 row defects are deliberately left to the current-
    monitoring lineage validator so they produce a structured unavailable
    bundle. Every other month in the relevant authority range must be present
    exactly once.
    """

    if date_col not in frame.columns:
        raise ValueError(f"{authority_label} requires a {date_col} column")
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{authority_label} contains invalid dates")
    if dates.empty:
        raise ValueError(f"{authority_label} is empty")

    periods = dates.dt.to_period("M")
    counts = _source_month_counts_by_period(frame, periods)
    visible_periods = set(counts)
    visible_start = min(visible_periods)
    visible_end = max(visible_periods)
    monitoring_periods = _current_monitoring_required_periods()

    sample_start = (
        pd.Timestamp(sample_start_date).to_period("M") if sample_start_date is not None else None
    )
    if sample_start is None and allow_full_history_origin and "inflation_yoy" in frame:
        earliest_rows = periods.eq(visible_start)
        if frame.loc[earliest_rows, "inflation_yoy"].notna().any():
            raise ValueError(
                f"{authority_label} is sample-trimmed; the common raw warm-up "
                "authority is required before scenario calculation"
            )

    coverage_start = visible_start
    if sample_start is not None:
        required = pd.period_range(
            sample_start - YOY_WARMUP_MONTHS,
            sample_start - 1,
            freq="M",
        )
        invalid_warmup = [period for period in required if counts.get(period, 0) != 1]
        if invalid_warmup:
            invalid = ", ".join(str(period) for period in invalid_warmup)
            raise ValueError(
                f"{authority_label} lacks the required {YOY_WARMUP_MONTHS}-month "
                f"pre-sample warm-up before {sample_start}: exactly "
                f"{YOY_WARMUP_MONTHS} consecutive pre-sample months are required; "
                f"missing or ambiguous: {invalid}"
            )
        coverage_start = required[0]

    expected = set(pd.period_range(coverage_start, visible_end, freq="M"))
    missing = sorted(expected - visible_periods - monitoring_periods)
    ambiguous = sorted(
        period
        for period, count in counts.items()
        if coverage_start <= period <= visible_end
        and period not in monitoring_periods
        and count != 1
    )
    if missing or ambiguous:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(str(period) for period in missing))
        if ambiguous:
            details.append("ambiguous " + ", ".join(str(period) for period in ambiguous))
        raise ValueError(
            f"{authority_label} must contain one coherent row per calendar month; "
            + "; ".join(details)
        )


def _mask_ambiguous_current_monitoring_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent a selected duplicate row from becoming CPI value authority."""

    out = frame.copy()
    periods = pd.to_datetime(out["date"], errors="coerce").dt.to_period("M")
    counts = _source_month_counts_by_period(out, periods)
    ambiguous_periods = {
        period for period in _current_monitoring_required_periods() if counts.get(period, 0) > 1
    }
    if not ambiguous_periods:
        return out

    ambiguous_rows = periods.isin(ambiguous_periods)
    for measure_key in CURRENT_MONITORING_MEASURE_KEYS:
        measure = INFLATION_MEASURES[measure_key]
        for value_col in (
            measure.series_id,
            measure.observed_level_col,
            measure.level_col,
        ):
            if value_col in out.columns:
                out.loc[ambiguous_rows, value_col] = np.nan
        if measure.value_provenance_col in out.columns:
            out.loc[
                ambiguous_rows,
                measure.value_provenance_col,
            ] = VALUE_PROVENANCE_UNVERIFIED
        if measure.release_timestamp_col in out.columns:
            out.loc[ambiguous_rows, measure.release_timestamp_col] = pd.NaT
        if measure.release_timestamp_provenance_col in out.columns:
            out.loc[
                ambiguous_rows,
                measure.release_timestamp_provenance_col,
            ] = RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED
        if measure.release_timing_status_col in out.columns:
            out.loc[
                ambiguous_rows,
                measure.release_timing_status_col,
            ] = TIMING_STATUS_UNAVAILABLE
    return out


def _rowwise_latest_timestamp(*values: pd.Series) -> pd.Series:
    """Return the latest known timestamp per row, preserving unknown rows as NaT."""

    if not values:
        return pd.Series(dtype="datetime64[ns, UTC]")
    timestamps = pd.concat(
        [
            pd.to_datetime(value, errors="coerce", utc=True).astype("datetime64[ns, UTC]")
            for value in values
        ],
        axis=1,
    )
    return pd.to_datetime(timestamps.max(axis=1), errors="coerce", utc=True)


def _utc_timestamps(values: object) -> pd.Series:
    """Convert timestamp metadata to UTC while retaining time-of-day."""

    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        return parsed.astype("datetime64[ns, UTC]")
    if isinstance(parsed, pd.DatetimeIndex):
        return pd.Series(parsed.astype("datetime64[ns, UTC]"))
    raise TypeError("Timestamp metadata must be one-dimensional")


def _has_explicit_timezone(value: object) -> bool:
    """Return whether a timestamp value carries an explicit UTC offset/timezone."""

    if value is None or pd.isna(value):
        return False
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return False
    return timestamp.tzinfo is not None and timestamp.utcoffset() is not None


def _incoming_release_timing_status(
    frame: pd.DataFrame,
    measure: InflationMeasure,
) -> pd.Series:
    """Resolve source release status without borrowing headline trust."""

    candidate_columns = [
        measure.release_timing_status_col,
        measure.yoy_timing_status_col,
    ]
    if measure.key == HEADLINE_INFLATION_MEASURE:
        candidate_columns.append("timing_status")

    status = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in candidate_columns:
        if column not in frame.columns:
            continue
        candidate = frame[column].astype("string").str.strip().replace("", pd.NA)
        status = status.fillna(candidate)
    return status


def _trusted_release_timestamps(
    values: pd.Series,
    provenance: pd.Series,
    timing_status: pd.Series,
    *,
    value_available: pd.Series,
) -> pd.Series:
    """Accept exact publication times only when every trust gate is explicit."""

    timezone_known = values.map(_has_explicit_timezone).astype(bool)
    actual_metadata = (
        provenance.astype("string").eq(RELEASE_TIMESTAMP_PROVENANCE_ACTUAL).fillna(False)
    )
    release_aligned = timing_status.astype("string").eq(TIMING_STATUS_RELEASE_ALIGNED).fillna(False)
    parsed = _utc_timestamps(values)
    return parsed.where(
        timezone_known
        & actual_metadata
        & release_aligned
        & value_available.fillna(False).astype(bool)
    )


def _resolved_release_timing_status(
    supplied_status: pd.Series,
    trusted_releases: pd.Series,
    *,
    value_available: pd.Series,
) -> pd.Series:
    """Preserve non-exact source status while downgrading unsupported exact claims."""

    available = value_available.fillna(False).astype(bool)
    supplied = supplied_status.astype("string").str.strip().replace("", pd.NA)
    status = pd.Series(
        TIMING_STATUS_UNAVAILABLE,
        index=supplied_status.index,
        dtype="string",
    )
    status.loc[available] = TIMING_STATUS_REFERENCE_MONTH_ONLY
    supplied_nonexact = (
        available & supplied.notna() & ~supplied.eq(TIMING_STATUS_RELEASE_ALIGNED).fillna(False)
    )
    status.loc[supplied_nonexact] = supplied.loc[supplied_nonexact]
    status.loc[trusted_releases.notna()] = TIMING_STATUS_RELEASE_ALIGNED
    return status


@dataclass(frozen=True)
class _MeasureObservedAuthority:
    """Normalized value and timing authority retained from one physical monthly row."""

    present: bool
    observed_level: pd.Series
    originally_missing: pd.Series
    source_unavailable: pd.Series
    source_estimated: pd.Series
    value_provenance: pd.Series
    releases: pd.Series
    release_provenance: pd.Series
    release_timing_status: pd.Series


def _measure_observed_authority(
    frame: pd.DataFrame,
    measure: InflationMeasure,
) -> _MeasureObservedAuthority:
    """Resolve official-value authority separately from conservative H5 timing."""

    has_source_column = any(
        column in frame.columns
        for column in (
            measure.series_id,
            measure.observed_level_col,
            measure.level_col,
        )
    )
    source_present = has_source_column
    if measure.source_present_col in frame.columns:
        source_present = source_present and bool(
            _boolean_values(frame[measure.source_present_col]).all()
        )

    canonical_fred_input = measure.series_id in frame.columns and source_present
    if not source_present:
        missing = pd.Series(np.nan, index=frame.index, dtype=float)
        false = pd.Series(False, index=frame.index, dtype=bool)
        unverified = pd.Series(
            VALUE_PROVENANCE_UNVERIFIED,
            index=frame.index,
            dtype="string",
        )
        unavailable_timing = pd.Series(
            TIMING_STATUS_UNAVAILABLE,
            index=frame.index,
            dtype="string",
        )
        return _MeasureObservedAuthority(
            present=False,
            observed_level=missing,
            originally_missing=false,
            source_unavailable=pd.Series(True, index=frame.index, dtype=bool),
            source_estimated=false,
            value_provenance=unverified,
            releases=pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]"),
            release_provenance=pd.Series(
                RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED,
                index=frame.index,
                dtype="string",
            ),
            release_timing_status=unavailable_timing,
        )

    if canonical_fred_input:
        observed_level = pd.to_numeric(frame[measure.series_id], errors="coerce")
    elif measure.observed_level_col in frame.columns:
        observed_level = pd.to_numeric(frame[measure.observed_level_col], errors="coerce")
    elif measure.level_col in frame.columns:
        observed_level = pd.to_numeric(frame[measure.level_col], errors="coerce")
    else:  # pragma: no cover - guarded by has_source_column above
        raise AssertionError(f"No value column resolved for {measure.key}")

    source_estimated = pd.Series(False, index=frame.index, dtype=bool)
    if measure.imputed_col in frame.columns:
        source_estimated |= _boolean_values(frame[measure.imputed_col])
    if "uses_estimated_input" in frame.columns:
        source_estimated |= _boolean_values(frame["uses_estimated_input"])
    if "uses_imputed_input" in frame.columns:
        source_estimated |= _boolean_values(frame["uses_imputed_input"])
    if "estimated_input_months" in frame.columns:
        source_estimated |= frame["estimated_input_months"].map(
            lambda value: (
                bool(value)
                if value is not None and value is not pd.NA and not isinstance(value, float)
                else False
            )
        )

    explicit_missing = source_estimated.copy()
    if measure.originally_missing_col in frame.columns:
        explicit_missing |= _boolean_values(frame[measure.originally_missing_col])
    observed_level, originally_missing, source_unavailable = _missingness_flags(
        observed_level,
        explicit_missing,
    )

    if measure.value_provenance_col in frame.columns:
        value_provenance = (
            frame[measure.value_provenance_col]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .fillna(VALUE_PROVENANCE_UNVERIFIED)
        )
    elif canonical_fred_input:
        value_provenance = pd.Series(
            VALUE_PROVENANCE_OFFICIAL_FRED,
            index=frame.index,
            dtype="string",
        )
    else:
        value_provenance = pd.Series(
            VALUE_PROVENANCE_UNVERIFIED,
            index=frame.index,
            dtype="string",
        )

    supplied_releases = frame.get(
        measure.release_timestamp_col,
        pd.Series(pd.NaT, index=frame.index),
    )
    supplied_release_provenance = frame.get(
        measure.release_timestamp_provenance_col,
        pd.Series(pd.NA, index=frame.index, dtype="string"),
    ).astype("string")
    supplied_release_timing_status = _incoming_release_timing_status(frame, measure)
    releases = _trusted_release_timestamps(
        supplied_releases,
        supplied_release_provenance,
        supplied_release_timing_status,
        value_available=observed_level.notna(),
    )
    release_provenance = pd.Series(
        RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED,
        index=frame.index,
        dtype="string",
    )
    release_provenance.loc[releases.notna()] = RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
    release_timing_status = _resolved_release_timing_status(
        supplied_release_timing_status,
        releases,
        value_available=observed_level.notna(),
    )
    return _MeasureObservedAuthority(
        present=True,
        observed_level=observed_level,
        originally_missing=originally_missing,
        source_unavailable=source_unavailable,
        source_estimated=source_estimated,
        value_provenance=value_provenance,
        releases=releases,
        release_provenance=release_provenance,
        release_timing_status=release_timing_status,
    )


def _exact_month_source_count(
    frame: pd.DataFrame,
    month: pd.Timestamp,
) -> int:
    """Return the durable physical source-row count for one exact month."""

    rows = frame.index[frame["reference_month"].eq(month)]
    if len(rows) == 0:
        return 0
    count = len(rows)
    if SOURCE_MONTH_OBSERVATION_COUNT_COLUMN in frame.columns:
        incoming = pd.to_numeric(
            frame.loc[rows, SOURCE_MONTH_OBSERVATION_COUNT_COLUMN],
            errors="coerce",
        ).dropna()
        if not incoming.empty:
            count = max(count, int(incoming.max()))
    return count


def _lineage_endpoint_values(
    frame: pd.DataFrame,
    authority: _MeasureObservedAuthority,
    month: pd.Timestamp,
) -> tuple[object, object, object, object, object]:
    """Return exact-month value/timing fields without using physical adjacency."""

    rows = frame.index[frame["reference_month"].eq(month)]
    if len(rows) != 1 or _exact_month_source_count(frame, month) != 1:
        return None, None, None, None, None
    row_index = rows[0]
    return (
        authority.observed_level.loc[row_index],
        authority.value_provenance.loc[row_index],
        authority.releases.loc[row_index],
        authority.release_provenance.loc[row_index],
        authority.release_timing_status.loc[row_index],
    )


def _current_monitoring_series_lineage(
    frame: pd.DataFrame,
    measure: InflationMeasure,
    authority: _MeasureObservedAuthority,
    scenario_id: CurrentMonitoringScenarioId,
) -> CurrentMonitoringSeriesLineage:
    """Validate exact official endpoints without making release precision a value gate."""

    target = pd.Timestamp(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH)
    september = pd.Timestamp(CURRENT_MONITORING_PREVIOUS_ENDPOINT_MONTH)
    november = pd.Timestamp(CURRENT_MONITORING_FOLLOWING_ENDPOINT_MONTH)
    failures: list[tuple[str, str]] = []

    if not authority.present:
        failures.append(
            (
                "required_series_absent",
                f"Required {measure.label} series {measure.series_id} is absent.",
            )
        )
    elif authority.source_estimated.any():
        failures.append(
            (
                "source_contains_estimated_rows",
                f"Required {measure.label} source authority contains estimated or imputed rows.",
            )
        )

    target_rows = frame.index[frame["reference_month"].eq(target)]
    target_source_count = _exact_month_source_count(frame, target)
    if target_source_count == 0:
        failures.append(
            (
                "missing_reference_month_absent",
                f"Required missing reference month {target.date()} is absent for {measure.label}.",
            )
        )
    elif target_source_count != 1:
        failures.append(
            (
                "missing_reference_month_ambiguous",
                f"Missing reference month {target.date()} is ambiguous for {measure.label}.",
            )
        )
    elif authority.present:
        target_index = target_rows[0]
        if pd.notna(authority.observed_level.loc[target_index]):
            failures.append(
                (
                    "missing_reference_month_has_official_value",
                    f"{measure.label} {target.date()} is not an official missing observation.",
                )
            )
        elif not bool(authority.originally_missing.loc[target_index]) or bool(
            authority.source_unavailable.loc[target_index]
        ):
            failures.append(
                (
                    "missing_reference_month_not_official_gap",
                    f"{measure.label} {target.date()} is not a coherent official interior gap.",
                )
            )

    for endpoint_label, endpoint_month in (
        ("September", september),
        ("November", november),
    ):
        endpoint_rows = frame.index[frame["reference_month"].eq(endpoint_month)]
        endpoint_source_count = _exact_month_source_count(frame, endpoint_month)
        endpoint_code = endpoint_label.lower()
        if endpoint_source_count == 0:
            failures.append(
                (
                    f"{endpoint_code}_endpoint_absent",
                    f"Exact {endpoint_label} {endpoint_month.year} endpoint is absent for {measure.label}.",
                )
            )
            continue
        if endpoint_source_count != 1:
            failures.append(
                (
                    f"{endpoint_code}_endpoint_ambiguous",
                    f"Exact {endpoint_label} {endpoint_month.year} endpoint is ambiguous for {measure.label}.",
                )
            )
            continue
        if not authority.present:
            continue
        endpoint_index = endpoint_rows[0]
        endpoint_value = authority.observed_level.loc[endpoint_index]
        if pd.isna(endpoint_value):
            failures.append(
                (
                    f"{endpoint_code}_endpoint_null",
                    f"Exact {endpoint_label} {endpoint_month.year} official value is null for {measure.label}.",
                )
            )
        elif float(endpoint_value) <= 0:
            failures.append(
                (
                    f"{endpoint_code}_endpoint_invalid_value",
                    f"Exact {endpoint_label} {endpoint_month.year} value is invalid for {measure.label}.",
                )
            )
        if bool(authority.source_estimated.loc[endpoint_index]):
            failures.append(
                (
                    f"{endpoint_code}_endpoint_estimated",
                    f"Exact {endpoint_label} {endpoint_month.year} value is estimated or imputed for {measure.label}.",
                )
            )
        if bool(authority.originally_missing.loc[endpoint_index]) or bool(
            authority.source_unavailable.loc[endpoint_index]
        ):
            failures.append(
                (
                    f"{endpoint_code}_endpoint_not_observed",
                    f"Exact {endpoint_label} {endpoint_month.year} value is not an observed official value for {measure.label}.",
                )
            )
        if authority.value_provenance.loc[endpoint_index] != VALUE_PROVENANCE_OFFICIAL_FRED:
            failures.append(
                (
                    f"{endpoint_code}_endpoint_invalid_provenance",
                    f"Exact {endpoint_label} {endpoint_month.year} lacks approved official value provenance for {measure.label}.",
                )
            )

    september_values = _lineage_endpoint_values(frame, authority, september)
    november_values = _lineage_endpoint_values(frame, authority, november)
    failure_code, failure_reason = failures[0] if failures else (None, None)
    retrieved_at = None
    november_rows = frame.index[frame["reference_month"].eq(november)]
    november_source_count = _exact_month_source_count(frame, november)
    if november_source_count == 1 and "retrieved_at" in frame.columns:
        retrieved_at = frame.loc[november_rows[0], "retrieved_at"]
    data_vintage_status = DATA_VINTAGE_STATUS_LATEST_REVISED
    if november_source_count == 1 and "data_vintage_status" in frame.columns:
        candidate_status = frame.loc[november_rows[0], "data_vintage_status"]
        if pd.notna(candidate_status):
            data_vintage_status = str(candidate_status)

    return CurrentMonitoringSeriesLineage(
        series_key=measure.key,
        series_label=measure.label,
        series_id=measure.series_id,
        missing_reference_month=target,
        september_endpoint_month=september,
        september_endpoint_value=(
            float(september_values[0]) if pd.notna(september_values[0]) else None
        ),
        september_value_provenance=(
            str(september_values[1]) if pd.notna(september_values[1]) else None
        ),
        september_release_timestamp=september_values[2],
        september_release_timestamp_provenance=(
            str(september_values[3]) if pd.notna(september_values[3]) else None
        ),
        september_timing_status=(
            str(september_values[4]) if pd.notna(september_values[4]) else None
        ),
        november_endpoint_month=november,
        november_endpoint_value=(
            float(november_values[0]) if pd.notna(november_values[0]) else None
        ),
        november_value_provenance=(
            str(november_values[1]) if pd.notna(november_values[1]) else None
        ),
        november_release_timestamp=november_values[2],
        november_release_timestamp_provenance=(
            str(november_values[3]) if pd.notna(november_values[3]) else None
        ),
        november_timing_status=(str(november_values[4]) if pd.notna(november_values[4]) else None),
        estimate_method=CURRENT_MONITORING_ESTIMATE_METHODS[scenario_id],
        estimate_value=None,
        estimate_attempted=False,
        available=not failures,
        failure_code=failure_code,
        failure_reason=failure_reason,
        uses_estimated_input=False,
        estimated_input_months=(),
        ex_post_status="ex_post_lookahead_required",
        data_vintage_status=data_vintage_status,
        retrieved_at=retrieved_at,
    )


def _scenario_estimate_value(
    lineage: CurrentMonitoringSeriesLineage,
    scenario_id: CurrentMonitoringScenarioId,
) -> float:
    """Calculate only from the exact validated September and November values."""

    september = float(lineage.september_endpoint_value)
    november = float(lineage.november_endpoint_value)
    if scenario_id == "low":
        return september
    if scenario_id == "high":
        return november
    return float(np.sqrt(september * november))


def _retrieval_timestamp_column(
    out: pd.DataFrame,
    retrieved_at: object | None,
) -> pd.Series:
    """Resolve actual load metadata without inventing a publication timestamp."""

    if "retrieved_at" in out.columns:
        values = pd.to_datetime(out["retrieved_at"], errors="coerce", utc=True)
        return values
    if retrieved_at is None:
        return pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    value = pd.to_datetime(retrieved_at, errors="coerce", utc=True)
    if pd.isna(value):
        return pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    return pd.Series(value, index=out.index, dtype="datetime64[ns, UTC]")


def build_base_frame(
    merged: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
    current_monitoring_scenario: CurrentMonitoringScenarioId | str | None = None,
    retrieved_at: object | None = None,
) -> pd.DataFrame:
    """Build the canonical monthly base frame from merged raw FRED series.

    ``observed_only`` leaves every officially missing inflation-index level
    missing. ``ex_post_continuity`` with an explicit current-monitoring scenario
    applies the approved low/base/high assumption only to the permanently
    missing October 2025 headline/core CPI levels. A call without an explicit
    scenario retains the legacy isolated-gap bridge for backward compatibility;
    current and historical research routing must use the explicit seams. The raw
    official level remains null and separate from every estimate. Rows before
    ``start_date`` feed the 12-month YoY change but never appear in the output.
    """

    policy = resolve_imputation_policy(imputation_policy)
    if policy == "observed_only" and current_monitoring_scenario is not None:
        raise ValueError("A current-monitoring scenario is valid only with ex_post_continuity")
    explicit_current_scenario = current_monitoring_scenario is not None
    scenario_id = (
        resolve_current_monitoring_scenario(current_monitoring_scenario)
        if policy == "ex_post_continuity" and explicit_current_scenario
        else None
    )
    scenario_formula = scenario_id or "base"
    out = _monthly_macro_physical_rows(merged)
    out = _mask_ambiguous_current_monitoring_values(out)
    out = out.rename(columns={"TB3MS": "tbill_3m"})
    out["reference_month"] = pd.to_datetime(out["date"], errors="coerce")
    out["retrieved_at"] = _retrieval_timestamp_column(out, retrieved_at)
    if "vintage_timestamp" in out.columns:
        out["vintage_timestamp"] = _utc_timestamps(out["vintage_timestamp"])
    else:
        out["vintage_timestamp"] = pd.Series(
            pd.NaT,
            index=out.index,
            dtype="datetime64[ns, UTC]",
        )
    out["data_vintage_status"] = DATA_VINTAGE_STATUS_LATEST_REVISED
    out["imputation_policy"] = policy
    out["scenario_id"] = pd.Series(
        scenario_id if scenario_id is not None else pd.NA,
        index=out.index,
        dtype="string",
    )
    out["calibration_policy"] = pd.Series(
        CURRENT_MONITORING_CALIBRATION_POLICY if scenario_id is not None else pd.NA,
        index=out.index,
        dtype="string",
    )

    authorities = {
        measure.key: _measure_observed_authority(out, measure)
        for measure in INFLATION_MEASURES.values()
    }
    scenario_lineages: dict[str, CurrentMonitoringSeriesLineage] = {}
    scenario_bundle_available = False
    if scenario_id is not None:
        for measure_key in CURRENT_MONITORING_MEASURE_KEYS:
            measure = INFLATION_MEASURES[measure_key]
            lineage = _current_monitoring_series_lineage(
                out,
                measure,
                authorities[measure_key],
                scenario_id,
            )
            if lineage.available:
                estimate_value = _scenario_estimate_value(lineage, scenario_id)
                lineage = replace(
                    lineage,
                    estimate_value=estimate_value,
                    estimate_attempted=True,
                    uses_estimated_input=True,
                    estimated_input_months=(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH,),
                )
            scenario_lineages[measure_key] = lineage
        scenario_bundle_available = all(lineage.available for lineage in scenario_lineages.values())

        serialized_lineage = {key: lineage.to_dict() for key, lineage in scenario_lineages.items()}
        failed_series = tuple(
            key for key, lineage in scenario_lineages.items() if not lineage.available
        )
        failure_reasons = tuple(
            str(lineage.failure_reason)
            for lineage in scenario_lineages.values()
            if lineage.failure_reason
        )
        attempted_estimate = any(
            lineage.estimate_attempted for lineage in scenario_lineages.values()
        )
        out["scenario_bundle_available"] = scenario_bundle_available
        out["scenario_bundle_failure_reason"] = pd.Series(
            pd.NA if scenario_bundle_available else "; ".join(failure_reasons),
            index=out.index,
            dtype="string",
        )
        out["scenario_failed_series"] = pd.Series(
            [failed_series for _ in out.index],
            index=out.index,
            dtype="object",
        )
        out["scenario_estimate_attempted"] = attempted_estimate
        out["scenario_ex_post_status"] = "ex_post_lookahead_required"
        out["scenario_series_lineage"] = pd.Series(
            [deepcopy(serialized_lineage) for _ in out.index],
            index=out.index,
            dtype="object",
        )
        for measure_key, lineage in serialized_lineage.items():
            out[f"{measure_key}_scenario_lineage"] = pd.Series(
                [deepcopy(lineage) for _ in out.index],
                index=out.index,
                dtype="object",
            )

    for measure in INFLATION_MEASURES.values():
        authority = authorities[measure.key]
        if not authority.present and measure.key not in CURRENT_MONITORING_MEASURE_KEYS:
            continue
        observed_level = authority.observed_level
        originally_missing = authority.originally_missing
        source_unavailable = authority.source_unavailable

        if measure.key in CURRENT_MONITORING_MEASURE_KEYS:
            out[measure.source_present_col] = authority.present
        out[measure.observed_level_col] = observed_level
        out[measure.originally_missing_col] = originally_missing.astype(bool)
        out[measure.source_unavailable_col] = source_unavailable.astype(bool)
        out[measure.value_provenance_col] = authority.value_provenance
        out[measure.level_col] = observed_level
        out[measure.imputed_col] = False
        out[measure.imputation_method_col] = pd.Series(
            pd.NA,
            index=out.index,
            dtype="string",
        )
        out[measure.imputation_available_at_col] = pd.Series(
            pd.NaT,
            index=out.index,
            dtype="object",
        )
        out[measure.imputation_availability_basis_col] = pd.Series(
            pd.NA,
            index=out.index,
            dtype="string",
        )
        out[measure.estimate_method_col] = pd.Series(
            pd.NA,
            index=out.index,
            dtype="string",
        )
        out[measure.estimated_reference_month_col] = pd.Series(
            pd.NaT,
            index=out.index,
            dtype="datetime64[ns]",
        )
        out[measure.estimate_value_col] = pd.Series(
            np.nan,
            index=out.index,
            dtype=float,
        )

        releases = authority.releases
        out[measure.release_timestamp_col] = releases
        out[measure.release_timestamp_provenance_col] = authority.release_provenance
        out[measure.release_timing_status_col] = authority.release_timing_status

        safe_log_level = observed_level.where(observed_level > 0)
        log_interp = np.sqrt(safe_log_level.shift(1) * safe_log_level.shift(-1))
        isna = observed_level.isna()
        single_gap = isna & ~isna.shift(1, fill_value=False) & ~isna.shift(-1, fill_value=False)
        if policy == "ex_post_continuity":
            if explicit_current_scenario:
                scenario_level = pd.Series(np.nan, index=out.index, dtype=float)
                if (
                    measure.key in scenario_lineages
                    and scenario_bundle_available
                    and scenario_lineages[measure.key].estimate_value is not None
                ):
                    scenario_level.loc[
                        out["reference_month"].eq(
                            pd.Timestamp(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH)
                        )
                    ] = scenario_lineages[measure.key].estimate_value
                imputed = (
                    out["reference_month"].eq(
                        pd.Timestamp(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH)
                    )
                    & scenario_bundle_available
                    & originally_missing
                    & ~source_unavailable
                    & scenario_level.notna()
                    & scenario_level.gt(0)
                )
            else:
                scenario_level = log_interp
                imputed = (
                    single_gap
                    & originally_missing
                    & ~source_unavailable
                    & safe_log_level.shift(1).notna()
                    & safe_log_level.shift(-1).notna()
                    & scenario_level.notna()
                    & scenario_level.gt(0)
                )
            out[measure.imputed_col] = imputed
            out[measure.level_col] = observed_level.where(~imputed, scenario_level)
            legacy_method = (
                CURRENT_MONITORING_ESTIMATE_METHODS[scenario_formula]
                if explicit_current_scenario
                else "log_linear_bridge"
            )
            estimate_method = CURRENT_MONITORING_ESTIMATE_METHODS[scenario_formula]
            out.loc[imputed, measure.imputation_method_col] = legacy_method
            out.loc[imputed, measure.estimate_method_col] = estimate_method
            out.loc[imputed, measure.estimated_reference_month_col] = out.loc[
                imputed,
                "reference_month",
            ]
            out.loc[imputed, measure.estimate_value_col] = out.loc[
                imputed,
                measure.level_col,
            ]

            if explicit_current_scenario:
                proxy_availability = pd.Series(pd.NaT, index=out.index, dtype="object")
                following_release = pd.Series(
                    pd.NaT,
                    index=out.index,
                    dtype="datetime64[ns, UTC]",
                )
                target_mask = out["reference_month"].eq(
                    pd.Timestamp(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH)
                )
                november_rows = out.index[
                    out["reference_month"].eq(
                        pd.Timestamp(CURRENT_MONITORING_FOLLOWING_ENDPOINT_MONTH)
                    )
                ]
                if (
                    _exact_month_source_count(
                        out,
                        pd.Timestamp(CURRENT_MONITORING_FOLLOWING_ENDPOINT_MONTH),
                    )
                    == 1
                ):
                    november_index = november_rows[0]
                    proxy_availability.loc[target_mask] = pd.Timestamp(
                        CURRENT_MONITORING_FOLLOWING_ENDPOINT_MONTH
                    ) + pd.offsets.MonthEnd(1)
                    following_release.loc[target_mask] = releases.loc[november_index]
            else:
                next_reference_month = pd.to_datetime(out["date"], errors="coerce").shift(-1)
                proxy_availability = next_reference_month + pd.offsets.MonthEnd(1)
                following_release = releases.shift(-1)
            availability = pd.Series(
                proxy_availability.astype("object"),
                index=out.index,
                dtype="object",
            )
            availability.loc[following_release.notna()] = following_release.loc[
                following_release.notna()
            ]
            basis = pd.Series(
                "following_reference_month_end_plus_one_month_proxy",
                index=out.index,
                dtype="string",
            )
            basis.loc[following_release.notna()] = "following_release_timestamp"

            out.loc[imputed, measure.imputation_available_at_col] = availability.loc[imputed]
            out.loc[imputed, measure.imputation_availability_basis_col] = basis.loc[imputed]

        out[measure.yoy_col] = (
            out[measure.level_col].pct_change(
                12,
                fill_method=None,
            )
            * 100
        )
        imputed_input = out[measure.imputed_col].fillna(False).astype(bool)
        missing_input = out[measure.originally_missing_col].fillna(False).astype(bool)
        out[measure.yoy_uses_imputed_input_col] = imputed_input | imputed_input.shift(
            12, fill_value=False
        )
        out[measure.yoy_uses_missing_input_col] = missing_input | missing_input.shift(
            12, fill_value=False
        )

        level_information = releases.where(observed_level.notna())
        if policy == "ex_post_continuity":
            exact_imputation_availability = pd.Series(
                pd.NaT,
                index=out.index,
                dtype="datetime64[ns, UTC]",
            )
            actual_release_basis = (
                out[measure.imputation_availability_basis_col] == "following_release_timestamp"
            )
            exact_imputation_availability.loc[actual_release_basis] = pd.to_datetime(
                out.loc[actual_release_basis, measure.imputation_available_at_col],
                errors="coerce",
                utc=True,
            )
            level_information = level_information.where(
                ~out[measure.imputed_col],
                exact_imputation_availability,
            )
        out[measure.level_information_timestamp_col] = level_information

        lagged_level_information = level_information.shift(12)
        yoy_information = _rowwise_latest_timestamp(
            level_information,
            lagged_level_information,
        )
        yoy_dependencies_known = (
            out[measure.level_col].notna()
            & out[measure.level_col].shift(12).notna()
            & level_information.notna()
            & lagged_level_information.notna()
        )
        out[measure.yoy_information_timestamp_col] = yoy_information.where(
            out[measure.yoy_col].notna() & yoy_dependencies_known
        )
        out[measure.yoy_information_timestamp_provenance_col] = pd.Series(
            INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
            index=out.index,
            dtype="string",
        )
        out.loc[
            out[measure.yoy_information_timestamp_col].notna(),
            measure.yoy_information_timestamp_provenance_col,
        ] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
        yoy_available = out[measure.yoy_col].notna()
        out[measure.yoy_timing_status_col] = pd.Series(
            TIMING_STATUS_UNAVAILABLE,
            index=out.index,
            dtype="string",
        )
        out.loc[
            yoy_available,
            measure.yoy_timing_status_col,
        ] = TIMING_STATUS_REFERENCE_MONTH_ONLY
        out.loc[
            yoy_available & out[measure.yoy_information_timestamp_col].notna(),
            measure.yoy_timing_status_col,
        ] = TIMING_STATUS_RELEASE_ALIGNED

    raw_series_cols = [
        measure.series_id
        for measure in INFLATION_MEASURES.values()
        if measure.series_id in out.columns and measure.series_id != measure.level_col
    ]
    if raw_series_cols:
        out = out.drop(columns=raw_series_cols)

    headline = INFLATION_MEASURES[HEADLINE_INFLATION_MEASURE]
    headline_estimates = out[headline.estimated_reference_month_col].notna()
    if scenario_id is not None:
        scenario_metadata_rows = out["reference_month"].ge(
            pd.Timestamp(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH)
        )
        out["estimate_available_at"] = pd.Series(
            pd.NaT,
            index=out.index,
            dtype="object",
        )
        out["estimate_availability_basis"] = pd.Series(
            pd.NA,
            index=out.index,
            dtype="string",
        )
        for measure_key in CURRENT_MONITORING_MEASURE_KEYS:
            measure = INFLATION_MEASURES[measure_key]
            lineage = scenario_lineages[measure_key]
            out.loc[
                scenario_metadata_rows,
                measure.estimate_method_col,
            ] = lineage.estimate_method
            if lineage.estimate_attempted:
                out.loc[
                    scenario_metadata_rows,
                    measure.estimated_reference_month_col,
                ] = pd.Timestamp(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH)
                out.loc[
                    scenario_metadata_rows,
                    measure.estimate_value_col,
                ] = lineage.estimate_value

        headline_lineage = scenario_lineages[HEADLINE_INFLATION_MEASURE]
        if headline_lineage.estimate_attempted:
            headline_estimates = out[headline.imputed_col].fillna(False).astype(bool)
            if headline_estimates.any():
                headline_estimate_row = out.loc[headline_estimates].iloc[-1]
                estimate_available_at = headline_estimate_row.get(
                    headline.imputation_available_at_col, pd.NaT
                )
                estimate_availability_basis = headline_estimate_row.get(
                    headline.imputation_availability_basis_col, pd.NA
                )
            elif pd.notna(headline_lineage.november_release_timestamp):
                estimate_available_at = headline_lineage.november_release_timestamp
                estimate_availability_basis = "following_release_timestamp"
            else:
                estimate_available_at = pd.Timestamp(
                    CURRENT_MONITORING_FOLLOWING_ENDPOINT_MONTH
                ) + pd.offsets.MonthEnd(1)
                estimate_availability_basis = "following_reference_month_end_plus_one_month_proxy"
            out.loc[scenario_metadata_rows, "estimate_available_at"] = estimate_available_at
            out.loc[
                scenario_metadata_rows,
                "estimate_availability_basis",
            ] = estimate_availability_basis
        headline_estimates = out[headline.imputed_col].fillna(False).astype(bool)
    estimated_months = tuple(
        sorted(
            pd.to_datetime(
                out.loc[headline_estimates, "reference_month"],
                errors="coerce",
            )
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .unique()
        )
    )
    out["estimated_input_months"] = pd.Series(
        [
            estimated_months if bool(uses_estimate) else ()
            for uses_estimate in out[headline.yoy_uses_imputed_input_col].fillna(False)
        ],
        index=out.index,
        dtype="object",
    )
    out["uses_estimated_input"] = (
        out[headline.yoy_uses_imputed_input_col].fillna(False).astype(bool)
    )
    out["uses_imputed_input"] = out["uses_estimated_input"]
    if scenario_id is not None and any(
        lineage.estimate_attempted for lineage in scenario_lineages.values()
    ):
        scenario_metadata_rows = out["reference_month"].ge(
            pd.Timestamp(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH)
        )
        if not scenario_bundle_available:
            out.loc[scenario_metadata_rows, "uses_estimated_input"] = True
            out.loc[scenario_metadata_rows, "uses_imputed_input"] = True
            out.loc[scenario_metadata_rows, "estimated_input_months"] = pd.Series(
                [
                    (CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH,)
                    for _ in out.index[scenario_metadata_rows]
                ],
                index=out.index[scenario_metadata_rows],
                dtype="object",
            )
    if headline.yoy_information_timestamp_col in out.columns:
        out["information_timestamp"] = pd.to_datetime(
            out[headline.yoy_information_timestamp_col],
            errors="coerce",
            utc=True,
        )
        out["information_timestamp_provenance"] = out[
            headline.yoy_information_timestamp_provenance_col
        ].astype("string")
        out["timing_status"] = out[headline.yoy_timing_status_col].astype("string")
    else:
        out["information_timestamp"] = pd.Series(
            pd.NaT,
            index=out.index,
            dtype="datetime64[ns, UTC]",
        )
        out["information_timestamp_provenance"] = INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED
        out["timing_status"] = TIMING_STATUS_UNAVAILABLE
    return slice_date_range(out, start_date=start_date, end_date=end_date)


def build_current_monitoring_scenario_frame(
    observed_warmup: pd.DataFrame,
    *,
    scenario_id: CurrentMonitoringScenarioId | str,
    start_date: str | None = None,
    end_date: str | None = None,
    warmup_sample_start_date: str | None = None,
) -> pd.DataFrame:
    """Build one approved current-monitoring scenario from observed authority.

    The input must remain observed-only. The returned working level may contain
    the bounded October 2025 scenario estimate, while its official
    ``*_observed_level`` column remains null.
    """

    if "imputation_policy" in observed_warmup.columns:
        policies = set(observed_warmup["imputation_policy"].dropna().astype(str))
        if policies and policies != {"observed_only"}:
            raise ValueError("Current-monitoring scenarios require observed_only source authority")
    validate_monthly_warmup_authority(
        observed_warmup,
        sample_start_date=start_date or warmup_sample_start_date,
        authority_label="Current-monitoring scenario warm-up authority",
        allow_full_history_origin=start_date is None and warmup_sample_start_date is None,
    )
    return build_base_frame(
        observed_warmup,
        start_date=start_date,
        end_date=end_date,
        imputation_policy="ex_post_continuity",
        current_monitoring_scenario=scenario_id,
    )


def _fetch_start(start_date: str | None, warmup_months: int = YOY_WARMUP_MONTHS) -> str | None:
    """Move the fetch start earlier than the sample start by the warm-up buffer."""

    if start_date is None:
        return None
    return (pd.to_datetime(start_date) - pd.DateOffset(months=warmup_months)).strftime("%Y-%m-%d")


def _base_frame_and_warmup_from_merged(
    merged: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    imputation_policy: ImputationPolicy,
    retrieved_at: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one observed authority and retain its pre-sample YoY warm-up rows."""

    warmup = build_base_frame(
        merged,
        start_date=_fetch_start(start_date),
        end_date=end_date,
        imputation_policy=imputation_policy,
        retrieved_at=retrieved_at,
    )
    selected = slice_date_range(
        warmup,
        start_date=start_date,
        end_date=end_date,
    )
    return selected, warmup


def _load_base_macro_data_from_api_frames(
    api_key: str,
    *,
    start_date: str | None,
    end_date: str | None,
    imputation_policy: ImputationPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = merge_fred_series_api(
        BASE_FRED_SERIES,
        api_key=api_key,
        start_date=_fetch_start(start_date),
        end_date=end_date,
    )
    return _base_frame_and_warmup_from_merged(
        raw,
        start_date=start_date,
        end_date=end_date,
        imputation_policy=imputation_policy,
        retrieved_at=pd.Timestamp.now(tz="UTC"),
    )


def _load_base_macro_data_from_csv_frames(
    *,
    start_date: str | None,
    end_date: str | None,
    imputation_policy: ImputationPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = merge_fred_series(
        BASE_FRED_SERIES,
        start_date=_fetch_start(start_date),
        end_date=end_date,
    )
    return _base_frame_and_warmup_from_merged(
        raw,
        start_date=start_date,
        end_date=end_date,
        imputation_policy=imputation_policy,
        retrieved_at=pd.Timestamp.now(tz="UTC"),
    )


def load_base_macro_data_from_api(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> pd.DataFrame:
    """Load core macro data through the official FRED API."""

    policy = resolve_imputation_policy(imputation_policy)
    data, _ = _load_base_macro_data_from_api_frames(
        api_key,
        start_date=start_date,
        end_date=end_date,
        imputation_policy=policy,
    )
    return data


def load_base_macro_data_from_csv(
    start_date: str | None = None,
    end_date: str | None = None,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> pd.DataFrame:
    """Load core macro data through public FRED CSV endpoints."""

    policy = resolve_imputation_policy(imputation_policy)
    data, _ = _load_base_macro_data_from_csv_frames(
        start_date=start_date,
        end_date=end_date,
        imputation_policy=policy,
    )
    return data


def load_base_macro_data(
    start_date: str | None = None,
    end_date: str | None = None,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> pd.DataFrame:
    """Load core macro data for an explicit inclusive date range.

    Prefer :func:`load_macro_data_for_mode` so date ranges stay tied to the
    named sample modes in ``config.SAMPLE_MODES``.

    Series:
    - CPIAUCSL: CPI level.
    - TB3MS: 3-month Treasury bill secondary market rate, monthly percent.
      Used as the bill control because FRED has no 1-month bill series before
      2001-07; TB3MS covers the full paper sample (history starts 1934).
    """

    return load_base_macro_data_from_csv(
        start_date=start_date,
        end_date=end_date,
        imputation_policy=imputation_policy,
    )


def load_macro_data_for_mode(
    mode: SampleMode | str = DEFAULT_SAMPLE_MODE,
    allow_demo: bool = False,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> pd.DataFrame:
    """Load core macro data for a named sample mode (see ``config.SAMPLE_MODES``)."""

    result = load_macro_data_for_mode_with_status(
        mode,
        imputation_policy=imputation_policy,
    )
    if result.data_source_used == "demo" and not allow_demo:
        raise RuntimeError(
            "Only emergency demo data is available. Use "
            "load_macro_data_for_mode_with_status() to disclose this explicitly."
        )
    return result.data


def load_macro_data_for_mode_with_status(
    mode: SampleMode | str = DEFAULT_SAMPLE_MODE,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> MacroDataLoadResult:
    """Load macro data with API -> CSV -> cache -> demo fallback order."""

    resolved = resolve_sample_mode(mode)
    policy = resolve_imputation_policy(imputation_policy)
    status_parts: list[str] = []
    key = fred_api_key()

    if key:
        try:
            data, warmup_data = _load_base_macro_data_from_api_frames(
                key,
                start_date=resolved.start_date,
                end_date=resolved.end_date,
                imputation_policy=policy,
            )
            status_parts.append("fred_api: ok")
            return MacroDataLoadResult(
                data=data,
                data_source_used="fred_api",
                live_fetch_status="; ".join(status_parts),
                warmup_data=warmup_data,
                api_key_configured=True,
                imputation_policy=policy,
            )
        except Exception as exc:
            status_parts.append(f"fred_api: failed ({_safe_error(exc, key)})")
    else:
        status_parts.append("fred_api: skipped (FRED_API_KEY not configured)")

    try:
        data, warmup_data = _load_base_macro_data_from_csv_frames(
            start_date=resolved.start_date,
            end_date=resolved.end_date,
            imputation_policy=policy,
        )
        status_parts.append("fred_csv: ok")
        return MacroDataLoadResult(
            data=data,
            data_source_used="fred_csv",
            live_fetch_status="; ".join(status_parts),
            warmup_data=warmup_data,
            api_key_configured=key is not None,
            imputation_policy=policy,
        )
    except Exception as exc:
        status_parts.append(f"fred_csv: failed ({_safe_error(exc)})")

    try:
        cache_path = find_cached_macro_data_file(resolved)
        data, warmup_data = _load_cached_macro_data_frames_for_mode(
            resolved,
            imputation_policy=policy,
        )
        status_parts.append("cached_fred: ok")
        return MacroDataLoadResult(
            data=data,
            data_source_used="cached_fred",
            live_fetch_status="; ".join(status_parts),
            warmup_data=warmup_data,
            cache_file_used=cache_path.name,
            api_key_configured=key is not None,
            imputation_policy=policy,
        )
    except Exception as exc:
        status_parts.append(f"cached_fred: failed ({_safe_error(exc)})")

    warmup_data = make_demo_data()
    data = apply_sample_mode(warmup_data, resolved)
    if data.empty:
        data = make_demo_data()
    data["imputation_policy"] = policy
    status_parts.append("demo: emergency fallback")
    return MacroDataLoadResult(
        data=data,
        data_source_used="demo",
        live_fetch_status="; ".join(status_parts),
        warmup_data=warmup_data,
        api_key_configured=key is not None,
        imputation_policy=policy,
    )


def cached_macro_data_path(mode: SampleMode | str = DEFAULT_SAMPLE_MODE) -> Path:
    """Return the preferred cached raw macro path for a sample mode."""

    resolved = resolve_sample_mode(mode)
    return RAW_DATA_DIR / f"{CACHED_BASE_MACRO_STEM}_{resolved.name}.csv"


def find_cached_macro_data_file(mode: SampleMode | str = DEFAULT_SAMPLE_MODE) -> Path:
    """Find a cached macro dataset usable for the requested sample mode.

    An exact mode-specific cache is preferred. If it is absent, the max-history
    cache is used as a superset and sliced to the requested sample window.
    """

    resolved = resolve_sample_mode(mode)
    exact = cached_macro_data_path(resolved)
    max_history = cached_macro_data_path("max_history")
    candidates = tuple(dict.fromkeys((exact, max_history)))
    existing: list[Path] = []
    validation_failures: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        existing.append(candidate)
        try:
            cached_authority = pd.read_csv(
                candidate,
                usecols=lambda column: column in {"date", SOURCE_MONTH_OBSERVATION_COUNT_COLUMN},
            )
        except (KeyError, ValueError):
            # Let the authoritative cache loader report the schema defect.
            return candidate
        if "date" not in cached_authority.columns:
            # Let the authoritative cache loader report the schema defect.
            return candidate
        try:
            validate_monthly_warmup_authority(
                cached_authority,
                sample_start_date=resolved.start_date,
                authority_label=f"Cached macro data {candidate.name}",
            )
        except ValueError as exc:
            validation_failures.append(str(exc))
            continue
        return candidate

    if existing:
        names = ", ".join(path.name for path in existing)
        detail = "; ".join(validation_failures)
        if resolved.start_date is None:
            raise ValueError(
                f"Cached macro data lacks coherent monthly authority for "
                f"{resolved.name}: {names}. {detail}"
            )
        raise ValueError(
            f"Cached macro data lacks the required {YOY_WARMUP_MONTHS}-month "
            f"pre-sample warm-up for {resolved.name}: {names}. {detail}"
        )

    raise FileNotFoundError(f"No cached macro data found. Expected {exact} or {max_history}.")


def build_macro_cache_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the versioned raw-authority frame written to the macro cache.

    Processed continuity levels are never cache authority. When a processed
    frame is supplied, the observed-level and original-missingness columns are
    used to reconstruct FRED-shaped source columns before serialization.
    """

    if "date" not in df.columns:
        raise KeyError("Macro cache data must include date")
    if "TB3MS" in df.columns:
        tbill = pd.to_numeric(df["TB3MS"], errors="coerce")
    elif "tbill_3m" in df.columns:
        tbill = pd.to_numeric(df["tbill_3m"], errors="coerce")
    else:
        raise KeyError("Macro cache data must include TB3MS or tbill_3m")

    cache = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            MACRO_CACHE_SCHEMA_COLUMN: MACRO_CACHE_SCHEMA_VERSION,
        }
    )
    if SOURCE_MONTH_OBSERVATION_COUNT_COLUMN in df.columns:
        cache[SOURCE_MONTH_OBSERVATION_COUNT_COLUMN] = (
            pd.to_numeric(
                df[SOURCE_MONTH_OBSERVATION_COUNT_COLUMN],
                errors="coerce",
            )
            .fillna(1)
            .clip(lower=1)
            .astype("Int64")
        )
    else:
        cache[SOURCE_MONTH_OBSERVATION_COUNT_COLUMN] = pd.Series(
            1,
            index=df.index,
            dtype="Int64",
        )
    for measure in INFLATION_MEASURES.values():
        if measure.series_id in df.columns:
            observed_level = pd.to_numeric(df[measure.series_id], errors="coerce")
        elif measure.observed_level_col in df.columns:
            observed_level = pd.to_numeric(df[measure.observed_level_col], errors="coerce")
        elif measure.level_col in df.columns:
            observed_level = pd.to_numeric(df[measure.level_col], errors="coerce")
        else:
            continue

        if measure.key in CURRENT_MONITORING_MEASURE_KEYS:
            source_present = True
            if measure.source_present_col in df.columns:
                source_present = bool(_boolean_values(df[measure.source_present_col]).all())
            cache[measure.source_present_col] = source_present

        explicit_missing = pd.Series(False, index=df.index, dtype=bool)
        if measure.originally_missing_col in df.columns:
            explicit_missing |= _boolean_values(df[measure.originally_missing_col])
        if measure.imputed_col in df.columns:
            explicit_missing |= _boolean_values(df[measure.imputed_col])
        observed_level, originally_missing, source_unavailable = _missingness_flags(
            observed_level,
            explicit_missing,
        )

        cache[measure.series_id] = observed_level
        cache[measure.originally_missing_col] = originally_missing.astype(bool)
        cache[measure.source_unavailable_col] = source_unavailable.astype(bool)
        if measure.value_provenance_col in df.columns:
            value_provenance = (
                df[measure.value_provenance_col]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
                .fillna(VALUE_PROVENANCE_UNVERIFIED)
            )
        elif measure.series_id in df.columns:
            value_provenance = pd.Series(
                VALUE_PROVENANCE_OFFICIAL_FRED,
                index=df.index,
                dtype="string",
            )
        else:
            value_provenance = pd.Series(
                VALUE_PROVENANCE_UNVERIFIED,
                index=df.index,
                dtype="string",
            )
        cache[measure.value_provenance_col] = value_provenance

    for column in ("retrieved_at", "vintage_timestamp"):
        if column in df.columns:
            cache[column] = _utc_timestamps(df[column])
    for measure in INFLATION_MEASURES.values():
        if measure.series_id not in cache.columns:
            continue
        release_col = measure.release_timestamp_col
        provenance_col = measure.release_timestamp_provenance_col
        status_col = measure.release_timing_status_col
        supplied_releases = df.get(
            release_col,
            pd.Series(pd.NaT, index=df.index),
        )
        supplied_provenance = df.get(
            provenance_col,
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ).astype("string")
        supplied_timing_status = _incoming_release_timing_status(df, measure)
        value_available = cache[measure.series_id].notna()
        trusted_releases = _trusted_release_timestamps(
            supplied_releases,
            supplied_provenance,
            supplied_timing_status,
            value_available=value_available,
        )
        cache[release_col] = trusted_releases
        cache[provenance_col] = pd.Series(
            RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED,
            index=df.index,
            dtype="string",
        )
        cache.loc[
            trusted_releases.notna(),
            provenance_col,
        ] = RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
        cache[status_col] = _resolved_release_timing_status(
            supplied_timing_status,
            trusted_releases,
            value_available=value_available,
        )
    if "data_vintage_status" in df.columns:
        cache["data_vintage_status"] = df["data_vintage_status"].astype("string")

    if "CPIAUCSL" not in cache.columns:
        raise KeyError("Macro cache data must include a headline CPI level")
    cache["TB3MS"] = tbill
    return cache


def _cached_macro_input_frame(cached: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Normalize versioned raw and provenance-aware legacy cache frames."""

    has_schema_marker = MACRO_CACHE_SCHEMA_COLUMN in cached.columns
    if has_schema_marker:
        parsed_versions = pd.to_numeric(
            cached[MACRO_CACHE_SCHEMA_COLUMN],
            errors="coerce",
        )
        versions = parsed_versions.dropna()
        if (
            versions.empty
            or parsed_versions.isna().any()
            or not versions.eq(MACRO_CACHE_SCHEMA_VERSION).all()
        ):
            found = sorted(set(versions.astype(int))) if not versions.empty else ["unknown"]
            raise ValueError(
                f"Unsupported macro cache schema in {path}: {found}; "
                f"expected {MACRO_CACHE_SCHEMA_VERSION}"
            )

    raw_shape = {"date", "CPIAUCSL", "TB3MS"}.issubset(cached.columns)
    normalized_shape = {"date", "cpi_level", "tbill_3m"}.issubset(cached.columns)
    if not raw_shape and not normalized_shape:
        expected = " or ".join(str(columns) for columns in CACHED_INPUT_COLUMN_SETS)
        raise ValueError(f"Cached macro data {path} must include one of: {expected}")
    if has_schema_marker and not raw_shape:
        raise ValueError(
            f"Versioned macro cache {path} must use the raw-authority CPIAUCSL/TB3MS schema"
        )

    if raw_shape:
        tbill = pd.to_numeric(cached["TB3MS"], errors="coerce")
    else:
        tbill = pd.to_numeric(cached["tbill_3m"], errors="coerce")

    merged = pd.DataFrame(
        {
            "date": pd.to_datetime(cached["date"]),
            "TB3MS": tbill,
        }
    )
    if has_schema_marker:
        merged[MACRO_CACHE_SCHEMA_COLUMN] = MACRO_CACHE_SCHEMA_VERSION
    if SOURCE_MONTH_OBSERVATION_COUNT_COLUMN in cached.columns:
        merged[SOURCE_MONTH_OBSERVATION_COUNT_COLUMN] = (
            pd.to_numeric(
                cached[SOURCE_MONTH_OBSERVATION_COUNT_COLUMN],
                errors="coerce",
            )
            .fillna(1)
            .clip(lower=1)
            .astype("Int64")
        )

    for measure in INFLATION_MEASURES.values():
        if measure.series_id in cached.columns:
            observed_level = pd.to_numeric(cached[measure.series_id], errors="coerce")
        elif measure.observed_level_col in cached.columns:
            observed_level = pd.to_numeric(cached[measure.observed_level_col], errors="coerce")
        elif measure.level_col in cached.columns:
            observed_level = pd.to_numeric(cached[measure.level_col], errors="coerce")
            provenance_columns = {
                measure.observed_level_col,
                measure.originally_missing_col,
                measure.imputed_col,
            }
            if not (provenance_columns & set(cached.columns)):
                raise ValueError(
                    f"Legacy normalized macro cache {path} lacks a trusted schema marker "
                    f"and missingness provenance for {measure.level_col}; refresh the raw cache"
                )
        else:
            continue

        if measure.key in CURRENT_MONITORING_MEASURE_KEYS:
            source_present = True
            if measure.source_present_col in cached.columns:
                source_present = bool(_boolean_values(cached[measure.source_present_col]).all())
            merged[measure.source_present_col] = source_present

        explicit_missing = pd.Series(False, index=cached.index, dtype=bool)
        if measure.originally_missing_col in cached.columns:
            explicit_missing |= _boolean_values(cached[measure.originally_missing_col])
        if measure.imputed_col in cached.columns:
            explicit_missing |= _boolean_values(cached[measure.imputed_col])
        observed_level, originally_missing, source_unavailable = _missingness_flags(
            observed_level,
            explicit_missing,
        )

        merged[measure.series_id] = observed_level
        merged[measure.originally_missing_col] = originally_missing.astype(bool)
        merged[measure.source_unavailable_col] = source_unavailable.astype(bool)
        if measure.value_provenance_col in cached.columns:
            value_provenance = (
                cached[measure.value_provenance_col]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
                .fillna(VALUE_PROVENANCE_UNVERIFIED)
            )
        elif measure.series_id in cached.columns:
            value_provenance = pd.Series(
                VALUE_PROVENANCE_OFFICIAL_FRED,
                index=cached.index,
                dtype="string",
            )
        else:
            value_provenance = pd.Series(
                VALUE_PROVENANCE_UNVERIFIED,
                index=cached.index,
                dtype="string",
            )
        merged[measure.value_provenance_col] = value_provenance

        release_col = measure.release_timestamp_col
        provenance_col = measure.release_timestamp_provenance_col
        status_col = measure.release_timing_status_col
        supplied_releases = cached.get(
            release_col,
            pd.Series(pd.NaT, index=cached.index),
        )
        supplied_provenance = cached.get(
            provenance_col,
            pd.Series(pd.NA, index=cached.index, dtype="string"),
        ).astype("string")
        supplied_timing_status = _incoming_release_timing_status(cached, measure)
        value_available = observed_level.notna()
        trusted_releases = _trusted_release_timestamps(
            supplied_releases,
            supplied_provenance,
            supplied_timing_status,
            value_available=value_available,
        )
        merged[release_col] = trusted_releases
        merged[provenance_col] = pd.Series(
            RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED,
            index=cached.index,
            dtype="string",
        )
        merged.loc[
            trusted_releases.notna(),
            provenance_col,
        ] = RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
        merged[status_col] = _resolved_release_timing_status(
            supplied_timing_status,
            trusted_releases,
            value_available=value_available,
        )
    for column in ("retrieved_at", "vintage_timestamp"):
        if column in cached.columns:
            merged[column] = _utc_timestamps(cached[column])
    if "data_vintage_status" in cached.columns:
        merged["data_vintage_status"] = cached["data_vintage_status"].astype("string")
    return merged


def _load_cached_macro_data_frames_for_mode(
    mode: SampleMode | str = DEFAULT_SAMPLE_MODE,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load selected and pre-sample-warm-up frames from one cached authority.

    This is an offline fallback for dashboards/tests when FRED cannot be
    reached. Versioned caches store original source levels and explicit
    missingness. Flagged legacy processed caches are migrated by restoring
    estimated rows to missing before the requested policy is reapplied.
    """

    policy = resolve_imputation_policy(imputation_policy)
    path = find_cached_macro_data_file(mode)
    cached = pd.read_csv(path)
    merged = _cached_macro_input_frame(cached, path)

    resolved = resolve_sample_mode(mode)
    return _base_frame_and_warmup_from_merged(
        merged,
        start_date=resolved.start_date,
        end_date=resolved.end_date,
        imputation_policy=policy,
    )


def load_cached_macro_data_for_mode(
    mode: SampleMode | str = DEFAULT_SAMPLE_MODE,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> pd.DataFrame:
    """Load cached raw macro data for a named sample mode."""

    data, _ = _load_cached_macro_data_frames_for_mode(
        mode,
        imputation_policy=imputation_policy,
    )
    return data


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Only .csv and .parquet are supported")
    return path


def make_demo_data(periods: int = 260, seed: int = 7) -> pd.DataFrame:
    """Create clearly labeled demo data for UI smoke tests when offline.

    Do not use this for research conclusions.
    """

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-31", periods=periods, freq="ME")
    inflation = 2.0 + np.sin(np.linspace(0, 8 * np.pi, periods)) * 1.0
    shock = np.zeros(periods)
    shock[180:225] = np.linspace(0, 4.0, 45)
    shock[225:] = np.linspace(4.0, 0.5, periods - 225)
    noise = rng.normal(0, 0.25, periods)
    inflation_yoy = inflation + shock + noise
    cpi_level = 100 * np.cumprod(1 + np.nan_to_num(inflation_yoy, nan=2.0) / 100 / 12)
    return pd.DataFrame(
        {
            "date": dates,
            "reference_month": dates,
            "cpi_observed_level": cpi_level,
            "cpi_value_provenance": VALUE_PROVENANCE_UNVERIFIED,
            "cpi_level": cpi_level,
            "cpi_originally_missing": False,
            "cpi_source_unavailable": False,
            "cpi_imputed": False,
            "imputation_method": pd.Series(pd.NA, index=range(periods), dtype="string"),
            "imputation_available_at": pd.NaT,
            "imputation_availability_basis": pd.Series(
                pd.NA,
                index=range(periods),
                dtype="string",
            ),
            "tbill_3m": 0.25 + np.maximum(inflation_yoy - 2.0, 0) * 0.25,
            "inflation_yoy": inflation_yoy,
            "inflation_yoy_uses_imputed_input": False,
            "inflation_yoy_uses_missing_input": False,
            "release_timestamp": pd.NaT,
            "release_timestamp_provenance": RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED,
            "information_timestamp": pd.NaT,
            "information_timestamp_provenance": (INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED),
            "vintage_timestamp": pd.NaT,
            "retrieved_at": pd.NaT,
            "timing_status": TIMING_STATUS_REFERENCE_MONTH_ONLY,
            "data_vintage_status": DATA_VINTAGE_STATUS_LATEST_REVISED,
            "imputation_policy": DEFAULT_IMPUTATION_POLICY,
            "is_demo_data": True,
        }
    )

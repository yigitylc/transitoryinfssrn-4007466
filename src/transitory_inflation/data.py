from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
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

ImputationPolicy = Literal["observed_only", "ex_post_continuity"]
VALID_IMPUTATION_POLICIES: tuple[ImputationPolicy, ...] = (
    "observed_only",
    "ex_post_continuity",
)
DEFAULT_IMPUTATION_POLICY: ImputationPolicy = "observed_only"

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
    def yoy_uses_imputed_input_col(self) -> str:
        return f"{self.yoy_col}_uses_imputed_input"

    @property
    def yoy_uses_missing_input_col(self) -> str:
        return f"{self.yoy_col}_uses_missing_input"


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
    dict.fromkeys(
        [measure.series_id for measure in INFLATION_MEASURES.values()] + ["TB3MS"]
    )
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


def resolve_imputation_policy(value: str) -> ImputationPolicy:
    """Validate and normalize the CPI missing-data policy."""

    policy = str(value)
    if policy not in VALID_IMPUTATION_POLICIES:
        raise ValueError(
            f"Unknown imputation policy: {value}. "
            f"Expected one of {list(VALID_IMPUTATION_POLICIES)}"
        )
    return cast(ImputationPolicy, policy)


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


def apply_sample_mode(df: pd.DataFrame, mode: SampleMode | str, date_col: str = "date") -> pd.DataFrame:
    """Slice a frame to a named sample mode's inclusive date window."""

    resolved = resolve_sample_mode(mode)
    return slice_date_range(df, start_date=resolved.start_date, end_date=resolved.end_date, date_col=date_col)


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


def build_base_frame(
    merged: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> pd.DataFrame:
    """Build the canonical monthly base frame from merged raw FRED series.

    ``observed_only`` leaves every officially missing inflation-index level
    missing. ``ex_post_continuity`` may log-linearly bridge one isolated
    interior gap, but the raw observed level, original missingness, method, and
    earliest known availability marker remain separate. Rows before
    ``start_date`` feed the 12-month YoY change but never appear in the output.
    """

    policy = resolve_imputation_policy(imputation_policy)
    out = monthly_last(merged)
    out = out.rename(columns={"TB3MS": "tbill_3m"})
    out["imputation_policy"] = policy

    for measure in INFLATION_MEASURES.values():
        if measure.series_id in out.columns:
            observed_level = pd.to_numeric(out[measure.series_id], errors="coerce")
        elif measure.observed_level_col in out.columns:
            observed_level = pd.to_numeric(out[measure.observed_level_col], errors="coerce")
        elif measure.level_col in out.columns:
            observed_level = pd.to_numeric(out[measure.level_col], errors="coerce")
        else:
            continue

        explicit_missing = pd.Series(False, index=out.index, dtype=bool)
        if measure.originally_missing_col in out.columns:
            explicit_missing |= _boolean_values(out[measure.originally_missing_col])
        observed_level, originally_missing, source_unavailable = _missingness_flags(
            observed_level,
            explicit_missing,
        )

        out[measure.observed_level_col] = observed_level
        out[measure.originally_missing_col] = originally_missing.astype(bool)
        out[measure.source_unavailable_col] = source_unavailable.astype(bool)
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
            dtype="datetime64[ns]",
        )
        out[measure.imputation_availability_basis_col] = pd.Series(
            pd.NA,
            index=out.index,
            dtype="string",
        )

        safe_log_level = observed_level.where(observed_level > 0)
        log_interp = np.exp(np.log(safe_log_level).interpolate(limit_area="inside"))
        isna = observed_level.isna()
        single_gap = (
            isna
            & ~isna.shift(1, fill_value=False)
            & ~isna.shift(-1, fill_value=False)
        )
        if policy == "ex_post_continuity":
            imputed = single_gap & log_interp.notna()
            out[measure.imputed_col] = imputed
            out[measure.level_col] = observed_level.where(~imputed, log_interp)
            out.loc[imputed, measure.imputation_method_col] = "log_linear_bridge"

            release_col = f"{measure.lineage_prefix}_release_timestamp"
            if release_col not in out.columns and "release_timestamp" in out.columns:
                release_col = "release_timestamp"
            next_reference_month = pd.to_datetime(out["date"], errors="coerce").shift(-1)
            proxy_availability = next_reference_month + pd.offsets.MonthEnd(1)
            if release_col in out.columns:
                following_release = pd.to_datetime(
                    out[release_col],
                    errors="coerce",
                ).shift(-1)
                availability = following_release.fillna(proxy_availability)
                basis = pd.Series(
                    "following_reference_month_end_plus_one_month_proxy",
                    index=out.index,
                    dtype="string",
                )
                basis.loc[following_release.notna()] = "following_release_timestamp"
            else:
                availability = proxy_availability
                basis = pd.Series(
                    "following_reference_month_end_plus_one_month_proxy",
                    index=out.index,
                    dtype="string",
                )

            out.loc[imputed, measure.imputation_available_at_col] = availability.loc[imputed]
            out.loc[imputed, measure.imputation_availability_basis_col] = basis.loc[imputed]

        out[measure.yoy_col] = out[measure.level_col].pct_change(
            12,
            fill_method=None,
        ) * 100
        imputed_input = out[measure.imputed_col].fillna(False).astype(bool)
        missing_input = out[measure.originally_missing_col].fillna(False).astype(bool)
        out[measure.yoy_uses_imputed_input_col] = (
            imputed_input | imputed_input.shift(12, fill_value=False)
        )
        out[measure.yoy_uses_missing_input_col] = (
            missing_input | missing_input.shift(12, fill_value=False)
        )

    raw_series_cols = [
        measure.series_id
        for measure in INFLATION_MEASURES.values()
        if measure.series_id in out.columns and measure.series_id != measure.level_col
    ]
    if raw_series_cols:
        out = out.drop(columns=raw_series_cols)
    return slice_date_range(out, start_date=start_date, end_date=end_date)


def _fetch_start(start_date: str | None, warmup_months: int = YOY_WARMUP_MONTHS) -> str | None:
    """Move the fetch start earlier than the sample start by the warm-up buffer."""

    if start_date is None:
        return None
    return (pd.to_datetime(start_date) - pd.DateOffset(months=warmup_months)).strftime("%Y-%m-%d")


def load_base_macro_data_from_api(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> pd.DataFrame:
    """Load core macro data through the official FRED API."""

    policy = resolve_imputation_policy(imputation_policy)
    raw = merge_fred_series_api(
        BASE_FRED_SERIES,
        api_key=api_key,
        start_date=_fetch_start(start_date),
        end_date=end_date,
    )
    return build_base_frame(
        raw,
        start_date=start_date,
        end_date=end_date,
        imputation_policy=policy,
    )


def load_base_macro_data_from_csv(
    start_date: str | None = None,
    end_date: str | None = None,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> pd.DataFrame:
    """Load core macro data through public FRED CSV endpoints."""

    policy = resolve_imputation_policy(imputation_policy)
    raw = merge_fred_series(
        BASE_FRED_SERIES,
        start_date=_fetch_start(start_date),
        end_date=end_date,
    )
    return build_base_frame(
        raw,
        start_date=start_date,
        end_date=end_date,
        imputation_policy=policy,
    )


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
            data = load_base_macro_data_from_api(
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
                api_key_configured=True,
                imputation_policy=policy,
            )
        except Exception as exc:
            status_parts.append(f"fred_api: failed ({_safe_error(exc, key)})")
    else:
        status_parts.append("fred_api: skipped (FRED_API_KEY not configured)")

    try:
        data = load_base_macro_data_from_csv(
            start_date=resolved.start_date,
            end_date=resolved.end_date,
            imputation_policy=policy,
        )
        status_parts.append("fred_csv: ok")
        return MacroDataLoadResult(
            data=data,
            data_source_used="fred_csv",
            live_fetch_status="; ".join(status_parts),
            api_key_configured=key is not None,
            imputation_policy=policy,
        )
    except Exception as exc:
        status_parts.append(f"fred_csv: failed ({_safe_error(exc)})")

    try:
        cache_path = find_cached_macro_data_file(resolved)
        data = load_cached_macro_data_for_mode(
            resolved,
            imputation_policy=policy,
        )
        status_parts.append("cached_fred: ok")
        return MacroDataLoadResult(
            data=data,
            data_source_used="cached_fred",
            live_fetch_status="; ".join(status_parts),
            cache_file_used=cache_path.name,
            api_key_configured=key is not None,
            imputation_policy=policy,
        )
    except Exception as exc:
        status_parts.append(f"cached_fred: failed ({_safe_error(exc)})")

    data = apply_sample_mode(make_demo_data(), resolved)
    if data.empty:
        data = make_demo_data()
    data["imputation_policy"] = policy
    status_parts.append("demo: emergency fallback")
    return MacroDataLoadResult(
        data=data,
        data_source_used="demo",
        live_fetch_status="; ".join(status_parts),
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

    exact = cached_macro_data_path(mode)
    if exact.exists():
        return exact

    max_history = cached_macro_data_path("max_history")
    if max_history.exists():
        return max_history

    raise FileNotFoundError(
        f"No cached macro data found. Expected {exact} or {max_history}."
    )


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
    for measure in INFLATION_MEASURES.values():
        if measure.series_id in df.columns:
            observed_level = pd.to_numeric(df[measure.series_id], errors="coerce")
        elif measure.observed_level_col in df.columns:
            observed_level = pd.to_numeric(df[measure.observed_level_col], errors="coerce")
        elif measure.level_col in df.columns:
            observed_level = pd.to_numeric(df[measure.level_col], errors="coerce")
        else:
            continue

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

    timestamp_cols = [
        "release_timestamp",
        *(
            f"{measure.lineage_prefix}_release_timestamp"
            for measure in INFLATION_MEASURES.values()
        ),
    ]
    for column in timestamp_cols:
        if column in df.columns:
            cache[column] = pd.to_datetime(df[column], errors="coerce")

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

        release_col = f"{measure.lineage_prefix}_release_timestamp"
        if release_col in cached.columns:
            merged[release_col] = pd.to_datetime(cached[release_col], errors="coerce")

    if "release_timestamp" in cached.columns:
        merged["release_timestamp"] = pd.to_datetime(
            cached["release_timestamp"],
            errors="coerce",
        )
    return merged


def load_cached_macro_data_for_mode(
    mode: SampleMode | str = DEFAULT_SAMPLE_MODE,
    imputation_policy: ImputationPolicy | str = DEFAULT_IMPUTATION_POLICY,
) -> pd.DataFrame:
    """Load cached raw macro data for a named sample mode.

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
    return build_base_frame(
        merged,
        start_date=resolved.start_date,
        end_date=resolved.end_date,
        imputation_policy=policy,
    )


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
            "cpi_observed_level": cpi_level,
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
            "imputation_policy": DEFAULT_IMPUTATION_POLICY,
            "is_demo_data": True,
        }
    )

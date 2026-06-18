from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from .config import DEFAULT_SAMPLE_MODE, PROJECT_ROOT, RAW_DATA_DIR, SampleMode, resolve_sample_mode

FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
CACHED_BASE_MACRO_STEM = "fred_base_macro"
BASE_MACRO_COLUMNS = ("date", "cpi_level", "tbill_3m", "inflation_yoy")

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
    """Return the latest date where a value column is actually available."""

    missing = [column for column in (date_col, value_col) if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    dates = pd.to_datetime(df.loc[df[value_col].notna(), date_col])
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())


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
) -> pd.DataFrame:
    """Build the canonical monthly base frame from merged raw FRED series.

    Order of operations: monthly-last resample, rename, single-month CPI gap
    bridging, YoY inflation in percentage points, then the inclusive
    ``[start_date, end_date]`` trim. Rows before ``start_date`` (the warm-up
    buffer) feed the 12-month YoY change but never appear in the output.

    Inflation-index gap bridging: a single missing interior month is filled by
    log-linear interpolation and flagged in the measure-specific imputation
    column; otherwise one missing month makes every strict rolling window that
    contains it NaN and can freeze the live signal for years. Multi-month gaps
    and missing tail months are never imputed.
    """

    out = monthly_last(merged)
    out = out.rename(columns={"TB3MS": "tbill_3m"})

    for measure in INFLATION_MEASURES.values():
        if measure.series_id in out.columns:
            out[measure.level_col] = pd.to_numeric(out[measure.series_id], errors="coerce")
        elif measure.level_col in out.columns:
            out[measure.level_col] = pd.to_numeric(out[measure.level_col], errors="coerce")
        else:
            continue

        log_interp = np.exp(np.log(out[measure.level_col]).interpolate(limit_area="inside"))
        isna = out[measure.level_col].isna()
        single_gap = (
            isna
            & ~isna.shift(1, fill_value=False)
            & ~isna.shift(-1, fill_value=False)
        )
        out[measure.imputed_col] = single_gap & log_interp.notna()
        out[measure.level_col] = out[measure.level_col].where(
            ~out[measure.imputed_col],
            log_interp,
        )
        out[measure.yoy_col] = out[measure.level_col].pct_change(12) * 100

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
) -> pd.DataFrame:
    """Load core macro data through the official FRED API."""

    raw = merge_fred_series_api(
        BASE_FRED_SERIES,
        api_key=api_key,
        start_date=_fetch_start(start_date),
        end_date=end_date,
    )
    return build_base_frame(raw, start_date=start_date, end_date=end_date)


def load_base_macro_data_from_csv(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load core macro data through public FRED CSV endpoints."""

    raw = merge_fred_series(
        BASE_FRED_SERIES,
        start_date=_fetch_start(start_date),
        end_date=end_date,
    )
    return build_base_frame(raw, start_date=start_date, end_date=end_date)


def load_base_macro_data(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Load core macro data for an explicit inclusive date range.

    Prefer :func:`load_macro_data_for_mode` so date ranges stay tied to the
    named sample modes in ``config.SAMPLE_MODES``.

    Series:
    - CPIAUCSL: CPI level.
    - TB3MS: 3-month Treasury bill secondary market rate, monthly percent.
      Used as the bill control because FRED has no 1-month bill series before
      2001-07; TB3MS covers the full paper sample (history starts 1934).
    """

    return load_base_macro_data_from_csv(start_date=start_date, end_date=end_date)


def load_macro_data_for_mode(
    mode: SampleMode | str = DEFAULT_SAMPLE_MODE,
    allow_demo: bool = False,
) -> pd.DataFrame:
    """Load core macro data for a named sample mode (see ``config.SAMPLE_MODES``)."""

    result = load_macro_data_for_mode_with_status(mode)
    if result.data_source_used == "demo" and not allow_demo:
        raise RuntimeError(
            "Only emergency demo data is available. Use "
            "load_macro_data_for_mode_with_status() to disclose this explicitly."
        )
    return result.data


def load_macro_data_for_mode_with_status(
    mode: SampleMode | str = DEFAULT_SAMPLE_MODE,
) -> MacroDataLoadResult:
    """Load macro data with API -> CSV -> cache -> demo fallback order."""

    resolved = resolve_sample_mode(mode)
    status_parts: list[str] = []
    key = fred_api_key()

    if key:
        try:
            data = load_base_macro_data_from_api(
                key,
                start_date=resolved.start_date,
                end_date=resolved.end_date,
            )
            status_parts.append("fred_api: ok")
            return MacroDataLoadResult(
                data=data,
                data_source_used="fred_api",
                live_fetch_status="; ".join(status_parts),
                api_key_configured=True,
            )
        except Exception as exc:
            status_parts.append(f"fred_api: failed ({_safe_error(exc, key)})")
    else:
        status_parts.append("fred_api: skipped (FRED_API_KEY not configured)")

    try:
        data = load_base_macro_data_from_csv(
            start_date=resolved.start_date,
            end_date=resolved.end_date,
        )
        status_parts.append("fred_csv: ok")
        return MacroDataLoadResult(
            data=data,
            data_source_used="fred_csv",
            live_fetch_status="; ".join(status_parts),
            api_key_configured=key is not None,
        )
    except Exception as exc:
        status_parts.append(f"fred_csv: failed ({_safe_error(exc)})")

    try:
        cache_path = find_cached_macro_data_file(resolved)
        data = load_cached_macro_data_for_mode(resolved)
        status_parts.append("cached_fred: ok")
        return MacroDataLoadResult(
            data=data,
            data_source_used="cached_fred",
            live_fetch_status="; ".join(status_parts),
            cache_file_used=cache_path.name,
            api_key_configured=key is not None,
        )
    except Exception as exc:
        status_parts.append(f"cached_fred: failed ({_safe_error(exc)})")

    data = apply_sample_mode(make_demo_data(), resolved)
    if data.empty:
        data = make_demo_data()
    status_parts.append("demo: emergency fallback")
    return MacroDataLoadResult(
        data=data,
        data_source_used="demo",
        live_fetch_status="; ".join(status_parts),
        api_key_configured=key is not None,
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


def load_cached_macro_data_for_mode(mode: SampleMode | str = DEFAULT_SAMPLE_MODE) -> pd.DataFrame:
    """Load cached raw macro data for a named sample mode.

    This is an offline fallback for dashboards/tests when FRED cannot be
    reached. Cached CPI levels and rates are treated as raw inputs and rebuilt
    through the same monthly, imputation, YoY, and sample-slicing path as live
    FRED data. Derived cached columns such as ``inflation_yoy`` or
    ``cpi_imputed`` are never trusted.
    """

    path = find_cached_macro_data_file(mode)
    cached = pd.read_csv(path)
    if {"date", "CPIAUCSL", "TB3MS"}.issubset(cached.columns):
        optional_raw_series = [
            measure.series_id
            for measure in INFLATION_MEASURES.values()
            if measure.series_id in cached.columns and measure.series_id != "CPIAUCSL"
        ]
        merged = cached[["date", "CPIAUCSL", "TB3MS", *optional_raw_series]].copy()
    elif {"date", "cpi_level", "tbill_3m"}.issubset(cached.columns):
        optional_level_cols = [
            measure.level_col
            for measure in INFLATION_MEASURES.values()
            if measure.level_col in cached.columns and measure.level_col != "cpi_level"
        ]
        merged = cached[["date", "cpi_level", "tbill_3m", *optional_level_cols]].rename(
            columns={"cpi_level": "CPIAUCSL", "tbill_3m": "TB3MS"}
        )
    else:
        expected = " or ".join(str(columns) for columns in CACHED_INPUT_COLUMN_SETS)
        raise ValueError(f"Cached macro data {path} must include one of: {expected}")

    merged["date"] = pd.to_datetime(merged["date"])
    for column in merged.columns:
        if column != "date":
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    resolved = resolve_sample_mode(mode)
    return build_base_frame(
        merged,
        start_date=resolved.start_date,
        end_date=resolved.end_date,
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
            "cpi_level": cpi_level,
            "cpi_imputed": False,
            "tbill_3m": 0.25 + np.maximum(inflation_yoy - 2.0, 0) * 0.25,
            "inflation_yoy": inflation_yoy,
            "is_demo_data": True,
        }
    )

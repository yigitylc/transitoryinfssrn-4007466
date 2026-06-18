from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_SAMPLE_MODE, RAW_DATA_DIR, SampleMode, resolve_sample_mode
from .data import (
    _merge_series_frames,
    _safe_error,
    fetch_fred_series_api,
    fetch_fred_series_csv,
    fred_api_key,
    monthly_last,
    slice_date_range,
)

CACHED_MARKET_DATA_STEM = "fred_market_rates"


@dataclass(frozen=True)
class MarketFredSeries:
    """Approved FRED market-rate series for Phase 4A descriptive linkage."""

    series_id: str
    variable: str
    label: str
    frequency: str = "daily"
    units: str = "percent"


MARKET_FRED_SERIES: dict[str, MarketFredSeries] = {
    "yield_2y": MarketFredSeries(
        series_id="DGS2",
        variable="yield_2y",
        label="2Y Treasury yield",
    ),
    "yield_10y": MarketFredSeries(
        series_id="DGS10",
        variable="yield_10y",
        label="10Y Treasury yield",
    ),
    "breakeven_5y": MarketFredSeries(
        series_id="T5YIE",
        variable="breakeven_5y",
        label="5Y breakeven inflation",
    ),
    "breakeven_10y": MarketFredSeries(
        series_id="T10YIE",
        variable="breakeven_10y",
        label="10Y breakeven inflation",
    ),
    "real_yield_5y": MarketFredSeries(
        series_id="DFII5",
        variable="real_yield_5y",
        label="5Y real yield",
    ),
    "real_yield_10y": MarketFredSeries(
        series_id="DFII10",
        variable="real_yield_10y",
        label="10Y real yield",
    ),
}
MARKET_VALUE_COLUMNS: tuple[str, ...] = tuple(MARKET_FRED_SERIES)
MARKET_FRED_SERIES_IDS: tuple[str, ...] = tuple(
    meta.series_id for meta in MARKET_FRED_SERIES.values()
)
APPROVED_MARKET_FRED_SERIES_IDS: frozenset[str] = frozenset(
    {"DGS2", "DGS10", "T5YIE", "T10YIE", "DFII5", "DFII10"}
)
APPROVED_MARKET_VARIABLES: frozenset[str] = frozenset(
    {
        "yield_2y",
        "yield_10y",
        "breakeven_5y",
        "breakeven_10y",
        "real_yield_5y",
        "real_yield_10y",
    }
)


@dataclass(frozen=True)
class MarketDataLoadResult:
    """Market data plus source metadata for dashboard disclosure."""

    data: pd.DataFrame
    market_data_source_used: str
    market_live_fetch_status: str
    market_cache_file_used: str | None = None
    available_market_variables: tuple[str, ...] = ()
    latest_valid_date_by_variable: dict[str, pd.Timestamp | None] | None = None
    api_key_configured: bool = False


def validate_market_series_registry() -> None:
    """Fail loudly if Phase 4A market metadata drifts beyond the approved FRED set."""

    ids = set(MARKET_FRED_SERIES_IDS)
    variables = set(MARKET_VALUE_COLUMNS)
    if ids != APPROVED_MARKET_FRED_SERIES_IDS:
        raise ValueError(
            "Market FRED series must match the approved Phase 4A set: "
            f"{sorted(APPROVED_MARKET_FRED_SERIES_IDS)}"
        )
    if variables != APPROVED_MARKET_VARIABLES:
        raise ValueError(
            "Market variables must match the approved Phase 4A set: "
            f"{sorted(APPROVED_MARKET_VARIABLES)}"
        )


def _market_result(
    data: pd.DataFrame,
    source: str,
    status: str,
    cache_file: Path | None = None,
    api_key_configured: bool = False,
) -> MarketDataLoadResult:
    return MarketDataLoadResult(
        data=data,
        market_data_source_used=source,
        market_live_fetch_status=status,
        market_cache_file_used=cache_file.name if cache_file else None,
        available_market_variables=available_market_variables(data),
        latest_valid_date_by_variable=latest_valid_dates_by_variable(data),
        api_key_configured=api_key_configured,
    )


def _empty_market_frame() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})


def _market_merge_series(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return _merge_series_frames(frames)


def merge_fred_market_series_api(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch approved market series through the official FRED API."""

    validate_market_series_registry()
    return _market_merge_series(
        fetch_fred_series_api(series_id, api_key, start_date=start_date, end_date=end_date)
        for series_id in MARKET_FRED_SERIES_IDS
    )


def merge_fred_market_series_csv(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch approved market series through public FRED CSV endpoints."""

    validate_market_series_registry()
    return _market_merge_series(
        fetch_fred_series_csv(series_id, start_date=start_date, end_date=end_date)
        for series_id in MARKET_FRED_SERIES_IDS
    )


def build_market_frame(
    merged: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Normalize raw or cached FRED market data to monthly month-end observations."""

    if "date" not in merged.columns:
        raise ValueError("Market data must include a date column")

    monthly = monthly_last(merged)
    out = pd.DataFrame({"date": monthly["date"]})

    for variable, meta in MARKET_FRED_SERIES.items():
        if variable in monthly.columns:
            source_col = variable
        elif meta.series_id in monthly.columns:
            source_col = meta.series_id
        else:
            continue
        out[variable] = pd.to_numeric(monthly[source_col], errors="coerce")

    if len(out.columns) == 1:
        expected = ", ".join([*MARKET_FRED_SERIES_IDS, *MARKET_VALUE_COLUMNS])
        raise ValueError(f"Market data must include at least one approved series: {expected}")

    return slice_date_range(out, start_date=start_date, end_date=end_date)


def load_market_data_from_api(
    api_key: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    raw = merge_fred_market_series_api(
        api_key=api_key,
        start_date=start_date,
        end_date=end_date,
    )
    return build_market_frame(raw, start_date=start_date, end_date=end_date)


def load_market_data_from_csv(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    raw = merge_fred_market_series_csv(start_date=start_date, end_date=end_date)
    return build_market_frame(raw, start_date=start_date, end_date=end_date)


def cached_market_data_path(mode: SampleMode | str = DEFAULT_SAMPLE_MODE) -> Path:
    """Return the preferred cached raw market path for a sample mode."""

    resolved = resolve_sample_mode(mode)
    return RAW_DATA_DIR / f"{CACHED_MARKET_DATA_STEM}_{resolved.name}.csv"


def find_cached_market_data_file(mode: SampleMode | str = DEFAULT_SAMPLE_MODE) -> Path:
    """Find a cached market dataset usable for the requested sample mode."""

    exact = cached_market_data_path(mode)
    if exact.exists():
        return exact

    max_history = cached_market_data_path("max_history")
    if max_history.exists():
        return max_history

    raise FileNotFoundError(
        f"No cached market data found. Expected {exact} or {max_history}."
    )


def load_cached_market_data_for_mode(mode: SampleMode | str = DEFAULT_SAMPLE_MODE) -> pd.DataFrame:
    """Load cached raw or normalized market data for a named sample mode."""

    path = find_cached_market_data_file(mode)
    cached = pd.read_csv(path)
    if "date" not in cached.columns:
        raise ValueError(f"Cached market data {path} must include a date column")

    approved_columns = [
        column
        for column in cached.columns
        if column == "date"
        or column in MARKET_VALUE_COLUMNS
        or column in MARKET_FRED_SERIES_IDS
    ]
    if approved_columns == ["date"]:
        expected = ", ".join([*MARKET_FRED_SERIES_IDS, *MARKET_VALUE_COLUMNS])
        raise ValueError(f"Cached market data {path} must include one of: {expected}")

    resolved = resolve_sample_mode(mode)
    return build_market_frame(
        cached.loc[:, approved_columns],
        start_date=resolved.start_date,
        end_date=resolved.end_date,
    )


def load_market_data_for_mode_with_status(
    mode: SampleMode | str = DEFAULT_SAMPLE_MODE,
) -> MarketDataLoadResult:
    """Load market data with API -> CSV -> cache -> unavailable fallback order."""

    resolved = resolve_sample_mode(mode)
    status_parts: list[str] = []
    key = fred_api_key()

    if key:
        try:
            data = load_market_data_from_api(
                key,
                start_date=resolved.start_date,
                end_date=resolved.end_date,
            )
            status_parts.append("fred_api: ok")
            return _market_result(
                data,
                source="fred_api",
                status="; ".join(status_parts),
                api_key_configured=True,
            )
        except Exception as exc:
            status_parts.append(f"fred_api: failed ({_safe_error(exc, key)})")
    else:
        status_parts.append("fred_api: skipped (FRED_API_KEY not configured)")

    try:
        data = load_market_data_from_csv(
            start_date=resolved.start_date,
            end_date=resolved.end_date,
        )
        status_parts.append("fred_csv: ok")
        return _market_result(
            data,
            source="fred_csv",
            status="; ".join(status_parts),
            api_key_configured=key is not None,
        )
    except Exception as exc:
        status_parts.append(f"fred_csv: failed ({_safe_error(exc)})")

    try:
        cache_path = find_cached_market_data_file(resolved)
        data = load_cached_market_data_for_mode(resolved)
        status_parts.append("cached_fred_market: ok")
        return _market_result(
            data,
            source="cached_fred_market",
            status="; ".join(status_parts),
            cache_file=cache_path,
            api_key_configured=key is not None,
        )
    except Exception as exc:
        status_parts.append(f"cached_fred_market: failed ({_safe_error(exc)})")

    status_parts.append("market_linkage: unavailable")
    return _market_result(
        _empty_market_frame(),
        source="unavailable",
        status="; ".join(status_parts),
        api_key_configured=key is not None,
    )


def available_market_variables(df: pd.DataFrame) -> tuple[str, ...]:
    """Return approved market variables with at least one usable observation."""

    return tuple(
        variable
        for variable in MARKET_VALUE_COLUMNS
        if variable in df.columns and pd.to_numeric(df[variable], errors="coerce").notna().any()
    )


def latest_valid_dates_by_variable(df: pd.DataFrame) -> dict[str, pd.Timestamp | None]:
    """Return latest valid market observation date by approved variable."""

    latest: dict[str, pd.Timestamp | None] = {}
    if "date" not in df.columns:
        return {variable: None for variable in MARKET_VALUE_COLUMNS}

    dates = pd.to_datetime(df["date"])
    for variable in MARKET_VALUE_COLUMNS:
        if variable not in df.columns:
            latest[variable] = None
            continue
        valid_dates = dates.loc[pd.to_numeric(df[variable], errors="coerce").notna()]
        latest[variable] = (
            pd.Timestamp(valid_dates.max()) if not valid_dates.empty else None
        )
    return latest


def market_data_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize availability for every approved market variable."""

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    else:
        dates = pd.Series(pd.NaT, index=df.index)

    rows: list[dict[str, object]] = []
    for variable, meta in MARKET_FRED_SERIES.items():
        if variable in df.columns:
            valid = pd.to_numeric(df[variable], errors="coerce").notna()
            valid_dates = dates.loc[valid]
        else:
            valid = pd.Series(dtype=bool)
            valid_dates = pd.Series(dtype="datetime64[ns]")

        rows.append(
            {
                "market_variable": variable,
                "fred_series_id": meta.series_id,
                "label": meta.label,
                "frequency": meta.frequency,
                "units": meta.units,
                "available": bool(valid.any()),
                "valid_observations": int(valid.sum()),
                "first_valid_date": (
                    pd.Timestamp(valid_dates.min()) if not valid_dates.empty else pd.NaT
                ),
                "latest_valid_date": (
                    pd.Timestamp(valid_dates.max()) if not valid_dates.empty else pd.NaT
                ),
            }
        )
    return pd.DataFrame(rows)


def current_market_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Return the latest available value for each approved market variable."""

    if "date" not in df.columns:
        return pd.DataFrame(
            columns=("market_variable", "fred_series_id", "label", "latest_valid_date", "value")
        )

    dates = pd.to_datetime(df["date"])
    rows: list[dict[str, object]] = []
    for variable, meta in MARKET_FRED_SERIES.items():
        if variable not in df.columns:
            rows.append(
                {
                    "market_variable": variable,
                    "fred_series_id": meta.series_id,
                    "label": meta.label,
                    "latest_valid_date": pd.NaT,
                    "value": float("nan"),
                    "units": meta.units,
                }
            )
            continue

        values = pd.to_numeric(df[variable], errors="coerce")
        valid = values.notna()
        if not valid.any():
            latest_date = pd.NaT
            value = float("nan")
        else:
            latest_index = dates.loc[valid].idxmax()
            latest_date = pd.Timestamp(dates.loc[latest_index])
            value = float(values.loc[latest_index])
        rows.append(
            {
                "market_variable": variable,
                "fred_series_id": meta.series_id,
                "label": meta.label,
                "latest_valid_date": latest_date,
                "value": value,
                "units": meta.units,
            }
        )
    return pd.DataFrame(rows)

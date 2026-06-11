from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DEFAULT_SAMPLE_MODE, SampleMode, resolve_sample_mode

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# Months fetched before a sample's start_date so 12-month YoY inflation is
# defined from the first sample row instead of 12 months later.
YOY_WARMUP_MONTHS = 12


@dataclass(frozen=True)
class FredSeries:
    series_id: str
    name: str
    frequency: str = "monthly"


def fetch_fred_series(
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


def merge_fred_series(
    series_ids: Iterable[str],
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch and outer-join multiple FRED series by date."""

    merged: pd.DataFrame | None = None
    for series_id in series_ids:
        current = fetch_fred_series(series_id, start_date=start_date, end_date=end_date)
        merged = current if merged is None else merged.merge(current, on="date", how="outer")

    if merged is None:
        raise ValueError("No series IDs provided")

    return merged.sort_values("date").reset_index(drop=True)


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

    CPI gap bridging: a single missing interior month (e.g. 2025-10, whose
    CPI release was canceled during the government shutdown) is filled by
    log-linear interpolation and flagged in ``cpi_imputed``; otherwise one
    missing month makes every strict rolling window that contains it NaN and
    freezes the live signal for years. Multi-month gaps and missing tail
    months are never imputed.
    """

    out = monthly_last(merged)
    out = out.rename(columns={"CPIAUCSL": "cpi_level", "TB3MS": "tbill_3m"})

    log_interp = np.exp(np.log(out["cpi_level"]).interpolate(limit_area="inside"))
    isna = out["cpi_level"].isna()
    single_gap = isna & ~isna.shift(1, fill_value=False) & ~isna.shift(-1, fill_value=False)
    out["cpi_imputed"] = single_gap & log_interp.notna()
    out["cpi_level"] = out["cpi_level"].where(~out["cpi_imputed"], log_interp)

    out["inflation_yoy"] = out["cpi_level"].pct_change(12) * 100
    return slice_date_range(out, start_date=start_date, end_date=end_date)


def _fetch_start(start_date: str | None, warmup_months: int = YOY_WARMUP_MONTHS) -> str | None:
    """Move the fetch start earlier than the sample start by the warm-up buffer."""

    if start_date is None:
        return None
    return (pd.to_datetime(start_date) - pd.DateOffset(months=warmup_months)).strftime("%Y-%m-%d")


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

    raw = merge_fred_series(
        ["CPIAUCSL", "TB3MS"],
        start_date=_fetch_start(start_date),
        end_date=end_date,
    )
    return build_base_frame(raw, start_date=start_date, end_date=end_date)


def load_macro_data_for_mode(mode: SampleMode | str = DEFAULT_SAMPLE_MODE) -> pd.DataFrame:
    """Load core macro data for a named sample mode (see ``config.SAMPLE_MODES``)."""

    resolved = resolve_sample_mode(mode)
    return load_base_macro_data(start_date=resolved.start_date, end_date=resolved.end_date)


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
            "tbill_3m": 0.25 + np.maximum(inflation_yoy - 2.0, 0) * 0.25,
            "inflation_yoy": inflation_yoy,
            "is_demo_data": True,
        }
    )

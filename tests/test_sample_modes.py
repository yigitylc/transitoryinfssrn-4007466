from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.config import DEFAULT_SAMPLE_MODE, SAMPLE_MODES, resolve_sample_mode
from transitory_inflation.data import apply_sample_mode, build_base_frame


def _monthly_frame(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="ME")
    return pd.DataFrame({"date": dates, "value": np.arange(len(dates))})


def test_sample_mode_registry_matches_spec() -> None:
    assert set(SAMPLE_MODES) == {"paper_replication", "live_dashboard", "max_history"}

    paper = SAMPLE_MODES["paper_replication"]
    assert (paper.start_date, paper.end_date) == ("1982-01-01", "2021-07-31")

    live = SAMPLE_MODES["live_dashboard"]
    assert (live.start_date, live.end_date) == ("1982-01-01", None)

    full = SAMPLE_MODES["max_history"]
    assert (full.start_date, full.end_date) == (None, None)

    assert DEFAULT_SAMPLE_MODE == "live_dashboard"


def test_resolve_sample_mode_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown sample mode"):
        resolve_sample_mode("paper")
    assert resolve_sample_mode("max_history") is SAMPLE_MODES["max_history"]
    assert resolve_sample_mode(SAMPLE_MODES["live_dashboard"]) is SAMPLE_MODES["live_dashboard"]


def test_apply_sample_mode_paper_is_inclusive_on_both_ends() -> None:
    df = _monthly_frame("1980-01-31", "2024-12-31")
    out = apply_sample_mode(df, "paper_replication")
    assert out["date"].iloc[0] == pd.Timestamp("1982-01-31")
    assert out["date"].iloc[-1] == pd.Timestamp("2021-07-31")


def test_apply_sample_mode_live_keeps_latest_observation() -> None:
    df = _monthly_frame("1980-01-31", "2024-12-31")
    out = apply_sample_mode(df, "live_dashboard")
    assert out["date"].iloc[0] == pd.Timestamp("1982-01-31")
    assert out["date"].iloc[-1] == df["date"].iloc[-1]


def test_apply_sample_mode_max_history_keeps_everything() -> None:
    df = _monthly_frame("1980-01-31", "2024-12-31")
    out = apply_sample_mode(df, "max_history")
    assert len(out) == len(df)
    assert out["date"].iloc[0] == df["date"].iloc[0]


def test_build_base_frame_warmup_defines_yoy_at_sample_start() -> None:
    # 12 warm-up months (1981) feed the YoY change, then are trimmed away.
    dates = pd.date_range("1981-01-01", "1983-12-01", freq="MS")
    cpi = 100.0 * (1.005 ** np.arange(len(dates)))  # constant 0.5% m/m growth
    merged = pd.DataFrame({"date": dates, "CPIAUCSL": cpi, "TB3MS": 5.0})

    out = build_base_frame(merged, start_date="1982-01-01", end_date=None)

    assert out["date"].iloc[0] == pd.Timestamp("1982-01-31")
    assert out["inflation_yoy"].notna().all()
    expected_yoy = (1.005**12 - 1) * 100
    assert abs(out["inflation_yoy"].iloc[0] - expected_yoy) < 1e-9


def test_build_base_frame_end_date_trims_inclusively() -> None:
    dates = pd.date_range("1981-01-01", "2022-12-01", freq="MS")
    cpi = 100.0 * (1.005 ** np.arange(len(dates)))
    merged = pd.DataFrame({"date": dates, "CPIAUCSL": cpi, "TB3MS": 5.0})

    out = build_base_frame(merged, start_date="1982-01-01", end_date="2021-07-31")

    assert out["date"].iloc[-1] == pd.Timestamp("2021-07-31")
    assert out["date"].iloc[0] == pd.Timestamp("1982-01-31")


def test_build_base_frame_bridges_single_month_cpi_gap_log_linearly() -> None:
    # Mirrors the canceled 2025-10 CPI release: one interior month missing.
    dates = pd.date_range("1981-01-01", "1983-12-01", freq="MS")
    cpi = 100.0 * (1.005 ** np.arange(len(dates), dtype=float))
    cpi[22] = np.nan  # 1982-11, interior single-month gap

    merged = pd.DataFrame({"date": dates, "CPIAUCSL": cpi, "TB3MS": 5.0})
    out = build_base_frame(merged, start_date="1982-01-01", end_date=None)

    gap_row = out.loc[out["date"] == pd.Timestamp("1982-11-30")].iloc[0]
    # Log-linear bridge = geometric mean of neighbors, which under constant
    # growth recovers the true level exactly.
    assert gap_row["cpi_imputed"]
    assert abs(gap_row["cpi_level"] - 100.0 * 1.005**22) < 1e-9
    assert int(out["cpi_imputed"].sum()) == 1
    assert out["inflation_yoy"].notna().all()


def test_build_base_frame_never_imputes_multi_month_or_tail_gaps() -> None:
    dates = pd.date_range("1981-01-01", "1983-12-01", freq="MS")
    cpi = 100.0 * (1.005 ** np.arange(len(dates), dtype=float))
    cpi[20] = np.nan  # 1982-09 \ two-month interior gap
    cpi[21] = np.nan  # 1982-10 / stays visible
    cpi[-1] = np.nan  # 1983-12 tail gap stays visible

    merged = pd.DataFrame({"date": dates, "CPIAUCSL": cpi, "TB3MS": 5.0})
    out = build_base_frame(merged, start_date="1982-01-01", end_date=None)

    assert not out["cpi_imputed"].any()
    assert int(out["cpi_level"].isna().sum()) == 3

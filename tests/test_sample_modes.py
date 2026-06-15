from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.config import DEFAULT_SAMPLE_MODE, SAMPLE_MODES, resolve_sample_mode
from transitory_inflation.data import (
    FRED_API_URL,
    apply_sample_mode,
    build_base_frame,
    find_cached_macro_data_file,
    latest_valid_observation_date,
    load_cached_macro_data_for_mode,
    load_macro_data_for_mode,
    load_macro_data_for_mode_with_status,
)
from transitory_inflation.features import add_transitory_inflation_features, latest_signal_snapshot


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


def test_latest_valid_observation_date_ignores_trailing_missing_values() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-03-31", "2026-04-30", "2026-05-31"]),
            "inflation_yoy": [3.2, 3.8, np.nan],
        }
    )

    assert latest_valid_observation_date(df, "inflation_yoy") == pd.Timestamp("2026-04-30")


def _workspace_temp_dir() -> TemporaryDirectory[str]:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    return TemporaryDirectory(dir=artifacts)


def _fred_observations(series_id: str, periods: int = 15, missing_index: int | None = None) -> list[dict[str, str]]:
    dates = pd.date_range("2020-01-01", periods=periods, freq="MS")
    observations = []
    for index, date in enumerate(dates):
        if missing_index is not None and index == missing_index:
            value = "."
        elif series_id == "CPIAUCSL":
            value = f"{100 + index:.3f}"
        else:
            value = f"{1 + index / 10:.3f}"
        observations.append({"date": date.strftime("%Y-%m-%d"), "value": value})
    return observations


def _fred_csv(series_id: str, periods: int = 15) -> str:
    dates = pd.date_range("2020-01-01", periods=periods, freq="MS")
    rows = ["observation_date," + series_id]
    for index, date in enumerate(dates):
        value = 100 + index if series_id == "CPIAUCSL" else 1 + index / 10
        rows.append(f"{date:%Y-%m-%d},{value:.3f}")
    return "\n".join(rows)


class _FakeResponse:
    def __init__(self, text: str = "", payload: dict[str, object] | None = None) -> None:
        self.text = text
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_cached_macro_loader_uses_max_history_superset(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        dates = pd.date_range("1981-01-31", "2022-12-31", freq="ME")
        cache = pd.DataFrame(
            {
                "date": dates,
                "cpi_level": np.arange(len(dates), dtype=float) + 100,
                "tbill_3m": 4.0,
                "inflation_yoy": np.arange(len(dates), dtype=float) / 10,
            }
        )
        cache.to_csv(cache_path / "fred_base_macro_max_history.csv", index=False)

        out = load_cached_macro_data_for_mode("paper_replication")

        assert (
            find_cached_macro_data_file("paper_replication").name
            == "fred_base_macro_max_history.csv"
        )
        assert out["date"].iloc[0] == pd.Timestamp("1982-01-31")
        assert out["date"].iloc[-1] == pd.Timestamp("2021-07-31")


def test_cached_macro_loader_prefers_exact_mode_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        exact = pd.DataFrame(
            {
                "date": pd.date_range("1982-01-31", periods=3, freq="ME"),
                "cpi_level": [100.0, 101.0, 102.0],
                "tbill_3m": [4.0, 4.1, 4.2],
                "inflation_yoy": [2.0, 2.1, 2.2],
                "cpi_imputed": [False, True, False],
            }
        )
        exact.to_csv(cache_path / "fred_base_macro_live_dashboard.csv", index=False)

        out = load_cached_macro_data_for_mode("live_dashboard")

        assert (
            find_cached_macro_data_file("live_dashboard").name
            == "fred_base_macro_live_dashboard.csv"
        )
        assert out["cpi_imputed"].tolist() == [False, False, False]
        assert not out["inflation_yoy"].equals(exact["inflation_yoy"])


def test_cache_fallback_rebuilds_raw_cache_and_bridges_isolated_cpi_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        monkeypatch.setenv("FRED_API_KEY", "secret-test-key")

        dates = pd.date_range("2018-01-31", "2026-05-31", freq="ME")
        cpi = 250.0 * (1.002 ** np.arange(len(dates), dtype=float))
        gap_date = pd.Timestamp("2025-10-31")
        trailing_date = pd.Timestamp("2026-05-31")
        cpi[dates == gap_date] = np.nan
        cpi[dates == trailing_date] = np.nan
        cache = pd.DataFrame(
            {
                "date": dates,
                "cpi_level": cpi,
                "tbill_3m": np.linspace(1.0, 4.0, len(dates)),
                "inflation_yoy": -999.0,
                "cpi_imputed": False,
            }
        )
        cache.to_csv(cache_path / "fred_base_macro_max_history.csv", index=False)

        def fake_get(url: str, **kwargs: object) -> _FakeResponse:
            raise RuntimeError("network down")

        monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

        result = load_macro_data_for_mode_with_status("max_history")

        assert result.data_source_used == "cached_fred"
        assert "is_demo_data" not in result.data.columns
        assert result.cache_file_used == "fred_base_macro_max_history.csv"

        gap_row = result.data.loc[result.data["date"] == gap_date].iloc[0]
        trailing_row = result.data.loc[result.data["date"] == trailing_date].iloc[0]
        assert gap_row["cpi_imputed"]
        assert pd.notna(gap_row["cpi_level"])
        assert not trailing_row["cpi_imputed"]
        assert pd.isna(trailing_row["cpi_level"])

        features = add_transitory_inflation_features(
            result.data,
            baseline_method="rolling_36_shifted",
        )
        snapshot = latest_signal_snapshot(features)

        assert snapshot["available"]
        assert snapshot["date"] == pd.Timestamp("2026-04-30")
        assert snapshot["date"] > gap_date


def test_cached_macro_loader_errors_when_no_cache_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", Path(cache_dir))

        with pytest.raises(FileNotFoundError, match="No cached macro data"):
            load_cached_macro_data_for_mode("live_dashboard")


def test_api_key_present_uses_official_api_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[str] = []
    monkeypatch.setenv("FRED_API_KEY", "secret-test-key")

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        calls.append(url)
        assert url == FRED_API_URL
        params = kwargs["params"]
        assert isinstance(params, dict)
        series_id = str(params["series_id"])
        return _FakeResponse(payload={"observations": _fred_observations(series_id)})

    monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

    result = load_macro_data_for_mode_with_status("max_history")

    assert result.data_source_used == "fred_api"
    assert result.api_key_configured
    assert len(calls) == 2
    assert "secret-test-key" not in result.live_fetch_status
    assert "secret-test-key" not in capsys.readouterr().out
    assert {"date", "cpi_level", "tbill_3m", "inflation_yoy", "cpi_imputed"}.issubset(
        result.data.columns
    )


def test_missing_api_key_falls_back_to_public_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv("FRED_API_KEY", "")

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        calls.append(url)
        assert url != FRED_API_URL
        series_id = url.rsplit("=", maxsplit=1)[-1]
        return _FakeResponse(text=_fred_csv(series_id))

    monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

    result = load_macro_data_for_mode_with_status("max_history")

    assert result.data_source_used == "fred_csv"
    assert not result.api_key_configured
    assert result.live_fetch_status.startswith("fred_api: skipped")
    assert len(calls) == 2


def test_api_failure_falls_back_to_csv_without_exposing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "secret-test-key")

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        if url == FRED_API_URL:
            raise RuntimeError("failure for secret-test-key")
        series_id = url.rsplit("=", maxsplit=1)[-1]
        return _FakeResponse(text=_fred_csv(series_id))

    monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

    result = load_macro_data_for_mode_with_status("max_history")

    assert result.data_source_used == "fred_csv"
    assert "secret-test-key" not in result.live_fetch_status
    assert "[redacted]" in result.live_fetch_status


def test_api_loaded_data_still_applies_cpi_imputation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "secret-test-key")

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        assert url == FRED_API_URL
        params = kwargs["params"]
        assert isinstance(params, dict)
        series_id = str(params["series_id"])
        missing_index = 6 if series_id == "CPIAUCSL" else None
        return _FakeResponse(
            payload={"observations": _fred_observations(series_id, missing_index=missing_index)}
        )

    monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

    result = load_macro_data_for_mode_with_status("max_history")

    assert result.data_source_used == "fred_api"
    assert result.data["cpi_imputed"].any()


def test_cache_fallback_still_works_when_api_and_csv_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        monkeypatch.setenv("FRED_API_KEY", "secret-test-key")
        cache = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-31", periods=15, freq="ME"),
                "cpi_level": np.arange(15, dtype=float) + 100,
                "tbill_3m": 4.0,
                "inflation_yoy": np.arange(15, dtype=float) / 10,
            }
        )
        cache.to_csv(cache_path / "fred_base_macro_max_history.csv", index=False)

        def fake_get(url: str, **kwargs: object) -> _FakeResponse:
            raise RuntimeError("network down")

        monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

        result = load_macro_data_for_mode_with_status("max_history")

        assert result.data_source_used == "cached_fred"
        assert result.cache_file_used == "fred_base_macro_max_history.csv"
        assert len(result.data) == len(cache)


def test_demo_is_final_emergency_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_temp_dir() as cache_dir:
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", Path(cache_dir))
        monkeypatch.setenv("FRED_API_KEY", "secret-test-key")

        def fake_get(url: str, **kwargs: object) -> _FakeResponse:
            raise RuntimeError("network down")

        monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

        result = load_macro_data_for_mode_with_status("live_dashboard")

        assert result.data_source_used == "demo"
        assert result.data["is_demo_data"].all()
        assert "demo: emergency fallback" in result.live_fetch_status


def test_plain_loader_does_not_silently_return_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_temp_dir() as cache_dir:
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", Path(cache_dir))
        monkeypatch.setenv("FRED_API_KEY", "secret-test-key")

        def fake_get(url: str, **kwargs: object) -> _FakeResponse:
            raise RuntimeError("network down")

        monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

        with pytest.raises(RuntimeError, match="emergency demo data"):
            load_macro_data_for_mode("live_dashboard")

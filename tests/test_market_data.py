from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from transitory_inflation.data import FRED_API_URL
from transitory_inflation.market_data import (
    MARKET_FRED_SERIES,
    MARKET_FRED_SERIES_IDS,
    MARKET_TIMESTAMP_COLUMN,
    MARKET_TIMESTAMP_PROVENANCE_ACTUAL,
    MARKET_TIMESTAMP_PROVENANCE_COLUMN,
    MARKET_TIMESTAMP_PROVENANCE_FRED_DATE_ONLY,
    MARKET_TIMESTAMP_STATUS_COLUMN,
    MARKET_TIMESTAMP_STATUS_DATE_ONLY,
    MARKET_TIMESTAMP_STATUS_EXACT,
    MARKET_VALUE_COLUMNS,
    build_market_close_frame,
    build_market_frame,
    load_market_data_for_mode_with_status,
    validate_market_series_registry,
)


class _FakeResponse:
    def __init__(self, text: str = "", payload: dict[str, object] | None = None) -> None:
        self.text = text
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def _fred_observations(series_id: str) -> list[dict[str, str]]:
    return [
        {"date": "2020-01-02", "value": "1.000"},
        {"date": "2020-01-31", "value": "1.250"},
        {"date": "2020-02-28", "value": "1.500"},
    ]


def _fred_csv(series_id: str) -> str:
    return "\n".join(
        [
            f"observation_date,{series_id}",
            "2020-01-02,1.000",
            "2020-01-31,1.250",
            "2020-02-28,1.500",
        ]
    )


def _workspace_temp_dir() -> TemporaryDirectory[str]:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    return TemporaryDirectory(dir=artifacts)


def test_market_series_registry_is_fred_rates_only() -> None:
    expected_ids = ("DGS2", "DGS10", "T5YIE", "T10YIE", "DFII5", "DFII10")
    expected_variables = (
        "yield_2y",
        "yield_10y",
        "breakeven_5y",
        "breakeven_10y",
        "real_yield_5y",
        "real_yield_10y",
    )
    forbidden = {
        "SPY",
        "QQQ",
        "SP500",
        "S&P",
        "NASDAQ",
        "VIX",
        "DXY",
        "GLD",
        "GOLD",
        "OIL",
        "CRUDE",
        "WTI",
        "COMMODITY",
        "FEDFUNDS",
        "DFF",
        "PNL",
        "BACKTEST",
        "TRADE SIGNAL",
    }

    validate_market_series_registry()

    assert expected_ids == MARKET_FRED_SERIES_IDS
    assert expected_variables == MARKET_VALUE_COLUMNS
    metadata_text = " ".join(
        f"{variable} {meta.series_id} {meta.label}"
        for variable, meta in MARKET_FRED_SERIES.items()
    ).upper()
    assert not (forbidden & set(MARKET_FRED_SERIES_IDS))
    assert all(term not in metadata_text for term in forbidden)


def test_api_key_present_uses_official_market_api_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    monkeypatch.setenv("FRED_API_KEY", "secret-market-key")

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        calls.append(url)
        assert url == FRED_API_URL
        params = kwargs["params"]
        assert isinstance(params, dict)
        series_id = str(params["series_id"])
        assert series_id in MARKET_FRED_SERIES_IDS
        return _FakeResponse(payload={"observations": _fred_observations(series_id)})

    monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

    result = load_market_data_for_mode_with_status("max_history")

    assert result.market_data_source_used == "fred_api"
    assert result.api_key_configured
    assert len(calls) == len(MARKET_FRED_SERIES_IDS)
    assert set(result.available_market_variables) == set(MARKET_VALUE_COLUMNS)
    assert result.data["date"].tolist() == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
    ]
    assert result.market_closes["date"].tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
    ]
    assert result.market_closes[MARKET_TIMESTAMP_COLUMN].isna().all()
    assert result.market_closes[MARKET_TIMESTAMP_STATUS_COLUMN].eq(
        MARKET_TIMESTAMP_STATUS_DATE_ONLY
    ).all()
    assert result.market_closes[MARKET_TIMESTAMP_PROVENANCE_COLUMN].eq(
        MARKET_TIMESTAMP_PROVENANCE_FRED_DATE_ONLY
    ).all()
    assert "secret-market-key" not in result.market_live_fetch_status
    assert "secret-market-key" not in capsys.readouterr().out


def test_missing_api_key_uses_public_market_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv("FRED_API_KEY", "")

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        calls.append(url)
        assert url != FRED_API_URL
        series_id = url.rsplit("=", maxsplit=1)[-1]
        assert series_id in MARKET_FRED_SERIES_IDS
        return _FakeResponse(text=_fred_csv(series_id))

    monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

    result = load_market_data_for_mode_with_status("max_history")

    assert result.market_data_source_used == "fred_csv"
    assert not result.api_key_configured
    assert result.market_live_fetch_status.startswith("fred_api: skipped")
    assert len(calls) == len(MARKET_FRED_SERIES_IDS)


def test_api_failure_redacts_key_and_falls_back_to_market_csv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "secret-market-key")

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        if url == FRED_API_URL:
            raise RuntimeError("failure for secret-market-key")
        series_id = url.rsplit("=", maxsplit=1)[-1]
        return _FakeResponse(text=_fred_csv(series_id))

    monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

    result = load_market_data_for_mode_with_status("max_history")

    assert result.market_data_source_used == "fred_csv"
    assert "[redacted]" in result.market_live_fetch_status
    assert "secret-market-key" not in result.market_live_fetch_status
    assert "secret-market-key" not in capsys.readouterr().out


def test_api_and_csv_failure_uses_valid_local_market_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.market_data.RAW_DATA_DIR", cache_path)
        monkeypatch.setenv("FRED_API_KEY", "secret-market-key")
        cache = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-31", periods=3, freq="ME"),
                "yield_2y": [1.0, 1.1, 1.2],
                "yield_10y": [2.0, 2.1, 2.2],
            }
        )
        cache.to_csv(cache_path / "fred_market_rates_max_history.csv", index=False)

        def fake_get(url: str, **kwargs: object) -> _FakeResponse:
            raise RuntimeError("network down")

        monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

        result = load_market_data_for_mode_with_status("max_history")

        assert result.market_data_source_used == "cached_fred_market"
        assert result.market_cache_file_used == "fred_market_rates_max_history.csv"
        assert result.available_market_variables == ("yield_2y", "yield_10y")
        assert "is_demo_data" not in result.data.columns


def test_no_market_cache_means_unavailable_not_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_temp_dir() as cache_dir:
        monkeypatch.setattr("transitory_inflation.market_data.RAW_DATA_DIR", Path(cache_dir))
        monkeypatch.setenv("FRED_API_KEY", "secret-market-key")

        def fake_get(url: str, **kwargs: object) -> _FakeResponse:
            raise RuntimeError("network down")

        monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

        result = load_market_data_for_mode_with_status("live_dashboard")

        assert result.market_data_source_used == "unavailable"
        assert result.data.empty
        assert result.available_market_variables == ()
        assert "is_demo_data" not in result.data.columns
        assert "market_linkage: unavailable" in result.market_live_fetch_status


def test_daily_fred_values_become_month_end_last_observations() -> None:
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-30", "2024-02-01", "2024-02-28"]
            ),
            "DGS2": [1.0, 1.2, 1.3, 1.6],
            "DGS10": [2.0, 2.2, 2.3, 2.6],
        }
    )

    out = build_market_frame(raw)

    assert out["date"].tolist() == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
    ]
    assert out["yield_2y"].tolist() == [1.2, 1.6]
    assert out["yield_10y"].tolist() == [2.2, 2.6]
    assert MARKET_TIMESTAMP_COLUMN not in out.columns


def test_explicit_market_close_timestamp_preserves_time_of_day() -> None:
    close = pd.Timestamp("2024-02-13 16:00:00-05:00")
    out = build_market_close_frame(
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-02-13")],
                "DGS2": [4.5],
                MARKET_TIMESTAMP_COLUMN: [close],
                MARKET_TIMESTAMP_PROVENANCE_COLUMN: [
                    MARKET_TIMESTAMP_PROVENANCE_ACTUAL
                ],
                MARKET_TIMESTAMP_STATUS_COLUMN: [MARKET_TIMESTAMP_STATUS_EXACT],
            }
        )
    )

    assert out.loc[0, MARKET_TIMESTAMP_COLUMN] == close.tz_convert("UTC")
    assert out.loc[0, MARKET_TIMESTAMP_COLUMN].hour == 21
    assert out.loc[0, MARKET_TIMESTAMP_STATUS_COLUMN] == MARKET_TIMESTAMP_STATUS_EXACT


def test_market_timestamp_without_explicit_status_fails_closed() -> None:
    out = build_market_close_frame(
        pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-02-13")],
                "DGS2": [4.5],
                MARKET_TIMESTAMP_COLUMN: [pd.Timestamp("2024-02-13 21:00:00+00:00")],
                MARKET_TIMESTAMP_PROVENANCE_COLUMN: [
                    MARKET_TIMESTAMP_PROVENANCE_ACTUAL
                ],
            }
        )
    )

    assert pd.isna(out.loc[0, MARKET_TIMESTAMP_COLUMN])
    assert out.loc[0, MARKET_TIMESTAMP_STATUS_COLUMN] == (
        MARKET_TIMESTAMP_STATUS_DATE_ONLY
    )

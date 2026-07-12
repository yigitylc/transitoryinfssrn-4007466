from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation import report as report_mod
from transitory_inflation.dashboard import (
    CURRENT_SIGNAL_IMPUTATION_NOTICE,
    build_dashboard_data_views,
)
from transitory_inflation.data import build_base_frame
from transitory_inflation.features import add_transitory_inflation_features
from transitory_inflation.report import (
    REGIME_PLAYBOOK,
    MacroResearchReport,
    TraderReport,
    build_macro_research_report,
    build_trader_report,
    next_print_flip_threshold,
)


def _raw_frame(months: int = 160, yoy: float | None = None) -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    values = yoy if yoy is not None else 3.0 + np.sin(np.linspace(0, 12, months))
    return pd.DataFrame(
        {"date": dates, "cpi_level": 100.0, "tbill_3m": 4.0, "inflation_yoy": values}
    )


def _market_frame(dates: pd.Series) -> pd.DataFrame:
    length = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "yield_2y": np.linspace(1.0, 3.0, length),
            "yield_10y": np.linspace(2.0, 4.0, length),
            "breakeven_5y": np.linspace(2.0, 2.6, length),
            "breakeven_10y": np.linspace(2.1, 2.7, length),
            "real_yield_5y": np.linspace(0.0, 1.0, length),
            "real_yield_10y": np.linspace(0.2, 1.2, length),
        }
    )


def test_playbook_covers_all_snapshot_regimes() -> None:
    assert set(REGIME_PLAYBOOK) == {
        "elevated rising",
        "elevated falling",
        "neutral",
        "disinflationary",
    }


def test_report_builds_and_is_structured() -> None:
    raw = _raw_frame()
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")
    report = build_trader_report(raw, df, "fed_target", "live_dashboard", decay_windows=(24,))

    assert isinstance(report, TraderReport)
    assert report.available
    assert report.headline
    assert report.state_lines and report.persistence_lines and report.robustness_lines
    labels = [label for label, _ in report.playbook]
    assert "Macro read" in labels
    assert report.playbook[-1][0] == "Term-structure modifier"
    assert any("not investment advice" in caveat for caveat in report.caveats)


def test_report_flags_ex_post_baseline() -> None:
    raw = _raw_frame()
    df = add_transitory_inflation_features(raw, baseline_method="full_sample")
    report = build_trader_report(raw, df, "full_sample", "paper_replication", decay_windows=(24,))

    assert report.available
    assert any("EX-POST" in caveat for caveat in report.caveats)


def test_report_unavailable_without_complete_rows() -> None:
    raw = _raw_frame(months=10)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")
    report = build_trader_report(raw, df, "fed_target", "live_dashboard", decay_windows=(24,))

    assert not report.available
    assert report.reason


def test_flip_threshold_fed_target() -> None:
    # Constant 3% YoY against the 2% target: eps = +1pp each month, so the next
    # print must land at 2.0 - 3.0 = -1.0% to zero out the 4-month average.
    raw = _raw_frame(yoy=3.0)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")
    assert next_print_flip_threshold(df, "fed_target") == pytest.approx(-1.0)


def test_flip_threshold_undefined_for_ex_post_baselines() -> None:
    raw = _raw_frame(yoy=3.0)
    df = add_transitory_inflation_features(raw, baseline_method="full_sample")
    assert next_print_flip_threshold(df, "full_sample") is None


def test_macro_research_report_builds_required_phase_five_sections() -> None:
    raw = _raw_frame(months=180)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")
    market = _market_frame(raw["date"])

    report = build_macro_research_report(
        raw,
        df,
        baseline_method="fed_target",
        sample_mode="live_dashboard",
        macro_status={"data_source_used": "unit"},
        market_monthly=market,
        market_status={"market_data_source_used": "unit_market"},
        benchmark_horizons=(3, 6),
        market_horizons=(3, 6),
        robustness_baselines=("fed_target",),
    )

    assert isinstance(report, MacroResearchReport)
    assert report.available
    assert report.current_regime_lines
    assert report.signal_confidence_lines
    assert report.robustness_lines
    assert report.historical_analog_lines
    assert report.market_linkage_lines
    assert report.caveats
    assert report.watchlist
    assert {"latest_valid_signal_date", "data_source_used", "current_regime"}.issubset(
        report.current_regime_table.columns
    )
    assert {"ar1", "cpi_persistence"} <= set(report.benchmark_comparisons["comparison_model"])
    assert any("AR(1)" in line for line in report.signal_confidence_lines)
    assert any("point-forecast" in line for line in report.caveats)
    assert any("not a trading signal" in line for line in report.caveats)


def test_macro_research_report_discloses_data_vintage_caveat() -> None:
    raw = _raw_frame(months=180)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")

    report = build_macro_research_report(
        raw,
        df,
        baseline_method="fed_target",
        sample_mode="live_dashboard",
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("fed_target",),
    )

    assert any("vintage" in line.lower() for line in report.caveats)
    assert any("latest-revised" in line.lower() for line in report.caveats)
    assert any("walk-forward" in line.lower() for line in report.caveats)


def test_macro_research_report_discloses_missing_measures_and_approved_market_channels() -> None:
    raw = _raw_frame(months=180)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")
    market = _market_frame(raw["date"])

    report = build_macro_research_report(
        raw,
        df,
        baseline_method="fed_target",
        sample_mode="live_dashboard",
        market_monthly=market,
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("fed_target",),
    )

    unavailable = report.inflation_measure_availability.loc[
        ~report.inflation_measure_availability["available"],
        "inflation_measure",
    ]
    assert {"core_cpi", "pce", "core_pce"} <= set(unavailable)
    assert any("Missing inflation measures" in line for line in report.robustness_lines)
    assert set(report.market_channel_summary["market_channel"]) <= {
        "nominal_rates",
        "breakevens",
        "real_yields",
    }
    assert "spy" not in set(report.market_channel_summary["market_channel"])


def test_macro_research_report_flags_weak_historical_analog_evidence() -> None:
    raw = _raw_frame(months=50, yoy=2.2)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")
    market = _market_frame(raw["date"])

    report = build_macro_research_report(
        raw,
        df,
        baseline_method="fed_target",
        sample_mode="live_dashboard",
        market_monthly=market,
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("fed_target",),
    )

    assert not report.historical_analogs.empty
    assert report.historical_analogs["weak_evidence"].fillna(False).astype(bool).any()
    assert report.historical_analogs["evidence_note"].str.contains("Fewer than 30").any()


def test_macro_report_routes_only_current_sections_to_ex_post_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    periods = 120
    gap_pos = periods - 2
    dates = pd.date_range("2015-01-31", periods=periods, freq="ME")
    levels = 100.0 * (1.002 ** np.arange(periods, dtype=float))
    levels[gap_pos] = np.nan
    observed = build_base_frame(
        pd.DataFrame({"date": dates, "CPIAUCSL": levels, "TB3MS": 3.0}),
        imputation_policy="observed_only",
    )
    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
    )
    captured: dict[str, object] = {}

    original_benchmarks = report_mod._benchmark_tables
    original_robustness = report_mod._robustness_tables
    original_analogs = report_mod._historical_analog_table
    original_market = report_mod._market_summary

    def capture_benchmarks(featured: pd.DataFrame, **kwargs: object):
        captured["benchmarks"] = featured
        return original_benchmarks(featured, **kwargs)

    def capture_robustness(sample_frames: dict[str, pd.DataFrame], **kwargs: object):
        captured["robustness"] = sample_frames
        return original_robustness(sample_frames, **kwargs)

    def capture_analogs(
        featured: pd.DataFrame,
        snapshot: dict[str, object],
        **kwargs: object,
    ):
        captured["analogs"] = featured
        captured["analog_snapshot"] = snapshot
        return original_analogs(featured, snapshot, **kwargs)

    def capture_market(featured: pd.DataFrame, **kwargs: object):
        captured["market"] = featured
        return original_market(featured, **kwargs)

    monkeypatch.setattr(report_mod, "_benchmark_tables", capture_benchmarks)
    monkeypatch.setattr(report_mod, "_robustness_tables", capture_robustness)
    monkeypatch.setattr(report_mod, "_historical_analog_table", capture_analogs)
    monkeypatch.setattr(report_mod, "_market_summary", capture_market)

    report = build_macro_research_report(
        views.research_raw,
        views.research_featured,
        baseline_method="rolling_36_shifted",
        sample_mode="live_dashboard",
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("rolling_36_shifted",),
        current_raw=views.current_raw,
        current_featured=views.current_featured,
    )

    assert report.available
    assert report.as_of == str(dates[-1].date())
    assert report.current_signal_notice == CURRENT_SIGNAL_IMPUTATION_NOTICE
    assert report.current_regime_table.iloc[0]["imputation_policy"] == "ex_post_continuity"
    assert report.current_regime_table.iloc[0]["current_signal_uses_imputed_input"]
    for key in ("benchmarks", "analogs", "market"):
        frame = captured[key]
        assert isinstance(frame, pd.DataFrame)
        assert frame["imputation_policy"].eq("observed_only").all()
    robustness_frames = captured["robustness"]
    assert isinstance(robustness_frames, dict)
    assert all(
        frame["imputation_policy"].eq("observed_only").all()
        for frame in robustness_frames.values()
    )
    analog_snapshot = captured["analog_snapshot"]
    assert isinstance(analog_snapshot, dict)
    assert analog_snapshot["date"] < dates[-1]
    if not report.historical_analogs.empty:
        assert report.historical_analogs["conditioning_signal_date"].eq(
            str(analog_snapshot["date"].date())
        ).all()

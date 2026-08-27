from __future__ import annotations

import json

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
from transitory_inflation.market_data import (
    MARKET_TIMESTAMP_COLUMN,
    MARKET_TIMESTAMP_PROVENANCE_ACTUAL,
    MARKET_TIMESTAMP_PROVENANCE_COLUMN,
    MARKET_TIMESTAMP_STATUS_COLUMN,
    MARKET_TIMESTAMP_STATUS_EXACT,
    MARKET_VALUE_COLUMNS,
    build_market_close_frame,
)
from transitory_inflation.market_linkage import (
    MARKET_ORIGIN_INFORMATION_TIMESTAMP,
    MARKET_ORIGIN_UNAVAILABLE,
    MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED,
    MARKET_TIMING_UNAVAILABLE,
)
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
    assert {
        "latest_valid_signal_date",
        "reference_month",
        "release_timestamp",
        "information_timestamp",
        "release_timestamp_provenance",
        "information_timestamp_provenance",
        "timing_status",
        "data_vintage_status",
        "data_source_used",
        "current_regime",
    }.issubset(report.current_regime_table.columns)
    assert {"ar1", "cpi_persistence"} <= set(report.benchmark_comparisons["comparison_model"])
    assert any("AR(1)" in line for line in report.signal_confidence_lines)
    assert any("point-forecast" in line for line in report.caveats)
    assert any("not a trading signal" in line for line in report.caveats)
    assert any("conservative month-end t+1" in line for line in report.market_linkage_lines)


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
    timing_caveats = " ".join(report.caveats).lower()
    assert "reference-month-only" in timing_caveats
    assert "never described as release-aligned or vintage-safe" in timing_caveats


def test_macro_report_returns_explicit_reference_month_and_information_time() -> None:
    dates = pd.date_range("2010-01-31", periods=180, freq="ME")
    releases = pd.Series((dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC"))
    raw = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": 100.0 * (1.002 ** np.arange(len(dates), dtype=float)),
                "TB3MS": 1.0,
                "release_timestamp": releases,
                "release_timestamp_provenance": "actual_release_metadata",
                "release_timing_status": "release_aligned",
            }
        )
    )
    featured = add_transitory_inflation_features(raw, baseline_method="fed_target")

    report = build_macro_research_report(
        raw,
        featured,
        baseline_method="fed_target",
        sample_mode="live_dashboard",
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("fed_target",),
    )

    assert report.available
    assert report.reference_month == str(dates[-1].date())
    assert report.information_timestamp == releases.iloc[-1].isoformat()
    assert report.timing_status == "release_aligned"
    assert report.as_of == report.reference_month
    assert "not a signal availability" in report.as_of_semantics
    detail = report.current_regime_table.iloc[0]
    assert detail["reference_month"] == report.reference_month
    assert detail["information_timestamp"] == report.information_timestamp
    assert detail["latest_valid_signal_date"] == report.reference_month
    assert "not a signal availability" in detail["latest_valid_signal_date_semantics"]
    returned_text = " ".join(
        [*report.current_regime_lines, *report.historical_analog_lines]
    ).lower()
    assert "reference month" in returned_text
    assert "information timestamp" in returned_text


def test_macro_report_discloses_unavailable_timing_when_channel_summary_is_empty() -> None:
    dates = pd.date_range("2010-01-31", periods=180, freq="ME")
    releases = pd.Series((dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC"))
    raw = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": 100.0 * (1.002 ** np.arange(len(dates), dtype=float)),
                "TB3MS": 1.0,
                "release_timestamp": releases,
                "release_timestamp_provenance": "actual_release_metadata",
                "release_timing_status": "release_aligned",
            }
        )
    )
    featured = add_transitory_inflation_features(raw, baseline_method="fed_target")
    stale_market_row: dict[str, object] = {
        "date": pd.Timestamp("2009-12-31"),
        MARKET_TIMESTAMP_COLUMN: pd.Timestamp("2009-12-31 21:00:00+00:00"),
        MARKET_TIMESTAMP_PROVENANCE_COLUMN: MARKET_TIMESTAMP_PROVENANCE_ACTUAL,
        MARKET_TIMESTAMP_STATUS_COLUMN: MARKET_TIMESTAMP_STATUS_EXACT,
    }
    stale_market_row.update(
        {variable: 1.0 + position for position, variable in enumerate(MARKET_VALUE_COLUMNS)}
    )
    market = build_market_close_frame(pd.DataFrame([stale_market_row]))

    report = build_macro_research_report(
        raw,
        featured,
        baseline_method="fed_target",
        sample_mode="live_dashboard",
        market_monthly=market,
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("fed_target",),
    )

    assert report.available
    assert report.market_channel_summary.empty
    disclosure = "\n".join(report.market_linkage_lines)
    assert "No channel summary is available" in disclosure
    assert MARKET_ORIGIN_UNAVAILABLE in disclosure
    assert MARKET_TIMING_UNAVAILABLE in disclosure
    assert "conditioning origin=unavailable" in disclosure
    assert MARKET_ORIGIN_INFORMATION_TIMESTAMP not in disclosure


def test_macro_report_returns_named_per_series_timing_for_partial_rows() -> None:
    dates = pd.date_range("2010-01-31", periods=180, freq="ME")
    releases = pd.Series((dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC"))
    raw = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": 100.0 * (1.002 ** np.arange(len(dates), dtype=float)),
                "TB3MS": 1.0,
                "release_timestamp": releases,
                "release_timestamp_provenance": "actual_release_metadata",
                "release_timing_status": "release_aligned",
            }
        )
    )
    featured = add_transitory_inflation_features(raw, baseline_method="fed_target")
    market_timestamps = pd.Series(
        (dates + pd.offsets.Day(20) + pd.offsets.Hour(21)).tz_localize("UTC")
    )
    market = build_market_close_frame(
        pd.DataFrame(
            {
                "date": dates + pd.offsets.Day(20),
                "DGS2": np.linspace(1.0, 4.0, len(dates)),
                "DGS10": np.nan,
                MARKET_TIMESTAMP_COLUMN: market_timestamps,
                MARKET_TIMESTAMP_PROVENANCE_COLUMN: (MARKET_TIMESTAMP_PROVENANCE_ACTUAL),
                MARKET_TIMESTAMP_STATUS_COLUMN: MARKET_TIMESTAMP_STATUS_EXACT,
            }
        )
    )

    report = build_macro_research_report(
        raw,
        featured,
        baseline_method="fed_target",
        sample_mode="live_dashboard",
        market_monthly=market,
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("fed_target",),
    )

    assert report.available
    summary = report.market_series_timing_summary
    exact_2y = summary.loc[
        (summary["market_variable"] == "yield_2y")
        & (summary["market_origin_basis"] == MARKET_ORIGIN_INFORMATION_TIMESTAMP)
        & (summary["market_timing_status"] == MARKET_TIMING_INFORMATION_TIMESTAMP_ALIGNED)
    ]
    unavailable_10y = summary.loc[
        (summary["market_variable"] == "yield_10y")
        & (summary["market_origin_basis"] == MARKET_ORIGIN_UNAVAILABLE)
        & (summary["market_timing_status"] == MARKET_TIMING_UNAVAILABLE)
    ]
    assert not exact_2y.empty
    assert not unavailable_10y.empty
    disclosure = "\n".join(report.market_linkage_lines)
    assert "instrument=yield_2y" in disclosure
    assert "instrument=yield_10y" in disclosure
    assert MARKET_ORIGIN_INFORMATION_TIMESTAMP in disclosure
    assert MARKET_ORIGIN_UNAVAILABLE in disclosure


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
    dates = pd.date_range("2014-01-31", "2026-06-30", freq="ME")
    gap_pos = int(np.flatnonzero(dates == pd.Timestamp("2025-10-31"))[0])
    levels = 100.0 * (1.002 ** np.arange(len(dates), dtype=float))
    core_levels = 95.0 * (1.0018 ** np.arange(len(dates), dtype=float))
    levels[gap_pos] = np.nan
    core_levels[gap_pos] = np.nan
    warmup = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": levels,
                "CPILFESL": core_levels,
                "TB3MS": 3.0,
            }
        ),
        imputation_policy="observed_only",
    )
    observed = warmup.loc[warmup["date"].ge(pd.Timestamp("2015-01-31"))].reset_index(drop=True)
    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=warmup,
    )
    captured: dict[str, object] = {"analog_snapshots": []}

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
        snapshots = captured["analog_snapshots"]
        assert isinstance(snapshots, list)
        snapshots.append(snapshot)
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
        current_monitoring=views.current_monitoring,
    )

    assert report.available
    assert report.as_of == str(dates[-1].date())
    assert report.current_signal_notice == CURRENT_SIGNAL_IMPUTATION_NOTICE
    assert report.current_regime_table.iloc[0]["imputation_policy"] == "ex_post_continuity"
    assert report.current_regime_table.iloc[0]["current_signal_uses_imputed_input"]
    assert set(report.current_scenario_table["scenario_id"]) == {"low", "base", "high"}
    assert report.current_scenario_table["uses_estimated_input"].all()
    assert report.current_scenario_table["reference_month"].eq(dates[-1]).all()
    assert report.current_scenario_table["historical_population_policy"].eq("observed_only").all()
    required_lineage = {
        "scenario_id",
        "estimate_method",
        "estimated_reference_month",
        "estimate_value",
        "uses_estimated_input",
        "estimated_input_months",
        "calibration_policy",
        "reference_month",
        "release_timestamp",
        "information_timestamp",
        "timing_status",
        "retrieved_at",
        "information_status",
        "data_vintage_status",
    }
    assert required_lineage.issubset(report.current_scenario_table.columns)
    assert not report.section_lineage.empty
    analog_lineage = report.section_lineage.loc[
        report.section_lineage["section"] == "historical_analogs"
    ].iloc[0]
    assert analog_lineage["estimated_inputs_allowed"]
    assert not analog_lineage["population_estimated_inputs_allowed"]
    assert analog_lineage["conditioning_estimated_inputs_allowed"]
    historical_lineage = report.section_lineage.loc[
        report.section_lineage["section"] == "benchmarks_robustness_market_linkage"
    ].iloc[0]
    assert not historical_lineage["estimated_inputs_allowed"]
    assert not historical_lineage["population_estimated_inputs_allowed"]
    lineage_payload = json.loads(json.dumps(report.scenario_lineage))
    assert lineage_payload["status"] == "available"
    assert set(lineage_payload["scenarios"]) == {"low", "base", "high"}
    for scenario in lineage_payload["scenarios"].values():
        assert set(scenario["snapshot"]["series_lineage"]) == {
            "headline_cpi",
            "core_cpi",
        }
    table_payload = json.loads(
        report.current_scenario_table.to_json(orient="records", date_format="iso")
    )
    assert len(table_payload) == 3
    assert all(set(row["series_lineage"]) == {"headline_cpi", "core_cpi"} for row in table_payload)
    for key in ("benchmarks", "analogs", "market"):
        frame = captured[key]
        assert isinstance(frame, pd.DataFrame)
        assert frame["imputation_policy"].eq("observed_only").all()
    robustness_frames = captured["robustness"]
    assert isinstance(robustness_frames, dict)
    assert all(
        frame["imputation_policy"].eq("observed_only").all() for frame in robustness_frames.values()
    )
    analog_snapshots = captured["analog_snapshots"]
    assert isinstance(analog_snapshots, list)
    assert len(analog_snapshots) == 3
    assert {snapshot["scenario_id"] for snapshot in analog_snapshots} == {
        "low",
        "base",
        "high",
    }
    assert all(snapshot["date"] == dates[-1] for snapshot in analog_snapshots)
    if not report.historical_analogs.empty:
        assert report.historical_analogs["conditioning_signal_date"].eq(str(dates[-1].date())).all()
        assert report.historical_analogs["historical_population_policy"].eq("observed_only").all()
        assert set(report.historical_analogs["scenario_id"]) == {
            "low",
            "base",
            "high",
        }


def test_paper_window_report_suppresses_missing_cpi_scenarios() -> None:
    dates = pd.date_range("1980-01-31", "2026-06-30", freq="ME")
    steps = np.arange(len(dates), dtype=float)
    authority = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": 100.0 * (1.002**steps),
                "CPILFESL": 95.0 * (1.0018**steps),
                "TB3MS": 3.0,
            }
        ),
        imputation_policy="observed_only",
    )
    views = build_dashboard_data_views(
        authority,
        baseline_method="full_sample",
        warmup_raw=authority,
        sample_mode="paper_replication",
    )

    report = build_macro_research_report(
        views.research_raw,
        views.research_featured,
        baseline_method="full_sample",
        sample_mode="paper_replication",
        current_monitoring=views.current_monitoring,
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("full_sample",),
    )

    assert report.available
    assert report.scenario_status == "not_applicable"
    assert report.scenario_stability["not_applicable"]
    assert not report.scenario_stability["fully_stable"]
    assert report.current_scenario_table.empty
    assert "not applicable" in report.headline.lower()
    assert "stable across" not in report.headline.lower()
    assert not any("scenario stability" in line.lower() for line in report.current_regime_lines)
    assert any("not applicable" in line.lower() for line in report.watchlist)
    assert report.section_lineage["input_policy"].eq("observed_only").all()
    assert not report.section_lineage["estimated_inputs_allowed"].astype(bool).any()
    payload = json.loads(json.dumps(report.scenario_lineage))
    assert payload["status"] == "not_applicable"
    assert payload["scenarios"] == {}
    assert payload["observed_current"]["scenario_id"] == "observed_only"


def test_unavailable_report_preserves_two_series_lineage_without_classification() -> None:
    dates = pd.date_range("1980-01-31", "2026-06-30", freq="ME")
    steps = np.arange(len(dates), dtype=float)
    headline = 100.0 * (1.002**steps)
    core = 95.0 * (1.0018**steps)
    headline[dates == pd.Timestamp("2025-10-31")] = np.nan
    core[dates == pd.Timestamp("2025-10-31")] = np.nan
    core[dates == pd.Timestamp("2025-11-30")] = np.nan
    authority = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": headline,
                "CPILFESL": core,
                "TB3MS": 3.0,
            }
        ),
        imputation_policy="observed_only",
    )
    views = build_dashboard_data_views(
        authority,
        baseline_method="rolling_36_shifted",
        warmup_raw=authority,
        sample_mode="live_dashboard",
    )

    report = build_macro_research_report(
        views.research_raw,
        views.research_featured,
        baseline_method="rolling_36_shifted",
        sample_mode="live_dashboard",
        current_monitoring=views.current_monitoring,
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("rolling_36_shifted",),
    )

    assert not report.available
    assert report.as_of == "2025-10-31"
    assert report.reference_month == "2025-10-31"
    assert report.information_timestamp == "unknown"
    assert report.timing_status == "derived_value_unavailable"
    assert report.scenario_status == "unavailable"
    assert report.scenario_stability["unavailable"]
    assert not report.current_scenario_table.empty
    assert report.current_scenario_table["regime"].isna().all()
    assert report.current_scenario_table["pressure"].isna().all()
    assert report.current_scenario_table["tinf_4m"].isna().all()
    unavailable_lineage = report.section_lineage.iloc[0]
    assert unavailable_lineage["scenario_status"] == "unavailable"
    assert unavailable_lineage["ex_post_status"] == ("ex_post_current_monitoring_sensitivity")
    assert unavailable_lineage["lookahead_status"] == ("unavailable_required_scenario_input")
    assert unavailable_lineage["signal_reference_month"] == pd.Timestamp("2025-10-31")
    assert pd.isna(unavailable_lineage["signal_information_timestamp"])
    assert unavailable_lineage["signal_timing_status"] == ("derived_value_unavailable")
    assert unavailable_lineage["data_vintage_status"] == ("latest_revised_non_vintage")
    assert "retrieved_at" in report.section_lineage.columns
    payload = json.loads(json.dumps(report.scenario_lineage))
    assert payload["status"] == "unavailable"
    for scenario in payload["scenarios"].values():
        lineage = scenario["snapshot"]["series_lineage"]
        assert set(lineage) == {"headline_cpi", "core_cpi"}
        assert lineage["core_cpi"]["failure_code"]
        assert scenario["snapshot"]["reference_month"] == "2025-10-31T00:00:00"
        assert scenario["snapshot"]["information_timestamp"] is None
        assert scenario["snapshot"]["timing_status"] == "derived_value_unavailable"
        assert scenario["snapshot"]["lookahead_status"] == ("unavailable_required_scenario_input")
    table_payload = json.loads(
        report.current_scenario_table.to_json(orient="records", date_format="iso")
    )
    assert all(row["regime"] is None for row in table_payload)
    assert all(set(row["series_lineage"]) == {"headline_cpi", "core_cpi"} for row in table_payload)


def test_full_sample_baseline_discloses_when_missing_cpi_zeroes_historical_evidence() -> None:
    """Post-H2 audit: full_sample's whole-column lineage must disclose, not raise.

    ``full_sample``'s baseline value depends on every row, so the permanent
    2025-10 CPI gap marks the *entire* historical population ineligible under
    that baseline (unlike windowed baselines, which only lose eligibility near
    the gap). Zero eligible historical evidence must still leave the report
    unavailable, but ``build_macro_research_report`` must disclose that
    cleanly instead of raising: an uncaught raise here would abort the whole
    Streamlit script and silently skip every tab whose source block runs
    later than the Report tab (Paper Framework, Decay/Convergence,
    Robustness), even though those tabs render earlier in the visual tab bar.
    """

    dates = pd.date_range("1981-01-31", "2026-06-30", freq="ME")
    gap = dates == pd.Timestamp("2025-10-31")
    headline = 100.0 * (1.002 ** np.arange(len(dates), dtype=float))
    core = 95.0 * (1.0018 ** np.arange(len(dates), dtype=float))
    headline[gap] = np.nan
    core[gap] = np.nan
    warmup = build_base_frame(
        pd.DataFrame({"date": dates, "CPIAUCSL": headline, "CPILFESL": core, "TB3MS": 3.0}),
        imputation_policy="observed_only",
    )

    for sample_mode, start in (("live_dashboard", "1982-01-31"), ("max_history", "1981-01-31")):
        observed = warmup.loc[warmup["date"].ge(pd.Timestamp(start))].reset_index(drop=True)
        views = build_dashboard_data_views(
            observed,
            baseline_method="full_sample",
            warmup_raw=warmup,
            sample_mode=sample_mode,
        )
        assert not report_mod.observed_only_historical_eligibility(views.research_featured).any(), (
            "fixture must reproduce the whole-sample eligibility wipeout"
        )

        report = build_macro_research_report(
            views.research_raw,
            views.research_featured,
            baseline_method="full_sample",
            sample_mode=sample_mode,
            current_monitoring=views.current_monitoring,
            benchmark_horizons=(3,),
            market_horizons=(3,),
            robustness_baselines=("full_sample",),
        )

        assert not report.available
        assert report.reason is not None
        assert "no observed-only eligible rows" in report.reason
        assert "full_sample" in report.reason
        assert "rolling_36_shifted" in report.reason
        assert report.as_of == "2026-06-30"


def test_windowed_baseline_keeps_historical_evidence_despite_missing_cpi_month() -> None:
    """Scope check: the disclosure path must not relax windowed baselines too."""

    dates = pd.date_range("1981-01-31", "2026-06-30", freq="ME")
    gap = dates == pd.Timestamp("2025-10-31")
    headline = 100.0 * (1.002 ** np.arange(len(dates), dtype=float))
    core = 95.0 * (1.0018 ** np.arange(len(dates), dtype=float))
    headline[gap] = np.nan
    core[gap] = np.nan
    warmup = build_base_frame(
        pd.DataFrame({"date": dates, "CPIAUCSL": headline, "CPILFESL": core, "TB3MS": 3.0}),
        imputation_policy="observed_only",
    )
    observed = warmup.loc[warmup["date"].ge(pd.Timestamp("1982-01-31"))].reset_index(drop=True)
    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=warmup,
        sample_mode="live_dashboard",
    )
    assert report_mod.observed_only_historical_eligibility(views.research_featured).any()

    report = build_macro_research_report(
        views.research_raw,
        views.research_featured,
        baseline_method="rolling_36_shifted",
        sample_mode="live_dashboard",
        current_monitoring=views.current_monitoring,
        benchmark_horizons=(3,),
        market_horizons=(3,),
        robustness_baselines=("rolling_36_shifted",),
    )
    assert report.available


def test_historical_analog_population_rejects_uses_estimated_input_only() -> None:
    raw = _raw_frame(months=120)
    featured = add_transitory_inflation_features(raw, baseline_method="fed_target")
    snapshot = report_mod.latest_signal_snapshot(featured)
    featured["imputation_policy"] = "observed_only"
    featured["uses_estimated_input"] = True
    featured["signal_uses_imputed_input"] = False

    with pytest.raises(ValueError, match="no observed-only eligible rows"):
        report_mod._require_observed_only_authority(
            featured,
            label="Historical analog population",
        )

    analogs = report_mod._historical_analog_table(
        featured,
        snapshot,
        market_monthly=None,
        horizons=(3,),
        threshold_pp=0.5,
    )
    assert analogs.empty


def test_historical_analog_production_path_excludes_authoritative_estimate_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_frame(months=120)
    featured = add_transitory_inflation_features(raw, baseline_method="fed_target")
    snapshot = report_mod.latest_signal_snapshot(featured)
    featured["imputation_policy"] = "observed_only"
    featured["uses_estimated_input"] = False
    featured.loc[featured.index[-10], "uses_estimated_input"] = True
    captured: dict[str, pd.DataFrame] = {}
    original = report_mod.validation_mod.build_historical_validation_frame

    def capture(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        result = original(frame, **kwargs)
        captured["validation"] = result
        return result

    monkeypatch.setattr(
        report_mod.validation_mod,
        "build_historical_validation_frame",
        capture,
    )

    report_mod._historical_analog_table(
        featured,
        snapshot,
        market_monthly=None,
        horizons=(3,),
        threshold_pp=0.5,
    )

    validation = captured["validation"]
    contaminated = validation.loc[validation.index == featured.index[-10]]
    assert len(contaminated) == 1
    assert not bool(report_mod.observed_only_historical_eligibility(contaminated).iloc[0])


def test_historical_analogs_keep_clean_origin_at_unaffected_horizon() -> None:
    raw = _raw_frame(months=160)
    featured = add_transitory_inflation_features(raw, baseline_method="fed_target")
    featured["imputation_policy"] = "observed_only"
    featured["uses_estimated_input"] = False
    featured.loc[80, "uses_estimated_input"] = True
    horizons = (3, 12)
    validation = report_mod.validation_mod.build_historical_validation_frame(
        featured,
        forward_horizons=horizons,
        label_horizons=horizons,
    )
    origin = validation.loc[68]
    pressure = str(origin["historical_short_term_pressure"])
    term_structure = {
        "firming": "accelerating",
        "cooling": "decelerating",
        "mixed": "mixed",
    }[pressure]
    snapshot = {
        "available": True,
        "scenario_id": "base",
        "date": origin["date"],
        "reference_month": origin["date"],
        "information_timestamp": pd.NaT,
        "timing_status": "reference_month_only",
        "regime": str(origin["historical_regime"]),
        "term_structure": term_structure,
        "tinf_4m": float(origin["tinf_4m"]),
    }
    state = report_mod._tinf_state(snapshot["tinf_4m"])
    signal_origin = validation["signal_observed_only_eligible"].fillna(False).astype(bool)
    group = (
        signal_origin
        & validation["historical_regime"].eq(snapshot["regime"])
        & validation["historical_short_term_pressure"].eq(pressure)
        & validation["tinf_4m"].map(report_mod._tinf_state).eq(state)
    )
    expected_3m_count = int((group & validation["cpi_yoy_change_3m"].notna()).sum())

    analogs = report_mod._historical_analog_table(
        featured,
        snapshot,
        market_monthly=None,
        horizons=horizons,
        threshold_pp=0.5,
    )
    actual_3m_count = int(
        analogs.loc[analogs["horizon_months"].eq(3), "future_inflation_count"].iloc[0]
    )

    assert validation.loc[68, "observed_only_eligible_3m"]
    assert not validation.loc[68, "observed_only_eligible_12m"]
    assert actual_3m_count == expected_3m_count


def test_historical_analog_rejects_fully_non_observed_population() -> None:
    raw = _raw_frame(months=120)
    featured = add_transitory_inflation_features(raw, baseline_method="fed_target")
    snapshot = report_mod.latest_signal_snapshot(featured)
    featured["imputation_policy"] = "ex_post_continuity"

    with pytest.raises(ValueError, match="no observed-only eligible rows"):
        report_mod._require_observed_only_authority(
            featured,
            label="Historical analog population",
        )

    analogs = report_mod._historical_analog_table(
        featured,
        snapshot,
        market_monthly=None,
        horizons=(3,),
        threshold_pp=0.5,
    )
    assert analogs.empty


def test_report_fallback_rejects_estimated_current_without_scenario_bundle() -> None:
    dates = pd.date_range("2014-01-31", "2026-06-30", freq="ME")
    levels = 100.0 * (1.002 ** np.arange(len(dates), dtype=float))
    levels[dates == pd.Timestamp("2025-10-31")] = np.nan
    warmup = build_base_frame(
        pd.DataFrame({"date": dates, "CPIAUCSL": levels, "TB3MS": 3.0}),
        imputation_policy="observed_only",
    )
    observed = warmup.loc[warmup["date"].ge(pd.Timestamp("2015-01-31"))].reset_index(drop=True)
    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=warmup,
    )

    with pytest.raises(ValueError, match="fallback without a scenario bundle"):
        build_macro_research_report(
            views.research_raw,
            views.research_featured,
            baseline_method="rolling_36_shifted",
            sample_mode="live_dashboard",
            current_raw=views.current_raw,
            current_featured=views.current_featured,
            benchmark_horizons=(3,),
            market_horizons=(3,),
            robustness_baselines=("rolling_36_shifted",),
        )


def test_report_rejects_estimated_robustness_override() -> None:
    raw = _raw_frame(months=120)
    raw["imputation_policy"] = "observed_only"
    featured = add_transitory_inflation_features(raw, baseline_method="fed_target")
    contaminated = raw.copy()
    contaminated["uses_estimated_input"] = True

    with pytest.raises(ValueError, match="robustness frame.*no observed-only eligible"):
        build_macro_research_report(
            raw,
            featured,
            baseline_method="fed_target",
            sample_mode="live_dashboard",
            robustness_sample_frames={"contaminated": contaminated},
            benchmark_horizons=(3,),
            market_horizons=(3,),
            robustness_baselines=("fed_target",),
        )


def test_macro_report_uses_common_origin_panels_and_neutral_loss_language() -> None:
    """H2: report comparisons are common-origin point estimates, not win claims."""

    raw = _raw_frame(months=180)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")

    report = build_macro_research_report(
        raw,
        df,
        baseline_method="fed_target",
        sample_mode="live_dashboard",
        macro_status={"data_source_used": "unit"},
        market_monthly=None,
        market_status=None,
        benchmark_horizons=(3, 6),
        market_horizons=(3,),
        robustness_baselines=("fed_target",),
    )

    comparisons = report.benchmark_comparisons
    assert not comparisons.empty
    assert {
        "common_origin_n",
        "common_origin_start",
        "common_origin_end",
        "tinf_lower_mae",
        "tinf_lower_rmse",
        "mae_differential_pp",
        "rmse_differential_pp",
    }.issubset(comparisons.columns)

    # Both sides of every comparison are scored on the same origins.
    assert (comparisons["tinf_count"] == comparisons["comparison_count"]).all()
    assert (comparisons["tinf_count"] == comparisons["common_origin_n"]).all()

    # Every benchmark model in the report is scored on its horizon's panel.
    metrics = report.benchmark_metrics
    for _, group in metrics.groupby("horizon_months", sort=False):
        assert group["count"].nunique() == 1
        assert group["common_origin_n"].nunique() == 1

    # Neutral point-estimate language only; significance is deferred to H10.
    prose = " ".join(
        [*report.signal_confidence_lines, *report.robustness_lines, *report.caveats]
    ).lower()
    for banned in (" beats ", " wins ", "win rate", "best by mae", "best by rmse"):
        assert banned not in prose
    assert "point estimate" in prose
    assert any("common-origin panel" in line for line in report.signal_confidence_lines)
    # "outperform" may appear only inside the explicit no-significance disclaimer.
    assert "not evidence that one model outperforms another" in prose
    assert prose.count("outperform") == 1

    offending = [
        column
        for frame in (report.benchmark_metrics, report.benchmark_comparisons,
                      report.robustness_verdict, report.robustness_lower_loss_rates)
        for column in frame.columns
        if "beats" in str(column).lower() or "win" in str(column).lower()
    ]
    assert not offending


def test_report_discloses_a_horizon_with_no_common_origin_panel() -> None:
    """H2: an unscoreable horizon is named, never silently omitted."""

    raw = _raw_frame(months=180)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")

    metrics, comparisons, unscored = report_mod._benchmark_tables(
        df,
        horizons=(3, 170),
        threshold_pp=0.50,
    )

    assert 170 in unscored
    assert 170 not in set(comparisons["horizon_months"])
    assert 170 not in set(metrics["horizon_months"])

    lines = report_mod._benchmark_lines(comparisons, unscored)
    assert any("170M" in line for line in lines)
    assert any("share no common forecast origin" in line for line in lines)

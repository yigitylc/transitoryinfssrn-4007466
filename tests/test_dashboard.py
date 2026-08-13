from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from transitory_inflation import dashboard as dashboard_mod
from transitory_inflation import validation as validation_mod
from transitory_inflation.dashboard import (
    CURRENT_SIGNAL_IMPUTATION_NOTICE,
    CurrentMonitoringBundle,
    ScenarioStability,
    build_dashboard_data_views,
    classify_scenario_stability,
    current_signal_imputation_notice,
)
from transitory_inflation.data import (
    INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
    RELEASE_TIMESTAMP_PROVENANCE_ACTUAL,
    RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED,
    TIMING_STATUS_UNAVAILABLE,
    VALUE_PROVENANCE_UNVERIFIED,
    build_base_frame,
)
from transitory_inflation.features import latest_signal_snapshot


def _observed_frame_with_official_october_gap(
    start: str = "2014-01-31",
    end: str = "2026-06-30",
) -> tuple[pd.DataFrame, pd.Timestamp]:
    dates = pd.date_range(start, end, freq="ME")
    periods = len(dates)
    gap_date = pd.Timestamp("2025-10-31")
    headline = 100.0 * (1.002 ** np.arange(periods, dtype=float))
    core = 95.0 * (1.0018 ** np.arange(periods, dtype=float))
    headline[dates == gap_date] = np.nan
    core[dates == gap_date] = np.nan
    raw = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": headline,
            "CPILFESL": core,
            "TB3MS": np.linspace(0.5, 4.0, periods),
        }
    )
    return build_base_frame(raw, imputation_policy="observed_only"), gap_date


def test_dashboard_views_restore_only_the_descriptive_current_scenarios() -> None:
    observed, gap_date = _observed_frame_with_official_october_gap()

    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=observed,
    )

    assert views.research_raw["imputation_policy"].eq("observed_only").all()
    research_gap = views.research_raw.loc[views.research_raw["date"] == gap_date].iloc[0]
    assert pd.isna(research_gap["cpi_observed_level"])
    assert pd.isna(research_gap["core_cpi_observed_level"])
    assert pd.isna(research_gap["cpi_level"])
    assert pd.isna(research_gap["core_cpi_level"])
    assert not views.research_raw["cpi_imputed"].any()
    assert not views.research_raw["core_cpi_imputed"].any()
    assert tuple(views.current_monitoring.scenarios) == ("low", "base", "high")

    for scenario_id, scenario in views.current_monitoring.scenarios.items():
        current_gap = scenario.raw.loc[scenario.raw["date"] == gap_date].iloc[0]
        assert scenario.raw["imputation_policy"].eq("ex_post_continuity").all()
        assert scenario.raw["scenario_id"].eq(scenario_id).all()
        assert pd.isna(current_gap["cpi_observed_level"])
        assert pd.isna(current_gap["core_cpi_observed_level"])
        assert current_gap["cpi_imputed"]
        assert current_gap["core_cpi_imputed"]
        assert pd.notna(current_gap["cpi_level"])
        assert pd.notna(current_gap["core_cpi_level"])
        assert scenario.snapshot["available"]
        assert scenario.snapshot["date"] == observed["date"].iloc[-1]

    research_snapshot = latest_signal_snapshot(views.research_featured)
    current_snapshot = views.current_snapshot
    assert current_snapshot["available"]
    assert current_snapshot["date"] == observed["date"].iloc[-1]
    assert current_snapshot["date"] > research_snapshot["date"]


def test_current_signal_keeps_derived_estimate_lineage_and_notice() -> None:
    observed, _ = _observed_frame_with_official_october_gap()
    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=observed,
    )

    snapshot = views.current_snapshot
    assert snapshot["baseline_uses_imputed_input"]
    assert snapshot["epsilon_uses_imputed_input"]
    assert snapshot["tinf_4m_uses_imputed_input"]
    assert snapshot["percentile_uses_imputed_input"]
    assert snapshot["regime_uses_imputed_input"]
    assert snapshot["uses_imputed_input"]
    assert snapshot["uses_estimated_input"]
    assert snapshot["scenario_id"] == "base"
    assert snapshot["estimated_reference_month"] == pd.Timestamp("2025-10-31")
    assert snapshot["estimated_input_months"] == ("2025-10-31",)
    assert snapshot["calibration_policy"] == "observed_only_eligible_history"
    assert snapshot["calibration_cutoff_policy"] == "strictly_prior_to_current_reference_month"
    assert not snapshot["observed_only_eligible"]
    assert snapshot["timing_status"] == "reference_month_only"
    assert pd.isna(snapshot["information_timestamp"])
    assert snapshot["data_vintage_status"] == "latest_revised_non_vintage"
    assert current_signal_imputation_notice(snapshot) == CURRENT_SIGNAL_IMPUTATION_NOTICE


def test_dashboard_snapshot_separates_reference_month_from_information_timestamp() -> None:
    dates = pd.date_range("2015-01-31", periods=72, freq="ME")
    releases = pd.Series((dates + pd.offsets.Day(13) + pd.offsets.Hour(13)).tz_localize("UTC"))
    observed = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": 100.0 * (1.002 ** np.arange(len(dates), dtype=float)),
                "TB3MS": 1.0,
                "release_timestamp": releases,
                "release_timestamp_provenance": RELEASE_TIMESTAMP_PROVENANCE_ACTUAL,
                "release_timing_status": "release_aligned",
            }
        ),
        imputation_policy="observed_only",
    )

    snapshot = build_dashboard_data_views(
        observed,
        baseline_method="fed_target",
        warmup_raw=observed,
    ).current_snapshot

    assert snapshot["reference_month"] == dates[-1]
    assert snapshot["information_timestamp"] == releases.iloc[-1]
    assert snapshot["reference_month"] != snapshot["information_timestamp"]
    assert snapshot["timing_status"] == "release_aligned"


def test_dashboard_research_consumers_remain_observed_only() -> None:
    observed, _ = _observed_frame_with_official_october_gap()
    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=observed,
    )

    historical = validation_mod.build_historical_validation_frame(
        views.research_featured,
        forward_horizons=(3,),
        label_horizons=(3,),
    )
    assert views.research_featured["imputation_policy"].eq("observed_only").all()
    assert not views.research_featured["signal_uses_imputed_input"].any()
    assert not historical["signal_uses_imputed_input"].any()
    assert current_signal_imputation_notice(latest_signal_snapshot(views.research_featured)) is None


def test_dashboard_views_reject_ex_post_data_as_research_authority() -> None:
    observed, _ = _observed_frame_with_official_october_gap()
    observed["imputation_policy"] = "ex_post_continuity"

    with pytest.raises(ValueError, match="research authority.*observed_only"):
        build_dashboard_data_views(
            observed,
            baseline_method="rolling_36_shifted",
            warmup_raw=observed,
        )


def test_legacy_ex_post_continuity_cannot_route_into_dashboard_research() -> None:
    dates = pd.date_range("2014-01-31", "2026-06-30", freq="ME")
    trend = np.arange(len(dates), dtype=float)
    headline = 100.0 * (1.002**trend)
    core = 95.0 * (1.0018**trend)
    gap = dates == pd.Timestamp("2025-10-31")
    headline[gap] = np.nan
    core[gap] = np.nan
    legacy = build_base_frame(
        pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": headline,
                "CPILFESL": core,
                "TB3MS": 1.0,
            }
        ),
        imputation_policy="ex_post_continuity",
    )

    assert legacy["cpi_imputed"].any()
    with pytest.raises(ValueError, match="research authority.*observed_only"):
        build_dashboard_data_views(
            legacy,
            baseline_method="rolling_36_shifted",
            warmup_raw=legacy,
        )


@pytest.mark.parametrize(
    ("regimes", "pressures", "expected"),
    [
        (
            ("neutral", "neutral", "neutral"),
            ("mixed", "mixed", "mixed"),
            (True, True, True, False),
        ),
        (
            ("neutral", "elevated rising", "neutral"),
            ("mixed", "mixed", "mixed"),
            (False, True, False, True),
        ),
        (
            ("neutral", "neutral", "neutral"),
            ("cooling", "firming", "cooling"),
            (True, False, False, True),
        ),
    ],
)
def test_scenario_stability_requires_exact_three_scenario_agreement(
    regimes: tuple[str, str, str],
    pressures: tuple[str, str, str],
    expected: tuple[bool, bool, bool, bool],
) -> None:
    snapshots = {
        scenario_id: {
            "available": True,
            "regime": regime,
            "pressure": pressure,
        }
        for scenario_id, regime, pressure in zip(
            ("low", "base", "high"),
            regimes,
            pressures,
            strict=True,
        )
    }

    stability = classify_scenario_stability(snapshots)

    assert (
        stability.regime_stable,
        stability.pressure_stable,
        stability.fully_stable,
        stability.scenario_sensitive,
    ) == expected
    assert not stability.unavailable


def test_scenario_stability_fails_closed_when_any_scenario_is_unavailable() -> None:
    snapshots = {
        "low": {"available": True, "regime": "neutral", "pressure": "mixed"},
        "base": {"available": False, "reason": "insufficient calibration"},
        "high": {"available": True, "regime": "neutral", "pressure": "mixed"},
    }

    stability = classify_scenario_stability(snapshots)

    assert stability.unavailable
    assert not stability.regime_stable
    assert not stability.pressure_stable
    assert not stability.fully_stable
    assert not stability.scenario_sensitive
    assert stability.label == "unavailable"


def test_scenario_specific_failure_atomically_invalidates_every_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed, _ = _observed_frame_with_official_october_gap()
    original = dashboard_mod.latest_signal_snapshot
    calls = 0

    def fail_first_scenario(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        snapshot = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            snapshot["available"] = False
            snapshot["reason"] = "Synthetic scenario-specific feature failure."
        return snapshot

    monkeypatch.setattr(dashboard_mod, "latest_signal_snapshot", fail_first_scenario)

    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=observed,
    )

    assert views.current_monitoring.status == "unavailable"
    assert views.current_monitoring.stability.unavailable
    assert views.current_chart_featured is None
    for view in views.current_monitoring.scenarios.values():
        assert not view.snapshot["available"]
        assert view.snapshot["classification_status"] == ("unavailable_required_scenario_input")
        for key in ("tinf_4m", "tinf_4m_percentile", "regime", "pressure"):
            assert view.snapshot[key] is None
        assert view.featured[["tinf_4m", "tinf_8m", "tinf_12m"]].isna().all().all()
    assert views.current_featured[["tinf_4m", "tinf_8m", "tinf_12m"]].isna().all().all()


def test_bundle_status_cannot_disagree_with_available_scenario_payloads() -> None:
    observed, _ = _observed_frame_with_official_october_gap()
    available = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=observed,
    ).current_monitoring

    with pytest.raises(ValueError, match="cannot contain an available scenario"):
        CurrentMonitoringBundle(
            scenarios=available.scenarios,
            stability=ScenarioStability(
                regime_stable=False,
                pressure_stable=False,
                fully_stable=False,
                scenario_sensitive=False,
                unavailable=True,
            ),
            historical_evidence_endpoint=available.historical_evidence_endpoint,
            status="unavailable",
            reason="Synthetic inconsistent bundle.",
            sample_mode=available.sample_mode,
            baseline_method=available.baseline_method,
        )

    with pytest.raises(ValueError, match="requires only an observed current view"):
        CurrentMonitoringBundle(
            scenarios=available.scenarios,
            stability=ScenarioStability(
                regime_stable=False,
                pressure_stable=False,
                fully_stable=False,
                scenario_sensitive=False,
                unavailable=False,
                not_applicable=True,
            ),
            historical_evidence_endpoint=available.historical_evidence_endpoint,
            status="not_applicable",
            reason="Synthetic Paper Window leak.",
            sample_mode="paper_replication",
            baseline_method=available.baseline_method,
            observed_current=None,
        )

    leaked_scenarios = {
        scenario_id: type(view)(
            scenario_id=view.scenario_id,
            raw=view.raw,
            featured=pd.DataFrame(),
            snapshot={
                **view.snapshot,
                "available": False,
                "tinf_4m": 0.66,
                "regime": "neutral",
                "pressure": "firming",
            },
        )
        for scenario_id, view in available.scenarios.items()
    }
    with pytest.raises(ValueError, match="cannot expose snapshot signal fields"):
        CurrentMonitoringBundle(
            scenarios=leaked_scenarios,
            stability=ScenarioStability(
                regime_stable=False,
                pressure_stable=False,
                fully_stable=False,
                scenario_sensitive=False,
                unavailable=True,
            ),
            historical_evidence_endpoint=available.historical_evidence_endpoint,
            status="unavailable",
            reason="Synthetic snapshot leak.",
            sample_mode=available.sample_mode,
            baseline_method=available.baseline_method,
        )


def test_current_bundle_fails_closed_until_following_official_cpi_exists() -> None:
    observed, gap_date = _observed_frame_with_official_october_gap()
    observed = observed.loc[observed["date"] <= gap_date].copy()

    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=observed,
    )

    assert views.current_monitoring.stability.unavailable
    assert not views.current_monitoring.stability.regime_stable
    assert not views.current_monitoring.stability.pressure_stable
    assert not views.current_monitoring.stability.scenario_sensitive
    for scenario in views.current_monitoring.scenarios.values():
        gap = scenario.raw.loc[scenario.raw["date"] == gap_date].iloc[0]
        assert not gap["cpi_imputed"]
        assert not gap["core_cpi_imputed"]
        assert not scenario.snapshot["available"]
        assert scenario.snapshot["reference_month"] == gap_date
        assert "Exact November 2025 endpoint is absent" in scenario.snapshot["reason"]


def test_dashboard_requires_the_common_warmup_authority() -> None:
    observed, _ = _observed_frame_with_official_october_gap()

    with pytest.raises(ValueError, match="requires warmup_raw"):
        build_dashboard_data_views(
            observed,
            baseline_method="rolling_36_shifted",
        )


def test_trimmed_warmup_cannot_silently_change_live_features() -> None:
    full_warmup, _ = _observed_frame_with_official_october_gap(
        "1981-01-31",
        "2026-06-30",
    )
    observed = full_warmup.loc[full_warmup["date"] >= pd.Timestamp("1982-01-31")].reset_index(
        drop=True
    )

    proper = build_dashboard_data_views(
        observed,
        baseline_method="expanding_shifted",
        warmup_raw=full_warmup,
        sample_mode="live_dashboard",
    )
    assert proper.current_snapshot["available"]
    assert pd.notna(proper.current_snapshot["baseline"])
    assert pd.notna(proper.current_snapshot["tinf_4m"])

    with pytest.raises(ValueError, match="sample-trimmed"):
        build_dashboard_data_views(
            observed,
            baseline_method="expanding_shifted",
            warmup_raw=observed,
            sample_mode="live_dashboard",
        )


def test_loader_processed_warmup_is_validated_by_explicit_sample_start() -> None:
    raw_dates = pd.date_range("1980-01-31", "2026-06-30", freq="ME")
    raw = pd.DataFrame(
        {
            "date": raw_dates,
            "CPIAUCSL": 100.0 * (1.002 ** np.arange(len(raw_dates), dtype=float)),
            "CPILFESL": 95.0 * (1.0018 ** np.arange(len(raw_dates), dtype=float)),
            "TB3MS": 3.0,
        }
    )
    gap = raw["date"].eq(pd.Timestamp("2025-10-31"))
    raw.loc[gap, ["CPIAUCSL", "CPILFESL"]] = np.nan
    loader_warmup = build_base_frame(
        raw,
        start_date="1981-01-01",
        imputation_policy="observed_only",
    )
    observed = loader_warmup.loc[loader_warmup["date"].ge(pd.Timestamp("1982-01-31"))].reset_index(
        drop=True
    )

    assert pd.notna(loader_warmup.iloc[0]["inflation_yoy"])
    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=loader_warmup,
        sample_mode="live_dashboard",
    )

    assert views.current_monitoring.available
    assert views.current_snapshot["reference_month"] == pd.Timestamp("2026-06-30")


def test_incomplete_warmup_calendar_fails_before_scenario_calculation() -> None:
    full_warmup, _ = _observed_frame_with_official_october_gap(
        "1981-01-31",
        "2026-06-30",
    )
    incomplete_warmup = full_warmup.loc[
        full_warmup["date"].eq(pd.Timestamp("1981-01-31"))
        | full_warmup["date"].ge(pd.Timestamp("1982-01-31"))
    ].reset_index(drop=True)
    observed = incomplete_warmup.loc[
        incomplete_warmup["date"].ge(pd.Timestamp("1982-01-31"))
    ].reset_index(drop=True)

    with pytest.raises(ValueError, match="12 consecutive pre-sample months"):
        build_dashboard_data_views(
            observed,
            baseline_method="expanding_shifted",
            warmup_raw=incomplete_warmup,
            sample_mode="live_dashboard",
        )


def test_all_sample_modes_slice_one_common_superset() -> None:
    warmup, _ = _observed_frame_with_official_october_gap(
        "1981-01-31",
        "2026-06-30",
    )
    selected = {
        "paper_replication": warmup.loc[
            warmup["date"].between("1982-01-31", "2021-07-31")
        ].reset_index(drop=True),
        "live_dashboard": warmup.loc[warmup["date"] >= pd.Timestamp("1982-01-31")].reset_index(
            drop=True
        ),
        "max_history": warmup.copy(),
    }

    views = {
        mode: build_dashboard_data_views(
            frame,
            baseline_method="rolling_36_shifted",
            warmup_raw=warmup,
            sample_mode=mode,
        )
        for mode, frame in selected.items()
    }

    assert {view.warmup_start for view in views.values()} == {pd.Timestamp("1981-01-31")}
    assert views["paper_replication"].research_raw["date"].iloc[-1] == pd.Timestamp("2021-07-31")
    assert views["live_dashboard"].research_raw["date"].iloc[0] == pd.Timestamp("1982-01-31")
    assert views["max_history"].research_raw["date"].iloc[0] == pd.Timestamp("1981-01-31")


def test_paper_window_suppresses_missing_cpi_scenarios() -> None:
    warmup, _ = _observed_frame_with_official_october_gap(
        "1981-01-31",
        "2021-07-31",
    )
    observed = warmup.loc[warmup["date"] >= pd.Timestamp("1982-01-31")].reset_index(drop=True)

    views = build_dashboard_data_views(
        observed,
        baseline_method="full_sample",
        warmup_raw=warmup,
        sample_mode="paper_replication",
    )

    bundle = views.current_monitoring
    assert bundle.status == "not_applicable"
    assert not bundle.applicable
    assert not bundle.available
    assert bundle.scenarios == {}
    assert bundle.scenario_table().empty
    assert bundle.stability.not_applicable
    assert not bundle.stability.regime_stable
    assert not bundle.stability.pressure_stable
    assert views.current_raw.equals(views.research_raw)
    assert views.current_featured.equals(views.research_featured)
    assert views.current_snapshot["scenario_applicability_status"] == "not_applicable"


@pytest.mark.parametrize(
    ("failure", "expected_series", "expected_code"),
    [
        ("headline_absent", "headline_cpi", "required_series_absent"),
        ("core_absent", "core_cpi", "required_series_absent"),
        ("headline_september", "headline_cpi", "september_endpoint_null"),
        ("core_september", "core_cpi", "september_endpoint_null"),
        ("core_november", "core_cpi", "november_endpoint_null"),
        ("headline_november", "headline_cpi", "november_endpoint_null"),
    ],
)
def test_required_series_or_endpoint_failure_invalidates_the_entire_bundle(
    failure: str,
    expected_series: str,
    expected_code: str,
) -> None:
    observed, _ = _observed_frame_with_official_october_gap()
    if failure.endswith("_absent"):
        prefix = "cpi" if failure.startswith("headline") else "core_cpi"
        observed = observed.drop(
            columns=[column for column in observed if column.startswith(prefix)]
        )
    else:
        series_prefix = "cpi" if failure.startswith("headline") else "core_cpi"
        month = pd.Timestamp("2025-09-30" if failure.endswith("september") else "2025-11-30")
        row = observed["date"].eq(month)
        observed.loc[row, f"{series_prefix}_observed_level"] = np.nan
        observed.loc[row, f"{series_prefix}_level"] = np.nan
        observed.loc[row, f"{series_prefix}_originally_missing"] = True

    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=observed,
    )

    bundle = views.current_monitoring
    assert bundle.status == "unavailable"
    assert bundle.stability.unavailable
    assert not bundle.available
    assert len(bundle.scenarios) == 3
    for scenario in bundle.scenarios.values():
        snapshot = scenario.snapshot
        assert not snapshot["available"]
        assert snapshot["classification_status"] == "unavailable_required_scenario_input"
        assert snapshot["date"] == pd.Timestamp("2025-10-31")
        assert snapshot["reference_month"] == pd.Timestamp("2025-10-31")
        assert snapshot["signal_reference_month"] == pd.Timestamp("2025-10-31")
        assert pd.isna(snapshot["release_timestamp"])
        assert snapshot["release_timestamp_provenance"] == (RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED)
        assert pd.isna(snapshot["information_timestamp"])
        assert snapshot["information_timestamp_provenance"] == (
            INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED
        )
        assert snapshot["timing_status"] == TIMING_STATUS_UNAVAILABLE
        assert pd.isna(snapshot["signal_information_timestamp"])
        assert snapshot["signal_timing_status"] == TIMING_STATUS_UNAVAILABLE
        assert snapshot["lookahead_status"] == "unavailable_required_scenario_input"
        for key in (
            "baseline",
            "epsilon",
            "tinf_4m",
            "tinf_4m_percentile",
            "regime",
            "pressure",
        ):
            assert snapshot[key] is None
        lineage = snapshot["series_lineage"][expected_series]
        assert not lineage["available"]
        assert lineage["failure_code"] == expected_code

    table = bundle.scenario_table()
    assert len(table) == 3
    assert table["bundle_status"].eq("unavailable").all()
    assert table["available"].eq(False).all()
    assert table["regime"].isna().all()
    assert table["pressure"].isna().all()


def test_current_chart_surface_fails_closed_but_preserves_paper_window() -> None:
    observed, _ = _observed_frame_with_official_october_gap()
    available = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=observed,
    )
    unavailable_authority = observed.copy()
    november = unavailable_authority["date"].eq(pd.Timestamp("2025-11-30"))
    unavailable_authority.loc[november, "core_cpi_value_provenance"] = VALUE_PROVENANCE_UNVERIFIED
    unavailable = build_dashboard_data_views(
        unavailable_authority,
        baseline_method="rolling_36_shifted",
        warmup_raw=unavailable_authority,
    )
    paper_warmup, _ = _observed_frame_with_official_october_gap(
        "1981-01-31",
        "2021-07-31",
    )
    paper_observed = paper_warmup.loc[
        paper_warmup["date"].ge(pd.Timestamp("1982-01-31"))
    ].reset_index(drop=True)
    paper = build_dashboard_data_views(
        paper_observed,
        baseline_method="full_sample",
        warmup_raw=paper_warmup,
        sample_mode="paper_replication",
    )

    assert available.current_chart_featured is not None
    assert available.current_chart_featured.equals(available.current_featured)
    assert unavailable.current_monitoring.status == "unavailable"
    assert unavailable.current_chart_featured is None
    assert paper.current_monitoring.status == "not_applicable"
    assert paper.current_chart_featured is not None
    assert paper.current_chart_featured.equals(paper.research_featured)


def test_available_and_unavailable_bundle_lineage_are_json_round_trip_safe() -> None:
    available_observed, _ = _observed_frame_with_official_october_gap()
    available = build_dashboard_data_views(
        available_observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=available_observed,
    ).current_monitoring

    unavailable_observed = available_observed.drop(
        columns=[column for column in available_observed if column.startswith("core_cpi")]
    )
    unavailable = build_dashboard_data_views(
        unavailable_observed,
        baseline_method="rolling_36_shifted",
        warmup_raw=unavailable_observed,
    ).current_monitoring

    for bundle, expected_status in (
        (available, "available"),
        (unavailable, "unavailable"),
    ):
        restored = json.loads(json.dumps(bundle.to_dict(), allow_nan=False))
        assert restored["status"] == expected_status
        assert set(restored["scenarios"]) == {"low", "base", "high"}
        for scenario in restored["scenarios"].values():
            assert set(scenario["snapshot"]["series_lineage"]) == {
                "headline_cpi",
                "core_cpi",
            }

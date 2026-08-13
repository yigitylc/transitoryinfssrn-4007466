from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.config import DEFAULT_SAMPLE_MODE, SAMPLE_MODES, resolve_sample_mode
from transitory_inflation.data import (
    BASE_FRED_SERIES,
    CURRENT_MONITORING_ESTIMATE_METHODS,
    CURRENT_MONITORING_SCENARIOS,
    FRED_API_URL,
    INFLATION_MEASURES,
    MACRO_CACHE_SCHEMA_COLUMN,
    MACRO_CACHE_SCHEMA_VERSION,
    RELEASE_TIMESTAMP_PROVENANCE_ACTUAL,
    TIMING_STATUS_RELEASE_ALIGNED,
    VALUE_PROVENANCE_OFFICIAL_FRED,
    VALUE_PROVENANCE_UNVERIFIED,
    MacroDataLoadResult,
    apply_sample_mode,
    build_base_frame,
    build_current_monitoring_scenario_frame,
    build_macro_cache_frame,
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


def _official_october_gap_raw(
    start: str = "2014-01-31",
    end: str = "2026-10-31",
) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="ME")
    trend = np.arange(len(dates), dtype=float)
    headline = 100.0 * (1.002**trend)
    core = 95.0 * (1.0018**trend)
    gap = dates == pd.Timestamp("2025-10-31")
    headline[gap] = np.nan
    core[gap] = np.nan
    return pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": headline,
            "CPILFESL": core,
            "TB3MS": np.linspace(0.5, 4.0, len(dates)),
        }
    )


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


def test_current_monitoring_scenarios_preserve_official_nulls_and_exact_formulas() -> None:
    observed = build_base_frame(
        _official_october_gap_raw(),
        imputation_policy="observed_only",
    )
    gap_date = pd.Timestamp("2025-10-31")
    prior_date = pd.Timestamp("2025-09-30")
    following_date = pd.Timestamp("2025-11-30")
    observed_gap = observed.loc[observed["date"] == gap_date].iloc[0]
    prior = observed.loc[observed["date"] == prior_date].iloc[0]
    following = observed.loc[observed["date"] == following_date].iloc[0]
    expected_headline = {
        "low": float(prior["cpi_observed_level"]),
        "base": float(
            np.sqrt(float(prior["cpi_observed_level"]) * float(following["cpi_observed_level"]))
        ),
        "high": float(following["cpi_observed_level"]),
    }
    expected_core = {
        "low": float(prior["core_cpi_observed_level"]),
        "base": float(
            np.sqrt(
                float(prior["core_cpi_observed_level"])
                * float(following["core_cpi_observed_level"])
            )
        ),
        "high": float(following["core_cpi_observed_level"]),
    }

    assert pd.isna(observed_gap["cpi_observed_level"])
    assert pd.isna(observed_gap["core_cpi_observed_level"])
    assert pd.isna(observed_gap["cpi_level"])
    assert pd.isna(observed_gap["core_cpi_level"])

    for scenario_id in CURRENT_MONITORING_SCENARIOS:
        scenario = build_current_monitoring_scenario_frame(
            observed,
            scenario_id=scenario_id,
        )
        gap = scenario.loc[scenario["date"] == gap_date].iloc[0]

        assert pd.isna(gap["cpi_observed_level"])
        assert pd.isna(gap["core_cpi_observed_level"])
        assert gap["cpi_originally_missing"]
        assert gap["core_cpi_originally_missing"]
        assert gap["cpi_imputed"]
        assert gap["core_cpi_imputed"]
        assert gap["cpi_level"] == pytest.approx(expected_headline[scenario_id])
        assert gap["core_cpi_level"] == pytest.approx(expected_core[scenario_id])
        assert gap["estimate_value"] == pytest.approx(expected_headline[scenario_id])
        assert gap["core_cpi_estimate_value"] == pytest.approx(expected_core[scenario_id])
        assert gap["estimate_method"] == CURRENT_MONITORING_ESTIMATE_METHODS[scenario_id]
        assert gap["estimated_reference_month"] == gap_date
        assert gap["scenario_id"] == scenario_id
        assert gap["scenario_bundle_available"]
        assert gap["uses_estimated_input"]
        assert gap["estimated_input_months"] == ("2025-10-31",)
        assert (
            gap["imputation_availability_basis"]
            == "following_reference_month_end_plus_one_month_proxy"
        )
        assert pd.isna(gap["cpi_level_information_timestamp"])
        assert gap["inflation_yoy_timing_status"] == "reference_month_only"
        pre_scenario = scenario.loc[scenario["date"] < gap_date]
        assert pre_scenario["estimate_method"].isna().all()
        assert pre_scenario["estimated_reference_month"].isna().all()
        assert pre_scenario["estimate_value"].isna().all()


def test_current_scenario_requires_exact_november_when_december_remains() -> None:
    observed = build_base_frame(
        _official_october_gap_raw(),
        imputation_policy="observed_only",
    )
    observed = observed.loc[~observed["date"].eq(pd.Timestamp("2025-11-30"))].reset_index(drop=True)

    for scenario_id in CURRENT_MONITORING_SCENARIOS:
        scenario = build_current_monitoring_scenario_frame(
            observed,
            scenario_id=scenario_id,
        )
        gap = scenario.loc[scenario["date"].eq(pd.Timestamp("2025-10-31"))].iloc[0]
        latest = scenario.iloc[-1]

        assert pd.Timestamp("2025-12-31") in set(scenario["date"])
        assert not latest["scenario_bundle_available"]
        assert set(latest["scenario_failed_series"]) == {
            "headline_cpi",
            "core_cpi",
        }
        assert not gap["cpi_imputed"]
        assert not gap["core_cpi_imputed"]
        assert pd.isna(gap["cpi_level"])
        assert pd.isna(gap["core_cpi_level"])
        assert pd.isna(gap["estimate_value"])
        assert pd.isna(gap["core_cpi_estimate_value"])
        for lineage in latest["scenario_series_lineage"].values():
            assert lineage["failure_code"] == "november_endpoint_absent"
            assert lineage["november_endpoint"]["month"] == ("2025-11-30T00:00:00")
            assert lineage["november_endpoint"]["value"] is None


@pytest.mark.parametrize(
    ("endpoint_month", "failure_code"),
    [
        (pd.Timestamp("2025-09-30"), "september_endpoint_ambiguous"),
        (pd.Timestamp("2025-11-30"), "november_endpoint_ambiguous"),
    ],
)
def test_current_scenario_preserves_raw_exact_endpoint_ambiguity(
    endpoint_month: pd.Timestamp,
    failure_code: str,
) -> None:
    raw = _official_october_gap_raw()
    duplicate = raw.loc[raw["date"].eq(endpoint_month)].copy()
    duplicate["CPIAUCSL"] = 999.0
    duplicate["CPILFESL"] = 888.0
    observed = build_base_frame(
        pd.concat([raw, duplicate], ignore_index=True),
        imputation_policy="observed_only",
    )

    scenario = build_current_monitoring_scenario_frame(
        observed,
        scenario_id="base",
    )
    latest = scenario.iloc[-1]

    assert observed["date"].eq(endpoint_month).sum() == 1
    assert not latest["scenario_bundle_available"]
    assert set(latest["scenario_failed_series"]) == {
        "headline_cpi",
        "core_cpi",
    }
    for lineage in latest["scenario_series_lineage"].values():
        assert not lineage["available"]
        assert lineage["failure_code"] == failure_code
        endpoint = (
            lineage["september_endpoint"]
            if endpoint_month.month == 9
            else lineage["november_endpoint"]
        )
        assert endpoint["month"] == endpoint_month.isoformat()
        assert endpoint["value"] is None


def test_current_scenario_value_acceptance_is_independent_of_release_precision() -> None:
    observed = build_base_frame(
        _official_october_gap_raw(),
        imputation_policy="observed_only",
    )
    scenario = build_current_monitoring_scenario_frame(
        observed,
        scenario_id="base",
    )
    lineage = scenario.iloc[-1]["scenario_series_lineage"]

    assert scenario.iloc[-1]["scenario_bundle_available"]
    for series_lineage in lineage.values():
        assert series_lineage["available"]
        assert series_lineage["september_endpoint"]["timing_status"] == ("reference_month_only")
        assert series_lineage["november_endpoint"]["timing_status"] == ("reference_month_only")
        assert series_lineage["november_endpoint"]["release_timestamp"] is None


def test_partial_endpoint_failure_retains_attempted_lineage_without_applying_it() -> None:
    observed = build_base_frame(
        _official_october_gap_raw(),
        imputation_policy="observed_only",
    )
    november = observed["date"].eq(pd.Timestamp("2025-11-30"))
    observed.loc[november, "core_cpi_value_provenance"] = VALUE_PROVENANCE_UNVERIFIED

    scenario = build_current_monitoring_scenario_frame(
        observed,
        scenario_id="base",
    )
    gap = scenario.loc[scenario["date"].eq(pd.Timestamp("2025-10-31"))].iloc[0]
    latest = scenario.iloc[-1]
    lineage = latest["scenario_series_lineage"]

    assert not latest["scenario_bundle_available"]
    assert latest["scenario_failed_series"] == ("core_cpi",)
    assert lineage["headline_cpi"]["estimate_attempted"]
    assert lineage["headline_cpi"]["estimate_value"] is not None
    assert not lineage["core_cpi"]["estimate_attempted"]
    assert lineage["core_cpi"]["failure_code"] == ("november_endpoint_invalid_provenance")
    assert not gap["cpi_imputed"]
    assert not gap["core_cpi_imputed"]
    assert pd.isna(gap["cpi_level"])
    assert pd.isna(gap["core_cpi_level"])
    assert latest["uses_estimated_input"]
    assert latest["uses_imputed_input"]
    assert latest["estimated_input_months"] == ("2025-10-31",)
    assert json.loads(json.dumps(lineage)) == lineage


def test_missing_preserved_value_provenance_fails_closed() -> None:
    observed = build_base_frame(
        _official_october_gap_raw(),
        imputation_policy="observed_only",
    ).drop(columns="core_cpi_value_provenance")

    scenario = build_current_monitoring_scenario_frame(
        observed,
        scenario_id="base",
    )
    core_lineage = scenario.iloc[-1]["core_cpi_scenario_lineage"]

    assert not scenario.iloc[-1]["scenario_bundle_available"]
    assert core_lineage["failure_code"] == "september_endpoint_invalid_provenance"
    assert core_lineage["september_endpoint"]["value_provenance"] == (VALUE_PROVENANCE_UNVERIFIED)
    assert not scenario["cpi_imputed"].any()
    assert not scenario["core_cpi_imputed"].any()


def test_missing_required_core_series_returns_structured_unavailable_frame() -> None:
    observed = build_base_frame(
        _official_october_gap_raw(),
        imputation_policy="observed_only",
    )
    core_columns = [column for column in observed.columns if column.startswith("core_cpi")]
    scenario = build_current_monitoring_scenario_frame(
        observed.drop(columns=core_columns),
        scenario_id="base",
    )
    latest = scenario.iloc[-1]

    assert not latest["scenario_bundle_available"]
    assert latest["scenario_failed_series"] == ("core_cpi",)
    assert latest["scenario_series_lineage"]["core_cpi"]["failure_code"] == (
        "required_series_absent"
    )
    assert "core_cpi_level" in scenario.columns
    assert scenario["core_cpi_level"].isna().all()
    assert not scenario["cpi_imputed"].any()


@pytest.mark.parametrize(
    ("missing_series", "expected_key", "expected_yoy"),
    [
        ("CPIAUCSL", "headline_cpi", "inflation_yoy"),
        ("CPILFESL", "core_cpi", "core_cpi_yoy"),
    ],
)
def test_raw_required_series_absence_remains_structured_through_scenario_routing(
    missing_series: str,
    expected_key: str,
    expected_yoy: str,
) -> None:
    observed = build_base_frame(
        _official_october_gap_raw().drop(columns=missing_series),
        imputation_policy="observed_only",
    )

    scenario = build_current_monitoring_scenario_frame(
        observed,
        scenario_id="base",
    )
    latest = scenario.iloc[-1]

    assert expected_yoy in observed.columns
    assert observed[expected_yoy].isna().all()
    assert not latest["scenario_bundle_available"]
    assert latest["scenario_series_lineage"][expected_key]["failure_code"] == (
        "required_series_absent"
    )


def test_direct_scenario_rejects_a_sample_trimmed_warmup_authority() -> None:
    full = build_base_frame(
        _official_october_gap_raw(start="1981-01-31"),
        imputation_policy="observed_only",
    )
    processed_trimmed = full.loc[full["date"].ge(pd.Timestamp("1982-01-31"))].reset_index(drop=True)
    raw_trimmed = _official_october_gap_raw(start="1982-01-31")

    for trimmed in (processed_trimmed, raw_trimmed):
        with pytest.raises(ValueError, match="12 consecutive pre-sample months"):
            build_current_monitoring_scenario_frame(
                trimmed,
                scenario_id="base",
                start_date="1982-01-01",
            )


def test_macro_data_load_result_preserves_former_six_positional_arguments() -> None:
    data = pd.DataFrame({"date": [pd.Timestamp("2025-01-31")]})
    result = MacroDataLoadResult(
        data,
        "cached_fred",
        "cache ok",
        "fred_base_macro_max_history.csv",
        True,
        "ex_post_continuity",
    )

    assert result.data is data
    assert result.data_source_used == "cached_fred"
    assert result.live_fetch_status == "cache ok"
    assert result.cache_file_used == "fred_base_macro_max_history.csv"
    assert result.api_key_configured is True
    assert result.imputation_policy == "ex_post_continuity"
    assert result.warmup_data is None


def test_scenarios_use_the_same_warmup_authority_before_sample_slicing() -> None:
    raw = _official_october_gap_raw("2024-12-31", "2026-10-31")
    observed_warmup = build_base_frame(raw, imputation_policy="observed_only")
    observed_selected = build_base_frame(
        raw,
        start_date="2026-01-01",
        imputation_policy="observed_only",
    )
    first_date = pd.Timestamp("2026-01-31")
    affected_date = pd.Timestamp("2026-10-31")

    assert observed_selected["date"].iloc[0] == first_date
    assert pd.notna(observed_selected["inflation_yoy"].iloc[0])

    for scenario_id in CURRENT_MONITORING_SCENARIOS:
        scenario = build_current_monitoring_scenario_frame(
            observed_warmup,
            scenario_id=scenario_id,
            start_date="2026-01-01",
        )

        assert scenario["date"].iloc[0] == first_date
        assert scenario["inflation_yoy"].iloc[0] == pytest.approx(
            observed_selected["inflation_yoy"].iloc[0]
        )
        unaffected = ~scenario["date"].isin([pd.Timestamp("2025-10-31"), affected_date])
        pd.testing.assert_series_equal(
            scenario.loc[unaffected, "inflation_yoy"].reset_index(drop=True),
            observed_selected.loc[unaffected, "inflation_yoy"].reset_index(drop=True),
        )
        scenario_featured = add_transitory_inflation_features(
            scenario,
            baseline_method="fed_target",
        )
        affected = scenario_featured.loc[scenario_featured["date"] == affected_date].iloc[0]
        expected_yoy = (
            float(affected["cpi_observed_level"]) / float(affected["estimate_value"]) - 1.0
        ) * 100.0
        assert affected["inflation_yoy"] == pytest.approx(expected_yoy)
        assert affected["inflation_yoy_uses_imputed_input"]
        assert affected["estimated_input_months"] == ("2025-10-31",)


def test_build_base_frame_end_date_trims_inclusively() -> None:
    dates = pd.date_range("1981-01-01", "2022-12-01", freq="MS")
    cpi = 100.0 * (1.005 ** np.arange(len(dates)))
    merged = pd.DataFrame({"date": dates, "CPIAUCSL": cpi, "TB3MS": 5.0})

    out = build_base_frame(merged, start_date="1982-01-01", end_date="2021-07-31")

    assert out["date"].iloc[-1] == pd.Timestamp("2021-07-31")
    assert out["date"].iloc[0] == pd.Timestamp("1982-01-31")


def test_build_base_frame_defaults_to_observed_only_with_explicit_lineage() -> None:
    dates = pd.date_range("2018-01-31", periods=30, freq="ME")
    levels = 100.0 + np.arange(30, dtype=float)
    gap_pos = 16
    levels[gap_pos] = np.nan

    out = build_base_frame(pd.DataFrame({"date": dates, "CPIAUCSL": levels, "TB3MS": 1.0}))

    gap = out.iloc[gap_pos]
    lagged_effect = out.iloc[gap_pos + 12]
    assert out["imputation_policy"].eq("observed_only").all()
    assert gap["cpi_originally_missing"]
    assert pd.isna(gap["cpi_observed_level"])
    assert pd.isna(gap["cpi_level"])
    assert not gap["cpi_imputed"]
    assert pd.isna(gap["inflation_yoy"])
    assert gap["inflation_yoy_uses_missing_input"]
    assert not gap["inflation_yoy_uses_imputed_input"]
    assert pd.isna(lagged_effect["inflation_yoy"])
    assert lagged_effect["inflation_yoy_uses_missing_input"]


def test_pre_series_rows_are_unavailable_not_officially_missing_inputs() -> None:
    dates = pd.date_range("2010-01-31", periods=30, freq="ME")
    levels = np.concatenate([np.full(5, np.nan), 100.0 + np.arange(25, dtype=float)])

    out = build_base_frame(pd.DataFrame({"date": dates, "CPIAUCSL": levels, "TB3MS": 1.0}))

    assert out.loc[:4, "cpi_source_unavailable"].all()
    assert not out.loc[:4, "cpi_originally_missing"].any()
    assert not out.loc[:4, "inflation_yoy_uses_missing_input"].any()
    assert not out.loc[5:, "cpi_source_unavailable"].any()


def test_build_base_frame_rejects_unknown_imputation_policy() -> None:
    merged = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=2, freq="ME"),
            "CPIAUCSL": [100.0, 101.0],
            "TB3MS": 1.0,
        }
    )

    with pytest.raises(ValueError, match="Unknown imputation policy"):
        build_base_frame(merged, imputation_policy="one_sided_nowcast")
    with pytest.raises(ValueError, match="Unknown imputation policy"):
        load_cached_macro_data_for_mode(
            "max_history",
            imputation_policy="one_sided_nowcast",
        )


def test_build_base_frame_ex_post_bridges_single_month_cpi_gap_log_linearly() -> None:
    # Mirrors the canceled 2025-10 CPI release: one interior month missing.
    dates = pd.date_range("1981-01-01", "1983-12-01", freq="MS")
    cpi = 100.0 * (1.005 ** np.arange(len(dates), dtype=float))
    cpi[22] = np.nan  # 1982-11, interior single-month gap

    merged = pd.DataFrame({"date": dates, "CPIAUCSL": cpi, "TB3MS": 5.0})
    out = build_base_frame(
        merged,
        start_date="1982-01-01",
        end_date=None,
        imputation_policy="ex_post_continuity",
    )

    gap_row = out.loc[out["date"] == pd.Timestamp("1982-11-30")].iloc[0]
    # Log-linear bridge = geometric mean of neighbors, which under constant
    # growth recovers the true level exactly.
    assert gap_row["cpi_imputed"]
    assert gap_row["cpi_originally_missing"]
    assert pd.isna(gap_row["cpi_observed_level"])
    assert gap_row["imputation_method"] == "log_linear_bridge"
    assert pd.notna(gap_row["imputation_available_at"])
    assert (
        gap_row["imputation_availability_basis"]
        == "following_reference_month_end_plus_one_month_proxy"
    )
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


def test_build_base_frame_adds_alternative_inflation_measure_columns() -> None:
    dates = pd.date_range("1981-01-01", "1983-12-01", freq="MS")
    trend = np.arange(len(dates), dtype=float)
    merged = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": 100.0 * (1.005**trend),
            "CPILFESL": 95.0 * (1.004**trend),
            "PCEPI": 90.0 * (1.003**trend),
            "PCEPILFE": 92.0 * (1.002**trend),
            "TB3MS": 5.0,
        }
    )

    out = build_base_frame(merged, start_date="1982-01-01", end_date=None)

    expected_columns = {
        "cpi_level",
        "inflation_yoy",
        "cpi_imputed",
        "core_cpi_level",
        "core_cpi_yoy",
        "core_cpi_imputed",
        "pce_level",
        "pce_yoy",
        "pce_imputed",
        "core_pce_level",
        "core_pce_yoy",
        "core_pce_imputed",
    }
    assert expected_columns.issubset(out.columns)
    assert out["inflation_yoy"].notna().all()
    assert out["core_cpi_yoy"].notna().all()
    assert out["pce_yoy"].notna().all()
    assert out["core_pce_yoy"].notna().all()


def test_build_base_frame_ex_post_imputes_alternative_measure_single_gap_not_tail() -> None:
    dates = pd.date_range("1981-01-01", "1983-12-01", freq="MS")
    trend = np.arange(len(dates), dtype=float)
    core_cpi = 95.0 * (1.004**trend)
    pce = 90.0 * (1.003**trend)
    core_cpi[22] = np.nan
    pce[-1] = np.nan
    merged = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": 100.0 * (1.005**trend),
            "CPILFESL": core_cpi,
            "PCEPI": pce,
            "TB3MS": 5.0,
        }
    )

    out = build_base_frame(
        merged,
        start_date="1982-01-01",
        end_date=None,
        imputation_policy="ex_post_continuity",
    )

    core_gap_row = out.loc[out["date"] == pd.Timestamp("1982-11-30")].iloc[0]
    pce_tail_row = out.loc[out["date"] == pd.Timestamp("1983-12-31")].iloc[0]
    assert core_gap_row["core_cpi_imputed"]
    assert pd.notna(core_gap_row["core_cpi_level"])
    assert not pce_tail_row["pce_imputed"]
    assert pd.isna(pce_tail_row["pce_level"])


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


def _fred_observations(
    series_id: str, periods: int = 15, missing_index: int | None = None
) -> list[dict[str, str]]:
    dates = pd.date_range("2020-01-01", periods=periods, freq="MS")
    observations = []
    for index, date in enumerate(dates):
        if missing_index is not None and index == missing_index:
            value = "."
        elif series_id in {"CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"}:
            value = f"{100 + index:.3f}"
        else:
            value = f"{1 + index / 10:.3f}"
        observations.append({"date": date.strftime("%Y-%m-%d"), "value": value})
    return observations


def _fred_csv(series_id: str, periods: int = 15) -> str:
    dates = pd.date_range("2020-01-01", periods=periods, freq="MS")
    rows = ["observation_date," + series_id]
    for index, date in enumerate(dates):
        value = (
            100 + index
            if series_id in {"CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"}
            else 1 + index / 10
        )
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


def test_versioned_cache_round_trip_preserves_raw_missingness_and_rebuilds_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        dates = pd.date_range("2018-01-31", periods=30, freq="ME")
        levels = 100.0 + np.arange(30, dtype=float)
        gap_pos = 16
        levels[gap_pos] = np.nan
        releases = (dates + pd.offsets.Day(15) + pd.offsets.Hour(13)).tz_localize("UTC")
        processed = build_base_frame(
            pd.DataFrame(
                {
                    "date": dates,
                    "CPIAUCSL": levels,
                    "TB3MS": 1.0,
                    "release_timestamp": releases,
                    "release_timestamp_provenance": (RELEASE_TIMESTAMP_PROVENANCE_ACTUAL),
                    "timing_status": TIMING_STATUS_RELEASE_ALIGNED,
                }
            ),
            imputation_policy="ex_post_continuity",
        )

        cache = build_macro_cache_frame(processed)
        assert cache[MACRO_CACHE_SCHEMA_COLUMN].eq(MACRO_CACHE_SCHEMA_VERSION).all()
        assert pd.isna(cache.loc[gap_pos, "CPIAUCSL"])
        assert cache.loc[gap_pos, "cpi_originally_missing"]
        assert cache["cpi_value_provenance"].eq(VALUE_PROVENANCE_OFFICIAL_FRED).all()
        assert "cpi_level" not in cache.columns
        assert "cpi_imputed" not in cache.columns
        assert cache.loc[gap_pos + 1, "release_timestamp"] == releases[gap_pos + 1]
        assert cache.loc[0, "release_timing_status"] == TIMING_STATUS_RELEASE_ALIGNED
        cache.to_csv(cache_path / "fred_base_macro_max_history.csv", index=False)

        observed = load_cached_macro_data_for_mode("max_history")
        continuity = load_cached_macro_data_for_mode(
            "max_history",
            imputation_policy="ex_post_continuity",
        )

        assert observed.loc[gap_pos, "cpi_originally_missing"]
        assert observed["cpi_value_provenance"].eq(VALUE_PROVENANCE_OFFICIAL_FRED).all()
        assert pd.isna(observed.loc[gap_pos, "cpi_level"])
        assert not observed.loc[gap_pos, "cpi_imputed"]
        assert continuity.loc[gap_pos, "cpi_originally_missing"]
        assert continuity.loc[gap_pos, "cpi_imputed"]
        assert pd.notna(continuity.loc[gap_pos, "cpi_level"])
        assert continuity.loc[gap_pos, "imputation_method"] == "log_linear_bridge"
        assert continuity.loc[gap_pos, "imputation_available_at"] == releases[gap_pos + 1]
        assert observed.iloc[-1]["timing_status"] == TIMING_STATUS_RELEASE_ALIGNED
        assert (
            continuity.loc[gap_pos, "imputation_availability_basis"]
            == "following_release_timestamp"
        )


def test_scenario_estimates_cannot_be_promoted_to_cache_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        gap_date = pd.Timestamp("2025-10-31")
        observed = build_base_frame(
            _official_october_gap_raw("2018-01-31", "2026-06-30"),
            imputation_policy="observed_only",
        )
        scenario = build_current_monitoring_scenario_frame(
            observed,
            scenario_id="base",
        )
        scenario_gap = scenario.loc[scenario["date"] == gap_date].iloc[0]
        assert scenario_gap["cpi_imputed"]
        assert scenario_gap["core_cpi_imputed"]
        assert pd.notna(scenario_gap["estimate_value"])
        assert pd.notna(scenario_gap["core_cpi_estimate_value"])

        cache = build_macro_cache_frame(scenario)
        cache_gap = cache.loc[cache["date"] == gap_date].iloc[0]

        assert pd.isna(cache_gap["CPIAUCSL"])
        assert pd.isna(cache_gap["CPILFESL"])
        assert cache_gap["cpi_originally_missing"]
        assert cache_gap["core_cpi_originally_missing"]
        assert {
            "cpi_level",
            "core_cpi_level",
            "cpi_imputed",
            "core_cpi_imputed",
            "scenario_id",
            "estimate_value",
            "core_cpi_estimate_value",
        }.isdisjoint(cache.columns)

        cache.to_csv(cache_path / "fred_base_macro_max_history.csv", index=False)
        reloaded = load_cached_macro_data_for_mode("max_history")
        reloaded_gap = reloaded.loc[reloaded["date"] == gap_date].iloc[0]

        assert pd.isna(reloaded_gap["cpi_observed_level"])
        assert pd.isna(reloaded_gap["core_cpi_observed_level"])
        assert pd.isna(reloaded_gap["cpi_level"])
        assert pd.isna(reloaded_gap["core_cpi_level"])
        assert not reloaded_gap["cpi_imputed"]
        assert not reloaded_gap["core_cpi_imputed"]


def test_cached_macro_loader_uses_max_history_superset(monkeypatch: pytest.MonkeyPatch) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        dates = pd.date_range("1981-01-31", "2022-12-31", freq="ME")
        cache = pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": np.arange(len(dates), dtype=float) + 100,
                "TB3MS": 4.0,
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
        exact_dates = pd.date_range("1981-01-31", "1982-03-31", freq="ME")
        cached_imputation = [False] * len(exact_dates)
        cached_imputation[-2] = True
        exact = pd.DataFrame(
            {
                "date": exact_dates,
                "cpi_level": 100.0 + np.arange(len(exact_dates), dtype=float),
                "tbill_3m": np.linspace(4.0, 4.2, len(exact_dates)),
                "inflation_yoy": -999.0,
                "cpi_imputed": cached_imputation,
            }
        )
        exact.to_csv(cache_path / "fred_base_macro_live_dashboard.csv", index=False)

        out = load_cached_macro_data_for_mode("live_dashboard")

        assert (
            find_cached_macro_data_file("live_dashboard").name
            == "fred_base_macro_live_dashboard.csv"
        )
        assert out["cpi_imputed"].tolist() == [False, False, False]
        assert out["cpi_originally_missing"].tolist() == [False, True, False]
        assert pd.isna(out.loc[1, "cpi_level"])
        assert not out["inflation_yoy"].eq(-999.0).any()


def test_cached_macro_loader_rejects_exact_cache_without_required_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        exact = pd.DataFrame(
            {
                "date": pd.date_range("1982-01-31", periods=3, freq="ME"),
                "CPIAUCSL": [100.0, 101.0, 102.0],
                "TB3MS": 4.0,
            }
        )
        exact.to_csv(cache_path / "fred_base_macro_live_dashboard.csv", index=False)

        with pytest.raises(ValueError, match="required 12-month pre-sample warm-up"):
            load_cached_macro_data_for_mode("live_dashboard")


def test_cached_macro_loader_rejects_an_incomplete_warmup_calendar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        dates = pd.DatetimeIndex([pd.Timestamp("1981-01-31")]).append(
            pd.date_range("1982-01-31", periods=3, freq="ME")
        )
        incomplete = pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": 100.0 + np.arange(len(dates), dtype=float),
                "TB3MS": 4.0,
            }
        )
        incomplete.to_csv(
            cache_path / "fred_base_macro_live_dashboard.csv",
            index=False,
        )

        with pytest.raises(ValueError, match="12 consecutive pre-sample months"):
            load_cached_macro_data_for_mode("live_dashboard")


def test_legacy_processed_cache_without_missingness_provenance_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        ambiguous = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-31", periods=15, freq="ME"),
                "cpi_level": np.arange(15, dtype=float) + 100,
                "tbill_3m": 4.0,
            }
        )
        ambiguous.to_csv(cache_path / "fred_base_macro_max_history.csv", index=False)

        with pytest.raises(
            ValueError,
            match="lacks a trusted schema marker and missingness provenance",
        ):
            load_cached_macro_data_for_mode("max_history")


def test_versioned_normalized_cache_cannot_claim_raw_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        invalid = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-31", periods=15, freq="ME"),
                MACRO_CACHE_SCHEMA_COLUMN: MACRO_CACHE_SCHEMA_VERSION,
                "cpi_level": np.arange(15, dtype=float) + 100,
                "tbill_3m": 4.0,
                "cpi_imputed": False,
            }
        )
        invalid.to_csv(cache_path / "fred_base_macro_max_history.csv", index=False)

        with pytest.raises(ValueError, match="must use the raw-authority"):
            load_cached_macro_data_for_mode("max_history")


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

        result = load_macro_data_for_mode_with_status(
            "max_history",
            imputation_policy="ex_post_continuity",
        )

        assert result.data_source_used == "cached_fred"
        assert result.imputation_policy == "ex_post_continuity"
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
    assert len(calls) == len(BASE_FRED_SERIES)
    assert "secret-test-key" not in result.live_fetch_status
    assert "secret-test-key" not in capsys.readouterr().out
    assert {"date", "cpi_level", "tbill_3m", "inflation_yoy", "cpi_imputed"}.issubset(
        result.data.columns
    )
    assert {"core_cpi_yoy", "pce_yoy", "core_pce_yoy"}.issubset(result.data.columns)


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
    assert len(calls) == len(BASE_FRED_SERIES)


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


def test_api_loaded_data_applies_cpi_imputation_only_when_ex_post_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    observed = load_macro_data_for_mode_with_status("max_history")
    result = load_macro_data_for_mode_with_status(
        "max_history",
        imputation_policy="ex_post_continuity",
    )

    observed_gap = observed.data.iloc[6]
    assert observed_gap["cpi_originally_missing"]
    assert pd.isna(observed_gap["cpi_level"])
    assert not observed_gap["cpi_imputed"]
    assert result.data_source_used == "fred_api"
    assert result.data["cpi_imputed"].any()


def test_api_loaded_data_applies_alternative_measure_imputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "secret-test-key")

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        assert url == FRED_API_URL
        params = kwargs["params"]
        assert isinstance(params, dict)
        series_id = str(params["series_id"])
        missing_index = 6 if series_id == "CPILFESL" else None
        return _FakeResponse(
            payload={"observations": _fred_observations(series_id, missing_index=missing_index)}
        )

    monkeypatch.setattr("transitory_inflation.data.requests.get", fake_get)

    result = load_macro_data_for_mode_with_status(
        "max_history",
        imputation_policy="ex_post_continuity",
    )

    assert result.data_source_used == "fred_api"
    assert result.data["core_cpi_imputed"].any()
    assert not result.data["cpi_imputed"].any()


def test_cached_macro_loader_preserves_optional_raw_alternative_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _workspace_temp_dir() as cache_dir:
        cache_path = Path(cache_dir)
        monkeypatch.setattr("transitory_inflation.data.RAW_DATA_DIR", cache_path)
        dates = pd.date_range("2020-01-31", periods=15, freq="ME")
        cache = pd.DataFrame(
            {
                "date": dates,
                "CPIAUCSL": np.arange(15, dtype=float) + 100,
                "CPILFESL": np.arange(15, dtype=float) + 95,
                "PCEPI": np.arange(15, dtype=float) + 90,
                "PCEPILFE": np.arange(15, dtype=float) + 92,
                "TB3MS": 4.0,
            }
        )
        cache.to_csv(cache_path / "fred_base_macro_max_history.csv", index=False)

        out = load_cached_macro_data_for_mode("max_history")

        for measure in INFLATION_MEASURES.values():
            assert measure.level_col in out.columns
            assert measure.yoy_col in out.columns
            assert measure.imputed_col in out.columns


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
                "CPIAUCSL": np.arange(15, dtype=float) + 100,
                "TB3MS": 4.0,
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

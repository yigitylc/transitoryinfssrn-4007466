"""Phase 0 research contracts and their implementation gate.

Implemented contracts pass normally. Unresolved contracts use strict expected failures,
so an XPASS fails the suite until its marker is deliberately removed.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from transitory_inflation import data as data_mod
from transitory_inflation import features as features_mod
from transitory_inflation import market_data as market_data_mod
from transitory_inflation import market_linkage as market_linkage_mod
from transitory_inflation.benchmarks import BENCHMARK_MODELS, build_benchmark_forecasts
from transitory_inflation.config import SAMPLE_MODES
from transitory_inflation.validation import (
    add_forward_outcomes,
    add_outcome_labels,
    forward_outcome_summary_by_regime,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CONTRACT_PATH = FIXTURE_DIR / "paper_replication_contract_v1.json"

PENDING_B1_REPLICATION = pytest.mark.xfail(
    strict=True,
    reason=(
        "B1 replication: paper-inspired naming and a separate literal-lag "
        "reconstruction path are pending"
    ),
)
PENDING_B2_IMPUTATION = pytest.mark.xfail(
    strict=True,
    reason=(
        "B2 imputation: observed-only CPI must remain missing and invariant "
        "to future-neighbor values"
    ),
)
PENDING_H1_CACHE_PROVENANCE = pytest.mark.xfail(
    strict=True,
    reason=(
        "H1 cache provenance: reload must preserve original missingness and "
        "ex-post imputation lineage"
    ),
)
PENDING_H2_COMMON_SAMPLE = pytest.mark.xfail(
    strict=True,
    reason=(
        "H2 common sample: every benchmark model must use one universal "
        "origin set per horizon"
    ),
)
PENDING_H4_DENOMINATORS = pytest.mark.xfail(
    strict=True,
    reason=(
        "H4 denominators: validation rates must expose metric-specific counts "
        "and preserve the 29/30 evidence boundary"
    ),
)
PENDING_H10_OVERLAP_UNCERTAINTY = pytest.mark.xfail(
    strict=True,
    reason=(
        "H10 overlap uncertainty: overlapping horizons must suppress naive "
        "independent-observation uncertainty"
    ),
)


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _candidate_frame() -> pd.DataFrame:
    contract = _contract()
    candidate = contract["candidate_input"]
    path = Path(__file__).parents[1] / candidate["path"]
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["inflation_yoy"] = frame["cpi_level"].pct_change(12, fill_method=None) * 100
    frame["baseline"] = frame["inflation_yoy"].rolling(36, min_periods=36).mean()
    frame["epsilon"] = frame["inflation_yoy"] - frame["baseline"]
    for window in (4, 8, 12):
        frame[f"tinf_{window}m"] = (
            frame["epsilon"].shift(2).rolling(window, min_periods=window).mean()
        )
    frame["tbill_monthly_pct"] = frame["tbill_3m"] / 12
    return frame


def _benchmark_feature_frame(periods: int = 90) -> pd.DataFrame:
    months = np.arange(periods, dtype=float)
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2015-01-31", periods=periods, freq="ME"),
            "inflation_yoy": 2.0
            + 0.35 * np.sin(months / 4.0)
            + 0.20 * np.cos(months / 9.0)
            + 0.01 * months,
        }
    )
    return features_mod.add_transitory_inflation_features(raw, baseline_method="fed_target")


def test_frozen_candidate_fixture_is_415_rows_but_not_a_golden_replication() -> None:
    contract = _contract()
    candidate = contract["candidate_input"]
    path = Path(__file__).parents[1] / candidate["path"]

    assert hashlib.sha256(path.read_bytes()).hexdigest() == candidate["sha256"]

    frame = _candidate_frame()
    analysis = frame.loc[frame["date"].between("1987-01-31", "2021-07-31")].copy()
    required = ["inflation_yoy", "tinf_4m", "tinf_8m", "tinf_12m", "tbill_monthly_pct"]
    common = analysis.dropna(subset=required)

    assert len(frame) == 475
    assert len(common) == 415
    assert common["date"].min() == pd.Timestamp("1987-01-31")
    assert common["date"].max() == pd.Timestamp("2021-07-31")
    assert common["date"].is_unique
    assert common[required].notna().all().all()

    targets = contract["published_targets"]["table_1"]["variables"]
    assert round(common["inflation_yoy"].mean(), 3) != targets["inflation_yoy"]["mean"]
    assert round(common["tinf_4m"].std(), 3) != targets["tinf_4m"]["std"]
    assert contract["overall_status"] == "unverified_candidate"


@PENDING_B1_REPLICATION
def test_current_paper_surface_is_explicitly_paper_inspired() -> None:
    assert "paper_inspired_window" in SAMPLE_MODES
    assert "paper_replication" not in SAMPLE_MODES


@PENDING_B1_REPLICATION
def test_paper_reconstruction_uses_literal_lag_and_a_separate_feature_path() -> None:
    builder = getattr(features_mod, "add_paper_reconstruction_features", None)
    assert callable(builder), "B1 requires a separate paper reconstruction feature path"

    raw = pd.DataFrame(
        {
            "date": pd.date_range("1980-01-31", periods=72, freq="ME"),
            "inflation_yoy": np.arange(72, dtype=float),
        }
    )
    out = builder(raw)
    target = 60
    for window in (4, 8, 12):
        expected = out["epsilon"].iloc[target - window - 1 : target - 1].mean()
        assert out[f"tinf_{window}m"].iloc[target] == expected


def test_observed_only_cpi_is_invariant_to_the_future_neighbor() -> None:
    signature = inspect.signature(data_mod.build_base_frame)
    assert "imputation_policy" in signature.parameters

    dates = pd.date_range("2018-01-31", periods=30, freq="ME")
    levels = 100.0 + np.arange(30, dtype=float)
    gap_pos = 16
    levels[gap_pos] = np.nan
    base = pd.DataFrame({"date": dates, "CPIAUCSL": levels, "TB3MS": 1.0})
    perturbed = base.copy()
    perturbed.loc[gap_pos + 1, "CPIAUCSL"] *= 1.10

    first = data_mod.build_base_frame(base, imputation_policy="observed_only")
    changed = data_mod.build_base_frame(perturbed, imputation_policy="observed_only")
    through_gap = first["date"] <= dates[gap_pos]

    pd.testing.assert_series_equal(
        first.loc[through_gap, "cpi_level"].reset_index(drop=True),
        changed.loc[through_gap, "cpi_level"].reset_index(drop=True),
    )
    gap = first.loc[first["date"] == dates[gap_pos]].iloc[0]
    assert pd.isna(gap["cpi_level"])
    assert gap["cpi_originally_missing"]
    assert not gap["cpi_imputed"]


def test_monthly_macro_normalization_keeps_one_physical_duplicate_row() -> None:
    dates = pd.date_range("2022-01-31", periods=15, freq="ME")
    regular = pd.DataFrame(
        {
            "date": dates[:-1],
            "CPIAUCSL": 100.0 + np.arange(len(dates) - 1, dtype=float),
            "TB3MS": 1.0,
            "release_timestamp": (
                dates[:-1] + pd.offsets.Day(13) + pd.offsets.Hour(13)
            ).tz_localize("UTC"),
            "release_timestamp_provenance": (
                data_mod.RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
            ),
            "timing_status": data_mod.TIMING_STATUS_RELEASE_ALIGNED,
        }
    )
    duplicate_rows = pd.DataFrame(
        {
            "date": [dates[-1], dates[-1]],
            "CPIAUCSL": [114.0, np.nan],
            "TB3MS": [np.nan, 9.0],
            "release_timestamp": [
                pd.Timestamp("2023-04-13 13:00:00+00:00"),
                pd.NaT,
            ],
            "release_timestamp_provenance": [
                data_mod.RELEASE_TIMESTAMP_PROVENANCE_ACTUAL,
                pd.NA,
            ],
            "timing_status": [
                data_mod.TIMING_STATUS_RELEASE_ALIGNED,
                data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY,
            ],
        }
    )

    out = data_mod.build_base_frame(pd.concat([regular, duplicate_rows], ignore_index=True))
    selected = out.loc[out["date"] == dates[-1]].iloc[0]

    assert pd.isna(selected["cpi_level"])
    assert selected["tbill_3m"] == 9.0
    assert pd.isna(selected["release_timestamp"])
    assert selected["release_timing_status"] == data_mod.TIMING_STATUS_UNAVAILABLE
    assert selected["timing_status"] == data_mod.TIMING_STATUS_UNAVAILABLE


def test_cache_reload_preserves_original_missingness_and_ex_post_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    signature = inspect.signature(data_mod.load_cached_macro_data_for_mode)
    assert "imputation_policy" in signature.parameters

    dates = pd.date_range("2019-01-31", periods=18, freq="ME")
    levels = 100.0 + np.arange(18, dtype=float)
    gap_pos = 8
    levels[gap_pos] = np.nan
    cached = pd.DataFrame({"date": dates, "CPIAUCSL": levels, "TB3MS": 1.0})
    path = tmp_path / "raw_macro.csv"
    cached.to_csv(path, index=False)
    monkeypatch.setattr(data_mod, "find_cached_macro_data_file", lambda mode: path)

    out = data_mod.load_cached_macro_data_for_mode(
        "max_history",
        imputation_policy="ex_post_continuity",
    )
    gap = out.loc[out["date"] == dates[gap_pos]].iloc[0]
    assert gap["cpi_originally_missing"]
    assert gap["cpi_imputed"]
    assert gap["imputation_method"] == "log_linear_bridge"
    assert pd.notna(gap["imputation_available_at"])


@pytest.mark.parametrize("timezone_aware", [True, False])
def test_cached_release_timestamp_timezone_trust_survives_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timezone_aware: bool,
) -> None:
    dates = pd.date_range("2022-01-31", periods=18, freq="ME")
    release_clock = dates + pd.offsets.Day(13) + pd.offsets.Hour(13)
    release_strings = [
        timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        + ("+00:00" if timezone_aware else "")
        for timestamp in release_clock
    ]
    cached = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": 100.0 + np.arange(len(dates), dtype=float),
            "TB3MS": 1.0,
            "release_timestamp": release_strings,
            "release_timestamp_provenance": (
                data_mod.RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
            ),
            "timing_status": data_mod.TIMING_STATUS_RELEASE_ALIGNED,
        }
    )

    writable_cache = data_mod.build_macro_cache_frame(cached)
    assert bool(writable_cache["release_timestamp"].notna().all()) is timezone_aware

    path = tmp_path / "raw_macro_with_release_timing.csv"
    writable_cache.to_csv(path, index=False)
    monkeypatch.setattr(data_mod, "find_cached_macro_data_file", lambda mode: path)

    out = data_mod.load_cached_macro_data_for_mode("max_history")
    latest = out.iloc[-1]
    if timezone_aware:
        assert latest["release_timestamp"] == pd.Timestamp(release_strings[-1])
        assert latest["information_timestamp"] == pd.Timestamp(release_strings[-1])
        assert latest["timing_status"] == data_mod.TIMING_STATUS_RELEASE_ALIGNED
    else:
        assert pd.isna(latest["release_timestamp"])
        assert pd.isna(latest["information_timestamp"])
        assert latest["timing_status"] == data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY


@pytest.mark.parametrize(
    ("incoming_status", "expected_cache_status"),
    [
        (
            data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY,
            data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY,
        ),
        ("proxy", "proxy"),
        ("unknown", "unknown"),
        (data_mod.TIMING_STATUS_UNAVAILABLE, data_mod.TIMING_STATUS_UNAVAILABLE),
        (None, data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY),
    ],
)
def test_cache_round_trip_never_promotes_nonexact_release_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incoming_status: str | None,
    expected_cache_status: str,
) -> None:
    dates = pd.date_range("2022-01-31", periods=18, freq="ME")
    raw = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": 100.0 + np.arange(len(dates), dtype=float),
            "TB3MS": 1.0,
            "release_timestamp": (
                dates + pd.offsets.Day(13) + pd.offsets.Hour(13)
            ).tz_localize("UTC"),
            "release_timestamp_provenance": (
                data_mod.RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
            ),
            "timing_status": incoming_status,
        }
    )

    cache = data_mod.build_macro_cache_frame(raw)
    assert cache["release_timestamp"].isna().all()
    assert cache["release_timing_status"].eq(expected_cache_status).all()

    path = tmp_path / "raw_macro_with_nonexact_timing.csv"
    cache.to_csv(path, index=False)
    monkeypatch.setattr(data_mod, "find_cached_macro_data_file", lambda mode: path)

    latest = data_mod.load_cached_macro_data_for_mode("max_history").iloc[-1]
    assert pd.isna(latest["release_timestamp"])
    assert latest["release_timing_status"] == expected_cache_status
    assert pd.isna(latest["information_timestamp"])
    assert latest["inflation_yoy_timing_status"] == (
        data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY
    )
    assert latest["timing_status"] == data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY


def test_measure_timing_and_provenance_remain_independent_through_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    headline_releases = (
        dates + pd.offsets.Day(13) + pd.offsets.Hour(13)
    ).tz_localize("UTC")
    core_releases = (
        dates + pd.offsets.Day(15) + pd.offsets.Hour(14)
    ).tz_localize("UTC")
    pce_releases = (
        dates + pd.offsets.Day(20) + pd.offsets.Hour(15)
    ).tz_localize("UTC")
    raw = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": 100.0 + np.arange(len(dates), dtype=float),
            "CPILFESL": 105.0 + np.arange(len(dates), dtype=float),
            "PCEPI": 110.0 + np.arange(len(dates), dtype=float),
            "TB3MS": 1.0,
            "release_timestamp": headline_releases,
            "release_timestamp_provenance": (
                data_mod.RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
            ),
            "release_timing_status": data_mod.TIMING_STATUS_RELEASE_ALIGNED,
            "core_cpi_release_timestamp": core_releases,
            "core_cpi_release_timestamp_provenance": (
                data_mod.RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
            ),
            "core_cpi_release_timing_status": (
                data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY
            ),
            "pce_release_timestamp": pce_releases,
            "pce_release_timestamp_provenance": (
                data_mod.RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
            ),
            "pce_release_timing_status": data_mod.TIMING_STATUS_RELEASE_ALIGNED,
        }
    )

    cache = data_mod.build_macro_cache_frame(raw)
    assert cache["release_timestamp"].notna().all()
    assert cache["core_cpi_release_timestamp"].isna().all()
    assert cache["core_cpi_release_timing_status"].eq(
        data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY
    ).all()
    assert cache["pce_release_timestamp"].notna().all()

    path = tmp_path / "raw_macro_with_measure_timing.csv"
    cache.to_csv(path, index=False)
    monkeypatch.setattr(data_mod, "find_cached_macro_data_file", lambda mode: path)

    latest = data_mod.load_cached_macro_data_for_mode("max_history").iloc[-1]
    assert latest["inflation_yoy_information_timestamp"] == headline_releases[-1]
    assert latest["inflation_yoy_timing_status"] == (
        data_mod.TIMING_STATUS_RELEASE_ALIGNED
    )
    assert pd.isna(latest["core_cpi_yoy_information_timestamp"])
    assert latest["core_cpi_yoy_timing_status"] == (
        data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY
    )
    assert latest["pce_yoy_information_timestamp"] == pce_releases[-1]
    assert latest["pce_yoy_timing_status"] == data_mod.TIMING_STATUS_RELEASE_ALIGNED
    assert latest["information_timestamp"] == headline_releases[-1]


def test_nonexact_timing_survives_cache_features_and_market_linkage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2018-01-31", periods=60, freq="ME")
    raw = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": 100.0 * np.cumprod(1.0 + np.linspace(0.001, 0.004, 60)),
            "TB3MS": 1.0,
            "release_timestamp": (
                dates + pd.offsets.Day(13) + pd.offsets.Hour(13)
            ).tz_localize("UTC"),
            "release_timestamp_provenance": (
                data_mod.RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
            ),
            "release_timing_status": "proxy",
        }
    )
    cache = data_mod.build_macro_cache_frame(raw)
    assert cache["release_timing_status"].eq("proxy").all()
    assert cache["release_timestamp"].isna().all()

    path = tmp_path / "nonexact_macro_cache.csv"
    cache.to_csv(path, index=False)
    monkeypatch.setattr(data_mod, "find_cached_macro_data_file", lambda mode: path)
    cached_base = data_mod.load_cached_macro_data_for_mode("max_history")
    featured = features_mod.add_transitory_inflation_features(
        cached_base,
        baseline_method="fed_target",
    )

    market_dates = pd.date_range("2018-02-28", periods=62, freq="ME")
    market_raw = pd.DataFrame(
        {
            "date": market_dates,
            "yield_2y": np.linspace(1.0, 3.0, len(market_dates)),
            market_data_mod.MARKET_TIMESTAMP_COLUMN: (
                market_dates + pd.offsets.Hour(20)
            ).tz_localize("UTC"),
            market_data_mod.MARKET_TIMESTAMP_PROVENANCE_COLUMN: (
                market_data_mod.MARKET_TIMESTAMP_PROVENANCE_ACTUAL
            ),
            market_data_mod.MARKET_TIMESTAMP_STATUS_COLUMN: (
                market_data_mod.MARKET_TIMESTAMP_STATUS_EXACT
            ),
        }
    )
    market = market_data_mod.build_market_close_frame(market_raw)
    panel = market_linkage_mod.build_market_linkage_panel(
        featured,
        market,
        horizons=(1,),
    )
    latest = panel.dropna(subset=["tinf_12m", "yield_2y"]).iloc[-1]

    assert latest["release_timing_status"] == "proxy"
    assert latest["inflation_yoy_timing_status"] == (
        data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY
    )
    assert pd.isna(latest["information_timestamp"])
    assert latest["timing_status"] == data_mod.TIMING_STATUS_REFERENCE_MONTH_ONLY
    assert latest["yield_2y_origin_basis"] == (
        market_linkage_mod.MARKET_ORIGIN_CONSERVATIVE_PROXY
    )
    assert latest["yield_2y_timing_status"] == (
        market_linkage_mod.MARKET_TIMING_CONSERVATIVE_PROXY
    )


def test_reference_month_is_not_silently_used_as_information_date() -> None:
    dates = pd.date_range("2023-01-31", periods=15, freq="ME")
    release_dates = (
        dates + pd.offsets.Day(13) + pd.offsets.Hour(13)
    ).tz_localize("UTC")
    merged = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": 100.0 + np.arange(15, dtype=float),
            "TB3MS": 1.0,
            "release_timestamp": release_dates,
            "release_timestamp_provenance": (
                data_mod.RELEASE_TIMESTAMP_PROVENANCE_ACTUAL
            ),
            "timing_status": data_mod.TIMING_STATUS_RELEASE_ALIGNED,
        }
    )

    out = data_mod.build_base_frame(merged)
    required = {
        "reference_month",
        "release_timestamp",
        "information_timestamp",
        "information_timestamp_provenance",
        "vintage_timestamp",
        "retrieved_at",
        "timing_status",
    }
    assert required <= set(out.columns)
    row = out.iloc[-1]
    assert row["reference_month"] != row["information_timestamp"]
    assert row["information_timestamp"] >= row["release_timestamp"]


def test_perfect_forecast_and_actual_use_the_same_origin_baseline() -> None:
    periods = 50
    target_pos = 40
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2018-01-31", periods=periods, freq="ME"),
            "inflation_yoy": 3.0,
            "baseline": 2.0,
            "epsilon": 1.0,
            "tinf_4m": np.linspace(0.5, 1.5, periods),
            "tinf_8m": np.linspace(0.4, 1.4, periods),
            "tinf_12m": np.linspace(0.3, 1.3, periods),
            "tinf_term_structure": "mixed",
        }
    )
    frame.loc[target_pos + 1, "baseline"] = 3.0
    frame.loc[target_pos + 1, "epsilon"] = 0.0

    forecasts = build_benchmark_forecasts(
        frame,
        horizon=1,
        ar_min_observations=8,
        bucket_min_observations=1,
    )
    row = forecasts.loc[
        (forecasts["model"] == "no_change")
        & (forecasts["date"] == frame.loc[target_pos, "date"])
    ].iloc[0]

    assert row["forecast_cpi_yoy"] == row["actual_cpi_yoy"] == 3.0
    assert bool(row["forecast_persistent_high_inflation"])
    assert bool(row["actual_persistent_high_inflation"])


@PENDING_H2_COMMON_SAMPLE
def test_benchmark_forecasts_use_one_universal_origin_set_for_all_models() -> None:
    forecasts = build_benchmark_forecasts(
        _benchmark_feature_frame(90),
        horizon=3,
        ar_min_observations=8,
        bucket_min_observations=1,
    )
    dates_by_model = {
        model: set(group["date"])
        for model, group in forecasts.groupby("model", sort=False)
    }

    assert set(dates_by_model) == set(BENCHMARK_MODELS)
    assert len({frozenset(dates) for dates in dates_by_model.values()}) == 1
    assert {len(dates) for dates in dates_by_model.values()} == {48}


def test_validation_rates_expose_metric_specific_numerators_and_denominators() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=5, freq="ME"),
            "historical_regime": "neutral",
            "inflation_yoy": [4.0, 1.0, 2.0, 2.0, 2.0],
            "epsilon": [2.0, -1.0, 0.0, 0.0, 0.0],
            "tinf_4m": [2.0, -1.0, 0.0, 0.0, 0.0],
        }
    )
    labelled = add_forward_outcomes(frame, horizons=(1,))
    labelled = add_outcome_labels(labelled, horizons=(1,), epsilon_threshold_pp=0.50)
    row = forward_outcome_summary_by_regime(labelled, horizons=(1,)).iloc[0]

    assert row["count"] == 4
    assert row["baseline_normalization_hit_rate_numerator"] == 3
    assert row["baseline_normalization_hit_rate_n_applicable"] == 4
    assert row["partial_decay_50_hit_rate_numerator"] == 2
    assert row["partial_decay_50_hit_rate_n_applicable"] == 2
    assert row["positive_shock_resolution_rate_numerator"] == 1
    assert row["positive_shock_resolution_rate_n_applicable"] == 1
    assert row["positive_shock_persistent_rate_numerator"] == 0
    assert row["positive_shock_persistent_rate_n_applicable"] == 1


def test_metric_evidence_strength_preserves_the_29_30_boundary() -> None:
    observed: list[tuple[int, str, bool]] = []
    for applicable in (29, 30):
        periods = applicable + 1
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2000-01-31", periods=periods, freq="ME"),
                "historical_regime": "neutral",
                "inflation_yoy": 3.0,
                "epsilon": 1.0,
                "tinf_4m": 1.0,
            }
        )
        labelled = add_forward_outcomes(frame, horizons=(1,))
        labelled = add_outcome_labels(labelled, horizons=(1,), epsilon_threshold_pp=0.50)
        row = forward_outcome_summary_by_regime(labelled, horizons=(1,)).iloc[0]
        observed.append(
            (
                int(row["positive_shock_persistent_rate_n_applicable"]),
                str(row["positive_shock_persistent_rate_evidence_strength"]),
                bool(row["positive_shock_persistent_rate_weak_evidence"]),
            )
        )

    assert observed == [(29, "weak", True), (30, "descriptive", False)]


@PENDING_H10_OVERLAP_UNCERTAINTY
def test_overlapping_horizons_do_not_emit_naive_uncertainty() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-31", periods=9, freq="ME"),
            "historical_regime": "neutral",
            "inflation_yoy": np.linspace(2.0, 3.0, 9),
            "epsilon": np.linspace(0.5, 1.5, 9),
            "tinf_4m": np.linspace(0.5, 1.5, 9),
        }
    )
    labelled = add_forward_outcomes(frame, horizons=(3,))
    labelled = add_outcome_labels(labelled, horizons=(3,), epsilon_threshold_pp=0.50)
    row = forward_outcome_summary_by_regime(labelled, horizons=(3,)).iloc[0]

    assert bool(row["overlapping_outcomes"])
    assert row["non_overlapping_count"] == 2
    assert row["uncertainty_status"] == "unavailable"


def test_unrelated_phase0_findings_remain_strict_xfail_gates() -> None:
    expected_pending_tests = {
        "test_current_paper_surface_is_explicitly_paper_inspired",
        "test_paper_reconstruction_uses_literal_lag_and_a_separate_feature_path",
        "test_benchmark_forecasts_use_one_universal_origin_set_for_all_models",
        "test_overlapping_horizons_do_not_emit_naive_uncertainty",
    }

    for test_name in expected_pending_tests:
        test_function = globals()[test_name]
        xfail_marks = [
            mark
            for mark in getattr(test_function, "pytestmark", ())
            if mark.name == "xfail"
        ]
        assert len(xfail_marks) == 1, f"{test_name} must retain one xfail marker"
        assert xfail_marks[0].kwargs.get("strict") is True

from __future__ import annotations

import numpy as np
import pandas as pd

import transitory_inflation.robustness as robustness_mod
from transitory_inflation.robustness import (
    DEFAULT_ROBUSTNESS_HORIZONS,
    DEFAULT_ROBUSTNESS_INFLATION_MEASURES,
    DEFAULT_ROBUSTNESS_THRESHOLDS,
    build_robustness_scorecard,
    inflation_measure_availability,
    robustness_tables,
    tinf_regime_verdict,
)


def _raw_cpi_frame(periods: int = 180) -> pd.DataFrame:
    months = np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2010-01-31", periods=periods, freq="ME"),
            "inflation_yoy": 2.0
            + 0.45 * np.sin(months / 5.0)
            + 0.20 * np.cos(months / 11.0)
            + 0.006 * months,
        }
    )


def _raw_multi_measure_frame(periods: int = 180) -> pd.DataFrame:
    months = np.arange(periods, dtype=float)
    base = _raw_cpi_frame(periods)
    base["core_cpi_yoy"] = 2.0 + 0.30 * np.sin(months / 6.0) + 0.004 * months
    base["pce_yoy"] = 1.8 + 0.25 * np.cos(months / 7.0) + 0.003 * months
    base["core_pce_yoy"] = 1.9 + 0.20 * np.sin(months / 8.0) + 0.002 * months
    return base


def test_robustness_scorecard_contains_expected_horizons_thresholds_and_models() -> None:
    scorecard = build_robustness_scorecard(
        {"unit_sample": _raw_cpi_frame()},
        baseline_methods=("rolling_36_shifted",),
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    assert set(scorecard["horizon_months"]) == set(DEFAULT_ROBUSTNESS_HORIZONS)
    assert set(scorecard["threshold_pp"]) == set(DEFAULT_ROBUSTNESS_THRESHOLDS)
    assert {"no_change", "mean_reversion", "ar1", "tinf_regime_bucket"}.issubset(
        set(scorecard["model"])
    )
    assert {"rank_by_mae", "rank_by_rmse"}.issubset(scorecard.columns)
    assert set(scorecard["inflation_measure"]) == set(DEFAULT_ROBUSTNESS_INFLATION_MEASURES)


def test_robustness_scorecard_includes_requested_inflation_measure_labels() -> None:
    scorecard = build_robustness_scorecard(
        {"unit_sample": _raw_multi_measure_frame()},
        horizons=(3,),
        thresholds=(0.50,),
        baseline_methods=("rolling_36_shifted",),
        inflation_measures=("headline_cpi", "core_cpi", "pce", "core_pce"),
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    assert set(scorecard["inflation_measure"]) == {
        "headline_cpi",
        "core_cpi",
        "pce",
        "core_pce",
    }
    assert set(scorecard["inflation_measure_label"]) == {
        "Headline CPI",
        "Core CPI",
        "PCE",
        "Core PCE",
    }
    assert (
        set(scorecard.loc[scorecard["inflation_measure"] == "headline_cpi", "paper_exact"])
        == {True}
    )
    assert (
        set(scorecard.loc[scorecard["inflation_measure"] != "headline_cpi", "paper_exact"])
        == {False}
    )


def test_robustness_scorecard_recovers_from_stale_benchmark_signature(
    monkeypatch,
) -> None:
    def stale_benchmark_comparison_tables(
        df,
        horizon: int,
        threshold_pp: float = 0.50,
        ar_min_observations: int = 24,
        bucket_min_observations: int = 8,
    ):
        raise AssertionError("stale benchmark function should be reloaded before use")

    monkeypatch.setattr(
        robustness_mod.benchmark_mod,
        "benchmark_comparison_tables",
        stale_benchmark_comparison_tables,
    )

    scorecard = robustness_mod.build_robustness_scorecard(
        {"unit_sample": _raw_multi_measure_frame()},
        horizons=(3,),
        thresholds=(0.50,),
        baseline_methods=("rolling_36_shifted",),
        inflation_measures=("headline_cpi", "core_cpi"),
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    assert not scorecard.empty
    assert {"headline_cpi", "core_cpi"} == set(scorecard["inflation_measure"])


def test_inflation_measure_availability_discloses_missing_measures() -> None:
    availability = inflation_measure_availability(
        {"unit_sample": _raw_cpi_frame()},
        inflation_measures=("headline_cpi", "core_cpi"),
    )

    by_measure = availability.set_index("inflation_measure")
    assert bool(by_measure.loc["headline_cpi", "available"])
    assert not bool(by_measure.loc["core_cpi", "available"])
    assert by_measure.loc["core_cpi", "valid_observations"] == 0


def test_full_sample_is_labeled_ex_post_when_included() -> None:
    scorecard = build_robustness_scorecard(
        {"unit_sample": _raw_cpi_frame()},
        horizons=(3,),
        thresholds=(0.50,),
        baseline_methods=("full_sample",),
        inflation_measures=("headline_cpi",),
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    assert not scorecard.empty
    assert set(scorecard["baseline_method"]) == {"full_sample"}
    assert set(scorecard["baseline_live_safe"]) == {False}
    assert set(scorecard["baseline_label"]) == {"ex-post / paper-style only"}


def test_tinf_regime_verdict_has_benchmark_beat_flags() -> None:
    scorecard = build_robustness_scorecard(
        {"unit_sample": _raw_cpi_frame()},
        horizons=(3,),
        thresholds=(0.50,),
        baseline_methods=("rolling_36_shifted",),
        inflation_measures=("headline_cpi",),
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    verdict = tinf_regime_verdict(scorecard)

    assert not verdict.empty
    assert {
        "beats_no_change_mae",
        "beats_no_change_rmse",
        "beats_mean_reversion_mae",
        "beats_mean_reversion_rmse",
        "beats_ar1_mae",
        "beats_ar1_rmse",
    }.issubset(verdict.columns)


def test_robustness_tables_do_not_introduce_phase_four_market_columns() -> None:
    scorecard, verdict, win_rates = robustness_tables(
        {"unit_sample": _raw_multi_measure_frame()},
        horizons=(3,),
        thresholds=(0.50,),
        baseline_methods=("rolling_36_shifted", "full_sample"),
        inflation_measures=("headline_cpi", "core_cpi"),
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    expected_scorecard_columns = {
        "sample_mode",
        "inflation_measure",
        "inflation_measure_label",
        "fred_series_id",
        "paper_exact",
        "baseline_method",
        "baseline_live_safe",
        "baseline_label",
        "threshold_pp",
        "model",
        "horizon_months",
        "count",
        "mae",
        "rmse",
        "directional_accuracy",
        "classification_count",
        "hit_rate",
        "false_positive_rate",
        "false_negative_rate",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "mae_improvement_vs_no_change_pct",
        "rmse_improvement_vs_no_change_pct",
        "mae_improvement_vs_mean_reversion_pct",
        "rmse_improvement_vs_mean_reversion_pct",
        "rank_by_mae",
        "rank_by_rmse",
    }
    expected_verdict_columns = {
        "sample_mode",
        "inflation_measure",
        "inflation_measure_label",
        "fred_series_id",
        "paper_exact",
        "baseline_method",
        "baseline_live_safe",
        "baseline_label",
        "horizon_months",
        "threshold_pp",
        "count",
        "tinf_mae",
        "tinf_rmse",
        "tinf_directional_accuracy",
        "tinf_rank_by_mae",
        "tinf_rank_by_rmse",
        "tinf_best_by_mae",
        "tinf_best_by_rmse",
        "mae_improvement_vs_no_change_pct",
        "rmse_improvement_vs_no_change_pct",
        "mae_improvement_vs_mean_reversion_pct",
        "rmse_improvement_vs_mean_reversion_pct",
        "beats_no_change_mae",
        "beats_no_change_rmse",
        "beats_mean_reversion_mae",
        "beats_mean_reversion_rmse",
        "beats_ar1_mae",
        "beats_ar1_rmse",
    }
    expected_win_rate_columns = {
        "sample_mode",
        "inflation_measure",
        "inflation_measure_label",
        "fred_series_id",
        "paper_exact",
        "baseline_method",
        "baseline_live_safe",
        "baseline_label",
        "settings_count",
        "tinf_best_by_mae_rate",
        "tinf_best_by_rmse_rate",
        "beats_no_change_mae_rate",
        "beats_no_change_rmse_rate",
        "beats_mean_reversion_mae_rate",
        "beats_mean_reversion_rmse_rate",
        "beats_ar1_mae_rate",
        "beats_ar1_rmse_rate",
    }

    assert set(scorecard.columns) <= expected_scorecard_columns
    assert set(verdict.columns) <= expected_verdict_columns
    assert set(win_rates.columns) <= expected_win_rate_columns

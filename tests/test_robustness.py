from __future__ import annotations

import numpy as np
import pandas as pd

from transitory_inflation.benchmarks import BENCHMARK_MODELS
from transitory_inflation.robustness import (
    DEFAULT_ROBUSTNESS_HORIZONS,
    DEFAULT_ROBUSTNESS_INFLATION_MEASURES,
    DEFAULT_ROBUSTNESS_THRESHOLDS,
    ROBUSTNESS_COVERAGE_COLUMNS,
    UNSCORED_CELL_REASONS,
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
    assert {
        "no_change",
        "mean_reversion",
        "ar1",
        "unconditional_drift",
        "tinf_regime_bucket",
    }.issubset(set(scorecard["model"]))
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


def test_tinf_regime_verdict_has_lower_loss_flags_and_differentials() -> None:
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
        "lower_mae_than_no_change",
        "lower_rmse_than_no_change",
        "lower_mae_than_mean_reversion",
        "lower_rmse_than_mean_reversion",
        "lower_mae_than_ar1",
        "lower_rmse_than_ar1",
        "lower_mae_than_unconditional_drift",
        "lower_rmse_than_unconditional_drift",
        "mae_differential_vs_no_change_pp",
        "rmse_differential_vs_no_change_pp",
        "mae_differential_vs_mean_reversion_pp",
        "rmse_differential_vs_mean_reversion_pp",
        "mae_differential_vs_ar1_pp",
        "rmse_differential_vs_ar1_pp",
        "mae_differential_vs_unconditional_drift_pp",
        "rmse_differential_vs_unconditional_drift_pp",
    }.issubset(verdict.columns)
    # Neutral point-estimate language only: no "beats"/"wins" claim columns.
    assert not [column for column in verdict.columns if "beats" in column or "win" in column]


def test_robustness_tables_do_not_introduce_phase_four_market_columns() -> None:
    scorecard, verdict, lower_loss_rates, coverage = robustness_tables(
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
        "common_origin_n",
        "common_origin_start",
        "common_origin_end",
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
        "mae_improvement_vs_unconditional_drift_pct",
        "rmse_improvement_vs_unconditional_drift_pct",
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
        "common_origin_n",
        "common_origin_start",
        "common_origin_end",
        "tinf_mae",
        "tinf_rmse",
        "tinf_directional_accuracy",
        "tinf_rank_by_mae",
        "tinf_rank_by_rmse",
        "tinf_lowest_mae",
        "tinf_lowest_rmse",
        "mae_improvement_vs_no_change_pct",
        "rmse_improvement_vs_no_change_pct",
        "mae_improvement_vs_mean_reversion_pct",
        "rmse_improvement_vs_mean_reversion_pct",
        "mae_improvement_vs_unconditional_drift_pct",
        "rmse_improvement_vs_unconditional_drift_pct",
        "lower_mae_than_no_change",
        "lower_rmse_than_no_change",
        "lower_mae_than_mean_reversion",
        "lower_rmse_than_mean_reversion",
        "lower_mae_than_ar1",
        "lower_rmse_than_ar1",
        "lower_mae_than_unconditional_drift",
        "lower_rmse_than_unconditional_drift",
        "mae_differential_vs_no_change_pp",
        "rmse_differential_vs_no_change_pp",
        "mae_differential_vs_mean_reversion_pp",
        "rmse_differential_vs_mean_reversion_pp",
        "mae_differential_vs_ar1_pp",
        "rmse_differential_vs_ar1_pp",
        "mae_differential_vs_unconditional_drift_pp",
        "rmse_differential_vs_unconditional_drift_pp",
    }
    expected_lower_loss_rate_columns = {
        "sample_mode",
        "inflation_measure",
        "inflation_measure_label",
        "fred_series_id",
        "paper_exact",
        "baseline_method",
        "baseline_live_safe",
        "baseline_label",
        "settings_count",
        "tinf_lowest_mae_rate",
        "tinf_lowest_rmse_rate",
        "lower_mae_than_no_change_rate",
        "lower_rmse_than_no_change_rate",
        "lower_mae_than_mean_reversion_rate",
        "lower_rmse_than_mean_reversion_rate",
        "lower_mae_than_ar1_rate",
        "lower_rmse_than_ar1_rate",
        "lower_mae_than_unconditional_drift_rate",
        "lower_rmse_than_unconditional_drift_rate",
    }

    assert set(scorecard.columns) <= expected_scorecard_columns
    assert set(verdict.columns) <= expected_verdict_columns
    assert set(lower_loss_rates.columns) <= expected_lower_loss_rate_columns
    assert set(coverage.columns) == set(ROBUSTNESS_COVERAGE_COLUMNS)
    assert not [
        column
        for frame in (scorecard, verdict, lower_loss_rates, coverage)
        for column in frame.columns
        if "beats" in column or "win" in column
    ]


def test_robustness_rejects_authoritative_estimate_only_rows() -> None:
    raw = _raw_cpi_frame()
    raw["imputation_policy"] = "observed_only"
    raw["uses_estimated_input"] = False
    raw["estimated_input_months"] = [()] * len(raw)
    raw["signal_uses_imputed_input"] = False
    raw["signal_uses_missing_input"] = False
    raw["signal_observed_only_eligible"] = True
    contaminated_pos = 100
    raw.loc[contaminated_pos, "uses_estimated_input"] = True
    changed = raw.copy()
    changed.loc[contaminated_pos, "inflation_yoy"] += 100.0

    kwargs = {
        "horizons": (3,),
        "thresholds": (0.50,),
        "baseline_methods": ("full_sample",),
        "inflation_measures": ("headline_cpi",),
        "ar_min_observations": 8,
        "bucket_min_observations": 1,
    }
    base_scorecard = build_robustness_scorecard({"unit_sample": raw}, **kwargs)
    changed_scorecard = build_robustness_scorecard({"unit_sample": changed}, **kwargs)
    availability = inflation_measure_availability(
        {"unit_sample": raw},
        inflation_measures=("headline_cpi",),
    ).iloc[0]

    pd.testing.assert_frame_equal(base_scorecard, changed_scorecard)
    assert availability["valid_observations"] == len(raw) - 1


def test_robustness_coverage_discloses_unscored_cells_and_common_panels() -> None:
    """H2: absent grid cells are visible as an absence, not silently dropped."""

    raw = _raw_cpi_frame()
    scorecard, _, _, coverage = robustness_tables(
        {"unit_sample": raw},
        # 3M is scoreable on this fixture; the long horizon leaves no shared origin.
        horizons=(3, 120),
        thresholds=(0.50,),
        baseline_methods=("rolling_36_shifted",),
        inflation_measures=("headline_cpi",),
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    assert not coverage.empty
    assert set(coverage["model"]) == set(BENCHMARK_MODELS)
    # Coverage is recorded once per horizon, for every requested horizon,
    # including the ones that produced no scored row.
    assert set(coverage["horizon_months"]) == {3, 120}

    scored = coverage.loc[coverage["horizon_months"] == 3]
    assert scored["scored"].all()
    assert scored["unscored_reason"].isna().all()
    assert scored["common_origin_n"].gt(0).all()
    assert scored["common_origin_start"].notna().all()
    assert scored["common_origin_end"].notna().all()

    unscored = coverage.loc[coverage["horizon_months"] == 120]
    assert not unscored["scored"].any()
    assert unscored["unscored_reason"].isin(set(UNSCORED_CELL_REASONS)).all()
    assert unscored["unscored_detail"].map(bool).all()
    # The unscored horizon is genuinely absent from the scored tables.
    assert 120 not in set(scorecard["horizon_months"])


def test_robustness_scorecard_scores_every_model_on_one_common_panel() -> None:
    scorecard, verdict, _, _ = robustness_tables(
        {"unit_sample": _raw_cpi_frame()},
        horizons=(3, 6),
        thresholds=(0.50,),
        baseline_methods=("rolling_36_shifted",),
        inflation_measures=("headline_cpi",),
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    setting_cols = ["sample_mode", "inflation_measure", "baseline_method", "horizon_months"]
    for _, group in scorecard.groupby(setting_cols, sort=False):
        assert set(group["model"]) == set(BENCHMARK_MODELS)
        assert group["count"].nunique() == 1
        assert group["classification_count"].nunique() == 1
        assert group["common_origin_n"].nunique() == 1
        assert int(group["count"].iloc[0]) == int(group["common_origin_n"].iloc[0])
        # Ranks are over models scored on identical origins.
        assert sorted(group["rank_by_mae"].tolist()) == list(range(1, len(BENCHMARK_MODELS) + 1))

    assert verdict["common_origin_n"].gt(0).all()


def test_unscored_cell_keeps_native_coverage_when_the_panel_is_empty() -> None:
    """An empty common panel is disclosed with each model's native counts intact."""

    _, _, _, coverage = robustness_tables(
        {"unit_sample": _raw_cpi_frame()},
        # No origin survives to a 120-month outcome once the regime warm-up bites,
        # yet the simplest models still produce forecasts.
        horizons=(120,),
        thresholds=(0.50,),
        baseline_methods=("rolling_36_shifted",),
        inflation_measures=("headline_cpi",),
        ar_min_observations=8,
        bucket_min_observations=1,
    )

    assert not coverage["scored"].any()
    assert (coverage["unscored_reason"] == "empty_common_origin_panel").all()
    assert (
        coverage["unscored_detail"] == UNSCORED_CELL_REASONS["empty_common_origin_panel"]
    ).all()
    # The panel is empty, but the diagnostic still reports what each model had.
    assert coverage["common_origin_n"].eq(0).all()
    assert coverage["native_count"].max() > 0
    by_model = coverage.set_index("model")
    assert int(by_model.loc["no_change", "native_count"]) > 0
    assert int(by_model.loc["tinf_regime_bucket", "native_count"]) == 0
    # Nothing is claimed as shared, so every native origin sits outside the panel.
    assert (coverage["origins_outside_common_panel"] == coverage["native_count"]).all()

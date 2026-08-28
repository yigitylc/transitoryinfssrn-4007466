from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from transitory_inflation import benchmarks as benchmark_mod
from transitory_inflation.data import (
    HEADLINE_INFLATION_MEASURE,
    INFLATION_MEASURES,
    InflationMeasure,
)
from transitory_inflation.features import (
    BASELINE_META,
    add_transitory_inflation_features,
    observed_only_historical_eligibility,
)

DEFAULT_ROBUSTNESS_HORIZONS: tuple[int, ...] = (3, 6, 12, 24, 36)
DEFAULT_ROBUSTNESS_THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
DEFAULT_ROBUSTNESS_BASELINES: tuple[str, ...] = (
    "rolling_36_shifted",
    "expanding_shifted",
    "full_sample",
)
DEFAULT_ROBUSTNESS_INFLATION_MEASURES: tuple[str, ...] = (HEADLINE_INFLATION_MEASURE,)

TINF_VERDICT_BENCHMARKS: tuple[str, ...] = (
    "no_change",
    "mean_reversion",
    "ar1",
    "unconditional_drift",
)


def _baseline_label(baseline_method: str) -> str:
    if baseline_method == "full_sample":
        return "ex-post / paper-style only"
    meta = BASELINE_META.get(baseline_method)
    if meta is not None and meta.live_safe:
        return "live-safe"
    return "not live-safe"


def _baseline_live_safe(baseline_method: str) -> bool:
    meta = BASELINE_META.get(baseline_method)
    return bool(meta.live_safe) if meta is not None else False


def _nonempty_tuple(values, label: str) -> tuple:
    out = tuple(values)
    if not out:
        raise ValueError(f"At least one {label} is required")
    return out


def _inflation_measure_configs(values) -> tuple[InflationMeasure, ...]:
    measure_keys = tuple(str(value) for value in _nonempty_tuple(values, "inflation measure"))
    unknown = [key for key in measure_keys if key not in INFLATION_MEASURES]
    if unknown:
        raise ValueError(
            f"Unknown inflation measures: {unknown}. Expected one of {list(INFLATION_MEASURES)}"
        )
    return tuple(INFLATION_MEASURES[key] for key in measure_keys)


def _benchmark_comparison_tables(
    featured: pd.DataFrame,
    horizon: int,
    threshold: float,
    inflation_col: str,
    ar_min_observations: int,
    bucket_min_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the benchmark tables for one robustness cell."""

    return benchmark_mod.benchmark_comparison_tables(
        featured,
        horizon=horizon,
        threshold_pp=threshold,
        inflation_col=inflation_col,
        ar_min_observations=ar_min_observations,
        bucket_min_observations=bucket_min_observations,
    )


#: Why one robustness cell produced no scored row. Cells are disclosed rather
#: than dropped silently, so an absent setting is visible as an absence.
UNSCORED_CELL_REASONS: dict[str, str] = {
    "no_eligible_origins": (
        "No model could forecast an observed-only origin with an available outcome in this "
        "setting, so there is nothing to score."
    ),
    "empty_common_origin_panel": (
        "Models produced forecasts but shared no common origin, so headline metrics were "
        "refused rather than reported on unequal samples."
    ),
}


def inflation_measure_availability(
    raw_frames_by_sample_mode: Mapping[str, pd.DataFrame],
    inflation_measures: tuple[str, ...] = tuple(INFLATION_MEASURES),
) -> pd.DataFrame:
    """Return measure availability by loaded sample frame."""

    measure_configs = _inflation_measure_configs(inflation_measures)
    rows: list[dict[str, object]] = []
    for sample_mode, raw in raw_frames_by_sample_mode.items():
        historical_eligible = observed_only_historical_eligibility(raw)
        for measure in measure_configs:
            has_yoy = measure.yoy_col in raw.columns
            valid = (
                raw[measure.yoy_col].notna() & historical_eligible
                if has_yoy
                else pd.Series(dtype=bool)
            )
            if has_yoy and valid.any() and "date" in raw.columns:
                latest_date = pd.Timestamp(pd.to_datetime(raw.loc[valid, "date"]).max())
            else:
                latest_date = pd.NaT
            imputation_applied = (
                bool(raw[measure.imputed_col].fillna(False).astype(bool).any())
                if measure.imputed_col in raw.columns
                else False
            )
            rows.append(
                {
                    "sample_mode": sample_mode,
                    "inflation_measure": measure.key,
                    "inflation_measure_label": measure.label,
                    "fred_series_id": measure.series_id,
                    "paper_exact": measure.paper_exact,
                    "available": bool(has_yoy and valid.any()),
                    "latest_valid_yoy_date": latest_date,
                    "valid_observations": int(valid.sum()) if has_yoy else 0,
                    "imputation_applied": imputation_applied,
                }
            )
    return pd.DataFrame(rows)


ROBUSTNESS_COVERAGE_COLUMNS: tuple[str, ...] = (
    "sample_mode",
    "inflation_measure",
    "inflation_measure_label",
    "baseline_method",
    "baseline_label",
    "horizon_months",
    "model",
    "native_count",
    "native_start",
    "native_end",
    "common_origin_n",
    "common_origin_start",
    "common_origin_end",
    "origins_outside_common_panel",
    "common_share_of_native_origins",
    "scored",
    "unscored_reason",
    "unscored_detail",
    "native_exclusion_reason",
)


def _cell_coverage(
    coverage: pd.DataFrame,
    sample_mode: str,
    measure: InflationMeasure,
    baseline_method: str,
    horizon: int,
    unscored_reason: str,
) -> pd.DataFrame:
    """Attach setting identity to one cell's per-model coverage diagnostic."""

    if coverage.empty:
        out = pd.DataFrame(
            {
                "model": list(benchmark_mod.BENCHMARK_MODELS),
                "horizon_months": int(horizon),
                "native_count": 0,
                "native_start": pd.NaT,
                "native_end": pd.NaT,
                "common_origin_n": 0,
                "common_origin_start": pd.NaT,
                "common_origin_end": pd.NaT,
                "origins_outside_common_panel": 0,
                "common_share_of_native_origins": float("nan"),
                "native_exclusion_reason": [
                    benchmark_mod.NATIVE_EXCLUSION_REASONS.get(model, "")
                    for model in benchmark_mod.BENCHMARK_MODELS
                ],
            }
        )
    else:
        out = coverage.copy()
        out["horizon_months"] = int(horizon)

    out.insert(0, "sample_mode", sample_mode)
    out.insert(1, "inflation_measure", measure.key)
    out.insert(2, "inflation_measure_label", measure.label)
    out.insert(3, "baseline_method", baseline_method)
    out.insert(4, "baseline_label", _baseline_label(baseline_method))
    out["scored"] = not unscored_reason
    out["unscored_reason"] = unscored_reason or pd.NA
    out["unscored_detail"] = UNSCORED_CELL_REASONS.get(unscored_reason, pd.NA)
    return out.loc[:, list(ROBUSTNESS_COVERAGE_COLUMNS)]


def _run_robustness_grid(
    raw_frames_by_sample_mode: Mapping[str, pd.DataFrame],
    horizons: tuple[int, ...] = DEFAULT_ROBUSTNESS_HORIZONS,
    thresholds: tuple[float, ...] = DEFAULT_ROBUSTNESS_THRESHOLDS,
    baseline_methods: tuple[str, ...] = DEFAULT_ROBUSTNESS_BASELINES,
    inflation_measures: tuple[str, ...] = DEFAULT_ROBUSTNESS_INFLATION_MEASURES,
    ar_min_observations: int = 24,
    bucket_min_observations: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the grid once and return the scorecard plus its coverage disclosure.

    Coverage is recorded per sample/measure/baseline/horizon. Thresholds change
    only classification labels, never a forecast or a common-origin panel, so
    coverage does not vary across the threshold dimension.
    """

    sample_items = tuple(raw_frames_by_sample_mode.items())
    if not sample_items:
        raise ValueError("At least one sample mode frame is required")
    horizons = tuple(int(value) for value in _nonempty_tuple(horizons, "horizon"))
    thresholds = tuple(float(value) for value in _nonempty_tuple(thresholds, "threshold"))
    baseline_methods = tuple(str(value) for value in _nonempty_tuple(baseline_methods, "baseline"))
    measure_configs = _inflation_measure_configs(inflation_measures)

    rows: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    for sample_mode, raw in sample_items:
        if raw.empty:
            continue
        historical_eligible = observed_only_historical_eligibility(raw)
        for measure in measure_configs:
            if measure.yoy_col not in raw.columns or raw[measure.yoy_col].dropna().empty:
                continue
            for baseline_method in baseline_methods:
                historical_raw = raw.copy()
                historical_raw.loc[~historical_eligible, measure.yoy_col] = float("nan")
                featured = add_transitory_inflation_features(
                    historical_raw,
                    inflation_col=measure.yoy_col,
                    baseline_method=baseline_method,
                )
                for horizon in horizons:
                    for threshold in thresholds:
                        unscored_reason = ""
                        coverage = pd.DataFrame()
                        try:
                            _, metrics, _, _, coverage = _benchmark_comparison_tables(
                                featured,
                                horizon=horizon,
                                threshold=threshold,
                                inflation_col=measure.yoy_col,
                                ar_min_observations=ar_min_observations,
                                bucket_min_observations=bucket_min_observations,
                            )
                        except benchmark_mod.EmptyCommonOriginPanelError as exc:
                            # Disclosed as an unscored cell rather than dropped
                            # silently or allowed to abort the whole grid. The
                            # error carries the native coverage, so the reason
                            # keeps its per-model detail.
                            metrics = pd.DataFrame()
                            coverage = exc.coverage
                            unscored_reason = "empty_common_origin_panel"
                        if metrics.empty and not unscored_reason:
                            unscored_reason = "no_eligible_origins"
                        if threshold == thresholds[0]:
                            coverage_rows.append(
                                _cell_coverage(
                                    coverage,
                                    sample_mode=sample_mode,
                                    measure=measure,
                                    baseline_method=baseline_method,
                                    horizon=horizon,
                                    unscored_reason=unscored_reason,
                                )
                            )
                        if metrics.empty:
                            continue
                        current = metrics.copy()
                        current.insert(0, "sample_mode", sample_mode)
                        current.insert(1, "inflation_measure", measure.key)
                        current.insert(2, "inflation_measure_label", measure.label)
                        current.insert(3, "fred_series_id", measure.series_id)
                        current.insert(4, "paper_exact", measure.paper_exact)
                        current.insert(5, "baseline_method", baseline_method)
                        current.insert(6, "baseline_live_safe", _baseline_live_safe(baseline_method))
                        current.insert(7, "baseline_label", _baseline_label(baseline_method))
                        current.insert(8, "threshold_pp", threshold)
                        rows.append(current)

    coverage_frame = (
        pd.concat(coverage_rows, ignore_index=True)
        if coverage_rows
        else pd.DataFrame(columns=list(ROBUSTNESS_COVERAGE_COLUMNS))
    )
    if not rows:
        return pd.DataFrame(), coverage_frame

    scorecard = pd.concat(rows, ignore_index=True)
    setting_cols = [
        "sample_mode",
        "inflation_measure",
        "baseline_method",
        "horizon_months",
        "threshold_pp",
    ]
    # Ranks are point-estimate orderings on one common-origin panel per setting,
    # not significance statements.
    scorecard["rank_by_mae"] = (
        scorecard.groupby(setting_cols)["mae"].rank(method="min", ascending=True).astype("Int64")
    )
    scorecard["rank_by_rmse"] = (
        scorecard.groupby(setting_cols)["rmse"].rank(method="min", ascending=True).astype("Int64")
    )
    return scorecard, coverage_frame


def build_robustness_scorecard(
    raw_frames_by_sample_mode: Mapping[str, pd.DataFrame],
    horizons: tuple[int, ...] = DEFAULT_ROBUSTNESS_HORIZONS,
    thresholds: tuple[float, ...] = DEFAULT_ROBUSTNESS_THRESHOLDS,
    baseline_methods: tuple[str, ...] = DEFAULT_ROBUSTNESS_BASELINES,
    inflation_measures: tuple[str, ...] = DEFAULT_ROBUSTNESS_INFLATION_MEASURES,
    ar_min_observations: int = 24,
    bucket_min_observations: int = 8,
) -> pd.DataFrame:
    """Run robustness benchmarks across reasonable settings and inflation measures.

    Every cell is scored on its own universal common-origin panel, so the ranks
    compare models on identical origins.
    """

    scorecard, _ = _run_robustness_grid(
        raw_frames_by_sample_mode,
        horizons=horizons,
        thresholds=thresholds,
        baseline_methods=baseline_methods,
        inflation_measures=inflation_measures,
        ar_min_observations=ar_min_observations,
        bucket_min_observations=bucket_min_observations,
    )
    return scorecard


def tinf_regime_verdict(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Return one row per setting comparing TINF/regime point loss to key baselines.

    Every comparison is a point-estimate ordering on that setting's common
    origins: ``lower_mae_than_<model>`` says TINF/regime's MAE point estimate is
    the smaller number, and ``mae_differential_vs_<model>_pp`` gives the signed
    size of that difference. No significance test is applied (see H10), so these
    are not statements that one model beats another.
    """

    if scorecard.empty:
        return pd.DataFrame()

    setting_cols = [
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
    ]
    rows: list[dict[str, object]] = []
    for keys, group in scorecard.groupby(setting_cols, dropna=False, sort=False):
        tinf = group.loc[group["model"] == "tinf_regime_bucket"]
        if tinf.empty:
            continue
        tinf_row = tinf.iloc[0]
        row = dict(zip(setting_cols, keys, strict=True))
        row.update(
            {
                "count": int(tinf_row["count"]),
                "common_origin_n": int(tinf_row.get("common_origin_n", tinf_row["count"])),
                "common_origin_start": tinf_row.get("common_origin_start", pd.NaT),
                "common_origin_end": tinf_row.get("common_origin_end", pd.NaT),
                "tinf_mae": float(tinf_row["mae"]),
                "tinf_rmse": float(tinf_row["rmse"]),
                "tinf_directional_accuracy": float(tinf_row["directional_accuracy"]),
                "tinf_rank_by_mae": int(tinf_row["rank_by_mae"]),
                "tinf_rank_by_rmse": int(tinf_row["rank_by_rmse"]),
                "tinf_lowest_mae": int(tinf_row["rank_by_mae"]) == 1,
                "tinf_lowest_rmse": int(tinf_row["rank_by_rmse"]) == 1,
                "mae_improvement_vs_no_change_pct": float(
                    tinf_row["mae_improvement_vs_no_change_pct"]
                ),
                "rmse_improvement_vs_no_change_pct": float(
                    tinf_row["rmse_improvement_vs_no_change_pct"]
                ),
                "mae_improvement_vs_mean_reversion_pct": float(
                    tinf_row["mae_improvement_vs_mean_reversion_pct"]
                ),
                "rmse_improvement_vs_mean_reversion_pct": float(
                    tinf_row["rmse_improvement_vs_mean_reversion_pct"]
                ),
                "mae_improvement_vs_unconditional_drift_pct": float(
                    tinf_row["mae_improvement_vs_unconditional_drift_pct"]
                ),
                "rmse_improvement_vs_unconditional_drift_pct": float(
                    tinf_row["rmse_improvement_vs_unconditional_drift_pct"]
                ),
            }
        )
        for benchmark in TINF_VERDICT_BENCHMARKS:
            other = group.loc[group["model"] == benchmark]
            if other.empty:
                row[f"lower_mae_than_{benchmark}"] = pd.NA
                row[f"lower_rmse_than_{benchmark}"] = pd.NA
                row[f"mae_differential_vs_{benchmark}_pp"] = float("nan")
                row[f"rmse_differential_vs_{benchmark}_pp"] = float("nan")
            else:
                other_row = other.iloc[0]
                row[f"lower_mae_than_{benchmark}"] = bool(tinf_row["mae"] < other_row["mae"])
                row[f"lower_rmse_than_{benchmark}"] = bool(tinf_row["rmse"] < other_row["rmse"])
                row[f"mae_differential_vs_{benchmark}_pp"] = float(
                    tinf_row["mae"] - other_row["mae"]
                )
                row[f"rmse_differential_vs_{benchmark}_pp"] = float(
                    tinf_row["rmse"] - other_row["rmse"]
                )
        rows.append(row)

    return pd.DataFrame(rows)


def robustness_lower_loss_rate_summary(verdict: pd.DataFrame) -> pd.DataFrame:
    """Aggregate how often TINF/regime posts the lower point loss across settings.

    Each rate is the share of settings in which TINF/regime's point estimate is
    the smaller number on that setting's common origins. It is a frequency of
    point-estimate orderings, not a win rate and not a significance result.
    """

    if verdict.empty:
        return pd.DataFrame()

    group_cols = [
        "sample_mode",
        "inflation_measure",
        "inflation_measure_label",
        "fred_series_id",
        "paper_exact",
        "baseline_method",
        "baseline_live_safe",
        "baseline_label",
    ]
    rate_cols = [
        "tinf_lowest_mae",
        "tinf_lowest_rmse",
        "lower_mae_than_no_change",
        "lower_rmse_than_no_change",
        "lower_mae_than_mean_reversion",
        "lower_rmse_than_mean_reversion",
        "lower_mae_than_ar1",
        "lower_rmse_than_ar1",
        "lower_mae_than_unconditional_drift",
        "lower_rmse_than_unconditional_drift",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in verdict.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys, strict=True))
        row["settings_count"] = int(len(group))
        for column in rate_cols:
            row[f"{column}_rate"] = float(group[column].dropna().astype(float).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_tables(
    raw_frames_by_sample_mode: Mapping[str, pd.DataFrame],
    horizons: tuple[int, ...] = DEFAULT_ROBUSTNESS_HORIZONS,
    thresholds: tuple[float, ...] = DEFAULT_ROBUSTNESS_THRESHOLDS,
    baseline_methods: tuple[str, ...] = DEFAULT_ROBUSTNESS_BASELINES,
    inflation_measures: tuple[str, ...] = DEFAULT_ROBUSTNESS_INFLATION_MEASURES,
    ar_min_observations: int = 24,
    bucket_min_observations: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return scorecard, verdict, lower-loss rates, and the coverage disclosure.

    The fourth table records every grid cell's per-model native coverage, its
    common-origin panel, and — for cells that produced no scored row — why, so
    an absent setting is visible rather than silently missing.
    """

    scorecard, coverage = _run_robustness_grid(
        raw_frames_by_sample_mode,
        horizons=horizons,
        thresholds=thresholds,
        baseline_methods=baseline_methods,
        inflation_measures=inflation_measures,
        ar_min_observations=ar_min_observations,
        bucket_min_observations=bucket_min_observations,
    )
    verdict = tinf_regime_verdict(scorecard)
    lower_loss_rates = robustness_lower_loss_rate_summary(verdict)
    return scorecard, verdict, lower_loss_rates, coverage

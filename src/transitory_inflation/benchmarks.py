from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from transitory_inflation import validation as validation_mod

DEFAULT_EPSILON_THRESHOLD_PP = validation_mod.DEFAULT_EPSILON_THRESHOLD_PP

BENCHMARK_MODELS: tuple[str, ...] = (
    "no_change",
    "cpi_persistence",
    "mean_reversion",
    "ar1",
    "unconditional_drift",
    "tinf_regime_bucket",
)

UNCONDITIONAL_DRIFT_SOURCE = "pooled_expanding_drift"
TINF_REGIME_SOURCE = "same_regime_mean"
TINF_FALLBACK_SOURCE = "pooled_expanding_drift_fallback"

FORECAST_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "forecast_source",
    "forecast_observation_count",
    "forecast_pooled_observation_count",
    "forecast_same_regime_observation_count",
)

RELATIVE_IMPROVEMENT_BASELINES: tuple[str, ...] = (
    "no_change",
    "mean_reversion",
    "unconditional_drift",
)

#: Keys identifying one forecast origin within a single-horizon benchmark frame.
COMMON_ORIGIN_KEYS: tuple[str, ...] = ("date", "horizon_months")

#: Why a model can lack a forecast at an otherwise scoreable origin. These are
#: structural warm-up requirements, not data defects, and they differ by model,
#: which is exactly why headline metrics need one common origin panel.
NATIVE_EXCLUSION_REASONS: dict[str, str] = {
    "no_change": "No warm-up requirement; available wherever CPI YoY is observed-only valid.",
    "cpi_persistence": "Requires CPI YoY at t-h, so the first h origins have no forecast.",
    "mean_reversion": "Requires a computed baseline, so baseline warm-up origins have no forecast.",
    "ar1": "Requires the expanding AR(1) minimum prior-observation count.",
    "unconditional_drift": (
        "Requires the minimum count of prior completed horizon-specific CPI YoY changes; "
        "it does not require a historical regime."
    ),
    "tinf_regime_bucket": (
        "Requires a walk-forward historical regime (expanding percentile warm-up), an origin at "
        "least h months into the sample, and a prior same-regime or unconditional bucket that "
        "meets the minimum observation count."
    ),
}


class EmptyCommonOriginPanelError(ValueError):
    """Raised when models produce forecasts but share no common forecast origin.

    This is deliberately fatal. Scoring would otherwise fall back silently to
    nothing while usable per-model forecasts existed, which is the failure mode
    the common-origin policy exists to prevent. A sample in which *no* model can
    forecast is a different situation and returns an empty frame instead.

    ``coverage`` carries the per-model native coverage diagnostic so a caller
    that turns this into a disclosed skip still knows what each model could
    have forecast on its own.
    """

    def __init__(self, message: str, coverage: pd.DataFrame | None = None) -> None:
        super().__init__(message)
        self.coverage = pd.DataFrame() if coverage is None else coverage


def _horizon(value: int) -> int:
    horizon = int(value)
    if horizon <= 0:
        raise ValueError(f"Horizon must be a positive month count: {value}")
    return horizon


def _suffix(horizon: int) -> str:
    return f"{int(horizon)}m"


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _nullable_bool(condition: pd.Series, valid: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=condition.index, dtype="boolean")
    result.loc[valid] = condition.loc[valid].astype(bool).to_numpy()
    return result


def _historical_validation_frame(
    df: pd.DataFrame,
    horizon: int,
    threshold_pp: float,
    inflation_col: str,
    baseline_col: str,
) -> pd.DataFrame:
    """Build the single-horizon validation frame used to score benchmarks."""

    return validation_mod.build_historical_validation_frame(
        df,
        forward_horizons=(horizon,),
        label_horizons=(horizon,),
        epsilon_threshold_pp=threshold_pp,
        fed_target_threshold_pp=threshold_pp,
        inflation_col=inflation_col,
        baseline_col=baseline_col,
    )


def _expanding_ar1_forecast(
    series: pd.Series,
    horizon: int,
    min_observations: int,
) -> pd.Series:
    """Forecast CPI YoY with an expanding AR(1) fit using data through t only."""

    horizon = _horizon(horizon)
    forecasts = pd.Series(np.nan, index=series.index, dtype=float)
    values = pd.to_numeric(series, errors="coerce")

    for end_pos in range(len(values)):
        history = values.iloc[: end_pos + 1].dropna()
        if len(history) < min_observations or pd.isna(values.iloc[end_pos]):
            continue

        y = history.iloc[1:].to_numpy(dtype=float)
        lagged = history.iloc[:-1].to_numpy(dtype=float)
        if len(y) < 2:
            continue

        x = np.column_stack([np.ones(len(lagged)), lagged])
        try:
            const, phi = np.linalg.lstsq(x, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        forecast = float(values.iloc[end_pos])
        for _ in range(horizon):
            forecast = float(const + phi * forecast)
        forecasts.iloc[end_pos] = forecast

    return forecasts


def _walk_forward_regime_bucket_forecast(
    df: pd.DataFrame,
    horizon: int,
    min_bucket_observations: int,
    inflation_col: str,
    regime_col: str,
) -> pd.Series:
    """Return the legacy TINF series while details retain source/count lineage."""

    return _walk_forward_regime_bucket_details(
        df,
        horizon=horizon,
        min_bucket_observations=min_bucket_observations,
        inflation_col=inflation_col,
        regime_col=regime_col,
    )["forecast_cpi_yoy"]


def _completed_change_history(
    df: pd.DataFrame,
    *,
    position: int,
    horizon: int,
    change_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return changes whose horizon outcomes are known at one forecast origin."""

    latest_known_origin = position - horizon
    if latest_known_origin < 0:
        empty = df.iloc[:0]
        return empty, empty[change_col].dropna()
    prior = df.iloc[: latest_known_origin + 1]
    return prior, prior[change_col].dropna()


def _empty_forecast_details(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_cpi_yoy": pd.Series(np.nan, index=index, dtype=float),
            "forecast_source": pd.Series(pd.NA, index=index, dtype="string"),
            "forecast_observation_count": pd.Series(pd.NA, index=index, dtype="Int64"),
            "forecast_pooled_observation_count": pd.Series(pd.NA, index=index, dtype="Int64"),
            "forecast_same_regime_observation_count": pd.Series(pd.NA, index=index, dtype="Int64"),
        },
        index=index,
    )


def _walk_forward_unconditional_drift_details(
    df: pd.DataFrame,
    horizon: int,
    min_observations: int,
    inflation_col: str,
) -> pd.DataFrame:
    """Forecast current CPI plus the mean of all prior completed h-step changes."""

    horizon = _horizon(horizon)
    suffix = _suffix(horizon)
    change_col = f"cpi_yoy_change_{suffix}"
    _require_columns(df, [inflation_col, change_col])

    details = _empty_forecast_details(df.index)
    forecast_col = details.columns.get_loc("forecast_cpi_yoy")
    source_col = details.columns.get_loc("forecast_source")
    observation_count_col = details.columns.get_loc("forecast_observation_count")
    pooled_count_col = details.columns.get_loc("forecast_pooled_observation_count")

    for pos in range(len(df)):
        current_inflation = df[inflation_col].iloc[pos]
        if pd.isna(current_inflation):
            continue

        _, prior_changes = _completed_change_history(
            df,
            position=pos,
            horizon=horizon,
            change_col=change_col,
        )
        pooled_count = len(prior_changes)
        details.iat[pos, pooled_count_col] = pooled_count
        if pooled_count < min_observations:
            continue

        details.iat[pos, forecast_col] = float(current_inflation + prior_changes.mean())
        details.iat[pos, source_col] = UNCONDITIONAL_DRIFT_SOURCE
        details.iat[pos, observation_count_col] = pooled_count

    return details


def _walk_forward_regime_bucket_details(
    df: pd.DataFrame,
    horizon: int,
    min_bucket_observations: int,
    inflation_col: str,
    regime_col: str,
) -> pd.DataFrame:
    """Forecast from a same-regime mean, falling back to the pooled drift mean."""

    horizon = _horizon(horizon)
    suffix = _suffix(horizon)
    change_col = f"cpi_yoy_change_{suffix}"
    _require_columns(df, [inflation_col, regime_col, change_col])

    details = _empty_forecast_details(df.index)
    forecast_col = details.columns.get_loc("forecast_cpi_yoy")
    source_col = details.columns.get_loc("forecast_source")
    observation_count_col = details.columns.get_loc("forecast_observation_count")
    pooled_count_col = details.columns.get_loc("forecast_pooled_observation_count")
    same_regime_count_col = details.columns.get_loc("forecast_same_regime_observation_count")

    for pos in range(len(df)):
        current_inflation = df[inflation_col].iloc[pos]
        current_regime = df[regime_col].iloc[pos]
        if pd.isna(current_inflation) or pd.isna(current_regime):
            continue

        # At month t, only rows ending no later than t have known t+h outcomes.
        prior, prior_changes = _completed_change_history(
            df,
            position=pos,
            horizon=horizon,
            change_col=change_col,
        )
        pooled_count = len(prior_changes)
        details.iat[pos, pooled_count_col] = pooled_count
        if prior_changes.empty:
            continue

        same_regime = prior.loc[prior[regime_col] == current_regime, change_col].dropna()
        same_regime_count = len(same_regime)
        details.iat[pos, same_regime_count_col] = same_regime_count
        if len(same_regime) >= min_bucket_observations:
            expected_change = float(same_regime.mean())
            source = TINF_REGIME_SOURCE
            observation_count = same_regime_count
        elif len(prior_changes) >= min_bucket_observations:
            expected_change = float(prior_changes.mean())
            source = TINF_FALLBACK_SOURCE
            observation_count = pooled_count
        else:
            continue

        details.iat[pos, forecast_col] = float(current_inflation + expected_change)
        details.iat[pos, source_col] = source
        details.iat[pos, observation_count_col] = observation_count

    return details


def build_native_benchmark_forecasts(
    df: pd.DataFrame,
    horizon: int,
    threshold_pp: float = DEFAULT_EPSILON_THRESHOLD_PP,
    inflation_col: str = "inflation_yoy",
    baseline_col: str = "baseline",
    regime_col: str = "historical_regime",
    ar_min_observations: int = 24,
    bucket_min_observations: int = 8,
) -> pd.DataFrame:
    """Build long-form no-lookahead benchmark forecasts on each model's own sample.

    Future CPI columns are created only inside the validation frame and are used
    for scoring. Forecast columns are computed from information available at
    month t, except the row is later dropped from scoring if t+h is unavailable.

    Every model keeps its **native** origin set here, so the models do not share
    a sample: warm-up requirements differ by model (see
    ``NATIVE_EXCLUSION_REASONS``). This output is a coverage diagnostic only.
    Cross-model metrics, ranks, and comparisons must use
    :func:`build_benchmark_forecasts`, which restricts it to one common panel.
    """

    horizon = _horizon(horizon)
    threshold_pp = float(threshold_pp)
    _require_columns(df, [inflation_col, baseline_col, "epsilon", "tinf_4m"])

    validation_df = _historical_validation_frame(
        df,
        horizon=horizon,
        threshold_pp=threshold_pp,
        inflation_col=inflation_col,
        baseline_col=baseline_col,
    )
    suffix = _suffix(horizon)
    actual_col = f"cpi_yoy_fwd_{suffix}"
    change_col = f"cpi_yoy_change_{suffix}"
    persistent_col = f"positive_shock_persistent_{suffix}"
    realized_gap_col = f"realized_gap_from_origin_baseline_{suffix}"
    _require_columns(
        validation_df,
        [
            actual_col,
            change_col,
            persistent_col,
            realized_gap_col,
            "positive_shock_eligible",
            regime_col,
        ],
    )

    current = validation_df[inflation_col]
    unconditional_drift = _walk_forward_unconditional_drift_details(
        validation_df,
        horizon=horizon,
        min_observations=bucket_min_observations,
        inflation_col=inflation_col,
    )
    tinf_regime_bucket = _walk_forward_regime_bucket_details(
        validation_df,
        horizon=horizon,
        min_bucket_observations=bucket_min_observations,
        inflation_col=inflation_col,
        regime_col=regime_col,
    )
    forecasts_by_model = {
        "no_change": current,
        "cpi_persistence": current + (current - current.shift(horizon)),
        "mean_reversion": validation_df[baseline_col],
        "ar1": _expanding_ar1_forecast(
            current,
            horizon=horizon,
            min_observations=ar_min_observations,
        ),
        "unconditional_drift": unconditional_drift["forecast_cpi_yoy"],
        "tinf_regime_bucket": tinf_regime_bucket["forecast_cpi_yoy"],
    }
    details_by_model = {
        "unconditional_drift": unconditional_drift,
        "tinf_regime_bucket": tinf_regime_bucket,
    }
    static_sources = {
        "no_change": "current_cpi_yoy",
        "cpi_persistence": "horizon_momentum",
        "mean_reversion": "origin_baseline",
        "ar1": "expanding_ar1",
    }

    rows: list[pd.DataFrame] = []
    base = pd.DataFrame(
        {
            "date": validation_df["date"] if "date" in validation_df.columns else validation_df.index,
            "horizon_months": horizon,
            "current_cpi_yoy": current,
            "baseline": validation_df[baseline_col],
            "epsilon": validation_df["epsilon"],
            "eligible_positive_shock": validation_df["positive_shock_eligible"],
            "historical_regime": validation_df[regime_col],
            "actual_cpi_yoy": validation_df[actual_col],
            "actual_cpi_yoy_change": validation_df[change_col],
            "actual_gap_from_origin_baseline": validation_df[realized_gap_col],
            "actual_persistent_high_inflation": validation_df[persistent_col],
        }
    )

    for model, forecast in forecasts_by_model.items():
        model_frame = base.copy()
        model_frame["model"] = model
        model_frame["forecast_cpi_yoy"] = forecast
        details = details_by_model.get(model)
        if details is None:
            model_frame["forecast_source"] = static_sources[model]
            for column in FORECAST_PROVENANCE_COLUMNS[1:]:
                model_frame[column] = pd.Series(pd.NA, index=model_frame.index, dtype="Int64")
        else:
            for column in FORECAST_PROVENANCE_COLUMNS:
                model_frame[column] = details[column]
        model_frame["forecast_cpi_yoy_change"] = forecast - current
        model_frame["forecast_error"] = forecast - validation_df[actual_col]
        forecast_gap = forecast - validation_df[baseline_col]
        classification_valid = (
            forecast.notna()
            & model_frame["eligible_positive_shock"].fillna(False).astype(bool)
        )
        model_frame["forecast_persistent_high_inflation"] = _nullable_bool(
            forecast_gap > threshold_pp,
            classification_valid,
        )
        valid = model_frame["actual_cpi_yoy"].notna() & model_frame["forecast_cpi_yoy"].notna()
        rows.append(model_frame.loc[valid])

    if not rows:
        return pd.DataFrame()

    output = pd.concat(rows, ignore_index=True)
    ordered = [
        "date",
        "horizon_months",
        "model",
        "current_cpi_yoy",
        "baseline",
        "epsilon",
        "eligible_positive_shock",
        "historical_regime",
        "forecast_cpi_yoy",
        *FORECAST_PROVENANCE_COLUMNS,
        "actual_cpi_yoy",
        "forecast_error",
        "forecast_cpi_yoy_change",
        "actual_cpi_yoy_change",
        "actual_gap_from_origin_baseline",
        "forecast_persistent_high_inflation",
        "actual_persistent_high_inflation",
    ]
    return output.loc[:, ordered]


def universal_common_origins(
    native_forecasts: pd.DataFrame,
    models: Iterable[str] = BENCHMARK_MODELS,
) -> pd.DataFrame:
    """Return the origins for which every registered model has a forecast.

    One universal panel per horizon: an origin qualifies only when all compared
    models can forecast it, so every headline metric is computed on identical
    origins and ranks stay mutually comparable.
    """

    expected = tuple(dict.fromkeys(str(model) for model in models))
    empty = pd.DataFrame(columns=list(COMMON_ORIGIN_KEYS))
    if native_forecasts.empty or not expected:
        return empty

    _require_columns(native_forecasts, [*COMMON_ORIGIN_KEYS, "model"])
    present = native_forecasts.loc[native_forecasts["model"].isin(expected)]
    if present.empty:
        return empty

    keys = list(COMMON_ORIGIN_KEYS)
    covered = present.groupby(keys, sort=True)["model"].nunique()
    qualifying = covered[covered == len(expected)]
    if qualifying.empty:
        return empty
    return qualifying.index.to_frame(index=False).loc[:, keys]


def _panel_bounds(panel: pd.DataFrame) -> tuple[int, object, object]:
    if panel.empty:
        return 0, pd.NaT, pd.NaT
    dates = pd.to_datetime(panel["date"], errors="coerce")
    return int(len(panel)), dates.min(), dates.max()


def _group_panel_identity(group: pd.DataFrame) -> tuple[int, object, object]:
    """Return the common-panel identity carried by one model's scored rows.

    Prefers the columns stamped by :func:`restrict_to_common_origins`, falls
    back to the group's own distinct origins, and degrades to an unknown panel
    on minimal frames that carry no origin keys at all.
    """

    if {"common_origin_n", "common_origin_start", "common_origin_end"}.issubset(group.columns):
        return (
            int(pd.to_numeric(group["common_origin_n"], errors="coerce").iloc[0]),
            group["common_origin_start"].iloc[0],
            group["common_origin_end"].iloc[0],
        )
    if "date" in group.columns:
        return _panel_bounds(group.loc[:, ["date"]].drop_duplicates())
    return 0, pd.NaT, pd.NaT


def benchmark_origin_coverage(
    native_forecasts: pd.DataFrame,
    models: Iterable[str] = BENCHMARK_MODELS,
) -> pd.DataFrame:
    """Return the per-model native coverage diagnostic behind the common panel.

    One row per model: how many origins it can forecast on its own, how many the
    common panel keeps, how many of its native origins the panel drops, and the
    structural reason its native sample differs. This table explains coverage;
    it must never be used to compare model accuracy.
    """

    expected = tuple(dict.fromkeys(str(model) for model in models))
    columns = [
        "model",
        "horizon_months",
        "native_count",
        "native_start",
        "native_end",
        "common_origin_n",
        "common_origin_start",
        "common_origin_end",
        "origins_outside_common_panel",
        "common_share_of_native_origins",
        "native_exclusion_reason",
    ]
    if native_forecasts.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(native_forecasts, [*COMMON_ORIGIN_KEYS, "model"])
    panel = universal_common_origins(native_forecasts, models=expected)
    common_n, common_start, common_end = _panel_bounds(panel)
    common_keys = set(map(tuple, panel.to_numpy())) if not panel.empty else set()
    horizons = pd.to_numeric(native_forecasts["horizon_months"], errors="coerce").dropna()
    horizon = int(horizons.iloc[0]) if not horizons.empty else 0

    rows: list[dict[str, object]] = []
    for model in expected:
        group = native_forecasts.loc[native_forecasts["model"] == model]
        dates = pd.to_datetime(group["date"], errors="coerce")
        native_count = int(len(group))
        if native_count and common_keys:
            outside = sum(
                1
                for key in map(tuple, group.loc[:, list(COMMON_ORIGIN_KEYS)].to_numpy())
                if key not in common_keys
            )
        else:
            outside = native_count
        rows.append(
            {
                "model": model,
                "horizon_months": horizon,
                "native_count": native_count,
                "native_start": dates.min() if native_count else pd.NaT,
                "native_end": dates.max() if native_count else pd.NaT,
                "common_origin_n": common_n,
                "common_origin_start": common_start,
                "common_origin_end": common_end,
                "origins_outside_common_panel": int(outside),
                "common_share_of_native_origins": (
                    float(common_n / native_count) if native_count else float("nan")
                ),
                "native_exclusion_reason": NATIVE_EXCLUSION_REASONS.get(model, ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def restrict_to_common_origins(
    native_forecasts: pd.DataFrame,
    horizon: int,
    models: Iterable[str] = BENCHMARK_MODELS,
) -> pd.DataFrame:
    """Restrict a native benchmark frame to its universal common-origin panel.

    Restriction is a strict subset of an already observed-only-gated frame: it
    only removes origins and never re-admits one, so observed-only trust,
    missing-CPI separation, and the frozen origin-baseline classification
    anchors are preserved unchanged.

    Raises :class:`EmptyCommonOriginPanelError` when models do have native
    forecasts but share no origin.
    """

    if native_forecasts.empty:
        return native_forecasts

    panel = universal_common_origins(native_forecasts, models=models)
    if panel.empty:
        coverage = benchmark_origin_coverage(native_forecasts, models=models)
        detail = ", ".join(
            f"{row['model']}={int(row['native_count'])}" for _, row in coverage.iterrows()
        )
        raise EmptyCommonOriginPanelError(
            f"No common forecast origin across all benchmark models at horizon "
            f"{_horizon(horizon)}m. Native origin counts: {detail or 'none'}. "
            "Headline metrics require one universal panel, so scoring is refused "
            "rather than silently reported on unequal samples.",
            coverage=coverage,
        )

    # An inner merge preserves the left frame's order, so the model-major row
    # layout the downstream summaries rely on survives the restriction.
    common = native_forecasts.merge(panel, on=list(COMMON_ORIGIN_KEYS), how="inner")
    common_n, common_start, common_end = _panel_bounds(panel)
    common["common_origin_n"] = common_n
    common["common_origin_start"] = common_start
    common["common_origin_end"] = common_end
    return common.reset_index(drop=True)


def build_benchmark_forecasts(
    df: pd.DataFrame,
    horizon: int,
    threshold_pp: float = DEFAULT_EPSILON_THRESHOLD_PP,
    inflation_col: str = "inflation_yoy",
    baseline_col: str = "baseline",
    regime_col: str = "historical_regime",
    ar_min_observations: int = 24,
    bucket_min_observations: int = 8,
    models: Iterable[str] = BENCHMARK_MODELS,
) -> pd.DataFrame:
    """Build benchmark forecasts restricted to one universal common-origin panel.

    Every registered model is scored on identical origins, so point-loss,
    directional, and classification denominators are equal across models and
    ranks are mutually comparable.

    Raises :class:`EmptyCommonOriginPanelError` when models do have native
    forecasts but share no origin. A sample where no model can forecast at all
    returns the empty native frame instead.
    """

    native = build_native_benchmark_forecasts(
        df,
        horizon=horizon,
        threshold_pp=threshold_pp,
        inflation_col=inflation_col,
        baseline_col=baseline_col,
        regime_col=regime_col,
        ar_min_observations=ar_min_observations,
        bucket_min_observations=bucket_min_observations,
    )
    return restrict_to_common_origins(native, horizon=horizon, models=models)


def benchmark_confusion_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Return persistent-high-inflation confusion counts by model."""

    if forecasts.empty:
        return pd.DataFrame(
            columns=("model", "true_positive", "false_positive", "true_negative", "false_negative")
        )

    _require_columns(
        forecasts,
        ["model", "actual_persistent_high_inflation", "forecast_persistent_high_inflation"],
    )
    rows: list[dict[str, object]] = []
    for model, group in forecasts.groupby("model", sort=False):
        classified = group.dropna(
            subset=["actual_persistent_high_inflation", "forecast_persistent_high_inflation"]
        )
        actual = classified["actual_persistent_high_inflation"].astype(bool)
        predicted = classified["forecast_persistent_high_inflation"].astype(bool)
        rows.append(
            {
                "model": model,
                "true_positive": int((predicted & actual).sum()),
                "false_positive": int((predicted & ~actual).sum()),
                "true_negative": int((~predicted & ~actual).sum()),
                "false_negative": int((~predicted & actual).sum()),
            }
        )
    return pd.DataFrame(rows)


def benchmark_metric_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Summarize forecast errors and classification quality by benchmark.

    Expects the common-origin frame from :func:`build_benchmark_forecasts`, so
    ``count`` and ``classification_count`` are identical across models and the
    values are point estimates on one shared panel. No significance test is
    applied, so a lower value means a lower point estimate, nothing more.
    """

    if forecasts.empty:
        return pd.DataFrame()

    _require_columns(
        forecasts,
        [
            "model",
            "actual_cpi_yoy",
            "forecast_cpi_yoy",
            "current_cpi_yoy",
            "forecast_cpi_yoy_change",
            "actual_cpi_yoy_change",
            "actual_persistent_high_inflation",
            "forecast_persistent_high_inflation",
        ],
    )
    confusion = benchmark_confusion_summary(forecasts).set_index("model")
    rows: list[dict[str, object]] = []

    for model, group in forecasts.groupby("model", sort=False):
        errors = group["forecast_cpi_yoy"] - group["actual_cpi_yoy"]
        direction = group[["forecast_cpi_yoy_change", "actual_cpi_yoy_change"]].dropna()
        direction_correct = np.sign(direction["forecast_cpi_yoy_change"]) == np.sign(
            direction["actual_cpi_yoy_change"]
        )

        counts = confusion.loc[model] if model in confusion.index else pd.Series(dtype=float)
        tp = int(counts.get("true_positive", 0))
        fp = int(counts.get("false_positive", 0))
        tn = int(counts.get("true_negative", 0))
        fn = int(counts.get("false_negative", 0))
        classification_count = tp + fp + tn + fn

        panel_n, panel_start, panel_end = _group_panel_identity(group)
        rows.append(
            {
                "model": model,
                "horizon_months": int(group["horizon_months"].iloc[0]),
                "count": int(len(group)),
                "common_origin_n": panel_n,
                "common_origin_start": panel_start,
                "common_origin_end": panel_end,
                "mae": float(errors.abs().mean()),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "directional_accuracy": float(direction_correct.mean())
                if len(direction_correct)
                else float("nan"),
                "classification_count": classification_count,
                "hit_rate": _safe_rate(tp + tn, classification_count),
                "false_positive_rate": _safe_rate(fp, fp + tn),
                "false_negative_rate": _safe_rate(fn, fn + tp),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
            }
        )

    return _add_relative_improvement_columns(pd.DataFrame(rows), forecasts)


def _paired_loss_differential(
    forecasts: pd.DataFrame,
    model: str,
    baseline_model: str,
) -> tuple[float, float]:
    """Return MAE/RMSE loss reduction versus a baseline, in percent.

    On the common-origin frame the paired join is the universal panel, so this
    is the panel loss differential. Positive means a lower point loss than the
    baseline; it is not a significance statement.
    """

    if "date" not in forecasts.columns:
        return float("nan"), float("nan")

    keys = ["date", "horizon_months"]
    left = forecasts.loc[forecasts["model"] == model, [*keys, "forecast_error"]].rename(
        columns={"forecast_error": "model_error"}
    )
    right = forecasts.loc[
        forecasts["model"] == baseline_model, [*keys, "forecast_error"]
    ].rename(columns={"forecast_error": "baseline_error"})
    common = left.merge(right, on=keys, how="inner")
    if common.empty:
        return float("nan"), float("nan")

    model_abs = common["model_error"].abs().mean()
    baseline_abs = common["baseline_error"].abs().mean()
    model_rmse = np.sqrt(np.mean(np.square(common["model_error"])))
    baseline_rmse = np.sqrt(np.mean(np.square(common["baseline_error"])))

    mae_improvement = (
        (baseline_abs - model_abs) / baseline_abs * 100 if baseline_abs else float("nan")
    )
    rmse_improvement = (
        (baseline_rmse - model_rmse) / baseline_rmse * 100 if baseline_rmse else float("nan")
    )
    return float(mae_improvement), float(rmse_improvement)


def _add_relative_improvement_columns(
    summary: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> pd.DataFrame:
    out = summary.copy()
    for baseline_model in RELATIVE_IMPROVEMENT_BASELINES:
        improvements = [
            _paired_loss_differential(forecasts, str(model), baseline_model)
            for model in out["model"]
        ]
        out[f"mae_improvement_vs_{baseline_model}_pct"] = [
            improvement[0] for improvement in improvements
        ]
        out[f"rmse_improvement_vs_{baseline_model}_pct"] = [
            improvement[1] for improvement in improvements
        ]
    return out


def benchmark_relative_improvement(summary: pd.DataFrame) -> pd.DataFrame:
    """Return long-form MAE/RMSE loss differentials versus naive baselines.

    Positive means a lower point loss than that baseline on the common panel.
    """

    if summary.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        for baseline_model in RELATIVE_IMPROVEMENT_BASELINES:
            rows.append(
                {
                    "model": row["model"],
                    "comparison_baseline": baseline_model,
                    "mae_improvement_pct": row.get(
                        f"mae_improvement_vs_{baseline_model}_pct", np.nan
                    ),
                    "rmse_improvement_pct": row.get(
                        f"rmse_improvement_vs_{baseline_model}_pct", np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def benchmark_comparison_tables(
    df: pd.DataFrame,
    horizon: int,
    threshold_pp: float = DEFAULT_EPSILON_THRESHOLD_PP,
    inflation_col: str = "inflation_yoy",
    ar_min_observations: int = 24,
    bucket_min_observations: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return common-panel forecasts, metrics, differentials, confusion, coverage.

    The first four tables are scored on one universal common-origin panel. The
    fifth is the per-model native coverage diagnostic that explains which
    origins each model could have forecast on its own and why they differ.
    """

    native = build_native_benchmark_forecasts(
        df,
        horizon=horizon,
        threshold_pp=threshold_pp,
        inflation_col=inflation_col,
        ar_min_observations=ar_min_observations,
        bucket_min_observations=bucket_min_observations,
    )
    coverage = benchmark_origin_coverage(native)
    forecasts = restrict_to_common_origins(native, horizon=horizon)
    metrics = benchmark_metric_summary(forecasts)
    improvements = benchmark_relative_improvement(metrics)
    confusion = benchmark_confusion_summary(forecasts)
    return forecasts, metrics, improvements, confusion, coverage

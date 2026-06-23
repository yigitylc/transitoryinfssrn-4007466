from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable

import numpy as np
import pandas as pd

from transitory_inflation import validation as validation_mod

DEFAULT_EPSILON_THRESHOLD_PP = validation_mod.DEFAULT_EPSILON_THRESHOLD_PP
BENCHMARK_VALIDATION_SIGNATURE_GUARD = True

BENCHMARK_MODELS: tuple[str, ...] = (
    "no_change",
    "cpi_persistence",
    "mean_reversion",
    "ar1",
    "tinf_regime_bucket",
)


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


def _historical_validation_frame(
    df: pd.DataFrame,
    horizon: int,
    threshold_pp: float,
    inflation_col: str,
) -> pd.DataFrame:
    """Call validation through the module so Streamlit stale imports can recover."""

    global validation_mod

    validation_func = validation_mod.build_historical_validation_frame
    if "inflation_col" not in inspect.signature(validation_func).parameters:
        validation_mod = importlib.reload(validation_mod)
        validation_func = validation_mod.build_historical_validation_frame

    return validation_func(
        df,
        forward_horizons=(horizon,),
        label_horizons=(horizon,),
        epsilon_threshold_pp=threshold_pp,
        fed_target_threshold_pp=threshold_pp,
        inflation_col=inflation_col,
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
    """Forecast with prior completed same-regime CPI YoY changes only."""

    horizon = _horizon(horizon)
    suffix = _suffix(horizon)
    change_col = f"cpi_yoy_change_{suffix}"
    _require_columns(df, [inflation_col, regime_col, change_col])

    forecasts = pd.Series(np.nan, index=df.index, dtype=float)
    for pos in range(len(df)):
        current_inflation = df[inflation_col].iloc[pos]
        current_regime = df[regime_col].iloc[pos]
        if pd.isna(current_inflation) or pd.isna(current_regime):
            continue

        # At month t, only rows ending no later than t have known t+h outcomes.
        latest_known_origin = pos - horizon
        if latest_known_origin < 0:
            continue
        prior = df.iloc[: latest_known_origin + 1]
        prior_changes = prior[change_col].dropna()
        if prior_changes.empty:
            continue

        same_regime = prior.loc[prior[regime_col] == current_regime, change_col].dropna()
        if len(same_regime) >= min_bucket_observations:
            expected_change = float(same_regime.mean())
        elif len(prior_changes) >= min_bucket_observations:
            expected_change = float(prior_changes.mean())
        else:
            continue

        forecasts.iloc[pos] = float(current_inflation + expected_change)

    return forecasts


def build_benchmark_forecasts(
    df: pd.DataFrame,
    horizon: int,
    threshold_pp: float = DEFAULT_EPSILON_THRESHOLD_PP,
    inflation_col: str = "inflation_yoy",
    baseline_col: str = "baseline",
    regime_col: str = "historical_regime",
    ar_min_observations: int = 24,
    bucket_min_observations: int = 8,
) -> pd.DataFrame:
    """Build long-form no-lookahead benchmark forecasts for one horizon.

    Future CPI columns are created only inside the validation frame and are used
    for scoring. Forecast columns are computed from information available at
    month t, except the row is later dropped from scoring if t+h is unavailable.
    """

    horizon = _horizon(horizon)
    threshold_pp = float(threshold_pp)
    _require_columns(df, [inflation_col, baseline_col, "epsilon", "tinf_4m"])

    validation_df = _historical_validation_frame(
        df,
        horizon=horizon,
        threshold_pp=threshold_pp,
        inflation_col=inflation_col,
    )
    suffix = _suffix(horizon)
    actual_col = f"cpi_yoy_fwd_{suffix}"
    change_col = f"cpi_yoy_change_{suffix}"
    persistent_col = f"positive_shock_persistent_{suffix}"
    _require_columns(validation_df, [actual_col, change_col, persistent_col, regime_col])

    current = validation_df[inflation_col]
    forecasts_by_model = {
        "no_change": current,
        "cpi_persistence": current + (current - current.shift(horizon)),
        "mean_reversion": validation_df[baseline_col],
        "ar1": _expanding_ar1_forecast(
            current,
            horizon=horizon,
            min_observations=ar_min_observations,
        ),
        "tinf_regime_bucket": _walk_forward_regime_bucket_forecast(
            validation_df,
            horizon=horizon,
            min_bucket_observations=bucket_min_observations,
            inflation_col=inflation_col,
            regime_col=regime_col,
        ),
    }

    rows: list[pd.DataFrame] = []
    base = pd.DataFrame(
        {
            "date": validation_df["date"] if "date" in validation_df.columns else validation_df.index,
            "horizon_months": horizon,
            "current_cpi_yoy": current,
            "baseline": validation_df[baseline_col],
            "epsilon": validation_df["epsilon"],
            "historical_regime": validation_df[regime_col],
            "actual_cpi_yoy": validation_df[actual_col],
            "actual_cpi_yoy_change": validation_df[change_col],
            "actual_persistent_high_inflation": validation_df[persistent_col],
        }
    )

    for model, forecast in forecasts_by_model.items():
        model_frame = base.copy()
        model_frame["model"] = model
        model_frame["forecast_cpi_yoy"] = forecast
        model_frame["forecast_cpi_yoy_change"] = forecast - current
        model_frame["forecast_error"] = forecast - validation_df[actual_col]
        model_frame["forecast_persistent_high_inflation"] = (
            forecast - validation_df[baseline_col]
        ) > threshold_pp
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
        "historical_regime",
        "forecast_cpi_yoy",
        "actual_cpi_yoy",
        "forecast_error",
        "forecast_cpi_yoy_change",
        "actual_cpi_yoy_change",
        "forecast_persistent_high_inflation",
        "actual_persistent_high_inflation",
    ]
    return output.loc[:, ordered]


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
    """Summarize forecast errors and classification quality by benchmark."""

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

        rows.append(
            {
                "model": model,
                "horizon_months": int(group["horizon_months"].iloc[0]),
                "count": int(len(group)),
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


def _common_sample_improvement(
    forecasts: pd.DataFrame,
    model: str,
    baseline_model: str,
) -> tuple[float, float]:
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
    for baseline_model in ("no_change", "mean_reversion"):
        improvements = [
            _common_sample_improvement(forecasts, str(model), baseline_model)
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
    """Return long-form MAE/RMSE improvement versus naive baselines."""

    if summary.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        for baseline_model in ("no_change", "mean_reversion"):
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return forecasts, metric summary, relative improvement, and confusion tables."""

    forecasts = build_benchmark_forecasts(
        df,
        horizon=horizon,
        threshold_pp=threshold_pp,
        inflation_col=inflation_col,
        ar_min_observations=ar_min_observations,
        bucket_min_observations=bucket_min_observations,
    )
    metrics = benchmark_metric_summary(forecasts)
    improvements = benchmark_relative_improvement(metrics)
    confusion = benchmark_confusion_summary(forecasts)
    return forecasts, metrics, improvements, confusion

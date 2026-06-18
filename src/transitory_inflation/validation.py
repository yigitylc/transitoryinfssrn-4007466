from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DEFAULT_FORWARD_HORIZONS: tuple[int, ...] = (3, 6, 12, 24, 36)
DEFAULT_LABEL_HORIZONS: tuple[int, ...] = (6, 12, 24, 36)
DEFAULT_EPSILON_THRESHOLD_PP = 0.50
DEFAULT_FED_TARGET_THRESHOLD_PP = 0.50
DEFAULT_FED_TARGET = 2.00
DEFAULT_REGIME_MIN_PRIOR_OBS = 36
DEFAULT_THRESHOLD_SENSITIVITY_LEVELS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

REGIME_ORDER: tuple[str, ...] = (
    "elevated rising",
    "elevated falling",
    "neutral",
    "disinflationary",
)
PRESSURE_ORDER: tuple[str, ...] = ("firming", "cooling", "mixed")
PRESSURE_LABELS: dict[str, str] = {
    "accelerating": "firming",
    "decelerating": "cooling",
    "mixed": "mixed",
}

TRANSITORY_SIGNAL_REGIMES: tuple[str, ...] = ("elevated falling",)
PERSISTENT_SIGNAL_REGIMES: tuple[str, ...] = ("elevated rising",)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon_months",
    "count",
    "avg_future_cpi_yoy_change",
    "median_future_cpi_yoy_change",
    "avg_future_epsilon_change",
    "baseline_normalization_hit_rate",
    "fed_target_normalization_hit_rate",
    "partial_decay_50_hit_rate",
    "partial_decay_80_hit_rate",
    "positive_shock_resolution_rate",
    "positive_shock_downside_overshoot_rate",
    "positive_shock_persistent_rate",
    "absolute_gap_persistent_rate",
    "persistent_rate",
    "reacceleration_rate",
)


def _horizons(values: Iterable[int]) -> tuple[int, ...]:
    horizons = tuple(int(value) for value in values)
    if not horizons:
        raise ValueError("At least one horizon is required")
    invalid = [value for value in horizons if value <= 0]
    if invalid:
        raise ValueError(f"Horizons must be positive month counts: {invalid}")
    return horizons


def _suffix(horizon: int) -> str:
    return f"{int(horizon)}m"


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _nullable_bool(condition: pd.Series, valid: pd.Series) -> pd.Series:
    result = pd.Series(pd.NA, index=condition.index, dtype="boolean")
    result.loc[valid] = condition.loc[valid].astype(bool).to_numpy()
    return result


def _hit_rate(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.astype(float).mean())


def _ordered_labels(values: Iterable[object], preferred_order: tuple[str, ...]) -> list[str]:
    labels = [str(value) for value in values if pd.notna(value)]
    preferred = [label for label in preferred_order if label in labels]
    extras = sorted(label for label in set(labels) if label not in preferred_order)
    return preferred + extras


def add_short_term_pressure_labels(
    df: pd.DataFrame,
    source_col: str = "tinf_term_structure",
    output_col: str = "historical_short_term_pressure",
) -> pd.DataFrame:
    """Add dashboard-facing pressure labels: firming, cooling, or mixed."""

    out = df.copy()
    if source_col in out.columns:
        out[output_col] = out[source_col].map(PRESSURE_LABELS).fillna("mixed")
    elif output_col not in out.columns:
        out[output_col] = "mixed"
    return out


def add_walk_forward_regime_labels(
    df: pd.DataFrame,
    tinf_col: str = "tinf_4m",
    output_col: str = "historical_regime",
    min_prior_observations: int = DEFAULT_REGIME_MIN_PRIOR_OBS,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.75,
) -> pd.DataFrame:
    """Add row-level historical regimes using expanding shifted TINF thresholds.

    The thresholds at month t are computed from valid TINF observations through
    t-1 only. This avoids the full-sample percentile lookahead used by an
    ex-post historical snapshot.
    """

    _require_columns(df, [tinf_col])
    out = df.copy()
    tinf = out[tinf_col]
    lower = tinf.expanding(min_periods=min_prior_observations).quantile(lower_quantile).shift(1)
    upper = tinf.expanding(min_periods=min_prior_observations).quantile(upper_quantile).shift(1)
    prev_tinf = tinf.shift(1)

    regime = pd.Series(pd.NA, index=out.index, dtype="string")
    valid = tinf.notna() & lower.notna() & upper.notna()
    elevated = valid & (tinf > upper) & prev_tinf.notna()
    disinflationary = valid & (tinf < lower)
    neutral = valid & ~elevated & ~disinflationary

    regime.loc[neutral] = "neutral"
    regime.loc[disinflationary] = "disinflationary"
    regime.loc[elevated & (tinf > prev_tinf)] = "elevated rising"
    regime.loc[elevated & (tinf <= prev_tinf)] = "elevated falling"

    out[output_col] = regime
    out[f"{output_col}_lower_threshold"] = lower
    out[f"{output_col}_upper_threshold"] = upper
    out[f"{output_col}_threshold_method"] = "expanding_shifted"
    return out


def add_forward_outcomes(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_FORWARD_HORIZONS,
    inflation_col: str = "inflation_yoy",
    epsilon_col: str = "epsilon",
    tinf_col: str = "tinf_4m",
    min_initial_gap_pp: float = DEFAULT_EPSILON_THRESHOLD_PP,
) -> pd.DataFrame:
    """Add future validation outcomes to a frame whose signal columns already exist.

    These columns must be used only to score historical outcomes. They are
    created after signal construction and should never feed back into baseline,
    epsilon, TINF, regime, or pressure features.
    """

    horizons = _horizons(horizons)
    _require_columns(df, [inflation_col, epsilon_col, tinf_col])
    out = df.copy()
    current_abs_gap = out[epsilon_col].abs()

    for horizon in horizons:
        suffix = _suffix(horizon)
        cpi_fwd = out[inflation_col].shift(-horizon)
        epsilon_fwd = out[epsilon_col].shift(-horizon)
        tinf_fwd = out[tinf_col].shift(-horizon)

        out[f"cpi_yoy_fwd_{suffix}"] = cpi_fwd
        out[f"cpi_yoy_change_{suffix}"] = cpi_fwd - out[inflation_col]
        out[f"epsilon_fwd_{suffix}"] = epsilon_fwd
        out[f"epsilon_change_{suffix}"] = epsilon_fwd - out[epsilon_col]
        out[f"tinf_4m_fwd_{suffix}"] = tinf_fwd
        out[f"tinf_4m_change_{suffix}"] = tinf_fwd - out[tinf_col]
        out[f"abs_epsilon_fwd_{suffix}"] = epsilon_fwd.abs()

        ratio = epsilon_fwd.abs().div(current_abs_gap)
        ratio = ratio.where(current_abs_gap >= min_initial_gap_pp)
        out[f"gap_decay_ratio_{suffix}"] = ratio.replace([np.inf, -np.inf], np.nan)

    return out


def add_outcome_labels(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
    epsilon_threshold_pp: float = DEFAULT_EPSILON_THRESHOLD_PP,
    fed_target_threshold_pp: float = DEFAULT_FED_TARGET_THRESHOLD_PP,
    fed_target: float = DEFAULT_FED_TARGET,
    reacceleration_threshold_pp: float | None = None,
    epsilon_col: str = "epsilon",
    inflation_col: str = "inflation_yoy",
) -> pd.DataFrame:
    """Add mechanical transitory/persistent outcome labels.

    A future month is baseline-normalized when abs(epsilon[t+h]) is within the
    threshold. Positive-shock labels ask whether a current above-baseline
    inflation shock faded. If a positive shock crosses below baseline, it is
    resolved with downside overshoot, not persistent high inflation. Absolute
    gap labels remain available as a secondary equilibrium/stability
    diagnostic. Decay labels require a meaningful current gap; rows where
    abs(epsilon[t]) is below the threshold are left nullable rather than being
    treated as failed decay events.
    Reacceleration is intentionally simple in Phase 1: CPI YoY rises by at
    least the configured threshold over the horizon.
    """

    horizons = _horizons(horizons)
    reacceleration_threshold_pp = (
        epsilon_threshold_pp
        if reacceleration_threshold_pp is None
        else float(reacceleration_threshold_pp)
    )
    _require_columns(df, [epsilon_col, inflation_col])
    out = df.copy()

    missing_outcomes = [
        f"epsilon_fwd_{_suffix(horizon)}"
        for horizon in horizons
        if f"epsilon_fwd_{_suffix(horizon)}" not in out.columns
    ]
    if missing_outcomes:
        out = add_forward_outcomes(
            out,
            horizons=horizons,
            inflation_col=inflation_col,
            epsilon_col=epsilon_col,
            min_initial_gap_pp=epsilon_threshold_pp,
        )

    current_abs_gap = out[epsilon_col].abs()
    meaningful_gap = current_abs_gap >= epsilon_threshold_pp
    positive_shock = out[epsilon_col] >= epsilon_threshold_pp

    for horizon in horizons:
        suffix = _suffix(horizon)
        epsilon_fwd_col = f"epsilon_fwd_{suffix}"
        cpi_fwd_col = f"cpi_yoy_fwd_{suffix}"
        cpi_change_col = f"cpi_yoy_change_{suffix}"

        future_valid = out[epsilon_fwd_col].notna() & out[cpi_fwd_col].notna()
        ratio = out[epsilon_fwd_col].abs().div(current_abs_gap)
        ratio = ratio.where(meaningful_gap)
        out[f"gap_decay_ratio_{suffix}"] = ratio.replace([np.inf, -np.inf], np.nan)

        baseline_condition = out[epsilon_fwd_col].abs() <= epsilon_threshold_pp
        fed_target_condition = (out[cpi_fwd_col] - fed_target).abs() <= fed_target_threshold_pp
        decay_50_condition = out[epsilon_fwd_col].abs() <= 0.50 * current_abs_gap
        decay_80_condition = out[epsilon_fwd_col].abs() <= 0.20 * current_abs_gap
        reaccelerated_condition = out[cpi_change_col] >= reacceleration_threshold_pp
        positive_shock_resolved_condition = out[epsilon_fwd_col] <= epsilon_threshold_pp
        downside_overshoot_condition = out[epsilon_fwd_col] <= -epsilon_threshold_pp
        positive_shock_persistent_condition = out[epsilon_fwd_col] > epsilon_threshold_pp

        baseline_normalized = _nullable_bool(baseline_condition, future_valid)
        partial_decay_50 = _nullable_bool(decay_50_condition, future_valid & meaningful_gap)

        out[f"baseline_normalized_{suffix}"] = baseline_normalized
        out[f"fed_target_normalized_{suffix}"] = _nullable_bool(
            fed_target_condition, future_valid
        )
        out[f"partial_decay_50_{suffix}"] = partial_decay_50
        out[f"partial_decay_80_{suffix}"] = _nullable_bool(
            decay_80_condition, future_valid & meaningful_gap
        )

        absolute_gap_persistent_valid = future_valid & partial_decay_50.notna()
        absolute_gap_persistent_condition = (
            ~baseline_normalized.fillna(False).astype(bool)
            & ~partial_decay_50.fillna(False).astype(bool)
        )
        out[f"absolute_gap_persistent_{suffix}"] = _nullable_bool(
            absolute_gap_persistent_condition,
            absolute_gap_persistent_valid,
        )
        out[f"positive_shock_resolved_{suffix}"] = _nullable_bool(
            positive_shock_resolved_condition,
            future_valid & positive_shock,
        )
        out[f"positive_shock_downside_overshoot_{suffix}"] = _nullable_bool(
            downside_overshoot_condition,
            future_valid & positive_shock,
        )
        out[f"positive_shock_persistent_{suffix}"] = _nullable_bool(
            positive_shock_persistent_condition,
            future_valid & positive_shock,
        )
        out[f"persistent_{suffix}"] = out[f"positive_shock_persistent_{suffix}"]
        out[f"reaccelerated_{suffix}"] = _nullable_bool(
            reaccelerated_condition, future_valid
        )

    return out


def build_historical_validation_frame(
    df: pd.DataFrame,
    forward_horizons: Iterable[int] = DEFAULT_FORWARD_HORIZONS,
    label_horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
    epsilon_threshold_pp: float = DEFAULT_EPSILON_THRESHOLD_PP,
    fed_target_threshold_pp: float = DEFAULT_FED_TARGET_THRESHOLD_PP,
    fed_target: float = DEFAULT_FED_TARGET,
    inflation_col: str = "inflation_yoy",
) -> pd.DataFrame:
    """Build the full validation frame from an already-computed signal frame."""

    out = add_short_term_pressure_labels(df)
    out = add_walk_forward_regime_labels(out)
    out = add_forward_outcomes(
        out,
        horizons=forward_horizons,
        inflation_col=inflation_col,
        min_initial_gap_pp=epsilon_threshold_pp,
    )
    return add_outcome_labels(
        out,
        horizons=label_horizons,
        epsilon_threshold_pp=epsilon_threshold_pp,
        fed_target_threshold_pp=fed_target_threshold_pp,
        fed_target=fed_target,
        inflation_col=inflation_col,
    )


def _summary_required_columns(suffix: str) -> list[str]:
    return [
        f"cpi_yoy_fwd_{suffix}",
        f"epsilon_fwd_{suffix}",
        f"cpi_yoy_change_{suffix}",
        f"epsilon_change_{suffix}",
        f"baseline_normalized_{suffix}",
        f"fed_target_normalized_{suffix}",
        f"partial_decay_50_{suffix}",
        f"partial_decay_80_{suffix}",
        f"positive_shock_resolved_{suffix}",
        f"positive_shock_downside_overshoot_{suffix}",
        f"positive_shock_persistent_{suffix}",
        f"absolute_gap_persistent_{suffix}",
        f"persistent_{suffix}",
        f"reaccelerated_{suffix}",
    ]


def _forward_outcome_summary_by_groups(
    df: pd.DataFrame,
    group_cols: Iterable[str],
    horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
) -> pd.DataFrame:
    """Summarize forward outcomes by one or more current-month grouping columns."""

    horizons = _horizons(horizons)
    group_cols = tuple(group_cols)
    if not group_cols:
        raise ValueError("At least one grouping column is required")
    _require_columns(df, group_cols)
    rows: list[dict[str, object]] = []

    for horizon in horizons:
        suffix = _suffix(horizon)
        required = _summary_required_columns(suffix)
        _require_columns(df, required)
        valid_groups = df.loc[:, list(group_cols)].notna().all(axis=1)
        valid = valid_groups & df[f"cpi_yoy_fwd_{suffix}"].notna()
        current = df.loc[valid].copy()

        for group_values, group in current.groupby(list(group_cols), dropna=True):
            if len(group_cols) == 1:
                group_values = (group_values,)
            row = dict(zip(group_cols, group_values, strict=True))
            row.update(
                {
                    "horizon_months": horizon,
                    "count": int(len(group)),
                    "avg_future_cpi_yoy_change": float(
                        group[f"cpi_yoy_change_{suffix}"].mean()
                    ),
                    "median_future_cpi_yoy_change": float(
                        group[f"cpi_yoy_change_{suffix}"].median()
                    ),
                    "avg_future_epsilon_change": float(
                        group[f"epsilon_change_{suffix}"].mean()
                    ),
                    "baseline_normalization_hit_rate": _hit_rate(
                        group[f"baseline_normalized_{suffix}"]
                    ),
                    "fed_target_normalization_hit_rate": _hit_rate(
                        group[f"fed_target_normalized_{suffix}"]
                    ),
                    "partial_decay_50_hit_rate": _hit_rate(
                        group[f"partial_decay_50_{suffix}"]
                    ),
                    "partial_decay_80_hit_rate": _hit_rate(
                        group[f"partial_decay_80_{suffix}"]
                    ),
                    "positive_shock_resolution_rate": _hit_rate(
                        group[f"positive_shock_resolved_{suffix}"]
                    ),
                    "positive_shock_downside_overshoot_rate": _hit_rate(
                        group[f"positive_shock_downside_overshoot_{suffix}"]
                    ),
                    "positive_shock_persistent_rate": _hit_rate(
                        group[f"positive_shock_persistent_{suffix}"]
                    ),
                    "absolute_gap_persistent_rate": _hit_rate(
                        group[f"absolute_gap_persistent_{suffix}"]
                    ),
                    "persistent_rate": _hit_rate(group[f"persistent_{suffix}"]),
                    "reacceleration_rate": _hit_rate(group[f"reaccelerated_{suffix}"]),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows, columns=(*group_cols, *SUMMARY_COLUMNS))


def forward_outcome_summary(
    df: pd.DataFrame,
    group_col: str,
    horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
) -> pd.DataFrame:
    """Summarize forward outcomes by a current-month grouping column."""

    return _forward_outcome_summary_by_groups(df, group_cols=(group_col,), horizons=horizons)


def forward_outcome_summary_by_regime(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
    regime_col: str = "historical_regime",
) -> pd.DataFrame:
    return forward_outcome_summary(df, group_col=regime_col, horizons=horizons)


def forward_outcome_summary_by_short_term_pressure(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
    pressure_col: str = "historical_short_term_pressure",
) -> pd.DataFrame:
    out = add_short_term_pressure_labels(df, output_col=pressure_col)
    return forward_outcome_summary(out, group_col=pressure_col, horizons=horizons)


def forward_outcome_summary_by_regime_and_pressure(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
    regime_col: str = "historical_regime",
    pressure_col: str = "historical_short_term_pressure",
) -> pd.DataFrame:
    """Summarize forward outcomes by historical regime crossed with pressure."""

    out = add_short_term_pressure_labels(df, output_col=pressure_col)
    return _forward_outcome_summary_by_groups(
        out,
        group_cols=(regime_col, pressure_col),
        horizons=horizons,
    )


def threshold_sensitivity_summary(
    df: pd.DataFrame,
    horizon: int,
    thresholds: Iterable[float] = DEFAULT_THRESHOLD_SENSITIVITY_LEVELS,
    fed_target: float = DEFAULT_FED_TARGET,
) -> pd.DataFrame:
    """Summarize outcome sensitivity across fixed mechanical thresholds.

    This is a robustness table, not a threshold-optimization routine. Each row
    rebuilds validation labels using the stated threshold and the selected
    horizon, while keeping signal construction unchanged.
    """

    horizon = _horizons((horizon,))[0]
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds:
        raise ValueError("At least one threshold is required")
    invalid = [value for value in thresholds if value <= 0]
    if invalid:
        raise ValueError(f"Thresholds must be positive percentage-point values: {invalid}")

    suffix = _suffix(horizon)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        validation_df = build_historical_validation_frame(
            df,
            forward_horizons=(horizon,),
            label_horizons=(horizon,),
            epsilon_threshold_pp=threshold,
            fed_target_threshold_pp=threshold,
            fed_target=fed_target,
        )
        _require_columns(validation_df, _summary_required_columns(suffix))
        valid = validation_df[f"cpi_yoy_fwd_{suffix}"].notna()
        current = validation_df.loc[valid]
        rows.append(
            {
                "threshold_pp": threshold,
                "horizon_months": horizon,
                "count": int(len(current)),
                "avg_future_cpi_yoy_change": float(
                    current[f"cpi_yoy_change_{suffix}"].mean()
                ),
                "median_future_cpi_yoy_change": float(
                    current[f"cpi_yoy_change_{suffix}"].median()
                ),
                "avg_future_epsilon_change": float(current[f"epsilon_change_{suffix}"].mean()),
                "baseline_normalization_hit_rate": _hit_rate(
                    current[f"baseline_normalized_{suffix}"]
                ),
                "fed_target_normalization_hit_rate": _hit_rate(
                    current[f"fed_target_normalized_{suffix}"]
                ),
                "partial_decay_50_hit_rate": _hit_rate(current[f"partial_decay_50_{suffix}"]),
                "partial_decay_80_hit_rate": _hit_rate(current[f"partial_decay_80_{suffix}"]),
                "positive_shock_resolution_rate": _hit_rate(
                    current[f"positive_shock_resolved_{suffix}"]
                ),
                "positive_shock_downside_overshoot_rate": _hit_rate(
                    current[f"positive_shock_downside_overshoot_{suffix}"]
                ),
                "positive_shock_persistent_rate": _hit_rate(
                    current[f"positive_shock_persistent_{suffix}"]
                ),
                "absolute_gap_persistent_rate": _hit_rate(
                    current[f"absolute_gap_persistent_{suffix}"]
                ),
                "persistent_rate": _hit_rate(current[f"persistent_{suffix}"]),
                "reacceleration_rate": _hit_rate(current[f"reaccelerated_{suffix}"]),
            }
        )

    return pd.DataFrame(rows, columns=("threshold_pp", *SUMMARY_COLUMNS))


def regime_transition_matrix(
    df: pd.DataFrame,
    horizon: int,
    regime_col: str = "historical_regime",
) -> pd.DataFrame:
    """Return transition probabilities from regime_t to regime_t+h."""

    horizon = _horizons((horizon,))[0]
    _require_columns(df, [regime_col])
    current = df[regime_col]
    future = current.shift(-horizon)
    valid = current.notna() & future.notna()
    if not valid.any():
        return pd.DataFrame()

    counts = pd.crosstab(current.loc[valid], future.loc[valid])
    row_order = _ordered_labels(counts.index, REGIME_ORDER)
    column_order = _ordered_labels(counts.columns, REGIME_ORDER)
    counts = counts.reindex(index=row_order, columns=column_order, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def validation_examples(
    df: pd.DataFrame,
    horizon: int,
    max_examples: int = 10,
    regime_col: str = "historical_regime",
) -> dict[str, pd.DataFrame]:
    """Return simple success/failure examples for Phase 1 validation.

    False-transitory examples use positive-shock persistence, not absolute
    distance to baseline. A high-inflation shock that crosses below baseline is
    a resolved shock with downside overshoot, even if it remains far from the
    baseline in absolute terms.
    """

    horizon = _horizons((horizon,))[0]
    suffix = _suffix(horizon)
    out = add_short_term_pressure_labels(df)
    required = [
        regime_col,
        f"baseline_normalized_{suffix}",
        f"partial_decay_50_{suffix}",
        f"positive_shock_resolved_{suffix}",
        f"positive_shock_downside_overshoot_{suffix}",
        f"positive_shock_persistent_{suffix}",
        f"persistent_{suffix}",
        f"cpi_yoy_fwd_{suffix}",
    ]
    _require_columns(out, required)

    transitory_signal = out[regime_col].isin(TRANSITORY_SIGNAL_REGIMES)
    persistent_signal = out[regime_col].isin(PERSISTENT_SIGNAL_REGIMES)
    positive_shock_resolved = out[f"positive_shock_resolved_{suffix}"].fillna(False).astype(bool)
    downside_overshoot = (
        out[f"positive_shock_downside_overshoot_{suffix}"].fillna(False).astype(bool)
    )
    positive_shock_persistent = out[f"positive_shock_persistent_{suffix}"].fillna(False).astype(bool)

    example_cols = [
        column
        for column in (
            "date",
            regime_col,
            "historical_short_term_pressure",
            "inflation_yoy",
            "epsilon",
            "tinf_4m",
            f"cpi_yoy_fwd_{suffix}",
            f"cpi_yoy_change_{suffix}",
            f"epsilon_fwd_{suffix}",
            f"epsilon_change_{suffix}",
            f"gap_decay_ratio_{suffix}",
            f"baseline_normalized_{suffix}",
            f"partial_decay_50_{suffix}",
            f"positive_shock_resolved_{suffix}",
            f"positive_shock_downside_overshoot_{suffix}",
            f"positive_shock_persistent_{suffix}",
            f"absolute_gap_persistent_{suffix}",
            f"persistent_{suffix}",
            f"reaccelerated_{suffix}",
        )
        if column in out.columns
    ]

    def take(mask: pd.Series) -> pd.DataFrame:
        examples = out.loc[mask & out[f"cpi_yoy_fwd_{suffix}"].notna(), example_cols].copy()
        if "date" in examples.columns:
            examples = examples.sort_values("date", ascending=False)
        return examples.head(max_examples).reset_index(drop=True)

    return {
        "false_transitory": take(transitory_signal & positive_shock_persistent),
        "false_persistent": take(persistent_signal & positive_shock_resolved),
        "successful_transitory": take(
            transitory_signal & positive_shock_resolved & ~downside_overshoot
        ),
        "successful_transitory_downside_overshoot": take(
            transitory_signal & downside_overshoot
        ),
        "successful_persistent": take(persistent_signal & positive_shock_persistent),
    }

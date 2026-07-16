from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .data import (
    INFORMATION_TIMESTAMP_PROVENANCE_RELEASES,
    INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
    TIMING_STATUS_REFERENCE_MONTH_ONLY,
    TIMING_STATUS_RELEASE_ALIGNED,
    TIMING_STATUS_UNAVAILABLE,
)
from .features import (
    _expanding_information_timestamp,
    _latest_timestamp,
    _trusted_information_timestamps,
)

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


def pressure_label(term_structure: object) -> str:
    """Map the internal TINF term-structure ordering to UI wording.

    accelerating -> firming, decelerating -> cooling, anything else -> mixed.
    """

    return PRESSURE_LABELS.get(str(term_structure), "mixed")


def _timing_status_for_value(
    values: pd.Series,
    exact_timestamps: pd.Series,
) -> pd.Series:
    status = pd.Series(
        TIMING_STATUS_UNAVAILABLE,
        index=values.index,
        dtype="string",
    )
    available = values.notna()
    status.loc[available] = TIMING_STATUS_REFERENCE_MONTH_ONLY
    status.loc[available & exact_timestamps.notna()] = TIMING_STATUS_RELEASE_ALIGNED
    return status


def _trusted_generic_information(df: pd.DataFrame) -> pd.Series:
    return _trusted_information_timestamps(
        df.get(
            "information_timestamp",
            pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]"),
        ),
        df.get(
            "information_timestamp_provenance",
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ),
        df.get(
            "timing_status",
            pd.Series(pd.NA, index=df.index, dtype="string"),
        ),
    )


def _consolidate_derived_timing(
    df: pd.DataFrame,
    *,
    values: pd.Series,
    derived_timestamp: pd.Series,
) -> None:
    """Make generic row timing include the newly derived label's dependencies."""

    incoming = _trusted_generic_information(df)
    available = values.notna()
    exact = available & incoming.notna() & derived_timestamp.notna()
    combined = _latest_timestamp(
        incoming,
        derived_timestamp,
    ).where(exact)

    if "information_timestamp" not in df.columns:
        df["information_timestamp"] = pd.Series(
            pd.NaT,
            index=df.index,
            dtype="datetime64[ns, UTC]",
        )
    if "information_timestamp_provenance" not in df.columns:
        df["information_timestamp_provenance"] = pd.Series(
            INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
            index=df.index,
            dtype="string",
        )
    if "timing_status" not in df.columns:
        df["timing_status"] = pd.Series(
            TIMING_STATUS_UNAVAILABLE,
            index=df.index,
            dtype="string",
        )

    # A missing optional label is not an input to later calculations. Preserve
    # the existing row trust in that case; only available derived labels can
    # extend or conservatively downgrade the generic dependency timestamp.
    nonexact_available = available & ~exact
    df.loc[nonexact_available, "information_timestamp"] = pd.NaT
    if exact.any():
        df.loc[exact, "information_timestamp"] = combined.loc[exact]
    df.loc[available, "information_timestamp_provenance"] = (
        INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED
    )
    df.loc[exact, "information_timestamp_provenance"] = (
        INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    )
    derived_status = _timing_status_for_value(values, combined)
    df.loc[available, "timing_status"] = derived_status.loc[available]

TRANSITORY_SIGNAL_REGIMES: tuple[str, ...] = ("elevated falling",)
PERSISTENT_SIGNAL_REGIMES: tuple[str, ...] = ("elevated rising",)

BINARY_RATE_SPECS: tuple[tuple[str, str], ...] = (
    ("baseline_normalization_hit_rate", "baseline_normalized"),
    ("fed_target_normalization_hit_rate", "fed_target_normalized"),
    ("partial_decay_50_hit_rate", "partial_decay_50"),
    ("partial_decay_80_hit_rate", "partial_decay_80"),
    ("positive_shock_resolution_rate", "positive_shock_resolved"),
    ("positive_shock_downside_overshoot_rate", "positive_shock_downside_overshoot"),
    ("positive_shock_persistent_rate", "positive_shock_persistent"),
    ("absolute_gap_persistent_rate", "absolute_gap_persistent"),
    ("persistent_rate", "persistent"),
    ("reacceleration_rate", "reaccelerated"),
)
BINARY_RATE_METADATA_SUFFIXES: tuple[str, ...] = (
    "numerator",
    "n_applicable",
    "evidence_strength",
    "weak_evidence",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon_months",
    "count",
    "avg_future_cpi_yoy_change",
    "median_future_cpi_yoy_change",
    "avg_future_epsilon_change",
    *(
        column
        for rate_name, _ in BINARY_RATE_SPECS
        for column in (
            rate_name,
            *(f"{rate_name}_{suffix}" for suffix in BINARY_RATE_METADATA_SUFFIXES),
        )
    ),
    "uses_imputed_input",
    "uses_missing_input",
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


def _lineage_flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].fillna(False).astype(bool)


def _mask_nonobserved_signal_inputs(
    df: pd.DataFrame,
    inflation_col: str,
) -> pd.DataFrame:
    """Blank lineage-contaminated signal values without removing calendar rows."""

    out = df.copy()
    if "signal_observed_only_eligible" in out.columns:
        eligible = out["signal_observed_only_eligible"].fillna(False).astype(bool)
    elif {
        "signal_uses_imputed_input",
        "signal_uses_missing_input",
    } & set(out.columns):
        eligible = ~(
            _lineage_flag(out, "signal_uses_imputed_input")
            | _lineage_flag(out, "signal_uses_missing_input")
        )
    else:
        return out

    invalid = ~eligible
    value_cols = [
        column
        for column in out.columns
        if column in {inflation_col, "baseline", "epsilon"}
        or (
            column.startswith("tinf_")
            and "_uses_" not in column
            and column != "tinf_term_structure"
        )
    ]
    if value_cols:
        out.loc[invalid, value_cols] = float("nan")
    if "tinf_term_structure" in out.columns:
        out.loc[invalid, "tinf_term_structure"] = pd.NA
    return out


def _evidence_strength(n_applicable: int) -> str:
    if n_applicable == 0:
        return "unavailable"
    if n_applicable < 10:
        return "sparse"
    if n_applicable < 30:
        return "weak"
    return "descriptive"


def _binary_rate_metrics(values: pd.Series, rate_name: str) -> dict[str, object]:
    clean = values.dropna()
    n_applicable = int(len(clean))
    numerator = int(clean.astype(bool).sum())
    rate = float(numerator / n_applicable) if n_applicable else float("nan")
    return {
        rate_name: rate,
        f"{rate_name}_numerator": numerator,
        f"{rate_name}_n_applicable": n_applicable,
        f"{rate_name}_evidence_strength": _evidence_strength(n_applicable),
        f"{rate_name}_weak_evidence": n_applicable < 30,
    }


def _binary_rate_summary(df: pd.DataFrame, suffix: str) -> dict[str, object]:
    summary: dict[str, object] = {}
    for rate_name, source_stem in BINARY_RATE_SPECS:
        summary.update(_binary_rate_metrics(df[f"{source_stem}_{suffix}"], rate_name))
    return summary


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
        source_available = out[source_col].notna()
        out[output_col] = out[source_col].map(PRESSURE_LABELS).where(source_available)
        out[output_col] = out[output_col].astype("string")
    elif output_col not in out.columns:
        out[output_col] = pd.Series(pd.NA, index=out.index, dtype="string")

    trusted_information = _trusted_generic_information(out).where(
        out[output_col].notna()
    )
    timestamp_col = f"{output_col}_information_timestamp"
    provenance_col = f"{output_col}_information_timestamp_provenance"
    status_col = f"{output_col}_timing_status"
    out[timestamp_col] = trusted_information
    out[provenance_col] = pd.Series(
        INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
        index=out.index,
        dtype="string",
    )
    out.loc[
        trusted_information.notna(),
        provenance_col,
    ] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    out[status_col] = _timing_status_for_value(
        out[output_col],
        trusted_information,
    )
    _consolidate_derived_timing(
        out,
        values=out[output_col],
        derived_timestamp=trusted_information,
    )
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

    information_col = f"{tinf_col}_information_timestamp"
    provenance_col = f"{tinf_col}_information_timestamp_provenance"
    status_col = f"{tinf_col}_timing_status"
    tinf_information = _trusted_information_timestamps(
        out.get(
            information_col,
            pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"),
        ),
        out.get(
            provenance_col,
            pd.Series(pd.NA, index=out.index, dtype="string"),
        ),
        out.get(
            status_col,
            pd.Series(pd.NA, index=out.index, dtype="string"),
        ),
    ).where(tinf.notna())
    prior_information, prior_information_exact = _expanding_information_timestamp(
        tinf,
        tinf_information,
        min_periods=min_prior_observations,
        shift=1,
    )
    regime_exact = (
        regime.notna()
        & tinf_information.notna()
        & prior_information_exact
        & prior_information.notna()
    )
    regime_information = _latest_timestamp(
        tinf_information,
        prior_information,
    ).where(regime_exact)
    regime_information_col = f"{output_col}_information_timestamp"
    regime_provenance_col = f"{output_col}_information_timestamp_provenance"
    regime_status_col = f"{output_col}_timing_status"
    out[regime_information_col] = regime_information
    out[regime_provenance_col] = pd.Series(
        INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
        index=out.index,
        dtype="string",
    )
    out.loc[
        regime_information.notna(),
        regime_provenance_col,
    ] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    out[regime_status_col] = _timing_status_for_value(
        regime,
        regime_information,
    )
    _consolidate_derived_timing(
        out,
        values=regime,
        derived_timestamp=regime_information,
    )
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
    origin_imputed = _lineage_flag(out, "signal_uses_imputed_input")
    origin_missing = _lineage_flag(out, "signal_uses_missing_input")

    for horizon in horizons:
        suffix = _suffix(horizon)
        cpi_fwd_imputed = _lineage_flag(
            out,
            f"{inflation_col}_uses_imputed_input",
        ).shift(-horizon, fill_value=False)
        epsilon_fwd_imputed = _lineage_flag(
            out,
            f"{epsilon_col}_uses_imputed_input",
        ).shift(-horizon, fill_value=False)
        tinf_fwd_imputed = _lineage_flag(
            out,
            f"{tinf_col}_uses_imputed_input",
        ).shift(-horizon, fill_value=False)
        cpi_fwd_missing = _lineage_flag(
            out,
            f"{inflation_col}_uses_missing_input",
        ).shift(-horizon, fill_value=False)
        epsilon_fwd_missing = _lineage_flag(
            out,
            f"{epsilon_col}_uses_missing_input",
        ).shift(-horizon, fill_value=False)
        tinf_fwd_missing = _lineage_flag(
            out,
            f"{tinf_col}_uses_missing_input",
        ).shift(-horizon, fill_value=False)

        out[f"cpi_yoy_fwd_{suffix}_uses_imputed_input"] = cpi_fwd_imputed
        out[f"epsilon_fwd_{suffix}_uses_imputed_input"] = epsilon_fwd_imputed
        out[f"tinf_4m_fwd_{suffix}_uses_imputed_input"] = tinf_fwd_imputed
        out[f"cpi_yoy_fwd_{suffix}_uses_missing_input"] = cpi_fwd_missing
        out[f"epsilon_fwd_{suffix}_uses_missing_input"] = epsilon_fwd_missing
        out[f"tinf_4m_fwd_{suffix}_uses_missing_input"] = tinf_fwd_missing

        out[f"outcome_{suffix}_uses_imputed_input"] = (
            origin_imputed | cpi_fwd_imputed | epsilon_fwd_imputed | tinf_fwd_imputed
        )
        out[f"outcome_{suffix}_uses_missing_input"] = (
            origin_missing | cpi_fwd_missing | epsilon_fwd_missing | tinf_fwd_missing
        )
        eligible = ~(
            out[f"outcome_{suffix}_uses_imputed_input"]
            | out[f"outcome_{suffix}_uses_missing_input"]
        )
        out[f"observed_only_eligible_{suffix}"] = eligible

        cpi_fwd = out[inflation_col].shift(-horizon).where(eligible)
        epsilon_fwd = out[epsilon_col].shift(-horizon).where(eligible)
        tinf_fwd = out[tinf_col].shift(-horizon).where(eligible)

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
    baseline_col: str = "baseline",
) -> pd.DataFrame:
    """Add mechanical transitory/persistent outcome labels.

    Positive-shock persistence freezes the origin baseline: eligibility uses
    inflation[t] - baseline[t] >= threshold and the realized label uses
    inflation[t+h] - baseline[t] > threshold. A future baseline remains only
    in explicitly ex-post normalization, absolute-gap, and decay diagnostics.
    If ``baseline_col`` is absent, the equivalent origin baseline is recovered
    as inflation - epsilon for compatibility with legacy validation inputs.
    Ineligible positive-shock origins receive nullable persistence labels.
    Decay labels require a meaningful current gap; rows where abs(epsilon[t])
    is below the threshold are left nullable rather than being treated as
    failed decay events.
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

    if baseline_col in out.columns:
        origin_baseline = pd.to_numeric(out[baseline_col], errors="coerce")
    else:
        origin_baseline = pd.to_numeric(out[inflation_col], errors="coerce") - pd.to_numeric(
            out[epsilon_col], errors="coerce"
        )
    origin_gap = pd.to_numeric(out[inflation_col], errors="coerce") - origin_baseline
    origin_valid = origin_gap.notna()
    positive_shock_eligible = _nullable_bool(
        origin_gap >= epsilon_threshold_pp,
        origin_valid,
    )
    out["persistence_origin_baseline"] = origin_baseline
    out["positive_shock_eligible"] = positive_shock_eligible

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
    positive_shock = positive_shock_eligible.fillna(False).astype(bool)

    for horizon in horizons:
        suffix = _suffix(horizon)
        epsilon_fwd_col = f"epsilon_fwd_{suffix}"
        cpi_fwd_col = f"cpi_yoy_fwd_{suffix}"
        cpi_change_col = f"cpi_yoy_change_{suffix}"

        future_valid = out[epsilon_fwd_col].notna() & out[cpi_fwd_col].notna()
        realized_origin_gap = out[cpi_fwd_col] - origin_baseline
        persistence_valid = realized_origin_gap.notna() & positive_shock
        out[f"realized_gap_from_origin_baseline_{suffix}"] = realized_origin_gap
        out[f"ex_post_gap_from_future_baseline_{suffix}"] = out[epsilon_fwd_col]
        ratio = out[epsilon_fwd_col].abs().div(current_abs_gap)
        ratio = ratio.where(meaningful_gap)
        out[f"gap_decay_ratio_{suffix}"] = ratio.replace([np.inf, -np.inf], np.nan)

        baseline_condition = out[epsilon_fwd_col].abs() <= epsilon_threshold_pp
        fed_target_condition = (out[cpi_fwd_col] - fed_target).abs() <= fed_target_threshold_pp
        decay_50_condition = out[epsilon_fwd_col].abs() <= 0.50 * current_abs_gap
        decay_80_condition = out[epsilon_fwd_col].abs() <= 0.20 * current_abs_gap
        reaccelerated_condition = out[cpi_change_col] >= reacceleration_threshold_pp
        positive_shock_resolved_condition = realized_origin_gap <= epsilon_threshold_pp
        downside_overshoot_condition = realized_origin_gap <= -epsilon_threshold_pp
        positive_shock_persistent_condition = realized_origin_gap > epsilon_threshold_pp

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
            persistence_valid,
        )
        out[f"positive_shock_downside_overshoot_{suffix}"] = _nullable_bool(
            downside_overshoot_condition,
            persistence_valid,
        )
        out[f"positive_shock_persistent_{suffix}"] = _nullable_bool(
            positive_shock_persistent_condition,
            persistence_valid,
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
    baseline_col: str = "baseline",
) -> pd.DataFrame:
    """Build the full validation frame from an already-computed signal frame."""

    out = _mask_nonobserved_signal_inputs(df, inflation_col=inflation_col)
    out = add_short_term_pressure_labels(out)
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
        baseline_col=baseline_col,
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
            if len(group_cols) == 1 and not isinstance(group_values, tuple):
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
                    **_binary_rate_summary(group, suffix),
                    "uses_imputed_input": bool(
                        _lineage_flag(group, f"outcome_{suffix}_uses_imputed_input").any()
                    ),
                    "uses_missing_input": bool(
                        _lineage_flag(group, f"outcome_{suffix}_uses_missing_input").any()
                    ),
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
                **_binary_rate_summary(current, suffix),
                "uses_imputed_input": bool(
                    _lineage_flag(
                        current,
                        f"outcome_{suffix}_uses_imputed_input",
                    ).any()
                ),
                "uses_missing_input": bool(
                    _lineage_flag(
                        current,
                        f"outcome_{suffix}_uses_missing_input",
                    ).any()
                ),
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

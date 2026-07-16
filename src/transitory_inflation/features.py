from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import (
    INFORMATION_TIMESTAMP_PROVENANCE_RELEASES,
    INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
    TIMING_STATUS_REFERENCE_MONTH_ONLY,
    TIMING_STATUS_RELEASE_ALIGNED,
    TIMING_STATUS_UNAVAILABLE,
    _has_explicit_timezone,
)

VALID_BASELINES = {
    "full_sample",
    "rolling_36_unshifted",
    "rolling_36_shifted",
    "expanding_shifted",
    "fed_target",
}


@dataclass(frozen=True)
class BaselineMeta:
    method: str
    description: str
    live_safe: bool
    warning: str | None = None


BASELINE_META: dict[str, BaselineMeta] = {
    "full_sample": BaselineMeta(
        method="full_sample",
        description="Full-sample historical mean. Useful for ex-post paper-style replication.",
        live_safe=False,
        warning="Uses future data when viewed historically. Do not use as a live signal.",
    ),
    "rolling_36_unshifted": BaselineMeta(
        method="rolling_36_unshifted",
        description="36-month rolling mean including the current month.",
        live_safe=False,
        warning="Includes current observation in its own baseline. Prefer shifted version for live signals.",
    ),
    "rolling_36_shifted": BaselineMeta(
        method="rolling_36_shifted",
        description="36-month rolling mean shifted by one month. Real-time safer baseline.",
        live_safe=True,
        warning=None,
    ),
    "expanding_shifted": BaselineMeta(
        method="expanding_shifted",
        description="Expanding historical mean shifted by one month.",
        live_safe=True,
        warning=None,
    ),
    "fed_target": BaselineMeta(
        method="fed_target",
        description="Fixed 2% inflation target baseline.",
        live_safe=True,
        warning=None,
    ),
}


def compute_baseline(
    inflation: pd.Series,
    method: str = "rolling_36_shifted",
    rolling_window: int = 36,
    expanding_min_periods: int = 120,
    fed_target: float = 2.0,
) -> pd.Series:
    """Compute mean-reversion inflation baseline."""

    if method not in VALID_BASELINES:
        raise ValueError(f"Unknown baseline method: {method}. Expected one of {sorted(VALID_BASELINES)}")

    if method == "full_sample":
        return pd.Series(inflation.mean(skipna=True), index=inflation.index, name="baseline")
    if method == "rolling_36_unshifted":
        return inflation.rolling(rolling_window, min_periods=rolling_window).mean().rename("baseline")
    if method == "rolling_36_shifted":
        return inflation.rolling(rolling_window, min_periods=rolling_window).mean().shift(1).rename("baseline")
    if method == "expanding_shifted":
        return inflation.expanding(min_periods=expanding_min_periods).mean().shift(1).rename("baseline")
    if method == "fed_target":
        return pd.Series(fed_target, index=inflation.index, name="baseline")

    raise AssertionError("Unreachable baseline branch")


def consecutive_true_count(flag: pd.Series) -> pd.Series:
    """Return run length of consecutive True values at each observation."""

    values = flag.fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(values), dtype=int)
    run = 0
    for i, value in enumerate(values):
        run = run + 1 if value else 0
        out[i] = run
    return pd.Series(out, index=flag.index, name="run_length_above")


def _lineage_flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].fillna(False).astype(bool)


def _baseline_lineage(
    flag: pd.Series,
    method: str,
    rolling_window: int = 36,
) -> pd.Series:
    """Propagate a CPI-input flag through the selected baseline dependency set."""

    numeric = flag.fillna(False).astype(int)
    if method == "full_sample":
        return pd.Series(bool(numeric.any()), index=flag.index, dtype=bool)
    if method == "rolling_36_unshifted":
        return numeric.rolling(rolling_window, min_periods=1).max().astype(bool)
    if method == "rolling_36_shifted":
        return (
            numeric.rolling(rolling_window, min_periods=1)
            .max()
            .shift(1, fill_value=0)
            .astype(bool)
        )
    if method == "expanding_shifted":
        return numeric.expanding(min_periods=1).max().shift(1, fill_value=0).astype(bool)
    if method == "fed_target":
        return pd.Series(False, index=flag.index, dtype=bool)
    raise ValueError(f"Unknown baseline method: {method}")


def _rolling_lineage(flag: pd.Series, window: int) -> pd.Series:
    return flag.astype(int).rolling(window, min_periods=1).max().astype(bool)


def _latest_timestamp(*values: pd.Series) -> pd.Series:
    timestamps = pd.concat(
        [
            pd.to_datetime(value, errors="coerce", utc=True).astype(
                "datetime64[ns, UTC]"
            )
            for value in values
        ],
        axis=1,
    )
    return pd.to_datetime(timestamps.max(axis=1), errors="coerce", utc=True)


def _trusted_information_timestamps(
    values: pd.Series,
    provenance: pd.Series,
    timing_status: pd.Series,
) -> pd.Series:
    """Retain exact times only when the incoming timing contract marks them exact."""

    timezone_known = values.map(_has_explicit_timezone).astype(bool)
    derived_from_releases = provenance.astype("string").eq(
        INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    ).fillna(False)
    release_aligned = timing_status.astype("string").eq(
        TIMING_STATUS_RELEASE_ALIGNED
    ).fillna(False)
    parsed = pd.to_datetime(values, errors="coerce", utc=True).astype(
        "datetime64[ns, UTC]"
    )
    return parsed.where(timezone_known & derived_from_releases & release_aligned)


def _component_timing_status(
    values: pd.Series,
    exact_timestamps: pd.Series,
    *,
    incoming_status: pd.Series | None = None,
) -> pd.Series:
    """Label component timing without promoting untrusted incoming status."""

    available = values.notna()
    status = pd.Series(
        TIMING_STATUS_UNAVAILABLE,
        index=values.index,
        dtype="string",
    )
    status.loc[available] = TIMING_STATUS_REFERENCE_MONTH_ONLY
    if incoming_status is not None:
        supplied = incoming_status.astype("string")
        preserve_nonexact = (
            available
            & supplied.notna()
            & supplied.ne(TIMING_STATUS_RELEASE_ALIGNED).fillna(False)
        )
        status.loc[preserve_nonexact] = supplied.loc[preserve_nonexact]
    status.loc[available & exact_timestamps.notna()] = TIMING_STATUS_RELEASE_ALIGNED
    return status


def _exact_rolling_timestamp_maximum(
    timestamps: pd.Series,
    *,
    window: int | None,
) -> pd.Series:
    """Return rolling/expanding timestamp maxima without float conversion."""

    parsed = pd.to_datetime(
        timestamps,
        errors="coerce",
        utc=True,
    ).astype("datetime64[ns, UTC]")
    timestamp_ns = parsed.array.asi8
    missing_ns = np.iinfo(np.int64).min
    latest_ns = np.full(len(parsed), missing_ns, dtype=np.int64)
    candidates: deque[int] = deque()

    for position, value_ns in enumerate(timestamp_ns):
        if window is not None:
            expired = position - window
            while candidates and candidates[0] <= expired:
                candidates.popleft()
        if value_ns != missing_ns:
            while candidates and timestamp_ns[candidates[-1]] <= value_ns:
                candidates.pop()
            candidates.append(position)
        if candidates:
            latest_ns[position] = timestamp_ns[candidates[0]]

    return pd.Series(
        pd.to_datetime(latest_ns, errors="coerce", unit="ns", utc=True),
        index=parsed.index,
        dtype="datetime64[ns, UTC]",
    )


def _rolling_information_timestamp(
    values: pd.Series,
    timestamps: pd.Series,
    *,
    window: int,
    min_periods: int,
    shift: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Return exact latest dependency time and whether every dependency is dated."""

    dependency_values = values.shift(shift)
    dependency_timestamps = pd.to_datetime(
        timestamps,
        errors="coerce",
        utc=True,
    ).astype("datetime64[ns, UTC]").shift(shift)
    valid = dependency_values.notna()
    dependency_timestamps = dependency_timestamps.where(valid)
    known = valid & dependency_timestamps.notna()
    valid_count = valid.astype(int).rolling(window, min_periods=1).sum()
    known_count = known.astype(int).rolling(window, min_periods=1).sum()
    exact = (valid_count >= min_periods) & (known_count == valid_count)

    latest = _exact_rolling_timestamp_maximum(
        dependency_timestamps,
        window=window,
    ).where(exact)
    return latest, exact


def _expanding_information_timestamp(
    values: pd.Series,
    timestamps: pd.Series,
    *,
    min_periods: int,
    shift: int = 0,
) -> tuple[pd.Series, pd.Series]:
    dependency_values = values.shift(shift)
    dependency_timestamps = pd.to_datetime(
        timestamps,
        errors="coerce",
        utc=True,
    ).astype("datetime64[ns, UTC]").shift(shift)
    valid = dependency_values.notna()
    dependency_timestamps = dependency_timestamps.where(valid)
    known = valid & dependency_timestamps.notna()
    valid_count = valid.astype(int).expanding(min_periods=1).sum()
    known_count = known.astype(int).expanding(min_periods=1).sum()
    exact = (valid_count >= min_periods) & (known_count == valid_count)

    latest = _exact_rolling_timestamp_maximum(
        dependency_timestamps,
        window=None,
    ).where(exact)
    return latest, exact


def _baseline_information_timestamp(
    inflation: pd.Series,
    inflation_information: pd.Series,
    method: str,
    *,
    rolling_window: int = 36,
    expanding_min_periods: int = 120,
) -> tuple[pd.Series, pd.Series]:
    if method == "full_sample":
        valid = inflation.notna()
        exact_value = bool(
            valid.any() and inflation_information.loc[valid].notna().all()
        )
        latest = (
            pd.to_datetime(
                inflation_information.loc[valid],
                errors="coerce",
                utc=True,
            ).max()
            if exact_value
            else pd.NaT
        )
        return (
            pd.Series(latest, index=inflation.index, dtype="datetime64[ns, UTC]"),
            pd.Series(exact_value, index=inflation.index, dtype=bool),
        )
    if method == "rolling_36_unshifted":
        return _rolling_information_timestamp(
            inflation,
            inflation_information,
            window=rolling_window,
            min_periods=rolling_window,
        )
    if method == "rolling_36_shifted":
        return _rolling_information_timestamp(
            inflation,
            inflation_information,
            window=rolling_window,
            min_periods=rolling_window,
            shift=1,
        )
    if method == "expanding_shifted":
        return _expanding_information_timestamp(
            inflation,
            inflation_information,
            min_periods=expanding_min_periods,
            shift=1,
        )
    if method == "fed_target":
        return (
            pd.Series(pd.NaT, index=inflation.index, dtype="datetime64[ns, UTC]"),
            pd.Series(True, index=inflation.index, dtype=bool),
        )
    raise ValueError(f"Unknown baseline method: {method}")


def add_transitory_inflation_features(
    df: pd.DataFrame,
    inflation_col: str = "inflation_yoy",
    baseline_method: str = "rolling_36_shifted",
    tinf_windows: tuple[int, ...] = (4, 8, 12),
) -> pd.DataFrame:
    """Add continuous TINF and diagnostic persistence flags.

    The continuous TINF signal is magnitude-based. Binary flags are only diagnostics.
    """

    if inflation_col not in df.columns:
        raise KeyError(f"Missing inflation column: {inflation_col}")

    out = df.copy()
    out["baseline_method"] = baseline_method
    out["baseline"] = compute_baseline(out[inflation_col], baseline_method)
    out["epsilon"] = out[inflation_col] - out["baseline"]

    inflation_information_col = f"{inflation_col}_information_timestamp"
    inflation_information_provenance_col = (
        f"{inflation_col}_information_timestamp_provenance"
    )
    if inflation_information_col in out.columns:
        supplied_inflation_information = out[inflation_information_col]
        supplied_inflation_provenance = out.get(
            inflation_information_provenance_col,
            pd.Series(pd.NA, index=out.index, dtype="string"),
        )
    elif inflation_col == "inflation_yoy" and "information_timestamp" in out.columns:
        supplied_inflation_information = out["information_timestamp"]
        supplied_inflation_provenance = out.get(
            "information_timestamp_provenance",
            pd.Series(pd.NA, index=out.index, dtype="string"),
        )
    else:
        supplied_inflation_information = pd.Series(
            pd.NaT,
            index=out.index,
            dtype="datetime64[ns, UTC]",
        )
        supplied_inflation_provenance = pd.Series(
            pd.NA,
            index=out.index,
            dtype="string",
        )
    inflation_timing_status_col = f"{inflation_col}_timing_status"
    if inflation_timing_status_col in out.columns:
        supplied_inflation_timing_status = out[inflation_timing_status_col]
    elif inflation_col == "inflation_yoy":
        supplied_inflation_timing_status = out.get(
            "timing_status",
            pd.Series(pd.NA, index=out.index, dtype="string"),
        )
    else:
        supplied_inflation_timing_status = pd.Series(
            pd.NA,
            index=out.index,
            dtype="string",
        )
    inflation_information = _trusted_information_timestamps(
        supplied_inflation_information,
        supplied_inflation_provenance,
        supplied_inflation_timing_status,
    ).where(out[inflation_col].notna())
    out[inflation_information_col] = inflation_information
    out[inflation_information_provenance_col] = pd.Series(
        INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
        index=out.index,
        dtype="string",
    )
    out.loc[
        inflation_information.notna(),
        inflation_information_provenance_col,
    ] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    out[inflation_timing_status_col] = _component_timing_status(
        out[inflation_col],
        inflation_information,
        incoming_status=supplied_inflation_timing_status,
    )

    baseline_information, baseline_information_exact = _baseline_information_timestamp(
        out[inflation_col],
        inflation_information,
        baseline_method,
    )
    baseline_information_exact &= out["baseline"].notna()
    out["baseline_information_timestamp"] = baseline_information.where(
        baseline_information_exact
    )
    out["baseline_information_timestamp_provenance"] = pd.Series(
        INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
        index=out.index,
        dtype="string",
    )
    out.loc[
        out["baseline_information_timestamp"].notna(),
        "baseline_information_timestamp_provenance",
    ] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    out["baseline_timing_status"] = _component_timing_status(
        out["baseline"],
        out["baseline_information_timestamp"],
    )

    inflation_information_exact = (
        out[inflation_col].notna() & inflation_information.notna()
    )
    epsilon_information_exact = (
        out["epsilon"].notna()
        & inflation_information_exact
        & baseline_information_exact
    )
    out["epsilon_information_timestamp"] = _latest_timestamp(
        inflation_information,
        out["baseline_information_timestamp"],
    ).where(epsilon_information_exact)
    out["epsilon_information_timestamp_provenance"] = pd.Series(
        INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
        index=out.index,
        dtype="string",
    )
    out.loc[
        out["epsilon_information_timestamp"].notna(),
        "epsilon_information_timestamp_provenance",
    ] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    out["epsilon_timing_status"] = _component_timing_status(
        out["epsilon"],
        out["epsilon_information_timestamp"],
    )

    inflation_imputed = _lineage_flag(
        out,
        f"{inflation_col}_uses_imputed_input",
    )
    inflation_missing = _lineage_flag(
        out,
        f"{inflation_col}_uses_missing_input",
    )
    out["baseline_uses_imputed_input"] = _baseline_lineage(
        inflation_imputed,
        baseline_method,
    )
    out["baseline_uses_missing_input"] = _baseline_lineage(
        inflation_missing,
        baseline_method,
    )
    out["epsilon_uses_imputed_input"] = (
        inflation_imputed | out["baseline_uses_imputed_input"]
    )
    out["epsilon_uses_missing_input"] = (
        inflation_missing | out["baseline_uses_missing_input"]
    )

    component_information_exact: dict[str, pd.Series] = {
        inflation_col: inflation_information_exact,
        "baseline": baseline_information_exact,
        "epsilon": epsilon_information_exact,
    }
    for window in tinf_windows:
        out[f"tinf_{window}m"] = out["epsilon"].rolling(window, min_periods=window).mean()
        tinf_information, tinf_information_exact = _rolling_information_timestamp(
            out["epsilon"],
            out["epsilon_information_timestamp"],
            window=window,
            min_periods=window,
        )
        tinf_col = f"tinf_{window}m"
        tinf_information_exact &= out[tinf_col].notna()
        out[f"{tinf_col}_information_timestamp"] = tinf_information.where(
            tinf_information_exact
        )
        out[f"{tinf_col}_information_timestamp_provenance"] = pd.Series(
            INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
            index=out.index,
            dtype="string",
        )
        out.loc[
            out[f"{tinf_col}_information_timestamp"].notna(),
            f"{tinf_col}_information_timestamp_provenance",
        ] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
        out[f"{tinf_col}_timing_status"] = _component_timing_status(
            out[tinf_col],
            out[f"{tinf_col}_information_timestamp"],
        )
        component_information_exact[tinf_col] = tinf_information_exact
        out[f"tinf_{window}m_uses_imputed_input"] = _rolling_lineage(
            out["epsilon_uses_imputed_input"],
            window,
        )
        out[f"tinf_{window}m_uses_missing_input"] = _rolling_lineage(
            out["epsilon_uses_missing_input"],
            window,
        )

    imputed_lineage_cols = [
        f"{inflation_col}_uses_imputed_input",
        "baseline_uses_imputed_input",
        "epsilon_uses_imputed_input",
        *(f"tinf_{window}m_uses_imputed_input" for window in tinf_windows),
    ]
    missing_lineage_cols = [
        f"{inflation_col}_uses_missing_input",
        "baseline_uses_missing_input",
        "epsilon_uses_missing_input",
        *(f"tinf_{window}m_uses_missing_input" for window in tinf_windows),
    ]
    out["signal_uses_imputed_input"] = pd.concat(
        [_lineage_flag(out, column) for column in imputed_lineage_cols],
        axis=1,
    ).any(axis=1)
    out["signal_uses_missing_input"] = pd.concat(
        [_lineage_flag(out, column) for column in missing_lineage_cols],
        axis=1,
    ).any(axis=1)
    out["signal_observed_only_eligible"] = ~(
        out["signal_uses_imputed_input"] | out["signal_uses_missing_input"]
    )

    derived_components = [
        inflation_col,
        "baseline",
        "epsilon",
        *(f"tinf_{window}m" for window in tinf_windows),
    ]
    information_columns = [
        inflation_information_col,
        "baseline_information_timestamp",
        "epsilon_information_timestamp",
        *(f"tinf_{window}m_information_timestamp" for window in tinf_windows),
    ]
    derived_available = pd.concat(
        [out[column].notna() for column in derived_components],
        axis=1,
    ).all(axis=1)
    derived_exact = pd.Series(True, index=out.index, dtype=bool)
    for component in derived_components:
        derived_exact &= ~out[component].notna() | component_information_exact[component]
    out["information_timestamp"] = _latest_timestamp(
        *(out[column] for column in information_columns)
    ).where(derived_available & derived_exact)
    out["information_timestamp_provenance"] = pd.Series(
        INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
        index=out.index,
        dtype="string",
    )
    out.loc[
        out["information_timestamp"].notna(),
        "information_timestamp_provenance",
    ] = INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
    out["timing_status"] = pd.Series(
        TIMING_STATUS_UNAVAILABLE,
        index=out.index,
        dtype="string",
    )
    out.loc[derived_available, "timing_status"] = TIMING_STATUS_REFERENCE_MONTH_ONLY
    out.loc[
        derived_available & out["information_timestamp"].notna(),
        "timing_status",
    ] = TIMING_STATUS_RELEASE_ALIGNED

    epsilon_available = out["epsilon"].notna()
    out["above_baseline"] = (
        (out["epsilon"] > 0).astype("boolean").where(epsilon_available)
    )
    out["run_length_above"] = (
        consecutive_true_count(out["above_baseline"])
        .astype("Int64")
        .where(epsilon_available)
    )
    out["short_regime_flag"] = (
        (out["run_length_above"] >= 4).astype("boolean").where(epsilon_available)
    )
    out["medium_regime_flag"] = (
        (out["run_length_above"] >= 8).astype("boolean").where(epsilon_available)
    )
    out["long_regime_flag"] = (
        (out["run_length_above"] >= 12).astype("boolean").where(epsilon_available)
    )

    if {"tinf_4m", "tinf_8m", "tinf_12m"}.issubset(out.columns):
        term_inputs_available = out[["tinf_4m", "tinf_8m", "tinf_12m"]].notna().all(axis=1)
        term_structure = pd.Series(pd.NA, index=out.index, dtype="string")
        term_structure.loc[term_inputs_available] = "mixed"
        term_structure.loc[
            term_inputs_available
            & (out["tinf_4m"] > out["tinf_8m"])
            & (out["tinf_8m"] > out["tinf_12m"])
        ] = "accelerating"
        term_structure.loc[
            term_inputs_available
            & (out["tinf_4m"] < out["tinf_8m"])
            & (out["tinf_8m"] < out["tinf_12m"])
        ] = "decelerating"
        out["tinf_term_structure"] = term_structure

    return out


def latest_signal_snapshot(df: pd.DataFrame) -> dict[str, object]:
    """Create a small current-signal dictionary for dashboards/reports."""

    cols = ["date", "inflation_yoy", "baseline", "epsilon", "tinf_4m", "tinf_8m", "tinf_12m"]
    clean = df.dropna(subset=[col for col in cols if col in df.columns]).copy()
    if clean.empty:
        return {"available": False, "reason": "No complete TINF observations available."}

    latest = clean.iloc[-1]
    tinf_hist = clean["tinf_4m"].dropna()
    percentile = float((tinf_hist <= latest["tinf_4m"]).mean() * 100)

    distribution_timestamps = pd.Series(
        pd.NaT,
        index=tinf_hist.index,
        dtype="datetime64[ns, UTC]",
    )
    if "tinf_4m_information_timestamp" in clean.columns:
        supplied_distribution_timestamps = clean.loc[
            tinf_hist.index,
            "tinf_4m_information_timestamp",
        ]
        supplied_distribution_provenance = clean.get(
            "tinf_4m_information_timestamp_provenance",
            pd.Series(pd.NA, index=clean.index, dtype="string"),
        ).loc[tinf_hist.index]
        supplied_distribution_status = clean.get(
            "tinf_4m_timing_status",
            pd.Series(pd.NA, index=clean.index, dtype="string"),
        ).loc[tinf_hist.index]
        distribution_timestamps = pd.to_datetime(
            supplied_distribution_timestamps,
            errors="coerce",
            utc=True,
        ).astype("datetime64[ns, UTC]")
        distribution_timestamp_exact = (
            supplied_distribution_timestamps.map(_has_explicit_timezone).astype(bool)
            & supplied_distribution_provenance.astype("string")
            .eq(INFORMATION_TIMESTAMP_PROVENANCE_RELEASES)
            .fillna(False)
            & supplied_distribution_status.astype("string")
            .eq(TIMING_STATUS_RELEASE_ALIGNED)
            .fillna(False)
            & distribution_timestamps.notna()
        )
        distribution_timestamps = distribution_timestamps.where(
            distribution_timestamp_exact
        )
    distribution_exact = bool(
        not tinf_hist.empty and distribution_timestamps.notna().all()
    )
    distribution_information_timestamp = (
        distribution_timestamps.max() if distribution_exact else pd.NaT
    )

    latest_information_timestamp = _trusted_information_timestamps(
        pd.Series([latest.get("information_timestamp", pd.NaT)]),
        pd.Series(
            [
                latest.get(
                    "information_timestamp_provenance",
                    INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
                )
            ],
            dtype="string",
        ),
        pd.Series(
            [latest.get("timing_status", TIMING_STATUS_REFERENCE_MONTH_ONLY)],
            dtype="string",
        ),
    ).iloc[0]
    snapshot_exact = pd.notna(latest_information_timestamp) and distribution_exact
    snapshot_information_timestamp = (
        max(latest_information_timestamp, distribution_information_timestamp)
        if snapshot_exact
        else pd.NaT
    )
    snapshot_information_provenance = (
        INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
        if snapshot_exact
        else INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED
    )
    snapshot_timing_status = (
        TIMING_STATUS_RELEASE_ALIGNED
        if snapshot_exact
        else TIMING_STATUS_REFERENCE_MONTH_ONLY
    )
    distribution_information_provenance = (
        INFORMATION_TIMESTAMP_PROVENANCE_RELEASES
        if distribution_exact
        else INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED
    )
    distribution_timing_status = (
        TIMING_STATUS_RELEASE_ALIGNED
        if distribution_exact
        else TIMING_STATUS_REFERENCE_MONTH_ONLY
    )

    prev = clean.iloc[-2] if len(clean) >= 2 else latest
    rising = latest["tinf_4m"] > prev["tinf_4m"]

    component_imputation_lineage = {
        "inflation_yoy_uses_imputed_input": bool(
            latest.get("inflation_yoy_uses_imputed_input", False)
        ),
        "baseline_uses_imputed_input": bool(
            latest.get("baseline_uses_imputed_input", False)
        ),
        "epsilon_uses_imputed_input": bool(
            latest.get("epsilon_uses_imputed_input", False)
        ),
        "tinf_4m_uses_imputed_input": bool(
            latest.get("tinf_4m_uses_imputed_input", False)
        ),
        "tinf_8m_uses_imputed_input": bool(
            latest.get("tinf_8m_uses_imputed_input", False)
        ),
        "tinf_12m_uses_imputed_input": bool(
            latest.get("tinf_12m_uses_imputed_input", False)
        ),
    }
    # Percentiles and the regime thresholds are full-sample descriptive reads.
    # Conservatively retain lineage when any TINF observation contributing to
    # their reference distribution depends on an estimated CPI input.
    distribution_uses_imputed_input = bool(
        _lineage_flag(clean, "tinf_4m_uses_imputed_input").any()
    )
    component_imputation_lineage["percentile_uses_imputed_input"] = (
        distribution_uses_imputed_input
    )
    component_imputation_lineage["regime_uses_imputed_input"] = (
        distribution_uses_imputed_input
    )
    uses_imputed_input = any(component_imputation_lineage.values())
    uses_missing_input = bool(latest.get("signal_uses_missing_input", False))

    if latest["tinf_4m"] > tinf_hist.quantile(0.75):
        regime = "elevated rising" if rising else "elevated falling"
    elif latest["tinf_4m"] < tinf_hist.quantile(0.25):
        regime = "disinflationary"
    else:
        regime = "neutral"

    return {
        "available": True,
        "date": latest["date"] if "date" in clean.columns else clean.index[-1],
        "reference_month": latest.get(
            "reference_month",
            latest["date"] if "date" in clean.columns else clean.index[-1],
        ),
        "release_timestamp": latest.get("release_timestamp", pd.NaT),
        "release_timestamp_provenance": latest.get(
            "release_timestamp_provenance",
            "release_metadata_unavailable_or_unverified",
        ),
        "information_timestamp": snapshot_information_timestamp,
        "information_timestamp_provenance": snapshot_information_provenance,
        "vintage_timestamp": latest.get("vintage_timestamp", pd.NaT),
        "retrieved_at": latest.get("retrieved_at", pd.NaT),
        "timing_status": snapshot_timing_status,
        "data_vintage_status": latest.get(
            "data_vintage_status",
            "latest_revised_non_vintage",
        ),
        "inflation_yoy": float(latest["inflation_yoy"]),
        "baseline": float(latest["baseline"]),
        "epsilon": float(latest["epsilon"]),
        "tinf_4m": float(latest["tinf_4m"]),
        "tinf_8m": float(latest["tinf_8m"]),
        "tinf_12m": float(latest["tinf_12m"]),
        "tinf_4m_percentile": percentile,
        "percentile_information_timestamp": distribution_information_timestamp,
        "percentile_information_timestamp_provenance": (
            distribution_information_provenance
        ),
        "percentile_timing_status": distribution_timing_status,
        "regime": regime,
        "regime_information_timestamp": distribution_information_timestamp,
        "regime_information_timestamp_provenance": (
            distribution_information_provenance
        ),
        "regime_timing_status": distribution_timing_status,
        "term_structure": latest.get("tinf_term_structure", "n/a"),
        **component_imputation_lineage,
        "uses_imputed_input": uses_imputed_input,
        "uses_missing_input": uses_missing_input,
        "observed_only_eligible": not (uses_imputed_input or uses_missing_input),
        "imputation_policy": latest.get("imputation_policy", "unspecified"),
    }

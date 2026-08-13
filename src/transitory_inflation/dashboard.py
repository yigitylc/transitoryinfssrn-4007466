from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .config import SampleMode, resolve_sample_mode
from .data import (
    CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH,
    CURRENT_MONITORING_SCENARIOS,
    INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED,
    RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED,
    TIMING_STATUS_UNAVAILABLE,
    CurrentMonitoringScenarioId,
    build_current_monitoring_scenario_frame,
    slice_date_range,
    validate_monthly_warmup_authority,
)
from .features import (
    add_transitory_inflation_features,
    latest_signal_snapshot,
    observed_only_historical_eligibility,
)

CURRENT_SIGNAL_IMPUTATION_NOTICE = (
    "Current monitoring uses unofficial low/base/high assumptions for the permanently "
    "missing October 2025 CPI levels. Base is the geometric midpoint of September and "
    "November; low/high use the September/November official levels. These are ex-post "
    "sensitivity scenarios, not confidence intervals, official observations, or a "
    "real-time nowcast. Inputs are latest-revised and non-vintage. Historical evidence "
    "and calibration remain observed-only and exclude estimated CPI inputs."
)

_UNAVAILABLE_SIGNAL_FIELDS = (
    "inflation_yoy",
    "baseline",
    "epsilon",
    "tinf_4m",
    "tinf_8m",
    "tinf_12m",
    "tinf_4m_percentile",
    "regime_lower_threshold",
    "regime_upper_threshold",
    "regime",
    "term_structure",
    "pressure",
)

_UNAVAILABLE_FEATURE_FIELDS = (
    "inflation_yoy",
    "baseline",
    "epsilon",
    "tinf_4m",
    "tinf_8m",
    "tinf_12m",
    "above_baseline",
    "run_length_above",
    "short_regime_flag",
    "medium_regime_flag",
    "long_regime_flag",
    "tinf_term_structure",
)


def _json_compatible(value: object) -> object:
    """Recursively normalize scenario metadata for JSON serialization."""

    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_compatible(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


@dataclass(frozen=True)
class ScenarioStability:
    """Exact approved classification-stability semantics."""

    regime_stable: bool
    pressure_stable: bool
    fully_stable: bool
    scenario_sensitive: bool
    unavailable: bool
    not_applicable: bool = False

    @property
    def label(self) -> str:
        if self.not_applicable:
            return "not applicable"
        if self.unavailable:
            return "unavailable"
        if self.fully_stable:
            return "stable across low/base/high scenarios"
        return "scenario-sensitive"


@dataclass(frozen=True)
class CurrentMonitoringScenarioView:
    """One current scenario plus its observed-history-calibrated snapshot."""

    scenario_id: str
    raw: pd.DataFrame
    featured: pd.DataFrame
    snapshot: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible in-memory scenario export."""

        return {
            "scenario_id": self.scenario_id,
            "snapshot": _json_compatible(self.snapshot),
        }


@dataclass(frozen=True)
class CurrentMonitoringBundle:
    """Low/base/high current paths and their classification stability."""

    scenarios: Mapping[CurrentMonitoringScenarioId, CurrentMonitoringScenarioView]
    stability: ScenarioStability
    historical_evidence_endpoint: pd.Timestamp | None
    status: str = "available"
    reason: str | None = None
    sample_mode: str = "custom"
    baseline_method: str = "unspecified"
    observed_current: CurrentMonitoringScenarioView | None = None

    def __post_init__(self) -> None:
        """Reject bundles whose public status disagrees with their signal payloads."""

        if self.status not in {"available", "unavailable", "not_applicable"}:
            raise ValueError(f"Unknown current-monitoring bundle status: {self.status}")
        if self.status == "not_applicable":
            if self.scenarios or self.observed_current is None or not self.stability.not_applicable:
                raise ValueError(
                    "A not-applicable current-monitoring bundle requires only an "
                    "observed current view"
                )
            return

        if set(self.scenarios) != set(CURRENT_MONITORING_SCENARIOS):
            raise ValueError(
                "Applicable current-monitoring bundles require exactly low/base/high scenarios"
            )
        snapshot_availability = {
            scenario_id: bool(self.scenarios[scenario_id].snapshot.get("available", False))
            for scenario_id in CURRENT_MONITORING_SCENARIOS
        }
        if self.status == "available":
            if self.stability.unavailable or not all(snapshot_availability.values()):
                raise ValueError(
                    "An available current-monitoring bundle requires three available scenarios"
                )
            return

        if not self.stability.unavailable or any(snapshot_availability.values()):
            raise ValueError(
                "An unavailable current-monitoring bundle cannot contain an available scenario"
            )
        leaked_snapshot_fields = {
            field
            for view in self.scenarios.values()
            for field in _UNAVAILABLE_SIGNAL_FIELDS
            if field in view.snapshot
            and view.snapshot[field] is not None
            and view.snapshot[field] is not pd.NA
            and not bool(pd.isna(view.snapshot[field]))
        }
        if leaked_snapshot_fields:
            leaked = ", ".join(sorted(leaked_snapshot_fields))
            raise ValueError(
                "An unavailable current-monitoring bundle cannot expose snapshot "
                f"signal fields: {leaked}"
            )
        leaked_fields = {
            field
            for view in self.scenarios.values()
            for field in _UNAVAILABLE_FEATURE_FIELDS
            if field in view.featured.columns and view.featured[field].notna().any()
        }
        if leaked_fields:
            leaked = ", ".join(sorted(leaked_fields))
            raise ValueError(
                "An unavailable current-monitoring bundle cannot expose derived signal "
                f"features: {leaked}"
            )

    @property
    def applicable(self) -> bool:
        return self.status != "not_applicable"

    @property
    def available(self) -> bool:
        return self.status == "available"

    @property
    def base(self) -> CurrentMonitoringScenarioView:
        if "base" in self.scenarios:
            return self.scenarios["base"]
        if self.observed_current is not None:
            return self.observed_current
        raise RuntimeError("Current monitoring has no base or observed current view")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible bundle without serializing research frames."""

        return _json_compatible(
            {
                "status": self.status,
                "applicable": self.applicable,
                "available": self.available,
                "reason": self.reason,
                "sample_mode": self.sample_mode,
                "baseline_method": self.baseline_method,
                "historical_evidence_endpoint": self.historical_evidence_endpoint,
                "stability": {
                    "regime_stable": self.stability.regime_stable,
                    "pressure_stable": self.stability.pressure_stable,
                    "fully_stable": self.stability.fully_stable,
                    "scenario_sensitive": self.stability.scenario_sensitive,
                    "unavailable": self.stability.unavailable,
                    "not_applicable": self.stability.not_applicable,
                },
                "scenarios": {
                    scenario_id: view.to_dict() for scenario_id, view in self.scenarios.items()
                },
                "observed_current": (
                    self.observed_current.to_dict() if self.observed_current is not None else None
                ),
            }
        )

    def scenario_table(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for scenario_id in CURRENT_MONITORING_SCENARIOS:
            view = self.scenarios.get(scenario_id)
            if view is None:
                continue
            snapshot = view.snapshot
            snapshot_lineage = snapshot.get("series_lineage", {})
            core_lineage = (
                snapshot_lineage.get("core_cpi", {})
                if isinstance(snapshot_lineage, Mapping)
                else {}
            )
            row: dict[str, object] = {
                "scenario_id": scenario_id,
                "bundle_status": self.status,
                "bundle_reason": self.reason,
                "scenario_applicable": self.applicable,
                "available": bool(snapshot.get("available", False)),
                "reference_month": snapshot.get("reference_month", pd.NaT),
                "release_timestamp": snapshot.get("release_timestamp", pd.NaT),
                "release_timestamp_provenance": snapshot.get("release_timestamp_provenance", pd.NA),
                "information_timestamp": snapshot.get("information_timestamp", pd.NaT),
                "information_timestamp_provenance": snapshot.get(
                    "information_timestamp_provenance", pd.NA
                ),
                "timing_status": snapshot.get("timing_status", pd.NA),
                "vintage_timestamp": snapshot.get("vintage_timestamp", pd.NaT),
                "retrieved_at": snapshot.get("retrieved_at", pd.NaT),
                "imputation_policy": snapshot.get("imputation_policy", "ex_post_continuity"),
                "estimate_method": snapshot.get("estimate_method", pd.NA),
                "estimated_reference_month": snapshot.get("estimated_reference_month", pd.NaT),
                "estimate_value": snapshot.get("estimate_value", float("nan")),
                "estimate_available_at": snapshot.get("estimate_available_at", pd.NaT),
                "estimate_availability_basis": snapshot.get("estimate_availability_basis", pd.NA),
                "core_cpi_estimate_value": (
                    core_lineage.get("estimate_value", float("nan"))
                    if isinstance(core_lineage, Mapping)
                    else float("nan")
                ),
                "uses_estimated_input": bool(snapshot.get("uses_estimated_input", False)),
                "estimated_input_months": snapshot.get("estimated_input_months", ()),
                "calibration_policy": snapshot.get("calibration_policy", pd.NA),
                "calibration_cutoff_policy": snapshot.get("calibration_cutoff_policy", pd.NA),
                "calibration_observation_count": snapshot.get("calibration_observation_count", 0),
                "calibration_end_month": snapshot.get("calibration_end_month", pd.NaT),
                "inflation_yoy": snapshot.get("inflation_yoy", float("nan")),
                "tinf_4m": snapshot.get("tinf_4m", float("nan")),
                "regime": snapshot.get("regime", pd.NA),
                "pressure": snapshot.get("pressure", pd.NA),
                "regime_stable": self.stability.regime_stable,
                "pressure_stable": self.stability.pressure_stable,
                "fully_stable": self.stability.fully_stable,
                "scenario_sensitive": self.stability.scenario_sensitive,
                "unavailable": self.stability.unavailable,
                "information_status": snapshot.get(
                    "information_status", snapshot.get("timing_status", pd.NA)
                ),
                "data_vintage_status": snapshot.get(
                    "data_vintage_status", "latest_revised_non_vintage"
                ),
                "historical_evidence_endpoint": self.historical_evidence_endpoint,
                "historical_endpoint_definition": (
                    "latest complete observed-only TINF4/TINF8/TINF12 and pressure row; "
                    "scored-outcome endpoints vary by horizon"
                ),
                "historical_population_policy": "observed_only",
                "sample_mode": snapshot.get("sample_mode", self.sample_mode),
                "baseline_method": snapshot.get("baseline_method", self.baseline_method),
                "ex_post_status": snapshot.get("ex_post_status", pd.NA),
                "lookahead_status": snapshot.get("lookahead_status", pd.NA),
                "signal_reference_month": snapshot.get(
                    "signal_reference_month",
                    snapshot.get("reference_month", pd.NaT),
                ),
                "signal_information_timestamp": snapshot.get(
                    "signal_information_timestamp",
                    snapshot.get("information_timestamp", pd.NaT),
                ),
                "signal_timing_status": snapshot.get(
                    "signal_timing_status", snapshot.get("timing_status", pd.NA)
                ),
                "series_lineage": snapshot.get("series_lineage", {}),
            }
            series_lineage = snapshot_lineage
            if isinstance(series_lineage, Mapping):
                for series_key in ("headline_cpi", "core_cpi"):
                    lineage = series_lineage.get(series_key, {})
                    if isinstance(lineage, Mapping):
                        for key, value in lineage.items():
                            row[f"{series_key}_{key}"] = value
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class DashboardDataViews:
    """Decision-safe historical authority plus explicit current scenarios."""

    research_raw: pd.DataFrame
    research_featured: pd.DataFrame
    current_monitoring: CurrentMonitoringBundle
    warmup_start: pd.Timestamp | None = None
    warmup_end: pd.Timestamp | None = None

    @property
    def current_raw(self) -> pd.DataFrame:
        """Compatibility alias: base scenario, or observed view when not applicable."""

        return self.current_monitoring.base.raw

    @property
    def current_featured(self) -> pd.DataFrame:
        """Compatibility alias: base scenario, or observed view when not applicable."""

        return self.current_monitoring.base.featured

    @property
    def current_chart_featured(self) -> pd.DataFrame | None:
        """Return current chart authority only when the current snapshot is available."""

        if not bool(self.current_snapshot.get("available", False)):
            return None
        return self.current_featured

    @property
    def current_snapshot(self) -> dict[str, object]:
        """Compatibility alias for the applicable base or observed current snapshot."""

        return self.current_monitoring.base.snapshot


def classify_scenario_stability(
    snapshots: Mapping[str, Mapping[str, object]],
) -> ScenarioStability:
    """Classify low/base/high agreement without modes or probabilities."""

    required = [snapshots.get(scenario_id, {}) for scenario_id in CURRENT_MONITORING_SCENARIOS]
    unavailable = any(not bool(snapshot.get("available", False)) for snapshot in required)
    if unavailable:
        return ScenarioStability(
            regime_stable=False,
            pressure_stable=False,
            fully_stable=False,
            scenario_sensitive=False,
            unavailable=True,
        )
    regime_stable = len({str(snapshot.get("regime")) for snapshot in required}) == 1
    pressure_stable = len({str(snapshot.get("pressure")) for snapshot in required}) == 1
    fully_stable = regime_stable and pressure_stable
    return ScenarioStability(
        regime_stable=regime_stable,
        pressure_stable=pressure_stable,
        fully_stable=fully_stable,
        scenario_sensitive=not fully_stable,
        unavailable=False,
    )


def _fail_closed_scenario_snapshot(
    current_raw: pd.DataFrame,
    snapshot: dict[str, object],
    *,
    sample_mode: str,
    baseline_method: str,
    forced_failure_reason: str | None = None,
) -> dict[str, object]:
    """Merge data-layer lineage and null classifications when the bundle failed."""

    target = pd.Timestamp(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH)
    metadata_row = pd.Series(dtype=object)
    if not current_raw.empty:
        reference_months = pd.to_datetime(
            current_raw.get("reference_month", current_raw.get("date")),
            errors="coerce",
        )
        target_rows = current_raw.loc[reference_months.eq(target)]
        metadata_row = target_rows.iloc[-1] if not target_rows.empty else current_raw.iloc[-1]

    series_lineage: Mapping[str, object] = {}
    if "scenario_series_lineage" in current_raw.columns:
        for value in reversed(current_raw["scenario_series_lineage"].tolist()):
            if isinstance(value, Mapping):
                series_lineage = value
                break

    bundle_available = (
        bool(metadata_row.get("scenario_bundle_available", True)) and forced_failure_reason is None
    )
    bundle_reason = forced_failure_reason or metadata_row.get(
        "scenario_bundle_failure_reason",
        metadata_row.get("scenario_bundle_reason", None),
    )
    if bundle_reason is pd.NA or (not isinstance(bundle_reason, str) and pd.isna(bundle_reason)):
        bundle_reason = None

    attempted = any(
        bool(lineage.get("estimate_attempted", False))
        for lineage in series_lineage.values()
        if isinstance(lineage, Mapping)
    )
    estimated_months = snapshot.get(
        "estimated_input_months",
        metadata_row.get("estimated_input_months", ()),
    )
    if attempted and not estimated_months:
        estimated_months = (CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH,)

    enriched = {
        **snapshot,
        "series_lineage": dict(series_lineage),
        "sample_mode": sample_mode,
        "baseline_method": baseline_method,
        "calibration_policy": snapshot.get("calibration_policy", "observed_only_eligible_history"),
        "historical_population_policy": "observed_only",
        "ex_post_status": "ex_post_current_monitoring_sensitivity",
        "lookahead_status": "uses_exact_following_november_official_observation",
        "signal_reference_month": snapshot.get("reference_month", pd.NaT),
        "signal_information_timestamp": snapshot.get("information_timestamp", pd.NaT),
        "signal_timing_status": snapshot.get("timing_status", pd.NA),
        "uses_estimated_input": bool(snapshot.get("uses_estimated_input", False) or attempted),
        "uses_imputed_input": bool(snapshot.get("uses_imputed_input", False) or attempted),
        "estimated_input_months": estimated_months,
    }
    if bundle_available:
        return enriched

    for key in _UNAVAILABLE_SIGNAL_FIELDS:
        enriched[key] = None
    enriched.update(
        {
            "available": False,
            "reason": str(bundle_reason or "Required missing-CPI scenario inputs are unavailable."),
            "date": target,
            "reference_month": target,
            "release_timestamp": pd.NaT,
            "release_timestamp_provenance": RELEASE_TIMESTAMP_PROVENANCE_UNVERIFIED,
            "information_timestamp": pd.NaT,
            "information_timestamp_provenance": (INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED),
            "timing_status": TIMING_STATUS_UNAVAILABLE,
            "information_status": TIMING_STATUS_UNAVAILABLE,
            "signal_reference_month": target,
            "signal_information_timestamp": pd.NaT,
            "signal_timing_status": TIMING_STATUS_UNAVAILABLE,
            "percentile_information_timestamp": pd.NaT,
            "percentile_information_timestamp_provenance": (
                INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED
            ),
            "percentile_timing_status": TIMING_STATUS_UNAVAILABLE,
            "regime_information_timestamp": pd.NaT,
            "regime_information_timestamp_provenance": (
                INFORMATION_TIMESTAMP_PROVENANCE_UNVERIFIED
            ),
            "regime_timing_status": TIMING_STATUS_UNAVAILABLE,
            "lookahead_status": "unavailable_required_scenario_input",
            "classification_status": "unavailable_required_scenario_input",
        }
    )
    return enriched


def _fail_closed_scenario_features(featured: pd.DataFrame) -> pd.DataFrame:
    """Remove derived current-signal values while retaining failure lineage flags."""

    out = featured.copy()
    for field in _UNAVAILABLE_FEATURE_FIELDS:
        if field in out.columns:
            out[field] = pd.NA
    return out


def _observed_authority(frame: pd.DataFrame, *, label: str) -> None:
    if "imputation_policy" in frame.columns:
        policies = set(frame["imputation_policy"].dropna().astype(str))
        if policies and policies != {"observed_only"}:
            raise ValueError(
                f"Dashboard {label} authority must use the observed_only imputation policy"
            )
    estimated_columns = [
        column for column in ("cpi_imputed", "core_cpi_imputed") if column in frame.columns
    ]
    if any(frame[column].fillna(False).astype(bool).any() for column in estimated_columns):
        raise ValueError(f"Dashboard {label} authority contains estimated CPI inputs")


def _require_common_warmup_authority(
    observed_raw: pd.DataFrame,
    warmup_raw: pd.DataFrame | None,
    sample_mode: SampleMode | str | None,
) -> pd.DataFrame:
    """Require one complete observed authority before any sample-mode slicing."""

    if warmup_raw is None:
        raise ValueError(
            "build_dashboard_data_views requires warmup_raw from the common raw "
            "warm-up authority; a sample-trimmed research frame is insufficient"
        )
    source = warmup_raw.copy()
    _observed_authority(source, label="warm-up")
    if "date" not in observed_raw.columns or "date" not in source.columns:
        raise ValueError("Dashboard observed and warm-up authorities require a date column")

    observed_months = pd.to_datetime(observed_raw["date"], errors="coerce").dt.to_period("M")
    source_months = pd.to_datetime(source["date"], errors="coerce").dt.to_period("M")
    if observed_months.isna().any() or source_months.isna().any():
        raise ValueError("Dashboard observed and warm-up authorities contain invalid dates")
    if observed_months.duplicated().any() or source_months.duplicated().any():
        raise ValueError("Dashboard warm-up authority must contain one coherent row per month")
    if not set(observed_months).issubset(set(source_months)):
        raise ValueError("Dashboard warm-up authority does not contain every observed sample month")

    observed_by_month = observed_raw.copy()
    observed_by_month.index = observed_months
    source_by_month = source.copy()
    source_by_month.index = source_months
    authority_columns = (
        "cpi_observed_level",
        "core_cpi_observed_level",
        "cpi_originally_missing",
        "core_cpi_originally_missing",
        "cpi_imputed",
        "core_cpi_imputed",
        "cpi_value_provenance",
        "core_cpi_value_provenance",
        "imputation_policy",
    )
    for column in authority_columns:
        if column not in observed_by_month.columns:
            continue
        if column not in source_by_month.columns:
            raise ValueError(
                f"Dashboard warm-up authority is missing observed authority column {column}"
            )
        observed_values = observed_by_month[column]
        source_values = source_by_month.loc[observed_values.index, column]
        equal = observed_values.eq(source_values) | (observed_values.isna() & source_values.isna())
        if not equal.fillna(False).all():
            raise ValueError(
                f"Dashboard warm-up authority disagrees with the observed sample for {column}"
            )

    if sample_mode is not None:
        resolved = resolve_sample_mode(sample_mode)
        if resolved.start_date is not None:
            required_start = pd.Timestamp(resolved.start_date).to_period("M").to_timestamp(
                "M"
            ) - pd.DateOffset(months=12)
            source_start = source_months.min().to_timestamp("M")
            if source_start > required_start:
                raise ValueError(
                    "Dashboard warm-up authority is sample-trimmed and does not include "
                    f"the required 12 pre-sample months through {required_start.date()}"
                )
        validation_start = resolved.start_date
    else:
        validation_start = None
    validate_monthly_warmup_authority(
        source,
        sample_start_date=validation_start,
        authority_label="Dashboard common warm-up authority",
        allow_full_history_origin=True,
    )
    if source_months.max() < observed_months.max():
        raise ValueError("Dashboard warm-up authority ends before the observed sample")
    return source


def _sample_bounds(
    observed_raw: pd.DataFrame,
    sample_mode: SampleMode | str | None,
) -> tuple[str | None, str | None]:
    if sample_mode is not None:
        resolved = resolve_sample_mode(sample_mode)
        return resolved.start_date, resolved.end_date
    dates = pd.to_datetime(observed_raw["date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def build_dashboard_data_views(
    observed_raw: pd.DataFrame,
    *,
    baseline_method: str,
    warmup_raw: pd.DataFrame | None = None,
    sample_mode: SampleMode | str | None = None,
) -> DashboardDataViews:
    """Build strict research and low/base/high views from one warm-up authority."""

    _observed_authority(observed_raw, label="research")
    source = _require_common_warmup_authority(observed_raw, warmup_raw, sample_mode)
    start_date, end_date = _sample_bounds(observed_raw, sample_mode)
    sample_mode_name = (
        resolve_sample_mode(sample_mode).name if sample_mode is not None else "custom"
    )

    research_raw = slice_date_range(
        source,
        start_date=start_date,
        end_date=end_date,
    )
    research_featured = add_transitory_inflation_features(
        research_raw,
        baseline_method=baseline_method,
    )
    complete_signal_columns = [
        column
        for column in ("tinf_4m", "tinf_8m", "tinf_12m", "tinf_term_structure")
        if column in research_featured.columns
    ]
    complete_signal = (
        research_featured[complete_signal_columns].notna().all(axis=1)
        if len(complete_signal_columns) == 4
        else pd.Series(False, index=research_featured.index, dtype=bool)
    )
    historical_eligible = observed_only_historical_eligibility(research_featured)
    historical_endpoint_values = pd.to_datetime(
        research_featured.loc[
            historical_eligible & complete_signal,
            "date",
        ],
        errors="coerce",
    )
    historical_endpoint = (
        pd.Timestamp(historical_endpoint_values.max())
        if not historical_endpoint_values.empty
        else None
    )

    source_dates = pd.to_datetime(source["date"], errors="coerce").dropna()
    research_dates = pd.to_datetime(research_raw["date"], errors="coerce").dropna()
    target = pd.Timestamp(CURRENT_MONITORING_ESTIMATED_REFERENCE_MONTH)
    if research_dates.empty or research_dates.max() < target:
        observed_snapshot = latest_signal_snapshot(research_featured)
        observed_snapshot.update(
            {
                "scenario_id": None,
                "scenario_applicability_status": "not_applicable",
                "scenario_applicability_reason": (
                    "The selected sample ends before the October 2025 missing-CPI month."
                ),
                "sample_mode": sample_mode_name,
                "baseline_method": baseline_method,
                "historical_population_policy": "observed_only",
                "ex_post_status": "observed_only_no_scenario",
                "lookahead_status": "not_applicable",
                "signal_reference_month": observed_snapshot.get("reference_month", pd.NaT),
                "signal_information_timestamp": observed_snapshot.get(
                    "information_timestamp", pd.NaT
                ),
                "signal_timing_status": observed_snapshot.get("timing_status", pd.NA),
                "series_lineage": {},
            }
        )
        observed_view = CurrentMonitoringScenarioView(
            scenario_id="observed_only",
            raw=research_raw,
            featured=research_featured,
            snapshot=observed_snapshot,
        )
        return DashboardDataViews(
            research_raw=research_raw,
            research_featured=research_featured,
            current_monitoring=CurrentMonitoringBundle(
                scenarios={},
                stability=ScenarioStability(
                    regime_stable=False,
                    pressure_stable=False,
                    fully_stable=False,
                    scenario_sensitive=False,
                    unavailable=False,
                    not_applicable=True,
                ),
                historical_evidence_endpoint=historical_endpoint,
                status="not_applicable",
                reason=("The selected sample ends before the October 2025 missing-CPI month."),
                sample_mode=sample_mode_name,
                baseline_method=baseline_method,
                observed_current=observed_view,
            ),
            warmup_start=(pd.Timestamp(source_dates.min()) if not source_dates.empty else None),
            warmup_end=(pd.Timestamp(source_dates.max()) if not source_dates.empty else None),
        )

    scenarios: dict[CurrentMonitoringScenarioId, CurrentMonitoringScenarioView] = {}
    for scenario_id in CURRENT_MONITORING_SCENARIOS:
        current_raw = build_current_monitoring_scenario_frame(
            source,
            scenario_id=scenario_id,
            warmup_sample_start_date=(start_date if sample_mode is not None else None),
        )
        current_raw = slice_date_range(
            current_raw,
            start_date=start_date,
            end_date=end_date,
        )
        current_featured = add_transitory_inflation_features(
            current_raw,
            baseline_method=baseline_method,
        )
        snapshot = latest_signal_snapshot(
            current_featured,
            calibration_df=research_featured,
        )
        snapshot = _fail_closed_scenario_snapshot(
            current_raw,
            snapshot,
            sample_mode=sample_mode_name,
            baseline_method=baseline_method,
        )
        scenarios[scenario_id] = CurrentMonitoringScenarioView(
            scenario_id=scenario_id,
            raw=current_raw,
            featured=current_featured,
            snapshot=snapshot,
        )

    stability = classify_scenario_stability(
        {scenario_id: view.snapshot for scenario_id, view in scenarios.items()}
    )
    bundle_status = "unavailable" if stability.unavailable else "available"
    bundle_reason = None
    if stability.unavailable:
        bundle_reason = next(
            (
                str(view.snapshot.get("reason"))
                for view in scenarios.values()
                if not view.snapshot.get("available") and view.snapshot.get("reason")
            ),
            "At least one required scenario is unavailable.",
        )
        scenarios = {
            scenario_id: CurrentMonitoringScenarioView(
                scenario_id=view.scenario_id,
                raw=view.raw,
                featured=_fail_closed_scenario_features(view.featured),
                snapshot=_fail_closed_scenario_snapshot(
                    view.raw,
                    view.snapshot,
                    sample_mode=sample_mode_name,
                    baseline_method=baseline_method,
                    forced_failure_reason=bundle_reason,
                ),
            )
            for scenario_id, view in scenarios.items()
        }
        stability = classify_scenario_stability(
            {scenario_id: view.snapshot for scenario_id, view in scenarios.items()}
        )
    return DashboardDataViews(
        research_raw=research_raw,
        research_featured=research_featured,
        current_monitoring=CurrentMonitoringBundle(
            scenarios=scenarios,
            stability=stability,
            historical_evidence_endpoint=historical_endpoint,
            status=bundle_status,
            reason=bundle_reason,
            sample_mode=sample_mode_name,
            baseline_method=baseline_method,
        ),
        warmup_start=(pd.Timestamp(source_dates.min()) if not source_dates.empty else None),
        warmup_end=(pd.Timestamp(source_dates.max()) if not source_dates.empty else None),
    )


def current_signal_imputation_notice(snapshot: Mapping[str, object]) -> str | None:
    """Return the required disclosure when a current signal uses an estimate."""

    if snapshot.get("available") and (
        snapshot.get("uses_estimated_input") or snapshot.get("uses_imputed_input")
    ):
        return CURRENT_SIGNAL_IMPUTATION_NOTICE
    return None

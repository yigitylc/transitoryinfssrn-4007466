from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import build_base_frame
from .features import add_transitory_inflation_features, latest_signal_snapshot

CURRENT_SIGNAL_IMPUTATION_NOTICE = (
    "This descriptive current signal uses an ex-post continuity estimate for an officially "
    "missing CPI input. It is not an official CPI observation and is excluded from "
    "observed-only historical testing."
)


@dataclass(frozen=True)
class DashboardDataViews:
    """Separate decision-safe research data from descriptive current monitoring."""

    research_raw: pd.DataFrame
    research_featured: pd.DataFrame
    current_raw: pd.DataFrame
    current_featured: pd.DataFrame

    @property
    def current_snapshot(self) -> dict[str, object]:
        return latest_signal_snapshot(self.current_featured)


def build_dashboard_data_views(
    observed_raw: pd.DataFrame,
    *,
    baseline_method: str,
) -> DashboardDataViews:
    """Build observed-only research and explicit ex-post current-monitoring views."""

    if "imputation_policy" in observed_raw.columns:
        policies = set(observed_raw["imputation_policy"].dropna().astype(str))
        if policies and policies != {"observed_only"}:
            raise ValueError(
                "Dashboard research authority must use the observed_only imputation policy"
            )

    research_raw = observed_raw.copy()
    research_featured = add_transitory_inflation_features(
        research_raw,
        baseline_method=baseline_method,
    )
    current_raw = build_base_frame(
        research_raw,
        imputation_policy="ex_post_continuity",
    )
    current_featured = add_transitory_inflation_features(
        current_raw,
        baseline_method=baseline_method,
    )
    return DashboardDataViews(
        research_raw=research_raw,
        research_featured=research_featured,
        current_raw=current_raw,
        current_featured=current_featured,
    )


def current_signal_imputation_notice(snapshot: dict[str, object]) -> str | None:
    """Return the required disclosure when a descriptive signal uses an estimate."""

    if snapshot.get("available") and snapshot.get("uses_imputed_input"):
        return CURRENT_SIGNAL_IMPUTATION_NOTICE
    return None

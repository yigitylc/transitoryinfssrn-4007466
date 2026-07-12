from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation import validation as validation_mod
from transitory_inflation.dashboard import (
    CURRENT_SIGNAL_IMPUTATION_NOTICE,
    build_dashboard_data_views,
    current_signal_imputation_notice,
)
from transitory_inflation.data import build_base_frame
from transitory_inflation.features import latest_signal_snapshot


def _observed_frame_with_tail_gap() -> tuple[pd.DataFrame, int]:
    periods = 120
    gap_pos = periods - 2
    dates = pd.date_range("2015-01-31", periods=periods, freq="ME")
    levels = 100.0 * (1.002 ** np.arange(periods, dtype=float))
    levels[gap_pos] = np.nan
    raw = pd.DataFrame(
        {
            "date": dates,
            "CPIAUCSL": levels,
            "TB3MS": np.linspace(0.5, 4.0, periods),
        }
    )
    return build_base_frame(raw, imputation_policy="observed_only"), gap_pos


def test_dashboard_views_restore_only_the_descriptive_current_signal() -> None:
    observed, gap_pos = _observed_frame_with_tail_gap()

    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
    )

    assert views.research_raw["imputation_policy"].eq("observed_only").all()
    assert pd.isna(views.research_raw.loc[gap_pos, "cpi_level"])
    assert not views.research_raw["cpi_imputed"].any()
    assert views.current_raw["imputation_policy"].eq("ex_post_continuity").all()
    assert views.current_raw.loc[gap_pos, "cpi_imputed"]
    assert pd.notna(views.current_raw.loc[gap_pos, "cpi_level"])

    research_snapshot = latest_signal_snapshot(views.research_featured)
    current_snapshot = views.current_snapshot
    assert current_snapshot["available"]
    assert current_snapshot["date"] == observed["date"].iloc[-1]
    assert current_snapshot["date"] > research_snapshot["date"]


def test_current_signal_keeps_derived_imputation_lineage_and_notice() -> None:
    observed, _ = _observed_frame_with_tail_gap()
    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
    )

    snapshot = views.current_snapshot
    assert snapshot["baseline_uses_imputed_input"]
    assert snapshot["epsilon_uses_imputed_input"]
    assert snapshot["tinf_4m_uses_imputed_input"]
    assert snapshot["percentile_uses_imputed_input"]
    assert snapshot["regime_uses_imputed_input"]
    assert snapshot["uses_imputed_input"]
    assert not snapshot["observed_only_eligible"]
    assert current_signal_imputation_notice(snapshot) == CURRENT_SIGNAL_IMPUTATION_NOTICE


def test_dashboard_research_consumers_remain_observed_only() -> None:
    observed, _ = _observed_frame_with_tail_gap()
    views = build_dashboard_data_views(
        observed,
        baseline_method="rolling_36_shifted",
    )

    historical = validation_mod.build_historical_validation_frame(
        views.research_featured,
        forward_horizons=(3,),
        label_horizons=(3,),
    )
    assert views.research_featured["imputation_policy"].eq("observed_only").all()
    assert not views.research_featured["signal_uses_imputed_input"].any()
    assert not historical["signal_uses_imputed_input"].any()
    assert current_signal_imputation_notice(
        latest_signal_snapshot(views.research_featured)
    ) is None


def test_dashboard_views_reject_ex_post_data_as_research_authority() -> None:
    observed, _ = _observed_frame_with_tail_gap()
    observed["imputation_policy"] = "ex_post_continuity"

    with pytest.raises(ValueError, match="research authority.*observed_only"):
        build_dashboard_data_views(
            observed,
            baseline_method="rolling_36_shifted",
        )

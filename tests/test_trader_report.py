from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transitory_inflation.features import add_transitory_inflation_features
from transitory_inflation.report import (
    REGIME_PLAYBOOK,
    TraderReport,
    build_trader_report,
    next_print_flip_threshold,
)


def _raw_frame(months: int = 160, yoy: float | None = None) -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=months, freq="ME")
    values = yoy if yoy is not None else 3.0 + np.sin(np.linspace(0, 12, months))
    return pd.DataFrame(
        {"date": dates, "cpi_level": 100.0, "tbill_3m": 4.0, "inflation_yoy": values}
    )


def test_playbook_covers_all_snapshot_regimes() -> None:
    assert set(REGIME_PLAYBOOK) == {
        "elevated rising",
        "elevated falling",
        "neutral",
        "disinflationary",
    }


def test_report_builds_and_is_structured() -> None:
    raw = _raw_frame()
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")
    report = build_trader_report(raw, df, "fed_target", "live_dashboard", decay_windows=(24,))

    assert isinstance(report, TraderReport)
    assert report.available
    assert report.headline
    assert report.state_lines and report.persistence_lines and report.robustness_lines
    labels = [label for label, _ in report.playbook]
    assert "Macro read" in labels
    assert report.playbook[-1][0] == "Term-structure modifier"
    assert any("not investment advice" in caveat for caveat in report.caveats)


def test_report_flags_ex_post_baseline() -> None:
    raw = _raw_frame()
    df = add_transitory_inflation_features(raw, baseline_method="full_sample")
    report = build_trader_report(raw, df, "full_sample", "paper_replication", decay_windows=(24,))

    assert report.available
    assert any("EX-POST" in caveat for caveat in report.caveats)


def test_report_unavailable_without_complete_rows() -> None:
    raw = _raw_frame(months=10)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")
    report = build_trader_report(raw, df, "fed_target", "live_dashboard", decay_windows=(24,))

    assert not report.available
    assert report.reason


def test_flip_threshold_fed_target() -> None:
    # Constant 3% YoY against the 2% target: eps = +1pp each month, so the next
    # print must land at 2.0 - 3.0 = -1.0% to zero out the 4-month average.
    raw = _raw_frame(yoy=3.0)
    df = add_transitory_inflation_features(raw, baseline_method="fed_target")
    assert next_print_flip_threshold(df, "fed_target") == pytest.approx(-1.0)


def test_flip_threshold_undefined_for_ex_post_baselines() -> None:
    raw = _raw_frame(yoy=3.0)
    df = add_transitory_inflation_features(raw, baseline_method="full_sample")
    assert next_print_flip_threshold(df, "full_sample") is None

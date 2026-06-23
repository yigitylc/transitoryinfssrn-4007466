from __future__ import annotations

import numpy as np
import pandas as pd

from transitory_inflation.diagnostics import ljung_box_table, stationarity_diagnostics


def test_ljung_box_table_structure() -> None:
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=200))
    out = ljung_box_table(series, lags=10)
    assert "lag" in out.columns
    assert "lb_pvalue" in out.columns
    assert len(out) == 1
    assert int(out["lag"].iloc[0]) == 10
    p_value = float(out["lb_pvalue"].iloc[0])
    assert 0.0 <= p_value <= 1.0


def test_ljung_box_table_short_series_fallback() -> None:
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(size=15))
    # lags=40 is larger than the series allows; the table should fall back to a
    # smaller lag rather than raise.
    out = ljung_box_table(series, lags=40)
    assert len(out) == 1
    assert int(out["lag"].iloc[0]) == 5


def test_stationarity_diagnostics_has_adf_and_kpss() -> None:
    rng = np.random.default_rng(2)
    series = pd.Series(rng.normal(size=120))
    out = stationarity_diagnostics(series)
    null_by_test = dict(zip(out["test"], out["null"], strict=True))
    assert {"ADF", "KPSS"} <= set(null_by_test)
    assert null_by_test["ADF"] == "unit root"
    assert null_by_test["KPSS"] == "stationary"


def test_stationarity_diagnostics_short_series_is_empty() -> None:
    # Fewer than 20 observations is not enough to run ADF/KPSS, so the table is
    # empty rather than reporting unreliable statistics.
    out = stationarity_diagnostics(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert out.empty

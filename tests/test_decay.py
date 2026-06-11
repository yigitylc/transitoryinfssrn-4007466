from __future__ import annotations

import numpy as np
import pandas as pd

from transitory_inflation.models import decay_curve, rolling_ar1_rho


def test_decay_curve_valid_values() -> None:
    curve = decay_curve(rho_T=1.15, mu=0.93, months=12)
    assert len(curve) == 12
    assert curve["decay_pct"].notna().all()


def test_decay_curve_invalid_values() -> None:
    curve = decay_curve(rho_T=-0.1, mu=0.93, months=12)
    assert curve["decay_pct"].isna().all()


def test_rolling_ar1_alignment() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=36, freq="ME"),
            "tinf_4m": np.linspace(0.0, 1.0, 36),
        }
    )
    rho = rolling_ar1_rho(df, window=24)
    assert rho["date"].iloc[0] == df["date"].iloc[23]

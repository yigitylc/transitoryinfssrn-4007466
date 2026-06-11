from __future__ import annotations

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss


def ljung_box_table(series: pd.Series, lags: int = 40) -> pd.DataFrame:
    """White-noise/autocorrelation diagnostic."""

    clean = series.dropna()
    if len(clean) <= lags + 1:
        lags = max(1, min(10, len(clean) // 3))
    result = acorr_ljungbox(clean, lags=[lags], return_df=True)
    result = result.reset_index(names="lag")
    return result


def stationarity_diagnostics(series: pd.Series) -> pd.DataFrame:
    """ADF and KPSS diagnostics in a compact table."""

    clean = series.dropna()
    rows = []

    if len(clean) >= 20:
        adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC")
        rows.append({"test": "ADF", "statistic": adf_stat, "p_value": adf_p, "null": "unit root"})
        try:
            kpss_stat, kpss_p, *_ = kpss(clean, regression="c", nlags="auto")
            rows.append({"test": "KPSS", "statistic": kpss_stat, "p_value": kpss_p, "null": "stationary"})
        except Exception as exc:
            rows.append({"test": "KPSS", "statistic": None, "p_value": None, "null": f"failed: {exc}"})

    return pd.DataFrame(rows)

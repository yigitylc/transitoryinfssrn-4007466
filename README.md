# Transitory Inflation Dashboard — SSRN 4007466

A Python/Streamlit implementation of the inflation framework from **Peron & Bonaparte, *Transitory Inflation and Projection of Future Inflation***, SSRN 4007466.

The project builds a local research dashboard that tracks U.S. CPI inflation relative to a mean-reversion baseline, computes short/medium/long transitory-inflation signals, estimates persistence, and summarizes the current inflation regime using live FRED data.

## Reference

**Paper:** Peron & Bonaparte, *Transitory Inflation and Projection of Future Inflation*
**Reference ID:** SSRN 4007466

The dashboard implements the paper’s core idea: inflation deviations from a mean-reversion baseline can be averaged over short, medium, and long windows to evaluate whether inflation pressure is fading, persisting, or re-accelerating.

## What the dashboard does

The app computes:

* CPI year-over-year inflation
* Mean-reversion baseline inflation
* Inflation deviation from baseline
* TINF 4M, 8M, and 12M signals
* Percentile/regime classification
* Short-term pressure across TINF horizons
* Rolling AR(1) persistence
* Paper-style decay/convergence estimates
* Robustness views across baseline definitions and sample windows

## Data

The dashboard uses FRED data:

* `CPIAUCSL` — Consumer Price Index for All Urban Consumers
* `TB3MS` — 3-Month Treasury Bill, used as a short-rate proxy

Data source priority:

1. Official FRED observations API using optional `FRED_API_KEY`
2. Public FRED CSV endpoints when the key is missing or the API request fails
3. Local cached raw FRED files under `data/raw/`
4. Clearly labeled demo data only as an emergency fallback

`FRED_API_KEY` is optional but recommended. Put real keys only in a local `.env`
file; `.env` is ignored and must not be committed.

## Methodology

The core signal is based on the paper’s transitory inflation concept:

```text
epsilon_t = inflation_t - baseline_t
TINF_n,t = n-month average of epsilon
```

where `n` is typically:

* `4M` for short-term transitory inflation
* `8M` for medium-term transitory inflation
* `12M` for long-term transitory inflation

The dashboard supports several baseline choices:

* `full_sample` — long-run average within selected sample
* `rolling_36_unshifted` — current 36-month rolling mean
* `rolling_36_shifted` — prior 36-month rolling mean, used as the default live-safe baseline
* `expanding_shifted` — expanding historical mean known as of the prior month
* `fed_target` — fixed 2% reference baseline

## Sample modes

The app separates historical implementation windows from live monitoring:

* `paper_window` — fixed historical window aligned with the reference study
* `live_dashboard` — 1982 onward through latest available FRED data
* `max_history` — maximum available FRED history

The default dashboard mode is `live_dashboard`.

## Important implementation notes

The live dashboard uses a trailing-current TINF calculation so the latest available CPI print is reflected in the current signal. 

The decay/convergence module follows the paper-style decay formula. The paper estimates an AR(1) process for rolling persistence and then applies a published decay function based on the latest persistence estimate and decay coefficient.

The dashboard is designed to fetch live FRED data. The app should distinguish between:

* latest raw CPI observation date
* latest raw short-rate observation date
* latest usable signal date after YoY/TINF/baseline calculations

These dates may differ when CPI has a trailing unpublished or incomplete observation.



## Status
This is a research implementation of a published inflation framework. It is intended for macro/inflation analysis, model inspection, and dashboard-based monitoring.

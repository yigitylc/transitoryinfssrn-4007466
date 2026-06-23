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
* Historical forward-outcome validation of the regime signal
* Benchmark comparison against no-change, CPI persistence, mean reversion, and AR(1)
* Robustness views across baselines, sample windows, inflation measures, horizons, and thresholds
* Descriptive market linkage to Treasury yields, breakevens, and real yields
* A synthesized macro research report

## Data

The dashboard uses FRED data.

Inflation measures (headline CPI is the paper/default measure; core CPI, PCE, and
core PCE are robustness checks, not paper-exact replication):

* `CPIAUCSL` — Headline CPI (Consumer Price Index for All Urban Consumers)
* `CPILFESL` — Core CPI (ex food and energy)
* `PCEPI` — PCE price index
* `PCEPILFE` — Core PCE price index

Short-rate control:

* `TB3MS` — 3-Month Treasury Bill secondary-market rate, used as a short-rate proxy

Market-linkage series (approved rates-only set, used only for the descriptive
Phase 4 market-linkage layer — never as a trading signal):

* `DGS2`, `DGS10` — 2Y and 10Y Treasury yields
* `T5YIE`, `T10YIE` — 5Y and 10Y breakeven inflation
* `DFII5`, `DFII10` — 5Y and 10Y real yields

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

* `paper_replication` — fixed 1982-01 to 2021-07 window aligned with the reference study
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



## Dashboard tabs and research phases

The app is organized into tabs that map to the project's research phases. All
layers are decision-support/research only — not a trading system, PnL backtest,
or buy/sell recommender.

* **Current Macro Signal** — latest live regime snapshot under the selected mode and baseline.
* **Historical Signal Validation** (Phase 1) — forward-outcome tables by regime and short-term pressure, threshold sensitivity, and worked examples.
* **Benchmark Comparison** (Phase 2) — TINF/regime vs no-change, CPI persistence, mean reversion, and AR(1) on MAE/RMSE/directional/hit-rate metrics.
* **Market Linkage** (Phase 4) — descriptive history of how Treasury yields, breakevens, and real yields moved after past TINF/regime states.
* **Paper Framework** — paper-style descriptive moments, correlations, and robust OLS regressions.
* **Decay / Convergence** — rolling AR(1) persistence and the paper-style decay curve.
* **Robustness** (Phase 3) — scorecard across baselines, sample modes, inflation measures, horizons, and thresholds.
* **Macro Research Report** (Phase 5) — a synthesized report combining current regime, signal confidence, robustness, historical analogs, market linkage, caveats, and a watchlist.

## Status
This is a research implementation of a published inflation framework. It is intended for macro/inflation analysis, model inspection, and dashboard-based monitoring.

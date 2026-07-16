# Production Roadmap

This roadmap is the canonical production plan for the project. `NEXT_TASKS.md`
tracks only the active gate. Do not start later phases until the current gate
passes and is committed.

## Phase 0 - Production Stabilization Before Commit

Goal: make the current dashboard stable, honest, reproducible, and safe for the
first production commit.

Required gate:

- The Streamlit app starts cleanly from the project root.
- Official FRED API loading works when `FRED_API_KEY` is present in the
  project-root `.env`.
- Data-source priority remains `fred_api -> fred_csv -> cached_fred -> demo`.
- Cached FRED fallback uses the same CPI cleaning, monthly preparation,
  isolated one-month imputation, YoY, baseline, and TINF feature path as live
  API/CSV data.
- Cached fallback does not trust precomputed `inflation_yoy` or `cpi_imputed`
  columns in raw cache files.
- If cleaned CPI data supports a 2026-04 signal reference month, the latest valid
  signal reference month must not remain stuck at 2025-09.
- Data status clearly shows:
  - source used
  - cache path when applicable
  - raw data end date
  - latest CPI observation date
  - latest valid CPI YoY date
  - latest valid signal reference month
  - latest trustworthy full signal-information timestamp and timing status
  - CPI imputation status
- Historical validation columns must be explicitly `historical_*` and must not
  overwrite live/current regime labels.
- Positive inflation-shock outcome logic must be added:
  - for positive `epsilon_t`, crossing below baseline should be treated as
    resolved or downside overshoot, not persistent inflation.
  - absolute distance to baseline remains a secondary diagnostic, not the
    primary positive-shock persistence label.
  - false transitory examples must use positive-shock logic, not only absolute
    distance to baseline.
- Dashboard explanation text must distinguish:
  - inflation direction / positive-shock resolution
  - absolute distance from baseline / equilibrium normalization
- Safe commit allowlist must exclude `.env`, `.venv/`, raw cache data,
  extracted paper text, generated logs, caches, `__pycache__/`, and the
  third-party PDF unless intentionally committed.

Exit criteria:

- `ruff check .` passes.
- `pytest` passes.
- `python -m compileall src app scripts` passes.
- Streamlit app has been restarted and confirmed to load without stale import
  errors.
- Git status and ignore checks confirm no secrets, local raw data, extracted
  paper text, generated logs, caches, virtualenv files, pycache, or unintended
  third-party files are staged.

## Phase 1 - Historical Validation Polish

Goal: make Phase 1 validation clearer and more decision-useful without adding
complex visuals.

Required work:

- Add a combined `historical_regime x historical_short_term_pressure` table.
- Add threshold sensitivity tables for `0.25`, `0.50`, `0.75`, and `1.00`
  percentage-point thresholds.
- Improve example categories:
  - successful transitory
  - successful transitory with downside overshoot
  - false transitory / persistent inflation
  - successful persistent
  - false persistent / shock faded
- Improve dashboard explanations for validation definitions and limitations.
- Keep visuals table-first; do not add complex charts yet.

## Phase 2 - Benchmark Comparison

Goal: test whether the TINF/regime signal adds information relative to simple
forecast baselines.

Compare TINF/regime signal performance against:

- no-change CPI forecast
- CPI persistence
- mean reversion to baseline
- simple AR(1)

Use these metrics:

- MAE
- RMSE
- directional accuracy
- hit rate
- false positive / false negative rates
- confusion matrix

## Phase 3 - Robustness

Goal: test whether conclusions survive reasonable methodological alternatives.

Required robustness dimensions:

- CPI vs core CPI vs PCE/core PCE.
- `rolling_36_shifted` vs `expanding_shifted`.
- trailing-current TINF vs paper-lagged formula.
- horizon and threshold sensitivity.

## Phase 4 - Market Linkage

Goal: connect inflation-regime information to market variables only after Phase
2 confirms the signal is useful.

Potential market links:

- 2Y and 10Y Treasury yields
- breakevens
- real yields
- Fed funds expectations
- DXY
- gold
- oil
- SPY/QQQ

Do not start this phase until Phase 2 establishes that the historical signal
has useful forward information.

## Phase 5 - Macro Research Report


Goal: synthesize the full dashboard into a practical macro research report that explains the current inflation regime, historical evidence, model confidence, robustness, market linkage, caveats, and data freshness.

This phase is a decision-support and reporting layer. It is not a trading strategy, not a buy/sell recommendation, and not a PnL backtest.

The report should answer:

What is the current inflation regime?
Is inflation pressure rising, cooling, neutral, or disinflationary?
How reliable has this signal been historically?
Does the signal beat simple benchmark models?
Is the result robust across horizons, thresholds, baselines, and inflation measures?
What historically happened to nominal yields, breakevens, and real yields after similar regimes?
What are the main caveats and model risks?
What should be monitored next?
Required sections
1. Current Regime

Show the current live macro signal:

latest valid signal reference month
full signal-information timestamp and timing status
data source used
inflation measure
sample mode
baseline method
current CPI / inflation reading
current TINF readings
current regime
current short-term pressure
latest data freshness status
CPI imputation status, if applicable

This section should be written as a quick “state of the world” snapshot.

2. Signal Confidence

Summarize Phase 2 benchmark comparison.

This section must clearly state whether TINF/regime beats:

no-change CPI forecast
CPI persistence
mean reversion to baseline
AR(1)

The report must not overstate predictive power. If AR(1) beats TINF/regime as a pure CPI point-forecast model, the report should say so clearly.

Suggested framing:

The TINF/regime signal may be useful as an interpretable inflation-regime diagnostic, but it does not robustly dominate simple AR(1) benchmarks as a CPI point-forecast model.

3. Robustness Summary

Summarize Phase 3 results across:

horizons: 3M, 6M, 12M, 24M, 36M
thresholds: 0.25, 0.50, 0.75, 1.00 pp
baselines: rolling_36_shifted, expanding_shifted, full_sample where applicable
inflation measures: headline CPI, core CPI, PCE, core PCE

This section should answer:

Does the conclusion survive reasonable methodological choices?
Is the signal only useful under one narrow setting?
Does the signal behave differently for core inflation or PCE?
4. Historical Analogs

Identify similar historical periods based on:

current regime
current short-term pressure
TINF state
inflation measure, where applicable

For each analog group, show what happened afterward:

future CPI / inflation change
future epsilon change
2Y yield change
10Y yield change
5Y / 10Y breakeven change
5Y / 10Y real yield change
count of observations
weak-evidence warning if sample size is low

This section should be descriptive and historical, not a trading recommendation.

5. Market Linkage Summary

Summarize Phase 4 market linkage.

Group market variables by channel:

nominal rates: 2Y and 10Y Treasury yields
breakevens: 5Y and 10Y inflation breakevens
real yields: 5Y and 10Y real yields

Explain:

A positive forward change means the market variable rose after that series' eligible market
origin, determined by the full signal information timestamp or the explicitly labelled proxy path.
Nominal yields, breakevens, and real yields represent different macro channels.
Market linkage can be useful even if TINF/regime is not the best CPI point-forecast model.
Market linkage is descriptive and historical, not a live trading signal.
6. Caveats / Model Risk

This section is required.

Include caveats such as:

This dashboard is historical research, not a trading signal.
TINF/regime does not robustly beat AR(1) as a CPI point-forecast model in current tests.
Market linkage is descriptive and may not hold out of sample.
Some regime buckets may have small sample sizes.
Full-sample baselines are ex-post / paper-style and are not row-lookahead-safe.
Row-lookahead safety, reference-month-only timing, release alignment, vintage safety,
and ex-post continuity are distinct; latest-revised FRED data are non-vintage.
Data freshness and timing status matter.
FRED publication lags can affect the latest trustworthy full signal-information
timestamp without changing the CPI reference month.
CPI imputation, if used, should be disclosed.
Future market changes are used only for evaluation, not signal construction.
7. Watchlist / What to Monitor Next

Show practical monitoring items:

next CPI release
next PCE release
change in TINF direction
change in short-term pressure
whether elevated inflation pressure is resolving or persisting
2Y yield reaction
10Y yield reaction
breakeven inflation reaction
real yield reaction
whether benchmark confidence improves or deteriorates
Phase 5 exit criteria

Phase 5 is complete when the dashboard has a clear research-report layer that:

synthesizes the current signal, validation, benchmarks, robustness, and market linkage
clearly separates signal interpretation from forecast accuracy
clearly separates descriptive market linkage from trading recommendations
includes caveats and data freshness
does not add PnL, trading rules, strategy backtests, or buy/sell recommendations
passes ruff, pytest, compileall, and Streamlit smoke checks

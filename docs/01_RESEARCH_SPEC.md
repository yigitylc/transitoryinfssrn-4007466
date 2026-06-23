# Research Specification

## Core concept

Define inflation deviation from a baseline:

```text
epsilon_t = inflation_yoy_t - baseline_t
```

Define transitory inflation as rolling average deviations:

```text
tinf_4m  = rolling_mean(epsilon, 4)
tinf_8m  = rolling_mean(epsilon, 8)
tinf_12m = rolling_mean(epsilon, 12)
```

The 4-month version is the paper's short-term transitory inflation proxy.

## Modes

Each research mode maps to a named sample mode defined in
`src/transitory_inflation/config.py` (see `docs/02_DATA_CONTRACT.md` for the
loading rules):

- Paper replication mode -> `paper_replication` (1982-01-01 through 2021-07-31, ex-post)
- Live signal mode, no full-sample lookahead -> `live_dashboard` (1982-01-01 through latest FRED; Streamlit default)
- Robustness checks -> `max_history` (earliest FRED through latest; not necessarily the default trading signal)

### 1. Paper replication mode

Purpose: reproduce the paper's tables/figures as closely as practical.

Sample: `paper_replication`, fixed 1982-01-01 through 2021-07-31. The slice is
applied at load time, before any baseline is computed, so the `full_sample`
baseline cannot see post-2021 data.

Expected outputs:

- CPI and TINF summary statistics
- correlation matrix
- CPI/TINF regressions with a T-bill control (implemented as 3-month `TB3MS`
  because FRED has no 1-month bill history before 2001-07 — disclosed deviation
  from the paper's stated 1-month control)
- white-noise/autocorrelation diagnostics
- rolling AR(1) persistence estimates
- decay/convergence table

### 2. Live signal mode (no full-sample lookahead)

Purpose: make the signal usable as a current macro indicator. "Live-safe" here
means no full-sample lookahead; it is not a real-time data-vintage backtest (the
loader uses latest-revised FRED data, not the data as first released).

Sample: `live_dashboard`, 1982-01-01 through the latest available FRED data.
Use `max_history` to check robustness of the same outputs over the longest
available sample.

Requirements:

- avoid full-sample lookahead
- prefer shifted rolling or shifted expanding baselines
- show what data was known at each point
- label results as live-safe or ex-post
- treat live-safe as no full-sample lookahead on latest-revised data, not a
  real-time data-vintage backtest

#### Historical signal validation

Status: research upgrade, not paper replication.

The validation layer asks whether the current-month TINF/regime signal contained
forward information about inflation persistence. Signal columns are constructed
first, then future CPI outcomes are appended only for scoring historical rows.
Future values must never feed back into baseline, epsilon, TINF, pressure, or
regime construction.

Default live-like settings:

- `sample_mode = live_dashboard`
- `baseline_method = rolling_36_shifted`
- historical regime thresholds use expanding shifted TINF 4M quantiles

Mechanical outcome definitions:

- baseline normalized: `abs(epsilon_(t+h)) <= 0.50pp`
- Fed target normalized: `abs(inflation_yoy_(t+h) - 2.00) <= 0.50pp`
- partial decay 50: `abs(epsilon_(t+h)) <= 0.50 * abs(epsilon_t)`
- partial decay 80: `abs(epsilon_(t+h)) <= 0.20 * abs(epsilon_t)`
- persistent: not baseline-normalized and not partial-decay-50, only when the
  current gap is large enough to make decay meaningful
- reaccelerated: CPI YoY rises by at least the configured threshold over the
  horizon

Rows where `abs(epsilon_t)` is near zero are not treated as meaningful shock
decay events; decay ratios and persistence labels are left undefined for those
rows.

`full_sample` is acceptable for ex-post paper-style replication but not for
judging live signal success, because its baseline and percentile/regime cutoffs
can use information unavailable at month t.

### 3. Trader research mode

Purpose: study relationships between inflation persistence and markets.

Possible assets/features:

- policy rates / T-bills
- Treasury yields
- breakevens / real yields
- SPY, QQQ, TLT, GLD, DXY, VIX
- value/growth and sector proxies

Do not treat this as a trading system unless explicitly requested.

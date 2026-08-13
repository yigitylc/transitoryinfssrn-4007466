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

Paper reconstruction uses official observed-only CPI. The October 2025 current-monitoring
scenarios are outside the fixed paper sample and may not enter the reconstruction path.

Expected outputs:

- CPI and TINF summary statistics
- correlation matrix
- CPI/TINF regressions with a T-bill control (implemented as 3-month `TB3MS`
  because FRED has no 1-month bill history before 2001-07 — disclosed deviation
  from the paper's stated 1-month control)
- white-noise/autocorrelation diagnostics
- rolling AR(1) persistence estimates
- decay/convergence table

### 2. Live signal mode (current monitoring; latest-revised and non-vintage)

Purpose: make the signal usable as a current macro indicator while keeping current monitoring
separate from historical evidence. A shifted baseline is row-lookahead-safe as a formula, but a
current signal that uses an October 2025 scenario is explicitly ex-post because the scenario bundle
uses the following November observation. That current signal is not live-like, release-aligned, or
vintage-safe. The loader uses latest-revised FRED data, and missing CPI release metadata remains
explicitly `reference_month_only`. Exact publication/information timing also requires explicit
provenance and a timezone-bearing timestamp; timestamp times are not reduced to calendar dates.

Sample: `live_dashboard`, 1982-01-01 through the latest available FRED data.
Use `max_history` to check robustness of the same outputs over the longest
available sample.

Requirements:

- avoid full-sample lookahead
- prefer shifted rolling or shifted expanding baselines
- keep the official October 2025 headline/core CPI levels null in raw data and caches
- derive observed and current-scenario views from one full raw warm-up superset before applying the
  selected sample window
- fail closed with a contract error when that common warm-up authority is absent or truncated; never
  substitute the already-sliced research frame
- classify current scenarios against prior observed-only eligible history from the selected sample
  mode; never let an estimate change the threshold population
- show only timing metadata actually supplied for each point and fail closed when it is unavailable
- label row-lookahead-safe and ex-post construction separately from timing/vintage status
- expose reference month, actual publication/information time when available, and the
  latest-revised non-vintage limitation

#### Current-monitoring scenarios

For the permanently unavailable October 2025 headline and core CPI levels, current-facing
descriptive surfaces use three explicit assumptions:

- `low`: October equals September;
- `base`: October is the geometric midpoint (log-linear bridge) of September and November;
- `high`: October equals November.

The assumptions are applied separately to headline and core CPI. The base bridge and endpoint
scenarios are unofficial, ex-post, latest-revised, and non-vintage; they are not probabilities or
confidence intervals. They may feed the Sidebar Current Reading, Current Macro Signal, current
TINF/regime/pressure, Trader Research current bucket, report headline/regime/watchlist, and the
separately labelled current baseline comparison.

September and November mean exactly 2025-09-30 and 2025-11-30 for each series. Physical adjacency
cannot substitute December or another row. Each endpoint must be one coherent normalized official
observation with a present value, approved value provenance, and no estimated/imputed lineage.
Official value authority is separate from release-time precision: a value may be used with
conservative `reference_month_only` timing, while its actual H5 status remains disclosed. Either
series failing any requirement makes the complete bundle unavailable and suppresses all current
classifications while preserving diagnostic lineage.

The dashboard reports every scenario's values and classification. It reports `regime_stable` only
when all three regimes agree, `pressure_stable` only when all three pressures agree, and
`fully_stable` only when both conditions hold. A disagreement is `scenario_sensitive`; if any
scenario cannot be calculated, the aggregate state is `unavailable`. No modal state or scenario
probability is reported.

`live_dashboard` and `max_history` each use their own selected-mode observed-only calibration
history. Changing the sample mode may therefore change thresholds and classifications, but not the
scenario formulas. Current-monitoring and historical-evidence endpoints are displayed separately.
All historical consumers and calibration use one fail-closed eligibility rule that treats
`uses_estimated_input=True` as authoritative contamination. Paper Window ends before the affected
month, so its scenario status is `not_applicable`; it keeps its observed-only calculations and does
not display low/base/high stability.

#### Historical signal validation

Status: research upgrade, not paper replication.

The validation layer asks whether the current-month TINF/regime signal contained
forward information about inflation persistence. Signal columns are constructed
first, then future CPI outcomes are appended only for scoring historical rows.
Future values must never feed back into baseline, epsilon, TINF, pressure, or
regime construction.

Historical validation, hit rates, benchmark scoring, historical robustness, historical
market-linkage origins, and historical analog populations remain strictly observed-only. A signal
origin or outcome that depends on estimated CPI is excluded with its reason and denominator impact
retained.

Default row-lookahead-safe settings:

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

Purpose: study, descriptively, how markets behaved after past inflation-persistence
states.

Scope (decided 2026-06-24; current-bucket routing amended 2026-07-16): descriptive only and
rates-only. It ships as the **Trader Research** Streamlit tab, which may condition on an explicitly
estimated current scenario bucket. That current bucket uses thresholds calibrated only from prior
observed-only eligible history. It queries the strict observed-only walk-forward analog population
(`historical_regime` / `historical_short_term_pressure`)
and reports, for the six approved FRED rate instruments, the forward-change
distribution (median, p25-p75, increase/decrease hit rates, sample count,
weak-evidence) plus the analog months behind it. It reuses the Phase 4 market-linkage
tables (`market_linkage.build_market_linkage_tables`) and adds no new market series.
The current estimate never enters the analog population or historical market-linkage origins. If
the low/base/high buckets differ, their conditional results remain separately selectable (and the
report lists them separately) rather than being collapsed to a modal bucket.
Exact market measurement begins at the first eligible market-close timestamp greater than or equal
to the full signal information timestamp, but only when both timestamps have explicit trustworthy
status, provenance, and timezone information. Standard FRED rate observations are date-only and
therefore cannot establish an exact close: when signal information timing is trustworthy, they use
the first observation date after that information timestamp as a clearly labelled conservative
proxy. Rows without trustworthy signal information timing use the clearly labelled conservative
month-end `t+1` proxy. Each required market series is routed independently before the row-level
exact, proxy, mixed, partial, or unavailable status is summarized.
None of these paths is a vintage backtest because the macro and market values are latest-revised
FRED data.

In scope (approved FRED rates only):

- 2Y / 10Y Treasury yields
- 5Y / 10Y breakeven inflation
- 5Y / 10Y real yields

Explicitly out of scope (future, not implemented):

- predictive / out-of-sample forecast scoring of market moves
- any PnL, sizing, timing, instruments, or trade recommendations
- other assets (e.g. SPY, QQQ, TLT, GLD, DXY, VIX, value/growth, sector proxies),
  which would require expanding the approved market registry

Do not treat this as a trading system unless explicitly requested.

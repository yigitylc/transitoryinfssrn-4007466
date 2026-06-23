# Paper Formula Reference

Reference PDF: `references/ssrn-4007466.pdf`

Local extraction command:

```powershell
python scripts/extract_reference_paper.py
```

The command writes full extracted text to
`artifacts/exports/ssrn-4007466_extracted.txt`. That artifact is local-only and
gitignored; do not commit full extracted copyrighted text if this repository may
become public.

## Formula Status Legend

- `paper exact`: directly verified from the extracted paper text.
- `paper ambiguous`: the paper supports the concept, but wording or appendix
  definitions leave implementation choices open.
- `research upgrade`: a project extension for live/research use, not part of the
  paper replication contract.

## Implementation Formulas

### CPI AR(1) Equation

Status: `paper exact`

The paper motivates inflation persistence with an AR(1):

```text
CPI_t = c + rho * CPI_(t-1) + epsilon_t
```

where `c` is a constant, `rho` is serial correlation, and `epsilon_t` is the
innovation/error term.

Implementation note: the project uses CPI YoY in percentage points as the
inflation variable.

### Deviation From Mean-Reversion Inflation

Status: `paper ambiguous`

The core deviation is:

```text
epsilon_t = CPI_t - CPI_bar
```

where `CPI_bar` is described as mean inflation or historical mean-reversion
inflation. The appendix also describes the created transitory inflation variables
relative to a 36-month moving average.

Implementation note: the project therefore keeps baseline methods explicit:
`full_sample`, `rolling_36_unshifted`, `rolling_36_shifted`,
`expanding_shifted`, and `fed_target`.

### TINF 4M / 8M / 12M

Status: `paper exact`

The paper defines transitory inflation as an `n`-period moving average of the
deviation term:

```text
TINF_t(n) = (1 / n) * sum_{i=1..n} epsilon_(t-i-1)
```

The extracted formula uses a lagged summation index. In implementation, this is
represented as a strict rolling average of the current epsilon series:

```text
tinf_4m  = rolling_mean(epsilon, 4)
tinf_8m  = rolling_mean(epsilon, 8)
tinf_12m = rolling_mean(epsilon, 12)
```

Interpretation:

- `n = 4`: short-term transitory inflation.
- `n = 8`: medium-term transitory inflation.
- `n = 12`: long-term transitory inflation.

### Rolling Rho AR(1)

Status: `paper exact`

For the short-term transitory inflation series, the paper estimates:

```text
ST_TINF_t = constant + rho * ST_TINF_(t-1) + epsilon_t
```

Because `rho` varies over time, the paper estimates this equation with rotating
windows to produce a time series:

```text
rho_1, rho_2, ..., rho_T
```

### AR(1) On Rolling Rho

Status: `paper exact`

The paper then models the rolling-rho sequence as:

```text
rho_t = c + mu * rho_(t-1) + epsilon_t
```

Implementation note: the paper-style decay formula below uses `rho_T` and `mu`.
Do not change the current model formulas in this documentation task.

### Paper-Style Decay Formula

Status: `paper exact`

If rho is treated as stationary, the paper first presents:

```text
Decay Rate % = 100 * (1 - rho_t)
```

After modeling time-varying rho, the paper-style policy function is:

```text
Decay Rate %_t = 100 * (1 - rho_T * mu^(t-1))
```

The paper reports semiannual and annual decay with:

```text
t = 6
t = 12
```

### Convergence Threshold

Status: `paper exact`

The paper defines convergence back to regular/mean-reversion inflation as a very
high decay rate:

```text
Decay Rate % > 95%
```

Solving the paper-style decay expression for the number of months to convergence:

```text
t_star = 1 + log(0.05 / rho_T) / log(mu)
t_star_years = t_star / 12
```

## Research Upgrades Not In The Paper

Status: `research upgrade`

These are live-dashboard additions rather than paper-exact replication steps:

- Named sample modes: `paper_replication`, `live_dashboard`, and `max_history`.
- Live-safe shifted baselines such as `rolling_36_shifted` and
  `expanding_shifted`.
- Fixed `fed_target` baseline for policy-target interpretation.
- Single-month interior CPI level imputation with a `cpi_imputed` flag.
- Current-signal snapshots that skip trailing rows with incomplete CPI/TINF
  fields.
- Historical signal validation: after signal columns are computed, future
  outcomes are appended with `shift(-h)` for horizons 3/6/12/24/36 months.
  These columns are validation-only and must not feed signal construction.
- Walk-forward historical regime labels: expanding shifted TINF 4M quantiles
  are used for historical validation, avoiding full-sample percentile
  lookahead.

### Historical Validation Labels

Status: `research upgrade`

For a validation horizon `h`, Phase 1 uses mechanical labels:

```text
cpi_yoy_change_h = cpi_yoy_(t+h) - cpi_yoy_t
epsilon_change_h = epsilon_(t+h) - epsilon_t
gap_decay_ratio_h = abs(epsilon_(t+h)) / abs(epsilon_t)
```

```text
baseline_normalized_h = abs(epsilon_(t+h)) <= 0.50
fed_target_normalized_h = abs(cpi_yoy_(t+h) - 2.00) <= 0.50
partial_decay_50_h = abs(epsilon_(t+h)) <= 0.50 * abs(epsilon_t)
partial_decay_80_h = abs(epsilon_(t+h)) <= 0.20 * abs(epsilon_t)
persistent_h = not baseline_normalized_h and not partial_decay_50_h
reaccelerated_h = cpi_yoy_change_h >= configured threshold
```

When `abs(epsilon_t)` is below the configured threshold, the decay ratio and
decay-based persistence labels are undefined rather than forced to a misleading
finite value.

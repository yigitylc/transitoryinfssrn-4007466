# Data Contract

## Canonical monthly data frame

Preferred index/columns:

```text
date                  backward-compatible alias for reference_month
reference_month       economic month measured by the CPI observation
release_timestamp     actual publication time, populated only from release metadata
release_timestamp_provenance actual_release_metadata only for a trusted exact timestamp
information_timestamp latest exact availability among inputs used by the derived row
information_timestamp_provenance derived_from_actual_release_metadata when exact
vintage_timestamp     stored-value vintage; null for current latest-revised FRED data
retrieved_at           time the project actually fetched or loaded the source
timing_status          release_aligned, reference_month_only, or derived_value_unavailable
data_vintage_status    latest_revised_non_vintage for the current FRED loader
cpi_level             CPI index level
cpi_imputed           bool: CPI level log-linearly bridged for a single missing interior month
inflation_yoy         YoY CPI inflation in percentage points
baseline              selected inflation baseline in percentage points
epsilon               inflation_yoy - baseline
tinf_4m               4-month rolling mean of epsilon
tinf_8m               8-month rolling mean of epsilon
tinf_12m              12-month rolling mean of epsilon
baseline_method       string label for baseline choice
live_safe             legacy baseline flag meaning no future-row/full-sample lookahead only
```

## Sample modes

Date ranges are selected through named sample modes defined in
`src/transitory_inflation/config.py` (`SAMPLE_MODES`). Do not hardcode ad-hoc
date filters elsewhere.

| Mode | start_date | end_date | Purpose |
|---|---|---|---|
| `paper_replication` | 1982-01-01 | 2021-07-31 | Reproduce the paper only (ex-post). |
| `live_dashboard` | 1982-01-01 | None (latest FRED) | Current macro signal. Default in Streamlit. |
| `max_history` | None (earliest FRED) | None (latest FRED) | Robustness over the longest sample. Not necessarily the default trading signal. |

Bounds are inclusive. `None` means unbounded on that side.

### Order of operations

1. Fetch FRED series. The source priority is official FRED API, public FRED
   CSV, local cached FRED, then demo data as an emergency fallback. When
   `start_date` is set, the fetch begins 12 months earlier
   (`YOY_WARMUP_MONTHS`) so `inflation_yoy` is defined from the first sample
   row instead of 12 months later.
2. Resample to monthly last observations.
3. Bridge single-month interior CPI gaps by log-linear interpolation and flag
   them in `cpi_imputed` (log-linear because CPI is an index level and YoY is
   ratio-based). Multi-month gaps and missing tail months are never imputed,
   so a longer outage stays visible instead of being silently estimated.
4. Compute `inflation_yoy` as the 12-month percent change, in percentage points.
5. Trim to the inclusive `[start_date, end_date]` window. Warm-up rows never
   appear in outputs.
6. Compute baseline/epsilon/TINF features on the trimmed frame. Baseline
   warm-up NaNs (for example 37 months for `rolling_36_shifted`) are
   intentional: row-lookahead-safe baselines must not reach back before the sample.

### Information-date and vintage rules

- CPI reference month is never silently used as its publication or information timestamp.
- Exact `release_timestamp` values are accepted only when actual release metadata, explicit
  provenance, a timezone-bearing timestamp, and an explicit `release_aligned` incoming timing
  status are supplied. Time-of-day is preserved. Each inflation measure carries its own release
  timing status and derived YoY timing status; core CPI, PCE, and core PCE never borrow headline CPI
  timing metadata.
- Cache serialization and reload validate the timestamp's original timezone before UTC
  conversion and persist the incoming per-measure timing status. A timezone-naive release string,
  a missing status, or any non-exact status remains untrusted and cannot acquire exact status merely
  because a parser attaches UTC or sees claimed provenance.
- CPI YoY, baseline, epsilon, and TINF availability is the latest exact availability among their
  dependencies. Incoming timing status is authoritative: only `release_aligned` dependencies can
  contribute exact timestamps. If any dependency actually used is `reference_month_only`, a proxy,
  unknown, or otherwise non-exact, `information_timestamp` stays null and derived `timing_status`
  fails closed to `reference_month_only`.
- Ex-post continuity retains its labelled month-end availability proxy when release metadata is
  absent, but that proxy is not promoted to an exact information timestamp.
- Monthly macro normalization selects the latest physically dated row within each month; the last
  stable input row breaks same-date ties. The selected row is retained whole, including nulls, so
  values, timestamps, status, and provenance cannot be spliced across complementary-null duplicates.
- Current FRED values are latest-revised and explicitly non-vintage. The project does not claim
  vintage safety without an actual vintage source.
- Exact market linkage starts at the first eligible market-close timestamp greater than or equal
  to the signal information timestamp. It requires explicit trustworthy signal timing plus an
  explicit timezone-bearing market-close timestamp, status, and provenance.
- Standard FRED market observations are date-only, not exact close timestamps. With trustworthy
  full signal-information timing they use the first observation date after the signal information
  timestamp as a labelled conservative next-observation proxy. Without trustworthy full
  signal-information timing they use the labelled conservative month-end `t+1` origin proxy.
- Duplicate market observation dates retain the last physical source row in stable input order.
  Values, nulls, timestamps, status, and provenance are selected from that one row; columns are
  never combined across duplicate rows.
- Multi-series linkage routes exact, next-observation proxy, month-end proxy, or unavailable
  treatment independently for each required series and retains each result's own origin timestamp
  or observation date, basis, and timing status. A missing series cannot demote another series with
  an eligible exact post-information observation. A shared row is exact only when every displayed
  series has an eligible exact origin; heterogeneous fully available origins are labelled mixed,
  mixed availability is labelled partial, and zero eligible series is unavailable. Exact series may
  have different origin timestamps. The shared origin records the latest selected per-series origin
  needed for the available set and does not imply simultaneity.
- When exact alignment is eligible but no trustworthy market observation exists at or after the
  signal information timestamp, the origin is explicitly unavailable rather than exact with a
  null timestamp/value. The same fail-closed rule applies when a selected proxy lookup finds no
  eligible observation after its required origin date.

### Series availability caveats

- `CPIAUCSL` begins 1947-01, so `max_history` CPI coverage starts there.
- The 2025-10 CPI release was canceled (federal government shutdown), so FRED
  has a permanent hole at 2025-10. The single-month imputation rule above
  bridges it; without the bridge, every strict 36-month rolling baseline
  window containing 2025-10 is NaN and the live signal freezes at 2025-09
  until 2028-11. Outputs touching 2025-10 are partly estimates and the app
  discloses this when the month is in the loaded sample.
- The paper's stated control is a 1-month T-bill, but FRED has no 1-month bill
  series before 2001-07 (`TB4WK`/`GS1M` start there, and the older `TB1MS` id
  does not exist on FRED at all). The project therefore uses `TB3MS` (3-month
  bill secondary market rate, history since 1934) as the `tbill_3m` control so
  the full paper sample is covered. This is a disclosed deviation from the
  paper. Rows without `tbill_3m` are dropped by the regression helpers, not by
  the loader.

### Raw file naming

`scripts/fetch_fred_data.py --mode <mode>` writes
`data/raw/fred_base_macro_<mode>.csv` so different date ranges never silently
overwrite each other.

### Data source priority

`FRED_API_KEY` is optional but recommended. If it is present in the environment
or project-root `.env`, the loader tries the official FRED observations API
first. The key must never be printed, logged, or committed.

Fallback order:

1. `fred_api`: official FRED observations API using `FRED_API_KEY`
2. `fred_csv`: public FRED CSV endpoint if the key is missing or API fetch fails
3. `cached_fred`: exact mode cache, or `fred_base_macro_max_history.csv` sliced
   through named sample-mode rules
4. `demo`: synthetic emergency data only when live and cached FRED are
   unavailable

The Streamlit app discloses `data_source_used`, `live_fetch_status`,
`cache_file_used`, raw data end date, latest CPI observation date, latest valid
CPI YoY reference month, signal reference month, information timestamp, timing
status, and whether CPI imputation was applied. A compatibility field named
`latest_valid_signal_date` is a reference-month alias, not a signal-availability
date; its companion semantics field states that explicitly.
Cache and demo fallbacks are visibly warned and do not fabricate fresh FRED
observations.

## Units

Use percentage points.

Correct:

```text
3.25 = 3.25% inflation
```

Avoid mixing with decimal returns:

```text
0.0325 = 3.25%
```

## Baselines

Baseline choice must be explicit in every exported table/figure:

- `full_sample`
- `rolling_36_unshifted`
- `rolling_36_shifted`
- `expanding_shifted`
- `fed_target`

## File policy

- raw external downloads -> `data/raw/`
- cleaned reusable data -> `data/processed/`
- intermediate/debug files -> `data/interim/`
- third-party/reference datasets -> `data/external/`
- generated charts/tables -> `reports/`

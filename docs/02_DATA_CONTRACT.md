# Data Contract

## Canonical monthly data frame

Preferred index/columns:

```text
date                  month-end or FRED monthly date
cpi_level             CPI index level
cpi_imputed           bool: CPI level log-linearly bridged for a single missing interior month
inflation_yoy         YoY CPI inflation in percentage points
baseline              selected inflation baseline in percentage points
epsilon               inflation_yoy - baseline
tinf_4m               4-month rolling mean of epsilon
tinf_8m               8-month rolling mean of epsilon
tinf_12m              12-month rolling mean of epsilon
baseline_method       string label for baseline choice
live_safe             bool or label indicating no-lookahead status
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
   intentional: live-safe baselines must not reach back before the sample.

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
CPI YoY date, latest valid signal date, and whether CPI imputation was applied.
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

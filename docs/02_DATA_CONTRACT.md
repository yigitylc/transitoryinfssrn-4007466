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
cpi_observed_level    official CPI index level; remains null for an unavailable observation
cpi_level             policy-specific working CPI level; never the raw/cache authority
cpi_imputed           compatibility bool: working CPI level contains an explicit estimate
scenario_id           low, base, or high for a current-monitoring scenario
estimate_method       deterministic method used for the scenario estimate
estimated_reference_month reference month whose official level is unavailable
estimate_value        scenario's working estimate for that reference month
uses_estimated_input  bool: the derived value depends on a scenario estimate
estimated_input_months reference months of scenario estimates used by the derived value
calibration_policy    threshold population policy; current scenarios use observed-only history
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
| `live_dashboard` | 1982-01-01 | None (latest FRED) | Current descriptive monitoring plus separate strict historical evidence. Default in Streamlit. |
| `max_history` | None (earliest FRED) | None (latest FRED) | Robustness over the longest sample. Not necessarily the default trading signal. |

Bounds are inclusive. `None` means unbounded on that side.

### Order of operations

1. Fetch one raw FRED superset. The source priority is official FRED API, public FRED CSV, local
   cached FRED, then demo data as an emergency fallback. When `start_date` is set, the superset
   begins at least 12 months earlier (`YOY_WARMUP_MONTHS`) so `inflation_yoy` is defined from the
   first sample row instead of 12 months later.
2. Resample the superset to monthly physical rows. Preserve official headline/core CPI nulls and
   original missingness as the raw/cache authority.
3. From that same full warm-up superset, derive one `observed_only` view and, for explicitly current
   descriptive use, the `low`, `base`, and `high` October 2025 scenario views. Never rebuild a
   scenario from an already-trimmed observed/research frame. Dashboard construction requires the
   common warm-up authority and raises a contract error when it is absent, truncated, duplicated,
   or inconsistent with the selected observed frame.
4. Compute each view's `inflation_yoy` as the 12-month percent change, in percentage points. The
   scenario working level may differ, but its official observed-level column remains null.
5. Trim every view to the same inclusive `[start_date, end_date]` sample window. Warm-up rows never
   appear in outputs.
6. Compute baseline/epsilon/TINF features on each trimmed frame. Baseline
   warm-up NaNs (for example 37 months for `rolling_36_shifted`) are
   intentional: row-lookahead-safe baselines must not reach back before the sample.
7. Calibrate a current scenario's percentile and regime thresholds only on prior complete
   `observed_only` rows eligible under the same selected sample mode. Estimate-dependent rows may
   not enter that calibration history.

The approved October 2025 scenario formulas, applied separately to headline and core CPI, use exact
calendar endpoints only:

- `low`: October equals September;
- `base`: October is the geometric midpoint/log-linear bridge,
  `sqrt(September * November)`;
- `high`: October equals November.

For each series, September means exactly 2025-09-30 and November means exactly 2025-11-30. Physical
row adjacency is never used, so a retained December row cannot substitute for a missing November.
An endpoint is accepted only when monthly normalization yields one coherent row, its official value
is present, approved FRED value provenance is present, and no estimated/imputed marker applies.
Endpoint value authority does not require `release_aligned` timing: missing exact release metadata
leaves H5 status conservatively `reference_month_only` while the authoritative official value may
still be used. Missing, null, ambiguous, estimated/imputed, or invalid-provenance endpoints fail the
series closed.

Headline and core CPI are both required. Any required-series, endpoint, or estimate failure makes
the whole scenario bundle unavailable and suppresses TINF, percentile, regime, pressure, and the
current Trader bucket. Attempted estimates and failure diagnostics remain serialized in lineage.

The base bridge and endpoint cases are deterministic scenario assumptions, not confidence
intervals, probabilities, forecasts, or official CPI values. They are explicitly unofficial,
ex-post, latest-revised, and non-vintage. Multi-month gaps, tail gaps, and other source-unavailable
periods remain missing in this phase.

### Policy routing

The scenario bundle is allowed only for the Sidebar Current Reading, Current Macro Signal, current
TINF/regime/pressure, Trader Research current bucket, report headline/regime/watchlist, and a
separately labelled current baseline comparison. An estimated current Trader Research bucket may
query an observed-only analog population.

Historical validation and hit rates, benchmark scoring, historical robustness, historical
market-linkage origins, historical analog populations, and paper reconstruction consume only the
observed view. Estimated CPI inputs are excluded rather than silently converted into admissible
history.

One canonical historical-eligibility predicate enforces that boundary. It rejects a row whenever
`uses_estimated_input`, `uses_imputed_input`, a non-empty `estimated_input_months` value, legacy
estimated/imputed/missing-input lineage, an incompatible data/population policy, or an existing
eligibility failure is present. Current scenario-aware surfaces remain separately routed.
Forward-outcome target lineage is not collapsed into that signal-origin predicate: each horizon's
`observed_only_eligible_{h}m` gate independently rejects a contaminated target while preserving an
otherwise clean origin's outcomes at unaffected horizons.

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
- All three October scenarios are ex-post current-monitoring assumptions. They are unavailable
  before the following November observation. They retain a labelled month-end availability proxy
  when exact November release metadata is absent, but that proxy is not promoted to an exact
  information timestamp.
- When the selected sample ends before October 2025, including Paper Window, the scenario bundle is
  `not_applicable`; no three-view scenario panel or cross-scenario stability claim is emitted.
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
  has a permanent official hole at 2025-10 for headline and core CPI. Under observed-only
  treatment, YoY CPI is missing at 2025-10 (missing numerator) and 2026-10 (missing 12-month
  denominator), then becomes continuously valid from 2026-11. Under the default shifted 36-month
  baseline, the baseline and epsilon recover at 2029-11; TINF 4M at 2030-02, TINF 8M at 2030-06,
  and TINF 12M/the complete strict snapshot at 2030-10. The current scenario bundle restores a
  descriptive current signal without changing any of those observed-only historical facts.
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
`cache_file_used`, raw data end date, latest official CPI observation date, current-monitoring and
historical-evidence endpoints, every scenario's value and classification, regime/pressure
stability, estimate method/month/value, estimated-input lineage, calibration policy, signal
reference month, information timestamp, timing status, and latest-revised non-vintage status. It
also states that the scenarios are unofficial and ex-post and that historical populations exclude
estimated CPI inputs. A compatibility field named
`latest_valid_signal_date` is a reference-month alias, not a signal-availability
date; its companion semantics field states that explicitly.
Per-scenario exports retain JSON-compatible headline and core lineage: series identity, missing
month, exact endpoint months/values/value provenance and actual H5 timing, estimate method/value,
attempt status, availability/failure reason, estimated-input flags/months, sample/baseline and
calibration policies, signal reference/information timing, retrieval, and latest-revised non-vintage
status. Unavailable bundles retain diagnostics but never retain a usable classification.
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

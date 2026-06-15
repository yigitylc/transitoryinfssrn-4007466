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
- If cleaned CPI data supports a 2026-04 signal, the latest valid signal date
  must not remain stuck at 2025-09.
- Data status clearly shows:
  - source used
  - cache path when applicable
  - raw data end date
  - latest CPI observation date
  - latest valid CPI YoY date
  - latest valid signal date
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

## Phase 5 - Trader Report

Goal: produce a practical macro research report that remains clear about model
confidence, freshness, and caveats.

Required sections:

- current signal
- historical analogs
- forward inflation odds
- benchmark confidence
- market implications
- caveats
- data freshness

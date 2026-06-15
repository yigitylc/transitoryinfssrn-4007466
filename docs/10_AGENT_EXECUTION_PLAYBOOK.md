# Agent Execution Playbook

This playbook is the operating manual for Codex, Claude, or another coding
agent working on this repository. The phase gates live in
[`docs/09_PRODUCTION_ROADMAP.md`](09_PRODUCTION_ROADMAP.md); this file explains
how to execute them without mixing methodology layers or scope.

## 1. Project Intent

This project is not just a paper replica. It has three separate layers:

1. Paper replication:
   Reproduce and document the Peron/Bonaparte transitory inflation paper as
   faithfully as possible.

2. Live macro signal:
   Use latest FRED inflation data to produce a current inflation-regime
   dashboard.

3. Historical validation:
   Test whether the signal historically helped distinguish transitory inflation
   shocks from persistent inflation shocks.

These three layers must not be mixed silently. Paper-replication choices can be
ex-post if they are clearly labeled. Live-signal choices must avoid lookahead.
Validation may use future outcomes only after signal features are built, and
only for scoring historical rows.

## 2. Current State

- A Streamlit dashboard exists.
- FRED data loading supports official API, public CSV, cached FRED, and demo
  fallback paths.
- Phase 1 historical validation exists but still needs production
  stabilization.
- The main current blocker is positive-shock logic and downside overshoot
  classification.
- The roadmap exists in `docs/09_PRODUCTION_ROADMAP.md`.
- `NEXT_TASKS.md` tracks the current gate only.

## 3. Current Limitations

- Historical validation currently risks confusing absolute baseline distance
  with positive inflation-shock resolution.
- A move from positive epsilon to negative epsilon should not be classified as
  persistent high inflation.
- Absolute distance to baseline is useful for equilibrium/stability, but not
  sufficient for false-transitory labeling.
- Dashboard explanations need to be more explicit.
- Cache fallback must be verified to pass through the same cleaning/imputation
  pipeline as live FRED data.
- Data-source status must be visible so stale data is not mistaken for live
  data.
- Phase 1 validation is diagnostic, not a complete forecast model.
- Phase 2 benchmark comparison has not been implemented yet, so hit rates
  should not be over-interpreted.
- Market linkage should not begin until benchmark comparison shows signal
  usefulness.

## 4. Methodology Rules

- Paper replication and live signal are separate tasks.
- Paper baseline is ambiguous: the main text suggests historical mean /
  full-sample mean-reversion; the appendix references a 36-month moving
  average.
- Live dashboard default should use `rolling_36_shifted`.
- Paper replication should use `full_sample` only inside the paper sample
  window.
- TINF timing is ambiguous: paper prose says moving average; the equation
  appears lagged.
- Live dashboard uses trailing-current TINF because the latest CPI print should
  inform the latest signal.
- Paper-lagged TINF is a future robustness check, not the current default.
- Paper-style decay should be labeled as paper-style because the paper
  estimates an intercept but the published decay formula uses `rho_T` and `mu`
  only.
- `TB3MS` is a short-rate proxy, not the exact 1-month T-bill.

## 5. Historical Validation Interpretation

Positive-shock resolution asks:
"When inflation was above baseline, did the high-inflation shock fade?"

Absolute baseline convergence asks:
"Did inflation end close to the baseline, regardless of direction?"

Important distinction: if epsilon moves from positive to negative, the positive
inflation shock resolved and overshot lower. It may fail absolute baseline
convergence because it is far below baseline, but it is not persistent high
inflation.

For trading interpretation, positive-shock resolution is usually more
actionable.

For policy/equilibrium interpretation, absolute baseline convergence is still
useful.

The dashboard should show both:

- positive-shock outcome labels
- absolute baseline convergence diagnostics

## 6. Dashboard Explanation Requirements

The Historical Signal Validation tab should include concise explanatory text
explaining by giving examples:

- what Outcome threshold (pp) means
- difference between positive-shock direction and absolute distance to baseline
- why regime and short-term pressure are grouped separately
- why combined regime x short-term pressure is more actionable
- why counts matter
- why hit rates should not be trusted without benchmark comparison
- why `full_sample` is ex-post and not valid for live-like historical
  validation

Suggested dashboard language:

"Positive-shock resolution asks whether above-baseline inflation pressure
faded. If epsilon moves from positive to negative, the high-inflation shock
resolved and overshot lower."

"Absolute baseline convergence asks whether inflation ended close to baseline,
regardless of direction. A negative overshoot may fail this test because
inflation is still far from baseline, but it is not persistent high inflation."

"For trading interpretation, positive-shock resolution is usually more
actionable. For equilibrium or policy-stability interpretation, absolute
baseline convergence remains useful."

## 7. Recommended Course of Action

Step 1: Finish Phase 0 production stabilization.

Step 2: Commit the stable version.

Step 3: Polish historical validation tables and explanations.

Step 4: Add threshold sensitivity.

Step 5: Move to Phase 2 benchmark comparison.

Step 6: Only after Phase 2, add robustness and market linkage.

Do not start Phase 2 until Phase 0 is committed and Phase 1 validation logic is
clean.

## 8. Better Ideas / Possible Improvements

Improvements that may be better than immediately adding more charts:

- Add positive-shock outcome logic before adding visuals.
- Add combined `historical_regime x historical_short_term_pressure` table.
- Add threshold sensitivity for `0.25`, `0.50`, `0.75`, and `1.00` pp.
- Add benchmark comparison before market linkage.
- Add simple explanation blocks inside Streamlit.
- Add data freshness status at the top of the dashboard.
- Add historical analogs only after validation definitions are stable.
- Add market linkage only after signal beats or complements benchmarks.
- Add core CPI / PCE robustness later.
- Keep absolute convergence and directional shock resolution as separate
  panels.

## 9. Recommended Next Prompts

Prompt A - Start current gate:

```text
Read docs/09_PRODUCTION_ROADMAP.md, docs/10_AGENT_EXECUTION_PLAYBOOK.md, and NEXT_TASKS.md. Execute only the current gate. First give me a plan. Do not start later phases.
```

Prompt B - Phase 0 implementation:

```text
Proceed with Phase 0 only. Fix app startup, FRED API/cache/demo fallback, cached CPI imputation, historical_* validation columns, positive-shock outcome logic, and dashboard explanatory text. Do not start Phase 1 or Phase 2. Run ruff, pytest, and compileall. Report changed files and whether safe to commit.
```

Prompt C - Read-only audit:

```text
Do a read-only audit of the current gate. Check for lookahead bias, stale data, mislabeled false transitory examples, secrets, ignored files, and test coverage. Do not edit files. Report P0/P1/P2 issues.
```

Prompt D - Commit readiness:

```text
Check git status, ignored files, secrets, generated artifacts, test results, Streamlit startup, and safe commit allowlist. Do not commit. Report whether this is safe for commit.
```

Prompt E - Phase 1 start:

```text
Start Phase 1 only after Phase 0 is committed. Add combined regime x pressure tables, threshold sensitivity, better example categories, and explanatory dashboard text. No benchmark models yet.
```

Prompt F - Phase 2 start:

```text
Start Phase 2 only after Phase 1 is accepted. Compare the signal against no-change, CPI persistence, mean reversion, and AR(1) benchmarks using MAE, RMSE, directional accuracy, hit rates, false positives, false negatives, and confusion matrices.
```

## 10. Agent Loop Rules

Codex should follow this loop:

1. Read roadmap, playbook, and `NEXT_TASKS.md`.
2. Inspect current code.
3. Produce a short implementation plan.
4. Ask questions if methodology is ambiguous.
5. Implement only the current gate.
6. Run checks.
7. Report files changed, assumptions, results, and remaining risks.
8. Do not start the next phase without approval.

## 11. Pre-Commit Checklist

- `ruff check .` passes
- `pytest` passes
- `python -m compileall src app scripts` passes
- Streamlit starts
- no `.env` staged
- no `.venv/` staged
- no raw cache CSV staged
- no extracted full paper text staged
- no generated logs staged
- no pycache/caches staged
- no API key printed or committed
- data source status visible
- latest signal date is not stale when data supports current calculation
- historical validation columns are `historical_*`
- false transitory examples do not misclassify downside overshoots

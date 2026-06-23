# Phase 0–5 Audit Plan — Transitory Inflation Macro Research

**Status:** Planning artifact. No audit executed yet, no source code changed.
**Date:** 2026-06-22
**Scope:** Read-only audit plan covering Phase 0 (production stabilization) through Phase 5 (macro research report).

## 0. Dashboard purpose (as understood)

A local Streamlit research tool that operationalizes the Peron & Bonaparte "transitory inflation"
paper (SSRN 4007466). It turns the phrase "transitory inflation" into a measurable state variable:
`epsilon = CPI_YoY − baseline`, then `TINF_n = n-month rolling mean of epsilon` (n = 4/8/12), in
**percentage points**. The project deliberately keeps three layers separate (per the playbook):
**paper replication** (ex-post, fixed 1982–2021 window), **live macro signal** (latest FRED,
no-lookahead baselines), and **historical validation** (does the signal carry forward
information). It is decision-support / research, explicitly **not** a trading system, PnL backtest,
or buy/sell recommender. The eight Streamlit tabs map to the phases: Current Signal, Historical
Validation (P1), Market Linkage (P4), Benchmark Comparison (P2), Paper Framework, Decay/Convergence,
Robustness (P3), Macro Research Report (P5).

## 1. Files and modules inspected

| Area | Files |
|---|---|
| Roadmap / governance | `docs/09_PRODUCTION_ROADMAP.md`, `docs/10_AGENT_EXECUTION_PLAYBOOK.md`, `NEXT_TASKS.md`, `docs/07_BACKLOG.md`, `docs/06_DECISION_LOG.md` |
| Spec / contract | `docs/00_PROJECT_CONTEXT.md`, `docs/01_RESEARCH_SPEC.md`, `docs/02_DATA_CONTRACT.md`, `docs/04_VALIDATION_PROTOCOL.md`, `docs/05_DASHBOARD_BRIEF.md` |
| Data layer | `src/.../config.py`, `data.py`, `market_data.py` |
| Signal / stats | `features.py`, `validation.py`, `models.py`, `diagnostics.py`, `plots.py` |
| Phase modules | `benchmarks.py` (P2), `robustness.py` (P3), `market_linkage.py` (P4), `report.py` (P5) |
| App | `app/streamlit_app.py` (1,297 lines) |
| Tests | `tests/test_{features,decay,validation,sample_modes,benchmarks,robustness,market_linkage,market_data,trader_report}.py` |
| Build / CI / config | `pyproject.toml`, `requirements.txt`, `.github/workflows/python-checks.yml`, `.gitignore`, `run_app.ps1`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `scripts/` |

## 2. Understanding of each phase (intent vs current implementation)

- **Phase 0 — Production stabilization.** Intent: app starts cleanly; FRED `api→csv→cache→demo`
  fallback with full disclosure; cached fallback reuses the same clean/impute/YoY/baseline path and
  never trusts precomputed `inflation_yoy`/`cpi_imputed`; isolated single-month CPI imputation so
  the live signal doesn't freeze at 2025-09; `historical_*` validation columns kept separate from
  live labels; positive-shock outcome logic; safe-commit hygiene. *Implementation present:*
  `data.load_macro_data_for_mode_with_status` (fallback + redaction), `build_base_frame` (single-gap
  log-linear bridge), `validation.add_walk_forward_regime_labels` (`historical_*`), `.gitignore`.
- **Phase 1 — Historical validation polish.** Intent: combined regime×pressure table, threshold
  sensitivity (0.25/0.50/0.75/1.00pp), richer example categories, positive-shock vs absolute-gap
  distinction. *Implementation:* `validation.py` (`forward_outcome_summary_by_*`,
  `threshold_sensitivity_summary`, `validation_examples`, positive-shock + downside-overshoot labels).
- **Phase 2 — Benchmark comparison.** Intent: test TINF/regime vs no-change, CPI persistence,
  mean-reversion, AR(1) on MAE/RMSE/directional/hit/FP/FN/confusion. *Implementation:* `benchmarks.py`
  (expanding AR(1), walk-forward regime bucket, relative-improvement, confusion).
- **Phase 3 — Robustness.** Intent: CPI vs core vs PCE/core PCE; rolling_36_shifted vs
  expanding_shifted vs full_sample; horizon/threshold sweeps; honest live-safe labels.
  *Implementation:* `robustness.py` (scorecard grid, `tinf_regime_verdict`, win-rate summary).
- **Phase 4 — Market linkage.** Intent (gated on P2): descriptive history of how 2Y/10Y yields,
  5Y/10Y breakevens, 5Y/10Y real yields moved after TINF/regime states; never a trade signal.
  *Implementation:* `market_data.py` (approved-FRED-rates-only registry + `validate_market_series_registry`),
  `market_linkage.py` (forward bp changes, channel summaries, correlations).
- **Phase 5 — Macro research report.** Intent: synthesize current regime, signal confidence,
  robustness, historical analogs, market linkage, caveats, watchlist; separate interpretation from
  forecast accuracy; no PnL/recommendations. *Implementation:* `report.build_macro_research_report`
  + report tab. (Note `report.build_trader_report`/`REGIME_PLAYBOOK` also exist but are **not** wired
  into the app.)

## 3. Phase-by-phase audit scope (what I will check)

- **P0:** fallback ordering + status disclosure correctness; cache path reuses clean/impute/YoY
  pipeline and ignores precomputed derived columns; single-month bridge vs multi/tail no-impute;
  latest-valid-signal-date advances past an imputed month; API-key never logged/committed; demo only
  as last resort; `.gitignore` truly excludes `.env`/caches/PDF/raw data.
- **P1:** regime/pressure label definitions; positive-shock vs absolute-gap separation; example
  bucket correctness (downside overshoot ≠ false transitory); threshold-sensitivity is sensitivity,
  not optimization; counts/weak-evidence surfaced.
- **P2:** each benchmark's definition and fairness; AR(1) iterated-forecast correctness; the
  persistent-classification mapping and its denominator; relative-improvement on common samples.
- **P3:** scorecard grid completeness; full_sample correctly flagged ex-post/not-live-safe; missing
  measures disclosed not backfilled; verdict/win-rate aggregation honesty (diagnostic, not a selector).
- **P4:** registry stays rates-only (no equities/FX/commodities/PnL); bp conversion; month-end
  alignment + one-to-one join; descriptive framing held; weak-evidence (<30) flagged.
- **P5:** all seven required sections present; AR(1)-may-beat-TINF framing honored; caveats include
  the mandated disclosures; "no trading recommendation" boundary held; freshness/imputation disclosed.

## 4. Methodology / math checks

- **Units:** every TINF/epsilon path stays in percentage points (`features.py`); bp = Δpercent×100 (`market_linkage.py:149`).
- **Baselines:** `rolling_36_shifted`/`expanding_shifted` are `.shift(1)` (live-safe); `full_sample`/
  `rolling_36_unshifted` flagged not-live-safe (`features.compute_baseline`, `BASELINE_META`).
- **Warm-up / YoY:** 12-month fetch warm-up defines YoY from sample start; rolling/expanding warm-up
  NaNs intentional (`data._fetch_start`, `build_base_frame`).
- **Forward alignment:** `shift(-h)` assigns t+h outcomes to row t; terminal rows NaN (tests confirm).
- **Decay:** paper-style `1 − rho_T·mu^(h−1)`, `t* = 1 + ln(0.05/rho_T)/ln(mu)`; intercept `c`
  estimated but intentionally unused (disclosed paper deviation); validity gate `rho_T>0 & 0<mu<1`.
- **AR(1):** params extracted by name not index (`models.extract_l1_param`); rolling rho end-date
  aligned; benchmark AR(1) fit on data through t then iterated h steps.
- **Stats:** HC1 robust OLS; Ljung-Box / ADF / KPSS used as diagnostics with correct null labels.
- **Candidate findings to confirm:** (a) `cpi_persistence = current + (current − current.shift(h))`
  is momentum extrapolation, not a random walk — confirm this matches intended "persistence";
  (b) `mean_reversion` forecast == baseline ⇒ `forecast − baseline = 0`, so it can **never** predict
  "persistent" — verify this structural artifact is disclosed, not silently penalizing it;
  (c) persistent-classification sample is conditional on current positive-shock rows — confirm
  denominators and captions match.

## 5. No-lookahead risks to look for

- **Strongest existing guards (re-verify):** perturbation tests prove future rows don't change
  signals/forecasts (`test_benchmarks`, `test_market_linkage`), and walk-forward regime thresholds
  use only prior TINF (`test_validation`). Confirm these still pass and cover report/analog paths.
- **Baseline lookahead:** ensure `full_sample`/unshifted never feed live claims; report + benchmark
  tabs warn when baseline not live-safe.
- **Validation feedback:** forward outcome columns (`*_fwd_*`, `*_change_*`) must never re-enter
  baseline/epsilon/TINF/regime/pressure — re-check the report analog path, which builds the market
  panel on a frame that already carries forward columns.
- **Regime-bucket leakage:** `_walk_forward_regime_bucket_forecast` gates on `latest_known_origin = pos − horizon`; verify no t+h outcome is used before it could be known.
- **Vintage vs revised data (CLAUDE.md trigger):** the project uses **latest-revised** FRED levels
  and treats month-t CPI as if known at t, while real CPI publishes ~mid month t+1. "Live-safe"
  here means *no full-sample lookahead*, not *real-time vintage*. I will flag whether this
  publication-lag/revision caveat is disclosed clearly enough.
- **Market timing:** end-of-month-t market levels are joined to the month-t CPI signal; assess the
  small contemporaneous-vs-publication-lag mismatch for the descriptive linkage.

## 6. UI/UX questions to evaluate

- **Tab order:** Market Linkage (P4) sits before Benchmark Comparison (P2); does that invert the
  roadmap's "no linkage until P2 confirms usefulness" narrative for a reader?
- **Live-safe vs ex-post clarity:** are warnings consistent across every tab (mostly strong today)?
- **Phase 5 boundary:** does the report stay descriptive (no trade calls)? Is `build_trader_report`/
  `REGIME_PLAYBOOK` (trade-flavored priors) meant to ship/be surfaced, or be removed?
- **Consistency:** market-linkage horizon options (3/6/12/24) omit 36M used elsewhere; duplicated
  helpers (`latest_valid_date`, `pressure_label`) across app/`data.py`/`report.py`.
- **Readability:** explanatory `section_notes` density; whether tables disclose counts/weak-evidence.
- **Performance/latency:** robustness + report tabs recompute large grids (baselines×measures×
  horizons×thresholds) with Python-loop AR(1) on each interaction and are not `@st.cache_data`-wrapped.

## 7. Tests and smoke checks planned

- `ruff check .` (config: E,F,I,UP,B,SIM; E501 ignored; E402 per-file for app/scripts).
- `pytest` (network-free via monkeypatched `requests.get`; `pythonpath=["src"]`, `testpaths=["tests"]`).
- `python -m compileall src app scripts`.
- **Coverage gap scan:** map which functions lack tests (e.g. `diagnostics.py`, `plots.py`,
  app-level wiring, report `next_print_flip_threshold` edge cases).
- **End-to-end smoke without network** (FRED CSV is flaky from this machine — keep it offline):
  build each phase's tables/report on synthetic/demo frames to confirm wiring; optionally a cached
  Streamlit import smoke. Avoid live FRED in CI-style checks.
- **Hygiene:** `git status` / `.gitignore` confirm no `.env`, `.venv`, caches, raw data, extracted
  paper text, or the third-party PDF are staged; grep for hardcoded absolute paths and secrets.

## 8. Recommended final audit-report structure

1. Executive summary + overall verdict (ship / fix-then-ship / blockers).
2. Severity-tagged findings: **P0** (correctness/lookahead/secrets), **P1** (methodology/clarity),
   **P2** (polish/code-health) — each with file:line, evidence, and recommended fix.
3. Phase-by-phase results (P0–P5): intent vs implementation, pass/fail per gate.
4. No-lookahead assessment (per-layer live-safe vs ex-post verdict).
5. Math/methodology verification log (formulas checked + outcome).
6. Test & smoke results (commands, pass/fail, coverage gaps).
7. UI/UX assessment.
8. Documentation/governance drift (roadmap vs `NEXT_TASKS.md` vs code vs README).
9. Prioritized remediation backlog + suggested next gate.

## Immediate red flags noticed during planning (to confirm in the audit)

- **R1 — Governance drift:** `NEXT_TASKS.md` says current gate = **Phase 2**, but git history shows
  committed work through **Phase 4** and uncommitted Phase 5 work (`report.py` et al.). The active-gate
  doc is stale vs the roadmap's "don't start later phases until the current gate is committed."
- **R2 — README drift:** lists only CPIAUCSL/TB3MS, names the mode `paper_window` (code:
  `paper_replication`), and omits benchmarks/robustness/market-linkage/Phase 5.
- **R3 — Orphaned trader layer:** `build_trader_report` + `REGIME_PLAYBOOK` (rates/curve/equities/FX
  priors) are shipped and tested but not surfaced in the app; scope + "no trade recommendation" question.
- **R4 — Tab ordering** inverts the P2→P4 gating narrative (Market Linkage shown before Benchmarks).
- **R5 — `mean_reversion` can never predict "persistent"** (forecast==baseline), a structural
  classification artifact to verify/disclose.
- **R6 — `importlib.reload` workarounds** scattered across app + `benchmarks.py` + `robustness.py`
  (Streamlit stale-import guards) — code-health smell to evaluate.
- **R7 — Vintage caveat:** uses latest-revised FRED data and treats month-t CPI as known at t;
  "live-safe" ≠ real-time vintage. Confirm this is disclosed prominently.

## Proposed next step if you approve

Run the audit in roadmap order (P0→P5): execute `ruff`/`pytest`/`compileall` + offline smoke first,
then work through the methodology/no-lookahead/UI checks above, and deliver the severity-tagged
report in the Section 8 structure. No code changes — findings only — unless you separately approve fixes.

# Phase 0-5 Audit Findings - Transitory Inflation Macro Research

**Status:** Audit executed. Read-only - no source code changed.
**Date:** 2026-06-22
**Auditor:** Claude Code (read-only deep audit per `reports/PHASE_0_5_AUDIT_PLAN.md`)
**Scope:** P0 (production stabilization) through P5 (macro research report): automated
checks, offline smoke, methodology / no-lookahead / math verification, UI/UX, hygiene,
and governance drift.

---

## 1. Executive summary + verdict

**Verdict: FIX-THEN-SHIP. No P0 blockers.**

The code is in good shape. There are **zero correctness, lookahead, or secret-leak (P0)
findings**. Automated checks all pass (`ruff` clean, `pytest` 84 passed, `compileall`
clean) and an offline end-to-end smoke builds every phase's outputs. The no-lookahead
discipline is genuinely strong and test-enforced: shifted baselines, walk-forward regime
thresholds, AR(1) fit on data through t only, regime-bucket gating on
`pos - horizon`, and forward outcomes used strictly for scoring.

What needs attention is **not** code correctness but (a) **process/governance drift** -
the repo has implemented Phases 2-5 while `NEXT_TASKS.md` still names Phase 2 as the
active gate, contradicting the project's own "commit each gate before starting the next"
rule; and (b) a handful of **interpretation-honesty clarity gaps** - most importantly the
vintage/revision caveat (the live signal uses latest-*revised* FRED data, not real-time
vintage) and two benchmark metrics that are degenerate by construction but not flagged
in the UI. The rest are polish/code-health items (eager recompute performance, tab order,
`importlib.reload` guards, duplicated helpers, README drift, an accidental stray file).

| Severity | Count | Headline items |
|---|---|---|
| **P0** correctness / lookahead / secrets | 0 | none found |
| **P1** methodology / clarity | 5 | governance drift, vintage caveat, mean_reversion degeneracy, dual regime definitions, orphaned trader layer |
| **P2** polish / code-health | 10 | eager recompute, tab order, reload guards, dup helpers, horizon options, README drift, stray file, test gaps, decay-note surfacing |

---

## 2. Severity-tagged findings

### P0 - correctness / lookahead / secrets

**None.** No hardcoded absolute paths in `src`/`scripts` (grep clean), no secrets tracked
(`git ls-files` matches only `.env.example`), `.gitignore` correctly excludes `.env`,
`.venv`, `data/raw/*`, caches, and `references/*.pdf`. No full-sample lookahead reaches
any live-signal path. See Section 4 for the per-layer no-lookahead assessment.

### P1 - methodology / clarity

**P1-1 Governance / gate drift (R1).**
`NEXT_TASKS.md:3` says `Current gate: Phase 2`, but git history has committed work through
Phase 4 (`b4fb216` P2, `47347b5`/`7d87284` P3, `19304d8`/`9da7a69` P4) and Phase 5 is
implemented (uncommitted `report.py` + report tab). `docs/09_PRODUCTION_ROADMAP.md:4-5`
and `docs/10_AGENT_EXECUTION_PLAYBOOK.md:145,213` require committing each gate before
starting the next. Many governance docs (`docs/00-06`, `METHODOLOGY.md`,
`PAPER_FORMULA_REFERENCE.md`, `AGENTS.md`, `CLAUDE.md`) are still untracked, and
`docs/09` itself is modified-uncommitted.
*Fix:* set `NEXT_TASKS.md` to the true active gate (Phase 5 closeout), commit the Phase
2-5 work + docs, and reconcile the roadmap to reality.

**P1-2 Vintage / revision caveat missing (R7).**
The live signal uses latest-*revised* FRED levels and treats month-t CPI as known at t,
while CPI actually publishes ~mid month t+1. The only disclosed timing caveat is
*publication lag* (`report.py:670`, `docs/09:247`). No doc uses the words
vintage/revision (grep across `docs/`), and `docs/01_RESEARCH_SPEC.md:50` labels live
mode "Real-time signal mode," which overstates it. "Live-safe" here means *no full-sample
lookahead*, not *real-time vintage*.
*Fix:* add an explicit caveat (report caveats + spec) that the signal is built on
latest-revised data and is not a true real-time-vintage backtest; soften the "real-time"
label.

**P1-3 `mean_reversion` can never predict "persistent"; `no_change` directional accuracy is ~0 (R5).**
`benchmarks.py:187` sets the `mean_reversion` forecast equal to `baseline`, and
`benchmarks.py:223-225` defines `forecast_persistent = (forecast - baseline) > threshold`
- which is *always False* for `mean_reversion`. Its confusion row (TP=FP=0) and hit rate
are therefore degenerate by construction. Likewise `no_change` forecasts zero CPI change,
so its `directional_accuracy` (`benchmarks.py:308`) is ~0 by construction. The benchmark
tab caption (`app/streamlit_app.py:887-892`, `872-877`) does not flag either artifact.
*Fix:* add a caption noting both structural artifacts so readers do not mis-rank the
naive baselines on those columns.

**P1-4 Two regime definitions share one label vocabulary.**
`features.latest_signal_snapshot` derives the *current* regime from in-sample tinf_4m
quantiles (`features.py:153-158`), while `validation.add_walk_forward_regime_labels`
derives *historical* regimes from expanding-shifted quantiles (`validation.py:124-137`).
Both emit the same strings ("elevated rising", etc.). `report._historical_analog_table`
then matches the in-sample current label against the walk-forward historical labels
(`report.py:424-431`). It works mechanically but mixes two definitions.
*Fix:* document the distinction prominently, or compute the current regime with the
walk-forward labeler so the analog match is apples-to-apples.

**P1-5 Orphaned, trade-flavored trader layer (R3).**
`report.build_trader_report` + `REGIME_PLAYBOOK` + `TraderReport` + `TERM_STRUCTURE_NOTES`
(`report.py:36-147,826-984`) are shipped and tested (`tests/test_trader_report.py`) but
never wired into the app (only `build_macro_research_report` is called,
`app/streamlit_app.py:916`). The playbook content opines on equities/FX/vol/gold/TIPS -
which conflicts with the deliberately **rates-only** market registry enforced by
`market_data.validate_market_series_registry` (`market_data.py:70-112`) and sits ahead of
its gate ("Trader research mode" is a *future* layer, `docs/01_RESEARCH_SPEC.md:100`,
`docs/00_PROJECT_CONTEXT.md:10`). It is caveated as "not investment advice" but is dead
code carrying broader-than-rates priors.
*Fix:* decide scope - either gate it explicitly as future/experimental (and reconcile
with the rates-only decision), or remove it until its gate is active.

### P2 - polish / code-health

**P2-1 Eager recompute on every rerun; heavy builders not cached (perf).**
Only data loaders are cached (`@st.cache_data` on `get_data`/`get_market_data`,
`app/streamlit_app.py:256,261`). Streamlit executes all eight tab bodies on every rerun,
so validation, benchmarks, market linkage, the robustness grid (sample_modes x baselines
x measures x 5 horizons x 4 thresholds, each with a Python-loop expanding AR(1)), **and**
the report tab - which internally recomputes benchmarks + a robustness pass again
(`report.py:716-733`) - all run on every widget change. *Fix:* wrap the heavy table
builders in `@st.cache_data` keyed by their inputs.

**P2-2 Tab order inverts the P2->P4 narrative (R4).** Market Linkage (P4) precedes
Benchmark Comparison (P2) in the tab list (`app/streamlit_app.py:343-344`), against the
roadmap's "no linkage until P2 confirms usefulness." *Fix:* reorder Benchmarks before
Market Linkage, or add a one-line note.

**P2-3 `importlib.reload` guards scattered (R6).** Seven reload guards at app import
(`app/streamlit_app.py:44-78`) plus inline guards in `benchmarks.py:55-60` and
`robustness.py:69-74`. Streamlit stale-import workarounds; fragile and confusing.
*Fix:* remove once module APIs are stable; rely on Streamlit's native reload.

**P2-4 Duplicated helpers.** `pressure_label`/`latest_valid_date`/`date_label`
(`app/streamlit_app.py:95,106,114`) duplicate `report._pressure_label`/`_latest_valid_date`/
`_date_label` (`report.py:158,217,223`), `data.latest_valid_observation_date`
(`data.py:275`), and `validation.PRESSURE_LABELS` (`validation.py:23`). Weak-evidence
constant/note duplicated in `market_linkage.py:22` and `report.py:23,247`. *Fix:*
centralize in one module.

**P2-5 Horizon option sets inconsistent across tabs.** Market-linkage selectbox offers
3/6/12/24 (`app/streamlit_app.py:203`) and omits 36M, yet the panel is built with the
default horizons including 36M (`app/streamlit_app.py:631`), so 36M is computed but never
displayable; validation/benchmark offer 6/12/24/36 and omit 3M. *Fix:* align option sets.

**P2-6 README drift (R2).** `README.md` lists only `CPIAUCSL`/`TB3MS` (lines 30-40),
names the mode `paper_window` (line 72; code: `paper_replication`), and omits core
CPI/PCE measures, benchmarks, robustness, market linkage, and the Phase 5 report. *Fix:*
refresh README to match `config.py` and the shipped phases.

**P2-7 Stray accidental file.** An untracked file literally named `tatus --short` (8192
bytes) sits in the repo root - a mistyped `git status --short` that redirected output to
a file. *Fix:* delete it.

**P2-8 Test coverage gaps.** `diagnostics.py` (`ljung_box_table`, `stationarity_diagnostics`)
and `plots.py` (figure builders) have no dedicated tests (grep across `tests/` matched
neither). `report.py` (`build_macro_research_report`, `next_print_flip_threshold`) is
covered inside `tests/test_trader_report.py`. *Fix:* add small smoke tests for
diagnostics/plots (they are only exercised through the app today).

**P2-9 Decay paper-deviation note not surfaced in the app.** The "intercept estimated but
unused" deviation is documented (`docs/10:75`) but the decay tab's notes only mention
"mu and c" (`app/streamlit_app.py:1072-1083`) without stating the published formula uses
`rho_T` and `mu` only. *Fix:* surface the one-line deviation note in the decay tab.

**P2-10 (note only) Explosive-but-valid decay.** `models.paper_decay_summary:167-173`
lets `rho_T > 1` with `0 < mu < 1` compute a decay path while emitting an "explosive"
warning. This is intentional and disclosed (the product still converges); no change
needed - recorded for completeness.

---

## 3. Phase-by-phase results (intent vs implementation)

| Phase | Gate intent | Result |
|---|---|---|
| **P0** Stabilization | API->CSV->cache->demo order; status disclosure; cache rebuilds clean/impute/YoY/baseline and ignores precomputed derived cols; single-month bridge; `historical_*` separation; positive-shock logic; secrets/gitignore | **PASS.** `data.load_macro_data_for_mode_with_status:435-501`; cache rebuild `:531-573`; single-gap bridge `build_base_frame:342-353`; redaction `_safe_error:113`; walk-forward `historical_*` labels `validation.py:106-143`. |
| **P1** Validation polish | combined regime x pressure; threshold sensitivity 0.25-1.00; richer examples; positive-shock vs absolute gap | **PASS.** `forward_outcome_summary_by_regime_and_pressure`, `threshold_sensitivity_summary` (sensitivity, not optimization, `validation.py:462-534`), 5 example buckets `validation_examples:559-638`. |
| **P2** Benchmarks | no_change / cpi_persistence / mean_reversion / AR(1) / regime-bucket on MAE/RMSE/dir/hit/FP/FN/confusion; common-sample improvement | **PASS** w/ clarity caveat (P1-3). AR(1) iterated `:99-102`; common-sample improvement `:343-373`; persistent denominator correctly conditional on positive-shock rows and that is disclosed (`app:887-892`). |
| **P3** Robustness | CPI/core/PCE; shifted vs full_sample; horizon/threshold sweeps; honest live-safe labels | **PASS.** `full_sample` tagged "ex-post / paper-style only" + `baseline_live_safe=False` (`robustness.py:28-39`); missing measures skipped not backfilled (`:147`); verdict/win-rate framed as diagnostics. |
| **P4** Market linkage | rates-only; bp; month-end one-to-one; descriptive; weak-evidence | **PASS.** Hard registry guard (`market_data.py:98-112`); `bp = (future-current)*100` (`market_linkage.py:149`); `merge(validate="one_to_one")` (`:175`); "unavailable" not demo on failure (`market_data.py:325-331`). |
| **P5** Report | 7 sections; AR(1)-may-beat framing; mandated caveats; no trade calls; freshness/imputation | **PASS** w/ caveat gaps. 7 sections present (smoke-verified); explicit "do not overstate TINF/regime ... AR(1) has lower error" (`report.py:322-332`); strong caveats (`:656-681`). Gaps: vintage caveat (P1-2), orphaned trader layer (P1-5). |

---

## 4. No-lookahead assessment (per layer)

| Layer | Live-safe verdict | Evidence |
|---|---|---|
| Paper replication (`full_sample`, paper window) | **Ex-post by design, correctly labeled** | `BASELINE_META` flags not-live-safe (`features.py:26-37`); robustness tags ex-post (`robustness.py:28-34`); app warns on every relevant tab. |
| Live signal (`rolling_36_shifted` / `expanding_shifted`) | **Live-safe re: no full-sample lookahead** - but uses latest-revised data, not real-time vintage (P1-2) | shifted baselines `.shift(1)` (`features.py:75-78`); walk-forward regime thresholds shifted (`validation.py:124-125`). |
| Validation / benchmarks | **No leakage** | forward cols built after signals, used only for scoring (`validation.py:146-184`); AR(1) on history through t (`benchmarks.py:83-84`); regime-bucket gates on `pos-horizon` (`:129`); perturbation tests pass. |
| Market linkage | **No leakage** | forward market changes via `shift(-h)` used only as outcomes (`market_linkage.py:147-149`); regime/pressure computed before forward cols, not recomputed in analog path (`report.py:424-443`). |

Caveat held back from "pass": the current-signal regime label uses in-sample quantiles
(P1-4) - not a future-data lookahead for the latest row, but a definitional inconsistency.

---

## 5. Math / methodology verification log

| Check | Outcome |
|---|---|
| Units in percentage points (epsilon = inflation - baseline; TINF = rolling mean) | PASS `features.py:114-117` |
| bp = delta_percent x 100 | PASS `market_linkage.py:149` |
| Live-safe baselines are `.shift(1)`; full_sample/unshifted flagged | PASS `features.py:75-80`, `BASELINE_META` |
| Forward alignment `shift(-h)`; terminal rows NaN | PASS `validation.py:168-170` |
| Decay `1 - rho_T*mu^(h-1)`; `t* = 1 + ln(0.05/rho_T)/ln(mu)`; gate `rho_T>0 & 0<mu<1` | PASS `models.py:176-178,167` |
| Intercept `c` estimated but unused (paper deviation) | PASS, disclosed `docs/10:75` (surface in app: P2-9) |
| AR(1) lag extracted by name not index | PASS `models.extract_l1_param:99-108` |
| Rolling rho end-date aligned | PASS `models.py:134` |
| HC1 robust OLS; ADF (null unit root) / KPSS (null stationary) labels | PASS `diagnostics.py:27-30`, `models.robust_ols:48` |
| Candidate (a): `cpi_persistence = current + (current - current.shift(h))` is momentum, not random walk | CONFIRMED `benchmarks.py:186` - defensible but name could mislead |
| Candidate (b): `mean_reversion` structurally never "persistent" | CONFIRMED -> P1-3 |
| Candidate (c): persistent-class denominator conditional on current positive-shock rows | CONFIRMED + disclosed `app:887-892` (`positive_shock_persistent` NA elsewhere) |

---

## 6. Test & smoke results

| Command | Result |
|---|---|
| `ruff check .` | **All checks passed!** |
| `pytest -q` | **84 passed in 18.63s** (network-free) |
| `python -m compileall src app scripts` | **clean** |
| Offline end-to-end smoke (demo + synthetic market, network disabled) | **PASS** - fallback chain disclosed (`fred_api: failed; fred_csv: failed; cached_fred: ok`); P1 frame 260x103; P2 all 5 models; P3 scorecard/verdict/win-rates; P4 panel+channels+bp col; P5 report with all 7 sections (with and without market) |

**Coverage gaps:** `diagnostics.py` and `plots.py` untested (P2-8). The Streamlit app body
is not import-smokeable headlessly (no `__main__` guard; runs top-to-bottom), so its
computational wiring was validated indirectly via the report build, which exercises the
same module calls.

---

## 7. UI / UX assessment

- **Strong:** live-safe vs ex-post warnings are consistent and prominent on every relevant
  tab (sidebar `:304`, validation `:427-429`, benchmarks `:812-816`, robustness `:1117`,
  report `:928-932`); data-source/freshness status line is thorough (`:318-338`); market
  linkage repeatedly states "descriptive, not a trading signal" (`:566-576`);
  `section_notes` explainers are dense but genuinely useful.
- **Fix-worthy:** tab order inverts P2->P4 (P2-2); `mean_reversion`/`no_change` degenerate
  metrics unflagged (P1-3); market-linkage horizon options omit 36M while it is computed
  (P2-5); eager recompute can make interactions slow (P2-1).
- **Consistency:** the "Current Macro Signal" regime can differ from the analog/validation
  regime because of the dual definition (P1-4).

---

## 8. Documentation / governance drift

- **R1 (P1-1):** `NEXT_TASKS.md` active gate (Phase 2) lags the committed Phases 2-4 and
  implemented Phase 5; roadmap "commit each gate first" rule violated.
- **R2 (P2-6):** README stale (2 series only, `paper_window` name, omits P2-P5).
- **R7 (P1-2):** vintage/revision caveat absent; "Real-time signal mode" label overstated.
- **R3 (P1-5):** trader layer implemented + tested ahead of its "Trader research mode"
  gate; broader-than-rates content vs the enforced rates-only registry.
- Many governance docs untracked; `docs/09` modified-uncommitted; the implementation is
  otherwise faithful to the playbook's methodology rules (`docs/10:60-77`: rolling_36_shifted
  default, full_sample only in paper window, trailing-current TINF, TB3MS proxy).

---

## 9. Prioritized remediation backlog + suggested next gate

**Suggested next gate: "Phase 5 closeout + governance reconciliation"** (no new features).

1. **Governance first (P1-1):** update `NEXT_TASKS.md` to the real gate; commit Phase 2-5
   code + docs; align `docs/09`. Removes the single biggest source of confusion.
2. **Honesty caveats (P1-2, P1-3):** add the vintage/revision caveat + soften "real-time";
   add the degenerate-metric note for `mean_reversion`/`no_change`. Low effort, high trust.
3. **Scope decision on the trader layer (P1-5):** gate-as-future or remove; reconcile with
   rates-only.
4. **Regime-definition consistency (P1-4):** document or unify the current vs walk-forward
   regime labels.
5. **Polish (P2):** cache heavy builders (P2-1); reorder tabs (P2-2); delete the stray file
   (P2-7); refresh README (P2-6); centralize duplicated helpers (P2-4); align horizon
   options (P2-5); add diagnostics/plots tests (P2-8); surface the decay deviation note
   (P2-9); retire `importlib.reload` guards once stable (P2-3).

No code was changed in this audit. Items above are recommendations only; awaiting separate
approval before any fix.

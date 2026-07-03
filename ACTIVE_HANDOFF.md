# ACTIVE HANDOFF - Audit-Fix Cycle

**As of:** 2026-07-02 · **Owner loop:** full read-only audit → audit-fix commit (pandas-3 validation labels + app reload guards + doc refresh); awaiting push approval

> **This is a LIVING document.** When a task completes, **update the relevant section
> in place** (flip status, rewrite "Next action"); do **not** append a running log.
> Its job: let a fresh Claude session resume after `/clear` with no prior chat context.
>
> Division of labor: `reports/PHASE_0_5_AUDIT_FINDINGS.md` is the **immutable audit record**
> (point-in-time, line-referenced). **This file** is the **live status** of the fix cycle.

---

## 0. Read first (orientation, ~5 min)

1. `CLAUDE.md` - project rules (ask-when-ambiguous, units in pp, no-commit-without-approval).
2. `reports/PHASE_0_5_AUDIT_FINDINGS.md` - the full audit (P0/P1/P2 findings, file:line evidence).
3. `NEXT_TASKS.md` - current active gate (kept in sync with this file).
4. `docs/09_PRODUCTION_ROADMAP.md` + `docs/10_AGENT_EXECUTION_PLAYBOOK.md` - phase gates & methodology rules.
5. `docs/DASHBOARD_UI_POLISH_PLAN.md` - the UI/UX polish plan (batches 1–4 committed + pushed through `0d9ddb3`; arc complete).

Project in one line: a Streamlit research tool operationalizing the Peron & Bonaparte
"transitory inflation" paper (SSRN 4007466); three separate layers - paper replication
(ex-post), live macro signal (no-lookahead), historical validation. Not a trading system.

---

## 1. Where we are right now (rewrite when state changes)

**Active work: 2026-07-02 audit-fix commit — COMMITTED locally, awaiting `git push` approval.**
A full read-only audit (verdict: PASS WITH MINOR FIXES) found
one visible bug already on `origin/main`: under **pandas 3**, `groupby(list-of-1)` yields
1-tuple keys, and `validation._forward_outcome_summary_by_groups` re-wrapped them, so the
single-key summaries (`forward_outcome_summary_by_regime` / `..._by_short_term_pressure`)
emitted tuple labels (`('neutral',)`). Impact: Validation-tab by-regime / by-pressure bar
charts showed tuple text as x-categories and silently dropped `REGIME_ORDER`/`PRESSURE_ORDER`;
string-equality filters on those frames returned empty. Rates/counts were correct; the combined
2-key summary and all market-linkage summaries (already guarded) were unaffected. The fix
mirrors `market_linkage.py`'s `isinstance` guard and adds a label regression test
(`test_single_key_summary_labels_are_plain_strings`). The same commit deletes the app-top
`importlib.reload` guard block that batch 1 (`affbb0a`) had accidentally reintroduced after
P2-3 (`e9462d0`) removed it — two of its conditions referenced dropped `*_SIGNATURE_GUARD`
attrs and were permanently true, silently re-importing `benchmarks`/`robustness` every rerun —
plus a doc refresh (this file, `NEXT_TASKS.md`, playbook §2/§3, project-context stage line,
`.env.example` `BASELINE_METHOD` removal). Gates green: ruff clean · pytest **118 passed** ·
compileall OK · offline AppTest smoke 0 exceptions with plain-string chart categories asserted.

Previous arc, unchanged below for context:

**Dashboard UI/UX polish — batch 4 ("report + light touches") is COMMITTED as
`0d9ddb3` and PUSHED (`origin/main` = the docs-refresh on top of `0d9ddb3`); batches 1–3 preceded
it, so the UI-polish arc is now COMPLETE.** Presentation-only; **no methodology, numbers, series,
or logic changed; every caveat's text preserved, only relocated into expanders.** What batch 4
added (per `docs/DASHBOARD_UI_POLISH_PLAN.md`), all reusing the batch-1/2/3 template and adding
**no new `plots.py` figure** (so `tests/test_plots.py` is unchanged):
- **Tab 9 (Macro Research Report)** — headline kept visible, current regime rendered as the tab-1
  snapshot **metric cards** (regime badge + CPI YoY/ε, Baseline, TINF 4M, TINF 4M percentile, all
  `delta_color="off"`) via `latest_signal_snapshot` (same values the report builder uses); a
  `st.divider()` between all 7 sections; every supporting DataFrame moved behind expanders;
  narrative bullets + empty-state `st.info` stay visible.
- **Tab 7 (Decay / Convergence)** — convergence headline as 3 **metric cards** (decayed in 6m /
  12m, time-to-95% t* with years as the delta) from the first valid window, before the charts;
  the `valid_decay` frame is reused for the decay-curve block (dropped a duplicate recompute).
  Both figures were already themed — untouched.
- **Tab 6 (Paper Framework)** — correlation matrix rendered via `plots_mod.heatmap_figure` (same
  RdBu / zmid=0 convention as the market-linkage grid), numeric table behind an expander;
  OLS / Ljung-Box left as tables.

Gates green (this loop): ruff clean · pytest **117 passed** (unchanged — no new/removed tests) ·
compileall OK · offline `AppTest` smoke renders all 9 tabs, **0 exceptions** for both the
`rolling_36_shifted` (live-safe) and `full_sample` (ex-post) baselines (the Framework heatmap and
the Report/Decay cards were each asserted to render; the heatmap trace type checked directly).
**Next:** the UI-polish arc is complete; project returns to maintenance — see §5.

---

Deep read-only audit is **complete** and **Phases 0-5 are closed**. **All five P1 findings
are fixed and committed** (`d2cb783`, "close out Phase 5 ... reconcile governance"). **All
eight open P2 code-health items are done and committed** (see section 3). The follow-on
**Streamlit width-deprecation cleanup is also done and committed** (`1c1d90c`: every
`use_container_width=True` -> `width="stretch"`; no methodology or output change). Checks are
green: ruff clean · pytest **92 passed** · compileall OK · plus an offline Streamlit `AppTest`
smoke (full app body, caching active, network forced offline -> demo/cache fallback, 0 exceptions).

**Committed and now pushed to `origin/main`:** the maintenance chain `e9462d0` (P2 code-health
polish), `b476ed7` (governance / research / audit docs), `5995a5e` (closeout reconciliation),
`1c1d90c` (Streamlit width cleanup), `dfa0ded` + `417eb4b` (handoff / NEXT_TASKS refresh), plus
the Trader Research feature below (`cbfb2a0`). HEAD is `76284e6` (the handoff-refresh commit
that landed after this paragraph was first written); `git status` is clean except `.claude/`
(project command defs, intentionally kept local/untracked).

**Trader Research mode — committed + pushed (`cbfb2a0`):** the **Trader Research Scope
Decision** was made by the user — **descriptive, rates-only, keep the shelved trader layer
un-wired, surface as a new Streamlit tab** — and shipped: new module
`src/transitory_inflation/trader_research.py` (a current-state-conditioned, live-safe reading
of the Phase 4 linkage; it conditions on the **walk-forward** bucket, NOT the ex-post snapshot
regime) + a new **Trader Research** tab in `app/streamlit_app.py` +
`tests/test_trader_research.py` (9 tests) + doc sync (decision log, research spec §3, README,
NEXT_TASKS). Checks green: ruff clean · pytest **101 passed** (92 prior + 9 new) · compileall
OK · offline `AppTest` smoke renders all 9 tabs incl. Trader Research, 0 exceptions.

---

## 2. Environment & checks (how to verify anything)

- **venv:** `.venv` (Python 3.12.10). Tools: `.venv\Scripts\ruff.exe`, `.venv\Scripts\python.exe`.
- **Run the full check sweep** (CI mirrors these):
  - `& .\.venv\Scripts\ruff.exe check .`
  - `& .\.venv\Scripts\python.exe -m pytest -q`
  - `& .\.venv\Scripts\python.exe -m compileall src app scripts -q`
- **Current check status:** ruff clean · **pytest 112 passed** · compileall OK · `git diff --check` clean.
- **Offline:** FRED CSV is flaky from this machine - keep work network-free. `pytest` is
  network-free (monkeypatched `requests.get`). An offline phase smoke is available via
  `data.make_demo_data()` + the phase builders (no network).
- **Windows/PowerShell:** never round-trip file content through PowerShell 5.1 (UTF-8
  corruption); use the Read/Write/Edit tools. `git diff` prints harmless LF->CRLF warnings.

---

## 3. Remediation status (THE tracker - flip OPEN/DONE here in place)

**P0 (correctness / lookahead / secrets): NONE.** No fixes needed.

| ID | Item | Status | Where it landed / note |
|---|---|---|---|
| P1-1 | Governance / gate drift | **DONE** | `NEXT_TASKS.md` rewritten to "Phase 5 closeout" gate |
| P1-2 | Vintage/revision caveat + soften "real-time" | **DONE** | caveat in `report._caveats` (+ test); `docs/01_RESEARCH_SPEC.md` -> "Live signal mode (no full-sample lookahead)" |
| P1-3 | Degenerate benchmark metrics unflagged | **DONE** | benchmark-tab captions in `app/streamlit_app.py` (mean_reversion never-persistent; no_change ~0 directional) |
| P1-4 | Dual regime definitions | **DONE** | clarity caveat in `report._caveats` (+ `walk-forward` test assert) |
| P1-5 | Orphaned trader layer | **DONE (gated)** | pre-exists in HEAD; kept + future/experimental + not-wired notes on `REGIME_PLAYBOOK` and `build_trader_report`; already un-wired from app |
| P2-1 | Eager recompute / heavy builders uncached | **DONE** | `@st.cache_data` wrappers in app for benchmarks/validation/threshold-sensitivity/market-linkage/robustness/report; non-hashable status dataclasses excluded via leading-`_` args |
| P2-2 | Tab order inverts P2->P4 | **DONE** | Benchmarks now precede Market Linkage in `st.tabs(...)` (handles + labels swapped; `with`-blocks unchanged, routed by handle) |
| P2-3 | `importlib.reload` guards | **DONE (re-fixed 2026-07-02)** | removed in `e9462d0`; UI-polish batch 1 (`affbb0a`) accidentally reintroduced the app-top block (two conditions referenced dropped `*_SIGNATURE_GUARD` attrs → always-true → reload every rerun); deleted again in the 2026-07-02 audit-fix commit |
| P2-4 | Duplicated helpers | **DONE** | `pressure_label` -> `validation.pressure_label`; `date_label` + `latest_valid_observation_date` -> `data` (latter made graceful on missing cols); app + report import them |
| P2-5 | Horizon option sets inconsistent | **DONE** | market-linkage selectbox gains 36M (already computed); validation + benchmark gain 3M (validation frame built with the selected horizon as a label horizon) |
| P2-6 | README drift | **DONE** | README now lists 5 inflation + 6 market series, `paper_replication`, and the P1-P5 tabs/phases |
| P2-7 | Stray file `tatus --short` | **DONE** | deleted (prior cycle) |
| P2-8 | Test coverage gaps | **DONE** | added `tests/test_diagnostics.py` + `tests/test_plots.py` (network-free) |
| P2-9 | Decay paper-deviation note not in app | **DONE** | decay-tab note: published formula uses `rho_T` + `mu` only; intercept `c` estimated-but-unused |
| P2-10 | Explosive-but-valid decay | **NOTE ONLY** | intentional + warned; no change |

What each P1 fix changed (so you don't need the chat):
- **P1-1:** active gate corrected from stale "Phase 2" to "Phase 5 closeout"; lists closeout tasks.
- **P1-2:** new caveat states the signal uses latest-revised FRED data and is not a real-time
  data-vintage backtest ("live-safe" = no full-sample lookahead). Spec label softened.
- **P1-3:** captions explain the two structural artifacts so naive baselines are not mis-ranked.
- **P1-4:** caveat states live-snapshot vs walk-forward labels are related-not-identical;
  analogs are historical comparisons, not exact regime-identity.
- **P1-5:** trader layer (`build_trader_report`/`REGIME_PLAYBOOK`) is `+626/-0` vs HEAD i.e.
  untouched/pre-existing; HEAD's app wired it, the Phase 5 working tree un-wired it. Kept &
  annotated as future/experimental; NOT surfaced in Streamlit.

---

## 4. Working tree + commit history (refresh when files change / after commit)

**Status: the 2026-07-02 audit-fix commit is COMMITTED locally on top of `dea3db2`
(= `origin/main`) and AWAITS PUSH APPROVAL. Files in it:
`src/transitory_inflation/validation.py` (isinstance guard), `tests/test_validation.py`
(label regression test), `app/streamlit_app.py` (reload-guard block deleted),
`NEXT_TASKS.md`, `ACTIVE_HANDOFF.md`, `docs/10_AGENT_EXECUTION_PLAYBOOK.md`,
`docs/00_PROJECT_CONTEXT.md`, `.env.example` (doc refresh). The working tree is otherwise
clean apart from the always-untracked `.claude/`.** Never `git add -A`.

Prior arc (all pushed): batches 1–4 (batch 4 feat = `0d9ddb3`, plus the docs-refresh
`dea3db2` on top).

**Batch 4 landed in `0d9ddb3`** (`app/streamlit_app.py` only — no `plots.py`/test change),
followed by this handoff/NEXT_TASKS/plan refresh. The offline AppTest smoke is run ad hoc (a temp
script kept outside the repo, not committed).

**What batch 4 changed (landed in `0d9ddb3`):**
- `app/streamlit_app.py` — Report tab: snapshot metric cards (via `latest_signal_snapshot`) +
  `st.divider()` between the 7 sections + all supporting DataFrames behind expanders. Decay tab:
  3 convergence metric cards from the first valid window (`valid_decay` reused for the curve
  block). Framework tab: correlation matrix via `plots_mod.heatmap_figure` + table behind an
  expander.
- No `src/transitory_inflation/plots.py` change and no `tests/test_plots.py` change — batch 4
  reuses already-tested figures (`heatmap_figure`); the gate is the existing 117-test suite plus
  the offline AppTest smoke.

**Dashboard UI-polish commit arc (all PUSHED to `origin/main` through `0d9ddb3` + this refresh):**
- `affbb0a` - batch 1 (shared theme, glossary, Current Macro Signal rebuild, Trader Research range
  plot) + plan doc.
- `9771837` - handoff + NEXT_TASKS refresh after batch 1.
- `1a0f5ab` - batch 2 (chart-ify Validation + Market Linkage: 4 new figures, tables behind
  expanders) + the three doc refreshes.
- `56b7037` - batch 3 (chart-ify Benchmark + Robustness: diverging improvement chart + verdict
  badges, win-rate bars, conditional formatting) + the three doc refreshes.
- `cc7f216` - handoff / NEXT_TASKS / plan refresh after batch 3.
- `0d9ddb3` - batch 4 (Report cards/dividers + expanders, Decay convergence cards, Framework
  correlation heatmap); app-only, no new figure/test.
- (this commit) - handoff / NEXT_TASKS / plan refresh after batch 4. **Last pushed.**

Earlier history (all pushed; see `git log` for the full chain): `4b66470` maintenance reconcile ·
`cbfb2a0` Trader Research feature · `e9462d0` P2 code-health · `b476ed7` governance/research docs ·
`1c1d90c` Streamlit width migration.

Working tree (current): **clean** apart from the always-untracked `.claude/`. Local `main` ==
`origin/main` == this refresh on top of `0d9ddb3`. Everything through batch 4 is committed and
pushed.

### Commit-readiness catches (do NOT miss)
1. **`.env`** is gitignored - never commit it.
2. **Generated logs are now gitignored** (`artifacts/*.log` + `artifacts/*.err`, added this
   cycle and verified with `git check-ignore`). Still never `git add -A`; use the explicit
   allowlist below.

### Commits that landed (the executed allowlist; we never used `git add -A`)

**Commit 1 - P2 code-health polish** (landed as `e9462d0`). `git add`: `README.md app/streamlit_app.py
src/transitory_inflation/benchmarks.py src/transitory_inflation/data.py
src/transitory_inflation/report.py src/transitory_inflation/robustness.py
src/transitory_inflation/validation.py tests/test_benchmarks.py tests/test_robustness.py
tests/test_diagnostics.py tests/test_plots.py .gitignore`
Message: `chore: complete Phase 5 P2 code-health backlog (cache, tab order, dedup, horizons, tests)`

**Commit 2 - governance / research docs** (landed as `b476ed7`). `git add`: `AGENTS.md CLAUDE.md ACTIVE_HANDOFF.md
NEXT_TASKS.md docs/00_PROJECT_CONTEXT.md docs/02_DATA_CONTRACT.md docs/03_AGENT_WORKFLOW.md
docs/04_VALIDATION_PROTOCOL.md docs/05_DASHBOARD_BRIEF.md docs/06_DECISION_LOG.md
docs/08_LOCALHOST_REVIEW.md docs/CLAUDE_PROMPTS.md docs/IMPLEMENTATION_PLAN.md
docs/METHODOLOGY.md docs/PAPER_FORMULA_REFERENCE.md docs/PAPER_NOTES.md
docs/RESEARCH_CHECKLIST.md docs/decisions/.gitkeep prompts/
reports/PHASE_0_5_AUDIT_PLAN.md reports/PHASE_0_5_AUDIT_FINDINGS.md
data/external/.gitkeep data/interim/.gitkeep data/processed/.gitkeep data/raw/.gitkeep
artifacts/exports/.gitkeep artifacts/snapshots/.gitkeep notebooks/archive/.gitkeep
reports/figures/.gitkeep reports/notes/.gitkeep reports/tables/.gitkeep`
Message: `docs: track governance, research, and audit docs`

`.claude/` (project skill/command defs) is optional - often kept local; add it only to share
the command defs in the repo. A single combined commit is also fine if preferred.

---

## 5. Next action (REWRITE this each loop iteration)

**This loop ran the full read-only audit and implemented the audit-fix commit** (§1 has the
full detail): pandas-3 tuple-label fix in `validation.py` + regression test, app reload-guard
block deleted, docs refreshed. Methodology/numbers unchanged except that the Validation tab's
by-regime / by-pressure charts and tables now show plain string labels with the intended
category order (the underlying rates/counts were always correct).

**Immediate next step:** get explicit user approval to `git push origin main` for the
audit-fix commit. After that, the project returns to maintenance. Any further visual work, or
either deferred item below, needs a fresh, separately-scoped user decision.

Deferred / out of scope until a fresh, separately-scoped user decision (unchanged by this loop):
- a **predictive** linkage — out-of-sample scoring of market moves vs descriptive base rates;
- any **asset-universe expansion** beyond the six approved FRED rate series.

---

## 6. Working rules (guardrails for every iteration)

- **Never commit, push, or `git add` without explicit user approval.** Never `git add -A`.
- Keep changes **small and surgical**; no new features, no broad refactors during this cycle.
- **Run the full check sweep (section 2) and confirm green before ending a turn.**
- Honor `CLAUDE.md`: inflation in **percentage points**; report ex-post vs live-safe vs
  experimental; don't hardcode absolute paths; don't commit secrets; don't delete data/
  reference files without permission.
- **Maintain this file in place:** flip statuses in section 3, refresh section 4 after any
  file change, rewrite section 5. Keep `NEXT_TASKS.md` consistent with section 3.
- End substantive work with the `CLAUDE.md` loop block (What changed / Files touched /
  Checks run / Open questions / Recommended next step).

---

## 7. Quick self-orientation commands (read-only)

```
git status --short                 # working tree (expect only .claude/ untracked; see section 4)
git diff --name-only               # tracked, unstaged changes
git log --oneline -8               # history (initial df93a57 .. Phase 4)
.venv\Scripts\python.exe -m pytest -q
```

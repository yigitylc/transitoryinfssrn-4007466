# ACTIVE HANDOFF - Audit-Fix Cycle

**As of:** 2026-06-27 · **Owner loop:** Dashboard UI/UX polish - batch 3 (evidence tabs)

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
5. `docs/DASHBOARD_UI_POLISH_PLAN.md` - **active work**: the UI/UX polish plan (batches 1–2 committed + pushed through `1a0f5ab`; batch 3 done/uncommitted; batch 4 next).

Project in one line: a Streamlit research tool operationalizing the Peron & Bonaparte
"transitory inflation" paper (SSRN 4007466); three separate layers - paper replication
(ex-post), live macro signal (no-lookahead), historical validation. Not a trading system.

---

## 1. Where we are right now (rewrite when state changes)

**Active work: Dashboard UI/UX polish — batch 3 ("evidence tabs") is implemented in the working
tree (UNCOMMITTED, needs review/commit approval). Batches 1–2 are COMMITTED and PUSHED to
`origin/main` through `1a0f5ab`.** Presentation-only; **no methodology, numbers, series, or logic
changed; every caveat's text preserved, only relocated into expanders.** What batch 3 added (per
`docs/DASHBOARD_UI_POLISH_PLAN.md`), all reusing the batch-1/2 template:
- **`plots.py`** — one new figure `improvement_diverging_figure` (horizontal diverging bars of the
  TINF model's MAE/RMSE improvement % vs each naive baseline; cold = TINF wins / lower error, hot =
  trails; dotted zero line). `hit_rate_bar_figure` gained two **additive** kwargs — `yaxis_title`
  and an optional `reference` guide line — so it also backs the robustness win-rate bars (batch-2
  callers unchanged; defaults preserve old behavior). Each change covered in `tests/test_plots.py`
  (return-type + trace-count + empty-frame + the new reference/axis params): **+5 tests**.
- **Tab 3 (Benchmark)** — two verdict **badges** (vs no-change / vs mean-reversion, metric cards),
  the diverging improvement chart as the headline; the metric summary stays visible; the
  improvement / classification / 50-row forecast-audit tables moved behind expanders; intro folded
  into `section_notes` + `scope_caveats()`.
- **Tab 8 (Robustness)** — **win-rate bars by setting** (MAE | RMSE side by side, each with a 0.5
  coin-flip reference line) reusing `hit_rate_bar_figure`; data-status / availability / scorecard /
  verdict / win-rate table moved behind expanders; the baseline quick-comparison table given
  hot/cold **regime conditional formatting** via a new `style_regime_cells()` Styler helper; intro
  folded into `section_notes` + `scope_caveats()`. Stationarity diagnostics left as a table.

Gates green (this loop): ruff clean · pytest **117 passed** (112 prior + 5 new plot tests) ·
compileall OK · offline `AppTest` smoke renders all 9 tabs, **0 exceptions / 0 errors** (fully
offline; the new Benchmark diverging chart, verdict badges, and Robustness win-rate chart sections
were each asserted to render their non-empty path). **Next:** review + commit batch 3 (then push,
when approved); then batch 4 (Report + light touches) — see §5. Read
`docs/DASHBOARD_UI_POLISH_PLAN.md` first.

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
| P2-3 | `importlib.reload` guards | **DONE** | removed app-top guards + inline guards in `benchmarks.py`/`robustness.py`; dropped now-unused `importlib`/`inspect`/`*_SIGNATURE_GUARD`; 2 obsolete reload-recovery tests removed |
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

**Status: batches 1–2 are COMMITTED and PUSHED to `origin/main` (HEAD = `1a0f5ab`). Batch-3 UI
polish (evidence tabs) is implemented in the WORKING TREE and is UNCOMMITTED — review/commit
approval pending. Both commit and push need explicit approval.** Never `git add -A`.

**Uncommitted now (batch 3 — to `git add` when approved):** `src/transitory_inflation/plots.py`,
`app/streamlit_app.py`, `tests/test_plots.py`, plus this doc refresh
`docs/DASHBOARD_UI_POLISH_PLAN.md`, `NEXT_TASKS.md`, `ACTIVE_HANDOFF.md`. The only always-untracked
path remains `.claude/`. The offline AppTest smoke is run ad hoc (a temp script kept outside the
repo, not committed).

**What batch 3 touches (for the commit allowlist):**
- `src/transitory_inflation/plots.py` — new `improvement_diverging_figure`; `hit_rate_bar_figure`
  gained additive `yaxis_title` + `reference` kwargs (batch-2 callers unchanged).
- `app/streamlit_app.py` — Benchmark verdict badges + diverging chart + expanders; Robustness
  win-rate bars + expanders; new `style_regime_cells()` helper + `WIN_RATE_SPECS_*` constants;
  `improvement_diverging_figure` added to the plot-reload guard.
- `tests/test_plots.py` — +5 tests (diverging return/empty/unknown-model; hit-rate reference+axis;
  hit-rate no-reference default).

**Dashboard UI-polish commit arc (all PUSHED to `origin/main` through `1a0f5ab`):**
- `affbb0a` - batch 1 (shared theme, glossary, Current Macro Signal rebuild, Trader Research range
  plot) + plan doc.
- `9771837` - handoff + NEXT_TASKS refresh after batch 1.
- `1a0f5ab` - batch 2 (chart-ify Validation + Market Linkage: 4 new figures, tables behind
  expanders) + the three doc refreshes. **Last pushed (HEAD == origin/main).**

Earlier history (all pushed; see `git log` for the full chain): `4b66470` maintenance reconcile ·
`cbfb2a0` Trader Research feature · `e9462d0` P2 code-health · `b476ed7` governance/research docs ·
`1c1d90c` Streamlit width migration.

Working tree (current): **batch-3 changes are uncommitted** (the three code/test files + the three
docs listed above), apart from the always-untracked `.claude/`. Local `main` == `origin/main` ==
`1a0f5ab`; batch 3 is not yet a commit. Everything earlier remains committed and pushed.

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

**This loop implemented Dashboard UI-polish batch 3** ("evidence tabs"): Benchmark verdict badges
(vs no-change / vs mean-reversion) + a diverging MAE/RMSE improvement chart (cold = TINF wins, hot
= trails) with the improvement / classification / forecast-audit tables behind expanders;
Robustness win-rate bars by setting (MAE | RMSE, each with a 0.5 reference line) with
scorecard / verdict / status behind expanders and hot/cold regime conditional formatting on the
baseline quick-comparison table (full per-item detail in §1 and
`docs/DASHBOARD_UI_POLISH_PLAN.md`). Presentation-only — **no methodology/numbers/series/logic
changed and every caveat's text kept, only relocated.** One new figure
(`improvement_diverging_figure`) plus additive `yaxis_title`/`reference` kwargs on
`hit_rate_bar_figure` (reused for the win-rate bars), each with tests. Gates green (ruff · pytest
**117 passed** · compileall · offline AppTest smoke 0 exceptions, fully-offline, new chart sections
asserted rendered). **Batch 3 is UNCOMMITTED.**

**Immediate next step:** review batch 3 (localhost eyeball per `docs/08_LOCALHOST_REVIEW.md` if
wanted). When approved, commit batch 3 (code + tests + the three doc refreshes) and push
(`git push origin main` — needs approval). Then start **batch 4 — report + light touches** (Macro
Research Report headline/regime as metric cards + `st.divider()` between sections with supporting
tables behind expanders; Decay metric cards from the first valid window; Paper Framework
correlation-matrix heatmap with the table behind an expander). Reuse the established template
(`apply_macro_theme`, `section_notes`, `scope_caveats`, `render_glossary`, `heatmap_figure`,
palette constants); add a return-type + trace-count + empty-frame test for every new figure. Keep
all methodology/numbers byte-identical and all caveat text (relocate into expanders, never drop).

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

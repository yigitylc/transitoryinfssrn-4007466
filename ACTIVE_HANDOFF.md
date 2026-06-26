# ACTIVE HANDOFF - Audit-Fix Cycle

**As of:** 2026-06-25 · **Owner loop:** Dashboard UI/UX polish - batch 1 (foundation + flagship)

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
5. `docs/DASHBOARD_UI_POLISH_PLAN.md` - **active work**: the UI/UX polish plan (batch 1 done/uncommitted; batch 2 next).

Project in one line: a Streamlit research tool operationalizing the Peron & Bonaparte
"transitory inflation" paper (SSRN 4007466); three separate layers - paper replication
(ex-post), live macro signal (no-lookahead), historical validation. Not a trading system.

---

## 1. Where we are right now (rewrite when state changes)

**Active work: Dashboard UI/UX polish — batch 1 ("Foundation + flagship") is COMMITTED as
`affbb0a` (NOT yet pushed — push needs separate approval; HEAD is 1 ahead of `origin/main`).**
Presentation-only; **no methodology, numbers, series, or logic changed.** What landed (per
`docs/DASHBOARD_UI_POLISH_PLAN.md`):
- **C1 shared theme** — `apply_macro_theme()` in `plots.py`; all 5 figures routed through it
  (consistent font, light gridlines, `hovermode="x unified"`, tight margins, top legend,
  hot/cold palette `HOT`/`COLD`/`NEUTRAL`).
- **C2 glossary + C5 sidebar** — `render_glossary()`/`GLOSSARY_ITEMS` in the sidebar; grouped
  header + compact "Current reading" mini-status line.
- **Tab 1 (Current Macro Signal) rebuilt** — `signal_headline()`, `regime_badge()`, metric cards
  with neutral semantic deltas (`delta_color="off"`); CPI figure epsilon-shaded + current marker;
  TINF figure zone-tinted above/below zero + current-value markers.
- **Tab 5 (Trader Research)** — new `forward_change_range_figure()` (median marker + p25–p75
  whisker, hot/cold by sign) as the headline with the table behind an expander; bucket **metric
  cards**; analog months behind an "Audit trail" expander; `section_notes` + `scope_caveats()`
  applied as the reusable C3/C4 template.
- **Reusable helpers for later batches:** `apply_macro_theme`, `render_glossary`, `scope_caveats`,
  `regime_badge`, `signal_headline`, and the palette constants.

Gates green (this loop): ruff clean · pytest **104 passed** (101 prior + 3 new plot tests) ·
compileall OK · offline `AppTest` smoke renders all 9 tabs, **0 exceptions / 0 errors**.
**Next:** push `affbb0a` (when approved), then batch 2 (Validation + Market Linkage charts) —
see §5. Read `docs/DASHBOARD_UI_POLISH_PLAN.md` first.

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
- **Current check status:** ruff clean · **pytest 104 passed** · compileall OK · `git diff --check` clean.
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

**Status: prior commits are PUSHED to `origin/main` (`4b66470`); batch-1 UI polish is COMMITTED
locally as `affbb0a` (code + plan doc) but NOT pushed. This handoff/NEXT_TASKS refresh is the
follow-on commit. `main` is ahead of `origin/main` by these two commits until pushed —
`git push origin main` needs explicit approval.** Never `git add -A`.

**Committed in `affbb0a` (Dashboard UI polish batch 1 — code + plan doc):**
- `src/transitory_inflation/plots.py` — `apply_macro_theme()`, hot/cold palette, epsilon shading
  on the CPI figure, zone tints + current markers on the TINF figure, new
  `forward_change_range_figure()`; all 5 figures themed.
- `app/streamlit_app.py` — glossary/`scope_caveats`/`regime_badge`/`signal_headline` helpers;
  sidebar glossary + "Current reading" mini-status; tab 1 executive rebuild; tab 5 range plot +
  cards + expanders.
- `tests/test_plots.py` — trace-count assertions updated for shading/markers; tests for
  `apply_macro_theme` and `forward_change_range_figure` (incl. empty-frame branch).
- `docs/DASHBOARD_UI_POLISH_PLAN.md` (new) — plan with batch 1 marked done.

**Committed in this follow-on refresh:** `ACTIVE_HANDOFF.md` + `NEXT_TASKS.md` (this status update).

**To push when approved:** `git push origin main` (sends `affbb0a` + this refresh). The only
always-untracked path is `.claude/` (project command defs, intentionally local).

History (pushed to `origin/main` through `4b66470`; the last two are local-only until pushed):
- `d2cb783` - Phase 5 code + the three then-modified docs (`NEXT_TASKS.md`,
  `docs/01_RESEARCH_SPEC.md`, `docs/09_PRODUCTION_ROADMAP.md`).
- `e9462d0` - P2 code-health polish (cache, tab order, dedup, horizons, tests).
- `b476ed7` - governance / research / audit docs.
- `5995a5e` - closeout reconciliation of the living status docs.
- `1c1d90c` - Streamlit width-parameter migration (`use_container_width` -> `width="stretch"`).
- `dfa0ded` + `417eb4b` - handoff / NEXT_TASKS refresh after maintenance cleanup.
- `cbfb2a0` - Trader Research feature (module + tab + 9 tests + doc sync).
- `76284e6` - handoff refresh after Trader Research.
- `4b66470` - NEXT_TASKS + handoff reconcile to pushed maintenance state (**last pushed**).
- `affbb0a` - Dashboard UI polish batch 1 (shared theme, glossary, tab 1 + tab 5) — **local-only**.
- (this commit) - handoff + NEXT_TASKS refresh after batch 1 — **local-only**.

Working tree (current): **clean after this refresh commit**, apart from the always-untracked
`.claude/` (project command defs). Local `main` carries the two unpushed commits above on top of
`origin/main` (`4b66470`); everything earlier remains committed and pushed.

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

**This loop implemented AND committed Dashboard UI-polish batch 1** ("Foundation + flagship"):
shared Plotly theme, glossary, rebuilt Current Macro Signal, and the Trader Research range plot
(full per-item detail in §1 and `docs/DASHBOARD_UI_POLISH_PLAN.md`). Presentation-only; gates
green (ruff · pytest **104 passed** · compileall · offline AppTest smoke 0 exceptions). Landed as
`affbb0a` plus this handoff refresh — **both local-only, NOT pushed** (push needs approval).

**Immediate next step:** when ready, `git push origin main` to publish `affbb0a` + this refresh
(verify with a localhost eyeball first per `docs/08_LOCALHOST_REVIEW.md` if not already done).
Then start **batch 2 — table-heavy tabs** (Validation: hit-rate bars, threshold-sensitivity line,
regime-transition heatmap; Market Linkage: grouped forward-change bars, correlation heatmap,
expanders). Reuse the batch-1 template (`apply_macro_theme`, `section_notes`, `scope_caveats`,
`render_glossary`, palette constants); add a return-type + trace-count + empty-frame test for
every new figure (mirror `forward_change_range_figure`). Keep all methodology/numbers
byte-identical and all caveat text (relocate into expanders, never drop).

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

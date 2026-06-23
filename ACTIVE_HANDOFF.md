# ACTIVE HANDOFF - Audit-Fix Cycle

**As of:** 2026-06-23 · **Owner loop:** Phase 5 closeout + governance reconciliation

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

Project in one line: a Streamlit research tool operationalizing the Peron & Bonaparte
"transitory inflation" paper (SSRN 4007466); three separate layers - paper replication
(ex-post), live macro signal (no-lookahead), historical validation. Not a trading system.

---

## 1. Where we are right now (rewrite when state changes)

Deep read-only audit is **complete**. **All five P1 findings are fixed and committed**
(`d2cb783`, "close out Phase 5 ... reconcile governance"). **All eight open P2 code-health
items are now done and committed** (see section 3). Checks are green:
ruff clean · pytest **92 passed** · compileall OK · plus an offline Streamlit `AppTest` smoke
(full app body, caching active, network forced offline -> demo/cache fallback, 0 exceptions).

**Committed** this cycle as `e9462d0` (P2 code-health polish) and `b476ed7` (governance /
research / audit docs). Working tree is clean except `.claude/` (project command defs,
intentionally kept local/untracked).

---

## 2. Environment & checks (how to verify anything)

- **venv:** `.venv` (Python 3.12.10). Tools: `.venv\Scripts\ruff.exe`, `.venv\Scripts\python.exe`.
- **Run the full check sweep** (CI mirrors these):
  - `& .\.venv\Scripts\ruff.exe check .`
  - `& .\.venv\Scripts\python.exe -m pytest -q`
  - `& .\.venv\Scripts\python.exe -m compileall src app scripts -q`
- **Current check status:** ruff clean · **pytest 85 passed** · compileall OK · `git diff --check` clean.
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

## 4. Uncommitted working tree + commit plan (refresh when files change / after commit)

**Status: NOT committed, NOT staged. A commit needs explicit user approval.**

`d2cb783` already committed the Phase 5 code + the three then-modified docs
(`NEXT_TASKS.md`, `docs/01_RESEARCH_SPEC.md`, `docs/09_PRODUCTION_ROADMAP.md`). What remains
is this P2 polish cycle plus the governance docs that commit left untracked.

Modified tracked (P2 polish + hygiene + living docs):
`.gitignore`, `NEXT_TASKS.md`, `README.md`, `app/streamlit_app.py`,
`src/transitory_inflation/{benchmarks,data,report,robustness,validation}.py`,
`tests/test_{benchmarks,robustness}.py`.

New + untracked: `tests/test_{diagnostics,plots}.py`; `ACTIVE_HANDOFF.md`, `AGENTS.md`,
`CLAUDE.md`, `docs/{00,02,03,04,05,06,08}_*.md`, `docs/CLAUDE_PROMPTS.md`,
`docs/IMPLEMENTATION_PLAN.md`, `docs/METHODOLOGY.md`, `docs/PAPER_FORMULA_REFERENCE.md`,
`docs/PAPER_NOTES.md`, `docs/RESEARCH_CHECKLIST.md`, `docs/decisions/.gitkeep`, `prompts/*`,
scaffolding `.gitkeep`s, `reports/PHASE_0_5_AUDIT_{PLAN,FINDINGS}.md`, `.claude/*`.

### Commit-readiness catches (do NOT miss)
1. **`.env`** is gitignored - never commit it.
2. **Generated logs are now gitignored** (`artifacts/*.log` + `artifacts/*.err`, added this
   cycle and verified with `git check-ignore`). Still never `git add -A`; use the explicit
   allowlist below.

### Two commits (explicit allowlist; never `git add -A`)

**Commit 1 - P2 code-health polish.** `git add`: `README.md app/streamlit_app.py
src/transitory_inflation/benchmarks.py src/transitory_inflation/data.py
src/transitory_inflation/report.py src/transitory_inflation/robustness.py
src/transitory_inflation/validation.py tests/test_benchmarks.py tests/test_robustness.py
tests/test_diagnostics.py tests/test_plots.py .gitignore`
Message: `chore: complete Phase 5 P2 code-health backlog (cache, tab order, dedup, horizons, tests)`

**Commit 2 - governance / research docs.** `git add`: `AGENTS.md CLAUDE.md ACTIVE_HANDOFF.md
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
the command defs in the repo. A single combined commit is also fine if preferred. Each commit
message ends with the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 5. Next action (REWRITE this each loop iteration)

P2 backlog is fully implemented, green, and **committed** as `e9462d0` (P2 polish) +
`b476ed7` (governance docs) on `main`. The Phase 5 gate is fully closed.

Next: the next roadmap item (Trader research mode / market trade priors) is unblocked but
should be a **deliberate, separately scoped decision** - it conflicts with the current
rates-only registry, so confirm scope before starting. Otherwise the project is in a
maintenance state.

Known non-blocking follow-up: the app passes `use_container_width=` to every `st.dataframe`/
`st.plotly_chart`, which Streamlit now deprecates (runtime warning). A future cleanup switches
to `width=`; out of scope for this P2 cycle.

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
git status --short                 # working tree (expect the M list in section 4)
git diff --name-only               # tracked, unstaged changes
git log --oneline -8               # history (initial df93a57 .. Phase 4)
.venv\Scripts\python.exe -m pytest -q
```

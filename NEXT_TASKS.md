# Next Tasks

**Active gate: NONE — the Dashboard UI/UX Polish arc (batches 1–4) is COMPLETE, committed + pushed
(`origin/main` = the docs-refresh on top of batch-4 feat `0d9ddb3`).** Project returns to
maintenance. See `docs/DASHBOARD_UI_POLISH_PLAN.md` for the full plan and `ACTIVE_HANDOFF.md`
§1/§4/§5 for status.

Batch 4 ("report + light touches") is committed as `0d9ddb3` and pushed (presentation-only; **no
methodology, numbers, series, or logic changed**; every caveat's text preserved, only relocated
into expanders). App-only (`app/streamlit_app.py`); **no `plots.py` or `tests/test_plots.py`
change** — it reuses already-tested figures:
- **Macro Research Report tab:** headline + current regime as the tab-1 snapshot metric cards (via
  `latest_signal_snapshot`); `st.divider()` between the 7 sections; all supporting DataFrames behind
  expanders (narrative bullets + empty-state `st.info` stay visible).
- **Decay / Convergence tab:** 3 convergence metric cards (decayed 6m/12m, time-to-95% t*) from the
  first valid window, before the charts; `valid_decay` reused for the decay-curve block.
- **Paper Framework tab:** correlation matrix as a `plots_mod.heatmap_figure` (RdBu / zmid=0), table
  behind an expander; OLS / Ljung-Box left as tables.

Batches 1–3 (foundation + flagship, table-heavy tabs, evidence tabs) preceded it; the whole
UI-polish arc through batch 4 is pushed to `origin/main`.

Gates (batch 4, at commit): ruff clean · pytest **117 passed** (unchanged — no new/removed tests) ·
compileall OK · offline `AppTest` smoke renders all 9 tabs for both the live-safe and ex-post
baselines (the Framework heatmap + Report/Decay cards asserted), 0 exceptions.

**Next:** none queued — the UI/UX polish arc is finished. Further visual work or either deferred
item (predictive linkage; asset-universe expansion) needs a fresh, separately-scoped user decision.

---

The Trader Research mode (descriptive, rates-only) is **shipped, committed, and pushed** to
`origin/main` (`cbfb2a0`), and the follow-up handoff refresh landed (`76284e6`). The Trader
Research Scope Decision is closed.

Scope decision (2026-06-24, user): descriptive only · rates-only (the six approved FRED rate
series) · keep the shelved `build_trader_report`/`REGIME_PLAYBOOK` layer un-wired · surface as
a new Streamlit tab. No forecasts, no PnL, no recommendations; registry unchanged.

Delivered and landed in `cbfb2a0`:

- `src/transitory_inflation/trader_research.py` - a current-state-conditioned, live-safe
  reading of the Phase 4 market linkage: today's walk-forward regime bucket, the forward
  rate-change distribution (median / p25-p75 / hit rates / count / weak-evidence), the channel
  roll-up, and the analog months behind it. Pure/in-memory; reuses
  `market_linkage.build_market_linkage_tables`.
- New **Trader Research** tab in `app/streamlit_app.py` (after Market Linkage), reusing the
  cached `get_market_data` / `get_market_linkage_tables` builders.
- `tests/test_trader_research.py` - 9 network-free tests.
- Doc sync: `docs/06_DECISION_LOG.md`, `docs/01_RESEARCH_SPEC.md` §3, `README.md`,
  `ACTIVE_HANDOFF.md`.

Checks: ruff clean · pytest **101 passed** (92 prior + 9 new) · compileall OK · offline
Streamlit `AppTest` smoke renders all 9 tabs incl. Trader Research, 0 exceptions.

The full prior work chain and the Trader Research feature are committed and pushed (`origin/main =
4b66470`); batch-1 UI polish (`affbb0a`) + this handoff refresh are committed **local-only** and
await `git push origin main` approval. The active gate is that batch — see the top of this file.
The only always-untracked path is `.claude/`.

Out of scope until a fresh, separately-scoped decision:

- Predictive / out-of-sample forecast scoring of market moves (vs descriptive base rates).
- Any market series beyond the approved FRED rates set
  (`market_data.validate_market_series_registry`) - e.g. equities/FX/vol/commodities.
- Wiring, editing, or deleting the shelved trader layer.

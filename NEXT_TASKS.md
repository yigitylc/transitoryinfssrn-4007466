# Next Tasks

**Active gate: Dashboard UI/UX Polish — batch 1 DONE (committed `affbb0a`, NOT pushed), batch 2
NEXT.** See `docs/DASHBOARD_UI_POLISH_PLAN.md` for the full plan and `ACTIVE_HANDOFF.md` §1/§4/§5
for status.

Batch 1 ("Foundation + flagship") is committed as `affbb0a` (presentation-only; **no methodology,
numbers, series, or logic changed**); local `main` is ahead of `origin/main` by `affbb0a` + the
handoff refresh until pushed (`git push origin main` needs approval):
- C1 shared Plotly theme `apply_macro_theme()` in `plots.py`; all 5 figures routed through it.
- C2 glossary (`render_glossary`) + C5 sidebar "Current reading" mini-status.
- Tab 1 Current Macro Signal rebuilt: headline, colored regime badge, semantic-delta cards,
  epsilon-shaded CPI chart, zone-tinted TINF chart.
- Tab 5 Trader Research: new `forward_change_range_figure()` range plot + bucket metric cards;
  `section_notes` + `scope_caveats()` applied as the reusable C3/C4 template.

Gates: ruff clean · pytest **104 passed** (101 prior + 3 new plot tests) · compileall OK ·
offline `AppTest` smoke renders all 9 tabs, 0 exceptions.

**Next:** push `affbb0a` + the handoff refresh when approved (`git push origin main`), then
batch 2 — table-heavy tabs (Validation hit-rate bars / sensitivity line / transition heatmap;
Market Linkage grouped bars / correlation heatmap / expanders). Reuse the batch-1 template; add a
return-type + trace-count + empty-frame test per new figure.

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

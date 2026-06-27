# Next Tasks

**Active gate: Dashboard UI/UX Polish — batches 1–3 committed + pushed (`origin/main` = `56b7037`),
batch 4 NEXT.** See `docs/DASHBOARD_UI_POLISH_PLAN.md` for the full plan and `ACTIVE_HANDOFF.md`
§1/§4/§5 for status.

Batch 3 ("evidence tabs") is committed as `56b7037` and pushed (presentation-only; **no methodology,
numbers, series, or logic changed**; every caveat's text preserved, only relocated into expanders).
Touched `src/transitory_inflation/plots.py`, `app/streamlit_app.py`, `tests/test_plots.py`:
- **Benchmark tab:** two verdict badges (vs no-change / vs mean-reversion) + a diverging MAE/RMSE
  improvement chart (cold = TINF wins, hot = trails) as the headline; metric summary stays visible;
  improvement / classification / forecast-audit tables behind expanders; intro folded into
  `section_notes` + `scope_caveats()`.
- **Robustness tab:** win-rate bars by setting (MAE | RMSE side by side, each with a 0.5 reference
  line); data-status / availability / scorecard / verdict / win-rate table behind expanders;
  hot/cold regime conditional formatting on the baseline quick-comparison table; intro folded into
  `section_notes` + `scope_caveats()`.
- **`plots.py`:** one new figure `improvement_diverging_figure`; `hit_rate_bar_figure` gained
  additive `yaxis_title` + `reference` kwargs and now also backs the win-rate bars (batch-2 callers
  unchanged). New `style_regime_cells()` Styler helper in the app. **+5 tests.**

Batches 1–2 ("Foundation + flagship" and "table-heavy tabs") preceded it; the whole UI-polish arc
through batch 3 is pushed to `origin/main` (`56b7037`).

Gates (batch 3, at commit): ruff clean · pytest **117 passed** (112 prior + 5 new plot tests) ·
compileall OK · offline `AppTest` smoke renders all 9 tabs (fully-offline; new chart sections
asserted rendered), 0 exceptions.

**Next:** start batch 4 — report + light touches (Macro Research Report cards/dividers + tables
behind expanders; Decay metric cards; Paper Framework correlation heatmap). Reuse the established
template (incl. `heatmap_figure`); add a return-type + trace-count + empty-frame test per new
figure. Keep methodology/numbers byte-identical and all caveat text (relocate into expanders).

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

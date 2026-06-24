# Next Tasks

**No active gate — project in maintenance.** The Trader Research mode (descriptive,
rates-only) is **shipped, committed, and pushed** to `origin/main` (`cbfb2a0`), and the
follow-up handoff refresh landed (`76284e6`). The Trader Research Scope Decision is closed.

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

No next gate. The project is in a maintenance state; the full work chain and the Trader
Research feature are committed and pushed (`HEAD = origin/main = 76284e6`, working tree clean
except the intentionally-local, untracked `.claude/`).

Out of scope until a fresh, separately-scoped decision:

- Predictive / out-of-sample forecast scoring of market moves (vs descriptive base rates).
- Any market series beyond the approved FRED rates set
  (`market_data.validate_market_series_registry`) - e.g. equities/FX/vol/commodities.
- Wiring, editing, or deleting the shelved trader layer.

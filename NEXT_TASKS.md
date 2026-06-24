# Next Tasks

Current gate: **Maintenance** - Phases 0-5 are closed. The next gate is the **Trader
Research Scope Decision**, not implementation.

Phase 5 (the macro research report layer) is implemented and committed (`d2cb783`).
The Phase 0-5 audit's P2 code-health backlog has been worked in full, and the follow-on
Streamlit width-deprecation cleanup is done:

- P2-1 cache heavy Streamlit builders - DONE
- P2-2 tab order (Benchmarks before Market Linkage) - DONE
- P2-3 retire importlib.reload guards - DONE
- P2-4 centralize duplicated helpers (pressure_label / date_label /
  latest_valid_observation_date) - DONE
- P2-5 align horizon option sets (36M market linkage; 3M validation/benchmark) - DONE
- P2-6 refresh README to match config + shipped phases - DONE
- P2-8 add diagnostics/plots smoke tests - DONE
- P2-9 surface decay paper-deviation note in app - DONE
- P2-7 stray file - DONE (prior cycle); P2-10 - note only, no change
- Streamlit width migration (`use_container_width=True` -> `width="stretch"`) - DONE

Committed (all local on `main`, **not pushed**):

- `e9462d0` - P2 code-health polish (code + tests + README + .gitignore).
- `b476ed7` - governance / research / audit docs + scaffolding .gitkeeps.
- `5995a5e` - reconcile living status docs after the closeout commit.
- `1c1d90c` - migrate the Streamlit width parameter (no methodology/output change).
- `dfa0ded` - refresh ACTIVE_HANDOFF after the maintenance cleanup.

The Phase 5 closeout gate is fully closed and the project is in a maintenance state.
The next roadmap item (Trader research mode) is unblocked but conflicts with the current
rates-only registry, so it must be a deliberate, separately-scoped decision before any
work starts. `.claude/` command defs are kept local (untracked).

Checks: ruff clean, pytest 92 passed, compileall OK, offline Streamlit AppTest smoke OK.

Out of scope until the Trader Research Scope Decision is made:

- Trader research mode / market trade priors beyond the rates-only registry.
- Any new market series beyond the approved FRED rates set
  (`market_data.validate_market_series_registry`).

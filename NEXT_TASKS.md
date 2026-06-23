# Next Tasks

Current gate: Phase 5 closeout - code-health polish + governance commit.

Phase 5 (the macro research report layer) is implemented and committed (`d2cb783`).
The Phase 0-5 audit's P2 code-health backlog has now been worked in full:

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

Remaining in this gate:

- Commit the P2 polish (code + tests + README + .gitignore) together with the
  still-untracked governance/research docs, after explicit approval. Use an explicit
  `git add` allowlist; never `git add -A`. Exclude `.env` and generated
  `artifacts/*.log` / `artifacts/*.err` (now gitignored).

Checks: ruff clean, pytest 92 passed, compileall OK, offline Streamlit AppTest smoke OK.

Out of scope until this gate is committed:

- Trader research mode / market trade priors beyond the rates-only registry.
- Any new market series beyond the approved FRED rates set
  (`market_data.validate_market_series_registry`).

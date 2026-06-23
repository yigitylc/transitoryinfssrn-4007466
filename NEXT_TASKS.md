# Next Tasks

Current gate: Phase 5 - Macro Research Report (closeout).

Phases 0-4 are implemented and committed (production stabilization, historical
validation, benchmark comparison, robustness, and market linkage). Phase 5 (the
macro research report layer) is implemented and is being finished and committed
under this gate. Per `docs/09_PRODUCTION_ROADMAP.md`, do not start later phases
(for example the trader research expansion) until this gate passes and is committed.

Phase 5 closeout tasks:

- Commit the Phase 2-5 source, tests, and governance docs together so the
  committed state matches the implemented phases.
- Keep the report descriptive: no PnL, trading rules, strategy backtests, or
  buy/sell recommendations.
- Disclose that the live signal is built on latest-revised FRED data with no
  full-sample lookahead - it is not a real-time data-vintage backtest.
- Decide the scope of the orphaned trader-report layer
  (`report.build_trader_report` / `REGIME_PLAYBOOK`): gate it explicitly as future
  scope or remove it until its gate is active.
- Refresh `README.md` to match `config.py` (five FRED series, `paper_replication`
  mode) and the shipped phases.

Out of scope until this gate is committed:

- Trader research mode / market trade priors beyond the rates-only registry.
- Any new market series beyond the approved FRED rates set
  (`market_data.validate_market_series_registry`).

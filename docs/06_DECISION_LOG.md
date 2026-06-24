# Decision Log

Use this file to record research decisions that affect results.

| Date | Decision | Reason | Alternatives considered | Owner |
|---|---|---|---|---|
| 2026-06-09 | Keep replication and live-signal modes separate | Avoid mixing ex-post and real-time interpretations | Single blended mode | User/agent |
| 2026-06-09 | Named sample modes: `paper_replication` (1982-01..2021-07), `live_dashboard` (1982-01..latest, Streamlit default), `max_history` (all FRED). Loader fetches a 12-month YoY warm-up before `start_date`; `run_paper_replication.py` now slices at load time, before the full-sample baseline | Make date handling explicit; remove post-2021 lookahead from the paper baseline; YoY defined from sample start | Free-form date inputs; slice after features; no warm-up buffer | User/agent |
| 2026-06-09 | T-bill control switched to `TB3MS` (3-month, since 1934) as column `tbill_3m`; the old `TB1MS` id returns 404 on FRED | Restore live fetch; cover the full 1982-2021 paper window with one control series | `TB4WK`/`GS1M` (true 1-month, but only from 2001-07); defer | User/agent |
| 2026-06-09 | Bridge single-month interior CPI gaps by log-linear interpolation, flagged in `cpi_imputed` (multi-month/tail gaps never imputed) | The canceled 2025-10 CPI release left a permanent FRED hole that froze the live signal at 2025-09 (strict rolling windows NaN until 2028-11); log-linear chosen because CPI is an index level and YoY is ratio-based | Linear interpolation; relaxing rolling `min_periods`; leaving the signal stale | User/agent |
| 2026-06-24 | Trader research mode = **descriptive + rates-only** (the six approved FRED rate series), surfaced as a new **Trader Research** Streamlit tab; the shelved `build_trader_report`/`REGIME_PLAYBOOK` layer stays un-wired | Keep the project descriptive / non-trading and inside the approved market registry; deliver a current-state-conditioned base-rate reading by reusing the Phase 4 linkage tables | Predictive (out-of-sample scoring) or tradable (PnL/sizing); expanding the asset universe within FRED or via an ETF vendor; wiring or deleting the trader layer | User |

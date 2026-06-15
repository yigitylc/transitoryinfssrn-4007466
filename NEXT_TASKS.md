# Next Tasks

Current gate: Phase 0 — Production Stabilization Before Commit.

Immediate Phase 0 tasks:

- Confirm the Streamlit app starts cleanly after a full process restart.
- Confirm `FRED_API_KEY` in `.env` drives the official FRED API path without
  exposing the key.
- Confirm fallback order remains `fred_api -> fred_csv -> cached_fred -> demo`.
- Confirm cached FRED fallback uses the same CPI cleaning, imputation, YoY,
  baseline, and TINF feature path as live data.
- Confirm the cached path advances the latest valid signal date past 2025-09
  when cleaned CPI data supports 2026-04.
- Add positive inflation-shock validation logic so downside overshoot is
  resolved / downside overshoot, not persistent inflation.
- Update false transitory examples to use positive-shock logic.
- Add dashboard text separating inflation direction / positive-shock resolution
  from absolute distance / equilibrium normalization.
- Verify safe commit allowlist excludes `.env`, `.venv/`, raw cache data,
  extracted paper text, generated logs, caches, pycache, and the third-party
  PDF unless intentionally committed.
- Run `ruff check .`, `pytest`, and `python -m compileall src app scripts`.

Do not start Phase 1/2/3/4/5 until Phase 0 passes and is committed.

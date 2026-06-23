# Claude Code Instructions

You are helping build a macro research tool, not just refactoring Python.

## First behavior

Before edits, do a read-only audit. Read:

- `README.md`
- `docs/00_PROJECT_CONTEXT.md`
- `docs/01_RESEARCH_SPEC.md`
- `docs/02_DATA_CONTRACT.md`
- `docs/03_AGENT_WORKFLOW.md`
- the relevant files in `src/`, `app/`, `scripts/`, and `tests/`

Ask before proceeding if anything is ambiguous.

## Ask when ambiguous

Ask when the next step depends on any of these:

- exact paper replication versus live macro signal
- baseline definition
- CPI versus core CPI versus PCE/Core PCE
- vintage/realtime data versus latest-revised data
- whether market linkage is descriptive, predictive, or tradable
- whether an output belongs in Streamlit, notebook, report table, or tests

## Working style

- Make small, testable changes.
- Prefer source modules over large notebooks.
- Keep notebooks as exploration/presentation only.
- Do not hardcode absolute local paths.
- Do not commit secrets or API keys.
- Do not delete data/reference files without permission.
- Do not push to Git without explicit approval.

## Research guardrails

- Inflation units should be percentage points, not raw decimals.
- Continuous TINF is magnitude-based: `epsilon = inflation_yoy - baseline`; then rolling means over 4/8/12 months.
- Binary consecutive-month flags are diagnostics only.
- Use shifted baselines for real-time signal mode.
- Extract model parameters by name, not fragile positional index.
- Report when a result is ex-post, live-safe, or purely experimental.

## Visualization guidance

Do not over-engineer visuals from instructions alone. Choose visuals that make the macro question clear. Favor clarity, explainability, and auditability over decoration.

## End every work loop with

```text
What changed:
Files touched:
Checks run:
Open questions:
Recommended next step:
```

# Agent Workflow

This project is designed for Claude Code or a similar coding agent.

## Loop

1. Audit current state.
2. Ask ambiguity questions if needed.
3. Propose a small plan.
4. Implement one coherent step.
5. Run checks.
6. Summarize changes and next step.

## Good loop size

A good loop changes one of these at a time:

- data loading
- feature construction
- one model/table
- one diagnostic
- one dashboard section
- one test suite improvement

Avoid large rewrites unless the user approves.

## Minimum checks

```powershell
pytest
python -m compileall src app scripts
```

For UI changes:

```powershell
streamlit run app/streamlit_app.py
```

## Handoff format

```text
What changed:
Files touched:
Checks run:
Open questions:
Recommended next step:
```

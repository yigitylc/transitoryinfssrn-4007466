# Agent Guide

This file is intentionally tool-agnostic so Claude Code, Codex, Cursor, or another coding agent can understand the project.

## Mission

Rebuild the transitory inflation paper correctly, then expand it into a live macro research dashboard that helps interpret inflation persistence and possible market implications.

## Do not over-direct

The repository provides methodology constraints and validation expectations. It does not prescribe every UI decision. Agents should propose sensible implementation choices and ask when choices affect research meaning.

## Important distinctions

- **Replication** means matching the paper's definitions and sample as closely as possible.
- **Research upgrade** means improving methodology, robustness, and live usability.
- **Trading dashboard** means interpretability and regime awareness, not automatic trade signals.

## Safety and data handling

- No secrets in code.
- No hardcoded `C:/Users/...` paths.
- No destructive file operations without approval.
- Generated outputs belong under `reports/` or `artifacts/`.
- Raw downloads belong under `data/raw/`.

# Dashboard UI/UX Polish Plan

**Status:** **batches 1–2 committed + pushed** (`origin/main` = `1a0f5ab`) · **batch 3 implemented**
(working tree, uncommitted — needs review/commit approval) · batch 4 still proposed · **As of:**
2026-06-27 · **Scope:** presentation/visualization only · **Methodology:** unchanged (no new models,
market series, vendors, signals, PnL, or recommendations) · **Color semantic:** hot = above baseline
/ inflationary (red), cold = below baseline / disinflationary (blue), neutral = gray.

> This is a design plan with batches 1–3 now built. It does **not** change any computed value,
> model, series, or research conclusion. Every change is a different *presentation* of outputs the
> app already produces. Remaining batches land in later, separately-approved steps.

> **Batch 1 — DONE (committed `affbb0a`, pushed).** Shared Plotly theme + glossary, rebuilt Current
> Macro Signal, and the Trader Research range plot. Files: `plots.py`, `app/streamlit_app.py`,
> `tests/test_plots.py`.
>
> **Batch 2 — DONE (committed `1a0f5ab`, pushed).** Table-heavy tabs converted to charts: Validation
> gained hit-rate bars (by regime + by pressure), a threshold-sensitivity line, and a
> regime-transition heatmap, with worked examples / rate tables / sensitivity / transition tables
> moved behind expanders; Market Linkage gained grouped forward-change bars (bp) by regime per
> channel and a signal-vs-market correlation heatmap, with availability / rankings / per-channel /
> summary tables behind expanders and the stacked disclaimers folded into one "Scope & caveats".
> Four new figures in `plots.py` (`hit_rate_bar_figure`, `threshold_sensitivity_figure`,
> `heatmap_figure`, `forward_change_by_regime_channel_figure`), each with return-type + trace-count
> + empty-frame tests.
>
> **Batch 3 — DONE (uncommitted).** Evidence tabs charted: Benchmark gained two verdict badges (vs
> no-change / vs mean-reversion) + a diverging MAE/RMSE improvement chart (cold = TINF wins, hot =
> trails), with improvement / classification / forecast-audit tables behind expanders; Robustness
> gained win-rate bars by setting (MAE | RMSE, each with a 0.5 reference line), with data-status /
> availability / scorecard / verdict / win-rate table behind expanders and hot/cold regime
> conditional formatting on the baseline quick-comparison table. One new figure
> (`improvement_diverging_figure`) plus additive `yaxis_title`/`reference` kwargs on
> `hit_rate_bar_figure` (reused for the win-rate bars) and a new `style_regime_cells()` Styler
> helper; +5 tests. Gates green: ruff clean · pytest **117 passed** (112 prior + 5 new) · compileall
> OK · offline `AppTest` smoke renders all 9 tabs, 0 exceptions (new chart sections asserted
> rendered). Files touched: `src/transitory_inflation/plots.py`, `app/streamlit_app.py`,
> `tests/test_plots.py`.

## Locked design decisions

1. **Executive summary** → enhance the existing **Current Macro Signal** tab into the polished
   executive summary. No new tab, no duplicated snapshot.
2. **Color system** → **hot/cold**: red = above baseline / inflationary, blue = below baseline /
   disinflationary, gray = neutral. No green/red good-bad scheme (above-baseline inflation is a
   regime, not a failure).
3. **First batch** → **Foundation + flagship**: shared Plotly theme + glossary, then rebuild
   Current Macro Signal and add the Trader Research range plot as showcases.

## Why

The dashboard is methodologically finished but reads like a stack of tables. A macro researcher
opening it cannot tell at a glance "what is inflation doing right now and how confident are we",
must scroll past repeated disclaimers, and meets ~10 wide DataFrames on the heaviest tabs with no
chart to anchor them. This plan raises visual hierarchy, converts the highest-value tables to
charts, hides detail behind expanders, and explains the core concepts inline — all on top of the
existing computed outputs.

## Key UI problems found

1. **Flat hierarchy / no executive read.** Outside tab 1's four metric cards, almost everything is
   `#### heading` + full-width DataFrame. No at-a-glance "current state + confidence".
2. **Tables where charts read better.** Forward rate-change distributions, hit-rate-by-regime,
   benchmark MAE/RMSE improvement, regime-transition matrix, correlations, and win-rates are all
   shown only as tables.
3. **Repeated disclaimer walls.** Each table-heavy tab opens with 2–4 `st.warning`/`st.markdown`
   blocks. The text is correct and must stay, but it buries the content.
4. **Inconsistent explainers.** The `section_notes` (➜/↳) helper is used on Signal, Report,
   Framework, and Decay — but **not** on the table-heaviest tabs (Validation, Benchmark, Market
   Linkage, Trader Research, Robustness).
5. **Metric cards only on tab 1.** Trader Research, the Report headline, and Decay all have
   card-worthy summary numbers rendered as prose.
6. **Unthemed, bare charts.** The four figures in `src/transitory_inflation/plots.py` have titles
   but no shared template, no unified hover, no semantic color, no current-value markers, no
   epsilon shading.
7. **No glossary.** TINF, epsilon, baseline, regime, short-term pressure, weak_evidence, and
   live-safe vs ex-post appear in prose and column names with no single definition surface.
8. **Over-long tables shown in full.** e.g. `forecasts.head(50)`, the five validation example
   tables, and the ~20-column robustness scorecard/verdict are always fully expanded.

## Cross-cutting improvements (the reusable foundation)

| # | Target | Current problem | Proposed change | Why it helps | Complexity | Changes methodology |
|---|--------|-----------------|-----------------|--------------|-----------|---------------------|
| C1 | all charts (`plots.py`) | 4 figures unthemed, default colors, no unified hover | Add `apply_macro_theme(fig)` (shared template: font, gridlines, `hovermode="x unified"`, tight margins, legend top, hot/cold palette) and route every figure through it | One consistent, legible chart language across the app | Low–Med | No |
| C2 | sidebar + all tabs | No concept definitions anywhere | One reusable **Concepts / glossary** expander (TINF, epsilon, baseline, regime, short-term pressure, weak_evidence, live-safe vs ex-post) — rendered in the sidebar and reusable inline | Readers self-serve definitions; prose gets shorter | Low | No |
| C3 | Validation, Benchmark, Market Linkage, Trader Research, Robustness | `section_notes` (➜/↳) missing on the table-heavy tabs | Apply the existing `section_notes` helper to those tabs | Consistent "what it answers / how to read it" rhythm | Low | No |
| C4 | every tab with stacked warnings | 2–4 disclaimer blocks push content down | Collapse the repeated warnings into one **"Scope & caveats"** expander per tab; keep a single one-line caption visible. **All text preserved**, only relocated | Recovers vertical space without losing guardrail language | Low | No |
| C5 | sidebar | Config only (mode + baseline) | Group config under a header, add a compact "current reading" mini-status line + a link to the glossary | Orientation without opening a tab | Low | No |

## Per-tab improvements

### Tab 1 — Current Macro Signal (flagship executive summary)

| Current problem | Proposed change | Why | Complexity | Methodology |
|-----------------|-----------------|-----|-----------|-------------|
| Cards are plain; no plain-English read; regime/pressure shown as a bare `st.write` line | Add a one-line plain-English **headline** ("Inflation is +0.74pp above baseline · pressure firming"); give the 4 cards semantic deltas (epsilon vs 0, percentile arrow) and a colored **regime badge** | Instant "what's happening + is it elevated" read | Low–Med | No |
| `cpi_vs_baseline_figure` is two flat lines | **Shade the epsilon gap** (fill between CPI and baseline, hot above / cold below), mark the **current point**, unified hover | The vertical gap *is* epsilon — show it, don't make readers infer it | Med | No |
| `tinf_term_structure_figure` is three flat lines | Tint above-zero (hot) / below-zero (cold), add current-value markers | Makes "firming vs cooling" legible at a glance | Low–Med | No |

### Tab 5 — Trader Research (flagship new chart)

| Current problem | Proposed change | Why | Complexity | Methodology |
|-----------------|-----------------|-----|-----------|-------------|
| The forward rate-change distribution (median, p25–p75) is shown **only as a table** | **Horizontal range/dot plot** per FRED instrument: p25–p75 as a whisker/bar, median as a marker, zero reference line, hot/cold by sign | Single highest-value chart in the app — turns a number grid into a distribution you can read | Med | No (plots `view.distribution`) |
| Current bucket rendered as a markdown sentence | **Metric cards**: regime, pressure, analog count, weak-evidence badge | Matches the executive-summary card language | Low | No |
| Channel roll-up + analog months are stacked tables | Channel roll-up → small bar chart; analog months → behind an **"Audit trail"** expander (kept fully) | Keeps the audit trail, de-clutters the default view | Low–Med | No |

### Tab 2 — Historical Signal Validation

| Current problem | Proposed change | Why | Complexity | Methodology |
|-----------------|-----------------|-----|-----------|-------------|
| 4 intro paragraphs + 2 warnings, then 9+ tables incl. 5 example tables | Intro → "How to read this tab" expander + 1-line caption; keep the **combined regime×pressure summary** visible | Recovers the top of the tab for signal, not prose | Low | No |
| Combined / by-regime / by-pressure summaries are tables | **Bar chart** of resolution/hit rate by regime (and by pressure) at the selected horizon | Compare buckets visually instead of scanning rows | Med | No |
| Threshold sensitivity is a table | **Line chart** across the 0.25/0.50/0.75/1.00 pp thresholds | Sensitivity is a trend — show the slope | Low–Med | No |
| Regime transition matrix is a table | **Heatmap** | Transition structure reads instantly as a grid | Med | No |
| 5 false-pos/neg example tables always expanded | Move worked examples + transition + sensitivity behind expanders | Keeps the audit material, hides the bulk | Low | No |

### Tab 3 — Benchmark Comparison

| Current problem | Proposed change | Why | Complexity | Methodology |
|-----------------|-----------------|-----|-----------|-------------|
| Improvement vs benchmarks is an `st.info` blob + tables | **Diverging bar chart** of TINF MAE/RMSE improvement vs each benchmark (hot when TINF loses, cold when it wins) for the selected horizon | "Does TINF beat the naive baselines" becomes one glance | Med | No |
| Confusion + 50-row forecast audit always shown | Keep the metric summary visible; move classification + forecast audit behind expanders | Headline metrics stay; detail is opt-in | Low | No |
| "Beats no-change / mean-reversion?" buried in prose | Two small colored **badges/cards** | Surfaces the verdict the info-box already computes | Low | No |

### Tab 4 — Market Linkage (heaviest tab, ~10 tables)

| Current problem | Proposed change | Why | Complexity | Methodology |
|-----------------|-----------------|-----|-----------|-------------|
| "What historically happened" is a wide table | **Grouped bar chart**: avg/median forward change (bp) by regime per channel at the selected horizon | The core descriptive result becomes a picture | Med | No |
| Correlations are a table | Small **heatmap** | Sign/strength reads at a glance | Low–Med | No |
| Availability, rankings (2-col), 3 per-channel sub-tables, summaries all expanded | Keep current snapshot + channel interpretation visible; move availability, rankings, per-channel detail, correlations behind expanders | Tame the longest tab without losing anything | Low | No |
| 3 disclaimer blocks at top | Consolidate into the "Scope & caveats" expander (C4) | Consistent with the rest | Low | No |

### Tab 6 — Paper Framework (light touch)

| Current problem | Proposed change | Why | Complexity | Methodology |
|-----------------|-----------------|-----|-----------|-------------|
| Correlation matrix is a table | **Heatmap** (keep table behind expander) | Faster read of co-movement | Low–Med | No |
| OLS / Ljung-Box / summary stats | **Leave as tables** — regression and diagnostic output belongs in tables | Charting them adds nothing | — | No |

### Tab 7 — Decay / Convergence (already chart-led)

| Current problem | Proposed change | Why | Complexity | Methodology |
|-----------------|-----------------|-----|-----------|-------------|
| Headline decay read (t*, % gone in 6/12m) is only in the table | **Metric cards** from the first valid window; apply the shared theme to both figures | Surfaces the convergence headline | Low | No |

### Tab 8 — Robustness

| Current problem | Proposed change | Why | Complexity | Methodology |
|-----------------|-----------------|-----|-----------|-------------|
| ~20-col scorecard + ~22-col verdict always shown | **Win-rates bar chart** (TINF beats X by MAE/RMSE); move scorecard + verdict behind expanders; keep win-rates + baseline comparison + stationarity visible | Lead with the conclusion, keep the wide grids on demand | Low–Med | No |
| Baseline quick-comparison is a plain table | Conditional formatting: color regime cells / highlight sign agreement | "Does the call survive baseline changes" reads instantly | Low–Med | No |

### Tab 9 — Macro Research Report

| Current problem | Proposed change | Why | Complexity | Methodology |
|-----------------|-----------------|-----|-----------|-------------|
| Headline + current regime are markdown lines | Render as **metric cards** (reuse the snapshot card component) | Consistent executive language | Low | No |
| 7 sections of bullets + inline tables | Keep narrative bullets visible; move supporting DataFrames behind expanders (extends the pattern already used for benchmark/robustness detail); add `st.divider()` between sections | Readable report flow; detail on demand | Low | No |

## Concept explanations to standardize (the glossary, C2)

These plain-language definitions back the glossary expander and the inline captions. Wording is
descriptive only and consistent with `docs/01_RESEARCH_SPEC.md` / `docs/METHODOLOGY.md`.

- **Baseline** — the mean-reversion anchor inflation is compared against (e.g. prior 36-month
  rolling mean). Shifted baselines use only data through the prior month (live-safe); the
  full-sample baseline is flat and ex-post.
- **epsilon (ε)** — the raw deviation `inflation_yoy − baseline`, in percentage points. Positive =
  inflation above baseline; negative = below.
- **TINF 4M / 8M / 12M** — the n-month rolling average of epsilon. 4M is recent transitory
  pressure; 8M/12M are slower-moving. Units: percentage points.
- **Regime** — a label combining level (vs the sample's 25th/75th-percentile band) with direction
  (vs the prior month): elevated rising, elevated falling, neutral, disinflationary.
- **Short-term pressure** — summarizes the 4M vs 8M vs 12M TINF ordering: firming (4M highest),
  cooling (4M lowest), or mixed.
- **weak_evidence** — a flag that a bucket has fewer than 30 complete observations; its rates are
  unstable and should be read cautiously.
- **live-safe vs ex-post** — live-safe = no full-sample lookahead (signal usable in real time);
  ex-post = uses information not available at the time (paper-style only). Note: "live-safe" means
  no full-sample lookahead, **not** a real-time data-vintage backtest.

## What should NOT change

- **No methodology, numbers, or logic.** Regime/pressure/baseline definitions, TINF/epsilon
  computation, benchmark/robustness math, and all column values stay byte-identical.
- **No new data, series, models, signals, PnL, or recommendations.** No recession shading or any
  other external series (that would be a new data dependency).
- **Keep every caveat/disclaimer's text** — only relocate it into expanders. The ex-post vs
  live-safe warnings and the "not a trading signal" language are research guardrails.
- **Keep the data-source/freshness status captions** (auditability).
- **Keep OLS / Ljung-Box / stationarity / scorecard as tables** — tabular output is correct there.
- **Don't reorder or rename tabs** (beyond cosmetic heading polish); tab order encodes the
  Phase 2→4 roadmap narrative.

## Recommended first implementation batch (smallest useful — "Foundation + flagship") — DONE (uncommitted)

1. **DONE — `apply_macro_theme()`** in `plots.py` (shared template: font, light gridlines,
   `hovermode="x unified"`, tight margins, top legend, hot/cold palette); all 5 figures routed
   through it (C1).
2. **DONE — Concepts / glossary** expander (`render_glossary()` over `GLOSSARY_ITEMS`), rendered
   in the sidebar; reusable inline (C2). Sidebar also gained a grouped header + a compact
   "Current reading" mini-status (C5).
3. **DONE — Current Macro Signal** rebuilt: plain-English `signal_headline()`, colored
   `regime_badge()`, metric cards with neutral (`delta_color="off"`) semantic deltas
   (epsilon-vs-baseline, percentile-vs-median); CPI chart now epsilon-shaded with a current
   marker; TINF chart zone-tinted above/below zero with current-value markers.
4. **DONE — Trader Research** forward rate-change **range/dot plot**
   (`forward_change_range_figure`, median marker + p25–p75 whisker, hot/cold by sign) as the
   headline, table moved behind an expander; bucket **metric cards** (regime / pressure / analog
   count / as-of); analog months behind an "Audit trail" expander.
5. **DONE — `section_notes` + "Scope & caveats" expander** (`scope_caveats()`) applied to Trader
   Research as the reusable template (C3/C4); Current Macro Signal already used `section_notes`.

Everything else (Validation / Benchmark / Market Linkage / Robustness charts, Report cards)
follows in later batches now that the template has landed.

**New reusable helpers available for later batches** — in `app/streamlit_app.py`:
`render_glossary()`, `scope_caveats(one_liner, caveats)`, `regime_badge(regime)`,
`signal_headline(snapshot)`; in `plots.py`: `apply_macro_theme(fig, ...)` and the
`HOT`/`COLD`/`NEUTRAL` palette constants. Reuse these instead of re-introducing ad-hoc styling.

### Files likely to change (batch 1)

- `src/transitory_inflation/plots.py` — theme helper + 1 new figure (Trader range plot); shading
  on the CPI and TINF figures.
- `app/streamlit_app.py` — glossary helper, tab 1 rebuild, tab 5 chart + cards.
- `tests/test_plots.py` — **update trace-count assertions** (shading/markers add traces) and add
  tests for the new figure(s).

### Tests / smoke checks needed

- `& .\.venv\Scripts\ruff.exe check .`
- `& .\.venv\Scripts\python.exe -m pytest -q` — **`test_plots.py` asserts exact trace counts
  (`len(fig.data) == 2 / 3 / 2`); these must be updated** when figures gain shading/marker traces;
  add return-type + trace-count + empty-frame tests for new figures.
- `& .\.venv\Scripts\python.exe -m compileall src app scripts -q`
- Offline `AppTest` smoke (ad-hoc, patched `requests.get` → offline; renders all 9 tabs, 0
  exceptions) — the project's standard gate. Note: run ad hoc, **not a committed test**.
- Manual `streamlit run app/streamlit_app.py` localhost review per `docs/08_LOCALHOST_REVIEW.md`.

## Later batches (sequence after the template lands)

The batch-1 template has landed (see above), so these are now unblocked. Reuse
`apply_macro_theme`, `section_notes`, `scope_caveats`, `render_glossary`, and the palette
constants rather than new ad-hoc styling.

- **Batch 2 — table-heavy tabs (DONE, committed `1a0f5ab`):** Validation (hit-rate bars, sensitivity
  line, transition heatmap) and Market Linkage (grouped bars, correlation heatmap, expanders). Each
  new figure has a return-type + trace-count + empty-frame test in `tests/test_plots.py`, mirroring
  `forward_change_range_figure`.
- **Batch 3 — evidence tabs (DONE, uncommitted):** Benchmark (diverging improvement bars + badges)
  and Robustness (win-rate bars, conditional formatting, expanders). New `improvement_diverging_figure`;
  `hit_rate_bar_figure` extended with `yaxis_title`/`reference` and reused for the win-rate bars;
  new `style_regime_cells()` helper. Each new/changed figure has a test.
- **Batch 4 — report + light touches (NEXT):** Macro Research Report cards/dividers, Decay cards,
  Paper Framework correlation heatmap (reuse `heatmap_figure`).

# Full Codebase Audit and Fix Plan

**Repository:** Transitory Inflation Macro Research

**Audit date:** 2026-07-10

**Scope:** full repository, methodology, data lineage, statistics, tests, scripts, configuration, CI, Git hygiene, report artifacts, and Streamlit UI/UX

**Audit mode:** skeptical and read-only; no fixes implemented

**Allowed write used:** this file only

## 1. Executive verdict

**Verdict: FIX BEFORE RESEARCH RELEASE.** The repository is a technically stable and unusually well-caveated **descriptive macro-research prototype**, but it is not ready to be represented as a faithful paper replication, a genuinely real-time/no-lookahead historical backtest, or an institutionally decision-safe predictive dashboard.

The automated engineering baseline is green: Ruff passes, all 118 tests pass, compileall succeeds, the paper script runs, and an offline full-app AppTest renders all nine tabs with zero exceptions. Those checks are real strengths, but they do not cover the highest-risk research semantics.

Two findings are release blockers:

1. The path named `paper_replication` does not reproduce the paper's effective sample, published descriptive statistics, regression anchors, or TINF construction. Its current output differs by orders that cannot be described as rounding or data revision.
2. The isolated CPI-gap bridge is a two-sided interpolation: the imputed value at month `t` depends on month `t+1`. That estimate is then allowed into historical validation and benchmark signal rows labeled live-like/live-safe. This is localized rather than pervasive full-sample leakage, but it is real look-ahead.

Several High findings also change displayed research conclusions or trust:

- cached macro data preserve the filled CPI value but erase the imputation flag;
- report and robustness winner logic compares unequal model samples and demonstrably flips verdicts versus common-date scoring;
- forecast and actual persistence classifications use different baseline dates;
- validation charts show a generic count that can differ radically from the denominator of the plotted conditional rate;
- market outcomes start from the CPI reference month-end, before that CPI was published;
- live FRED results can remain cached indefinitely;
- a cold full-app render took about 127 seconds because all tab bodies execute and expensive layers are duplicated;
- long-horizon overlapping outcomes are treated as raw independent counts without uncertainty.

The correct release posture is therefore:

- **Suitable now:** supervised local description, diagnosis, historical exploration, and scenario framing, with an expert reading caveats.
- **Not suitable now:** paper-replication claims, vintage-safe backtest claims, unattended live deployment, predictive market claims, trade timing, sizing, or automated recommendations.

## 2. Project and pipeline map

```text
FRED macro series
  API with optional key -> public CSV -> local macro cache -> labeled demo fallback
      |
      v
monthly-last alignment -> isolated CPI gap bridge -> YoY inflation (pp) -> sample slice
      |
      v
baseline -> epsilon = inflation_yoy - baseline -> trailing-current TINF 4/8/12
      |                                      |
      |                                      +-> current in-sample percentile/regime snapshot
      v
walk-forward historical regime + pressure labels
      |
      +-> validation outcomes and examples
      +-> benchmark forecasts and classifications
      +-> robustness grid
      |
      v
FRED rates-only market data
  API -> public CSV -> local market cache -> unavailable (never synthetic market data)
      |
      v
monthly-last join -> forward rate changes (bp) -> regime / pressure / channel summaries
      |
      +-> Trader Research current-bucket view
      +-> Macro Research Report analog and market sections

Paper-style models
  summary/correlation/OLS -> rolling AR(1) rho -> AR(1) on rho -> decay/convergence

All layers -> nine-tab Streamlit dashboard
```

### End-to-end intent reconstructed from docs and code

| Layer | Intended role | Current implementation |
|---|---|---|
| Data acquisition | Live FRED with honest fallbacks | Macro: API -> CSV -> cache -> demo. Market: API -> CSV -> cache -> unavailable. |
| CPI transformation | Headline CPI YoY in percentage points | Monthly CPI level, 12-month percentage change times 100; alternative core/PCE measures for robustness. |
| Baseline and epsilon | Explicit mean-reversion anchor | Full-sample, 36M rolling unshifted/shifted, expanding shifted, or fixed 2% reference; `epsilon = inflation_yoy - baseline`. |
| TINF | 4M/8M/12M average deviations | One shared trailing-current rolling implementation for every mode. |
| Current regime | Current descriptive state | Full-loaded-sample TINF percentile plus prior-complete-row direction; pressure from 4M/8M/12M ordering. |
| Historical validation | Test future inflation behavior without feeding outcomes into signals | Walk-forward regimes, forward CPI/epsilon outcomes, mechanical positive-shock and convergence labels, transitions, examples. |
| Benchmarks | Compare TINF/regime to simple alternatives | No-change, horizon momentum labeled CPI persistence, mean reversion, expanding AR(1), and historical-regime bucket mean. |
| Robustness | Vary samples, measures, baselines, horizons, thresholds | Grid over those dimensions, but no paper-lagged TINF variant and point-error rows are duplicated across thresholds. |
| Market Linkage | Descriptive rates-only history | Six approved FRED nominal, breakeven, and real-yield series; forward bp distributions, rankings, channels, correlations. |
| Trader Research | Current-state-conditioned rates history | Walk-forward current regime; optional pressure conditioning; distributions and analog months; no PnL or recommendations. |
| Paper framework | Descriptive replication-style outputs | Summary statistics, correlations, HC1 OLS, Ljung-Box, rolling AR(1), and paper decay arithmetic. |
| Macro report | Synthesize state, confidence, robustness, analogs, market linkage, caveats, watchlist | In-memory seven-section report rendered in Streamlit. |
| Intended decision use | Interpretability and regime awareness | Strongest for description/diagnosis; weak for extrapolation and not fit for trade decisions. |

## 3. Git state and validation results

### Git preflight

| Check | Result |
|---|---|
| `git status --short` at audit start | Only `?? .claude/` |
| `git log --oneline --decorate -12` | `HEAD -> main` and `origin/main` both at `3c5fe41` |
| `git diff --check` | Passed; no whitespace errors |
| `git rev-list --left-right --count origin/main...main` | `0 0` |
| Tracked secret / hardcoded-user-path scan | No populated secret assignment or hardcoded user path found in tracked code |
| Ignore checks | `.env`, `.venv/`, raw cache, PDF, extracted paper text, caches, and logs are ignored |

### Required and focused validation

| Command / diagnostic | Result |
|---|---|
| `python -m ruff check .` | Passed: `All checks passed!` |
| `python -m pytest` | **118 passed** in 21.65s |
| `python -m compileall src app scripts` | Passed |
| Focused high-risk tests | **7 passed** after correcting one initially mistyped pytest node name |
| `python scripts/run_paper_replication.py` | Executed successfully, but produced material paper-anchor mismatches documented in B1 |
| Offline Streamlit AppTest | 9 tabs, 0 exceptions, 17 metrics, 12 Plotly elements, 35 DataFrames, 3 warnings, 0 errors; about **127 seconds** cold |
| Macro/cache/imputation diagnostics | Confirmed future-neighbor interpolation and loss of imputation provenance |
| Benchmark diagnostics | Confirmed unequal counts and common-sample verdict reversals |
| Validation denominator diagnostic | Confirmed displayed counts can be unrelated to conditional-rate denominators |
| `python -m pip check` | No broken requirements |

Non-failing warnings:

- KPSS reported its normal lookup-table-bound interpolation warning.
- Pytest could not update `.pytest_cache` under the audit sandbox (`WinError 5`); test execution itself was unaffected.

The current runtime was Python 3.12.10, pandas 3.0.3, NumPy 2.4.6, SciPy 1.17.1, statsmodels 0.14.6, Streamlit 1.58.0, Plotly 6.8.0, pytest 9.0.3, and Ruff 0.15.16. The repository does not lock these versions.

An attempted hidden localhost Streamlit launch for browser-based visual review was rejected by the execution environment's approval/usage limit. It was not retried through a workaround. The successful full AppTest and source-level UI audit provide current smoke evidence, but there is no new manual browser screenshot or HTTP smoke from this audit.

## 4. Ratings (1-10)

| Area | Rating | Rationale |
|---|---:|---|
| Overall repository | **5/10** | Strong scaffold and disclosure culture; material methodology and lineage defects remain. |
| Paper replication fidelity | **2/10** | Current path is paper-inspired and does not match published anchors. |
| Data integrity and provenance | **4/10** | Good fallback design, but cache/imputation lineage is incorrect. |
| No-lookahead discipline | **4/10** | Shifted baselines and walk-forward thresholds are good; imputation and information-date issues prevent a clean pass. |
| Validation methodology | **5/10** | Positive-shock logic is thoughtful; denominators and uncertainty are not credible enough. |
| Benchmark fairness | **3/10** | Common-sample helper exists, but report/robustness conclusions still use unequal samples and inconsistent classification anchors. |
| Statistical rigor | **4/10** | Useful diagnostics, but no overlap-aware inference, forecast-loss uncertainty, or HAC regression view. |
| Decay implementation | **7/10** | Published arithmetic is implemented correctly for ordinary valid cases; inference and edge handling are thin. |
| Market Linkage / Trader Research | **5/10** | Clean descriptive rates-only implementation, but not decision-date aligned and statistically over-interpretable. |
| Report synthesis | **5/10** | Complete structure and caveats; benchmark verdicts and freshness synthesis can be wrong/incomplete. |
| Software architecture | **5/10** | Modular research code, but a 1,981-line eager Streamlit controller and duplicated heavy computation. |
| Test engineering | **6/10** | 118 fast, network-free tests; highest-risk integration semantics and committed AppTest are missing. |
| Streamlit UI/UX | **7/10** | Clear hierarchy, semantic color, charts, and caveats; performance, formatting, uncertainty, and conditioning clarity need work. |
| Documentation/governance | **5/10** | Extensive docs, but current handoff/gate state is stale and some claims are stronger than code evidence. |
| Git/secrets hygiene | **9/10** | Strong ignore rules, no tracked secrets, no hardcoded user paths, and disciplined local-only artifacts. |
| Release readiness | **4/10** | Supervised local research only; not ready for institutional or replication release. |
| Descriptive decision usefulness | **6/10** | Helpful for state description and research questions. |
| Predictive credibility | **2/10** | No clean common-sample universal scoring, no vintage/release alignment, and no market out-of-sample test. |

## 5. Findings summary by severity

Severity means:

- **Blocker:** invalidates a core project claim or the live-like historical research contract.
- **High:** can change a displayed conclusion, conceal material data lineage, or make live use unsafe.
- **Medium:** material limitation, latent bug, missing capability, or institutional-quality gap.
- **Low:** contained technical debt, edge case, or optional enhancement.

| Severity | Count | Headline |
|---|---:|---|
| Blocker | 2 | Paper replication is not a replication; two-sided CPI interpolation leaks future information into historical signals. |
| High | 10 | Cache provenance, common samples, classification anchors, denominators, information dates, stale caches, performance, benchmark verdict UX, fallback attribution, and inference. |
| Medium | 25 | Missing lagged robustness, pseudo-replication, conditioning inconsistencies, date contracts, fallback resilience, report provenance, metadata/docs/CI/UI gaps, and decay edge handling. |
| Low | 6 | Diagnostics edges, dormant code, unused config, duplicated constants, demo coherence, and launch/review ergonomics. |

## 6. Detailed finding register

### Blocker

#### B1 - The advertised paper-replication path does not replicate the published paper anchors

- **Issue:** `paper_replication` is a date-window label around the live feature path, not a frozen reconstruction of the paper's variables and tables.
- **Evidence:** `scripts/run_paper_replication.py:22-45` selects 1982-01 through 2021-07, uses `full_sample`, then calls the shared trailing-current feature builder. `features.py:111-117` includes current epsilon in current TINF. The extracted paper equation uses a lagged summation, while its appendix refers to a 36-month moving-average anchor. The current diagnostic produced 475 CPI rows and TINF counts 472/468/464; paper Table 1 reports 415 for every variable. Current CPI mean/TINF4 standard deviation/T-bill mean were 2.742/1.356/3.645 versus paper 2.343/0.443/0.243. The current short regression was TINF coefficient 0.959, bill coefficient 0.005, R-squared 0.896, n=472 versus paper 0.380, 3.205, 0.575, n=415.
- **Affected files/functions:** `scripts/run_paper_replication.py`; `config.SAMPLE_MODES`; `features.add_transitory_inflation_features`; `models.summary_stats`; `models.run_paper_style_regressions`; paper-method docs.
- **Why it matters:** the current contemporaneous regressor mechanically contains CPI, making the fit far stronger than the published relationship. A user cannot use these outputs to verify the paper.
- **Type:** paper mismatch / bug / docs drift.
- **Recommended fix:** freeze a replication contract covering source series and vintage, CPI transformation, effective common sample, 36M/full-mean ambiguity, exact lag indexing, rate maturity and units, missing-data rules, and Table 1-4 tolerances. Implement a separate paper feature path and golden regression tests. Until it passes, rename this surface `paper_inspired_window` / `Paper Framework` and remove replication claims.
- **Owner:** Claude.
- **Priority:** P0 / stop-ship for replication claims.

#### B2 - Two-sided CPI interpolation introduces future information into historical live-like signals

- **Issue:** an isolated missing CPI month is filled from both its prior and following observations, then used as if available at the missing reference month.
- **Evidence:** `data.build_base_frame` at `data.py:354-365` uses interior log interpolation. Perturbing only CPI at `t+1` changed imputed CPI at `t` from 111.597216 to 117.044147 and YoY at `t` from 6.167781pp to 11.349708pp. Validation and benchmarks do not exclude imputed origins or propagate an affected-window lineage flag.
- **Affected files/functions:** `data.build_base_frame`; `features.add_transitory_inflation_features`; all validation, benchmark, robustness, market-linkage, analog, and report paths that consume the affected rows.
- **Why it matters:** ex-post interpolation is acceptable for a clearly labeled continuity estimate after the next observation exists. It is not information known at historical month `t`, so it invalidates a strict no-lookahead backtest for affected signal origins and downstream windows.
- **Type:** bug / look-ahead / methodology limitation.
- **Recommended fix:** separate an ex-post continuity series from a vintage-safe research series. For validation, either skip unavailable origins and every feature window influenced by them, or use a documented one-sided nowcast available at the decision date. Propagate `uses_imputed_input`, affected start/end dates, and availability date through baseline/TINF/outcome tables.
- **Owner:** Claude.
- **Priority:** P0 / stop-ship for live-like backtest claims.

### High

#### H1 - Macro cache round-tripping erases imputation provenance while retaining the estimate

- **Issue:** the cache writer saves an already cleaned/imputed frame under `data/raw`, while the loader discards saved derived flags and cannot reconstruct original missingness.
- **Evidence:** `scripts/fetch_fred_data.py:31-36` writes `result.data`; imputation has already occurred in `data.py:354-365`. `load_cached_macro_data_for_mode` at `data.py:562-570` reloads filled levels and intentionally ignores the cached `cpi_imputed`. In the current local cache, 2025-10 is stored with `cpi_imputed=True`; reload returns the identical filled level with `cpi_imputed=False`, and the total flag count falls from 1 to 0.
- **Affected files/functions:** `scripts/fetch_fred_data.py`; `data.load_cached_macro_data_for_mode`; app/report data-status disclosures; cache tests.
- **Why it matters:** offline/cached mode can explicitly tell the user that no imputation occurred while using an imputed value.
- **Type:** bug / data lineage.
- **Recommended fix:** cache original FRED source columns and missingness before transformation; version the cache schema; store retrieval/source metadata; migrate legacy processed caches; add fetch -> save -> reload lineage tests.
- **Owner:** Codex.
- **Priority:** P0.

#### H2 - Report and robustness benchmark winners are computed on unequal samples

- **Issue:** models drop unavailable forecasts independently, but raw MAE/RMSE values are ranked and compared as though evaluated on the same origins.
- **Evidence:** independent row dropping occurs at `benchmarks.py:216-217`; metrics at `benchmarks.py:295-330`; raw ranking at `robustness.py:180-185`; raw winner flags at `robustness.py:238-246`; report comparisons at `report.py:250-271`. A correct pairwise helper exists at `benchmarks.py:333-363` but is used only against no-change and mean-reversion. At 12M, current counts were no-change 521, mean-reversion 485, AR(1) 498, TINF 446. Common-date rescoring flipped the 12M TINF/no-change MAE verdict, the 12M TINF/mean-reversion RMSE verdict, and the 24M TINF/no-change RMSE verdict.
- **Affected files/functions:** `benchmarks.benchmark_metric_summary`; `_common_sample_improvement`; `robustness.build_robustness_scorecard`; `tinf_regime_verdict`; `report._benchmark_tables`; report/robustness UI.
- **Why it matters:** the dashboard and report can state the wrong model winner.
- **Type:** bug / unfair benchmark comparison.
- **Recommended fix:** construct one common-origin evaluation panel per horizon, or explicit paired panels per model comparison, and use those values for every rank, card, report sentence, and robustness win. Show paired `n`, start/end dates, and missing-forecast reason.
- **Owner:** Codex.
- **Priority:** P0.

#### H3 - Forecast and actual persistence labels use different baseline dates

- **Issue:** actual persistence is defined using future epsilon relative to `baseline_(t+h)`, while forecast persistence compares forecast CPI with `baseline_t`.
- **Evidence:** `validation.py:253-300` derives actual positive-shock persistence from future epsilon. `benchmarks.py:213-215` computes forecast persistence as `(forecast - current baseline) > threshold`. A perfect CPI forecast can be scored as a false positive if the rolling baseline changes between origin and target.
- **Affected files/functions:** `validation.add_outcome_labels`; `benchmarks.build_benchmark_forecasts`; confusion metrics and report interpretation.
- **Why it matters:** classification errors can reflect inconsistent target definitions rather than forecast quality.
- **Type:** bug / methodology.
- **Recommended fix:** choose one decision-consistent target. Either freeze `baseline_t` for both forecast and realized classifications, or explicitly forecast the future baseline with information available at `t` and use the same forecast anchor for predicted and realized scoring. Document the economic meaning before implementation.
- **Owner:** Claude.
- **Priority:** P0.

#### H4 - Validation counts do not represent the denominator of conditional outcome rates

- **Issue:** each summary row exposes a generic future-CPI count, while individual rates silently drop non-applicable nullable observations.
- **Evidence:** `validation.py:374-385` sets `count=len(group)`; rates at `:395-420` use `_hit_rate(...dropna())`; threshold summaries repeat the pattern at `:504-539`. Under the current default 24M view, the disinflationary row showed count 81 but the positive-shock persistence rate used **1** applicable row; neutral showed count 227 but used 34.
- **Affected files/functions:** validation summaries, sensitivity tables, hit-rate charts, report analog interpretation.
- **Why it matters:** a displayed 100% conditional rate can look supported by 81 observations when it is based on one.
- **Type:** bug / UX / methodology.
- **Recommended fix:** emit denominator columns for every rate; apply consistent weak-evidence flags and confidence intervals; visually suppress or gray rates below a policy minimum; put `n_applicable` in chart labels/hover.
- **Owner:** Codex.
- **Priority:** P1.

#### H5 - CPI reference months are treated as decision/information dates, especially in Market Linkage

- **Issue:** the project has a `date` for the economic reference month but no release date, information date, or vintage date.
- **Evidence:** `data.monthly_last` at `data.py:314-321` rewrites rows to month-end. `market_linkage.py:153-176` joins signal month `t` to market month-end `t`; `:129-150` measures market changes from that point. CPI for reference month `t` is published in `t+1`. `report.py:653-656` acknowledges that month-t CPI is treated as known within month and latest-revised data are used.
- **Affected files/functions:** every historical signal origin; market linkage; Trader Research; analog tables; live-safe wording.
- **Why it matters:** market outcomes begin before the signal was observable. The result is an ex-post economic-month association, not a tradable or decision-date-safe forward distribution.
- **Type:** methodology limitation / timing look-ahead.
- **Recommended fix:** add `reference_month`, `release_date`, `information_date`, and vintage metadata. Start market measurement after the publication timestamp; add a one-month-lag sensitivity immediately and ALFRED/release-calendar support for a genuine vintage backtest.
- **Owner:** Claude.
- **Priority:** P0 before any predictive/trading use; P1 for descriptive release.

#### H6 - Live macro and market caches have no expiry or refresh control

- **Issue:** Streamlit caches both live data loaders indefinitely.
- **Evidence:** `app/streamlit_app.py:320-327` uses `@st.cache_data` without `ttl`, `max_entries`, or a refresh-generation key. No refresh control or cache clear exists.
- **Affected files/functions:** `get_data`; `get_market_data`; all dependent cached tables and report status.
- **Why it matters:** a long-running dashboard can remain stale after a CPI release, market update, cache refresh, or API-key configuration change.
- **Type:** bug / data freshness / UX.
- **Recommended fix:** add a bounded TTL, coherent manual refresh action, retrieval timestamp, and visible stale-age badge. Invalidate dependent tables as one generation and cap parameterized cache entries.
- **Owner:** Codex.
- **Priority:** P1.

#### H7 - Cold app rendering is too slow for institutional use and duplicates expensive work

- **Issue:** all nine tab bodies execute on every script run, and the report rebuilds layers already built elsewhere.
- **Evidence:** top-level tab bodies begin at `app/streamlit_app.py:525,594,815,1101,1313,1454,1568,1651,1727`. `report.py:702-733` recomputes benchmarks, robustness, analogs, and market linkage. Offline AppTest with immediate network failure took about 127 seconds on a cold run.
- **Affected files/functions:** `app/streamlit_app.py`; report builder; benchmark/robustness loops.
- **Why it matters:** a mode/baseline change or fresh process can impose a roughly two-minute wait before an analyst can use the dashboard.
- **Type:** architecture / performance / UX.
- **Recommended fix:** use navigation or conditional/lazy section rendering; put robustness behind an explicit form/run action; share computed artifacts with the report; cache a canonical forecast panel; establish cold/warm performance budgets and tests.
- **Owner:** Codex.
- **Priority:** P1.

#### H8 - The benchmark headline calls mixed error evidence a TINF win and omits AR(1)

- **Issue:** the card says `TINF wins` if **either** MAE or RMSE improvement is positive.
- **Evidence:** `app/streamlit_app.py:1368-1388` uses logical OR. At the current default 12M setting, TINF's common-sample no-change MAE improvement was about -0.16% while RMSE improvement was +2.72%; the UI still declares a win. The most important comparator, AR(1), has no corresponding headline card or common-sample improvement.
- **Affected files/functions:** Benchmark tab verdict cards and explanatory caption.
- **Why it matters:** a binary executive label overstates ambiguous evidence and hides the benchmark that often wins.
- **Type:** UX / predictive-credibility bug.
- **Recommended fix:** use `wins both`, `mixed`, `trails both`, and `indistinguishable` states; show MAE/RMSE separately; add a common-sample AR(1) card and practical significance threshold.
- **Owner:** Codex.
- **Priority:** P1.

#### H9 - The model labeled TINF/regime silently falls back to an unconditional forecast

- **Issue:** when same-regime history has fewer than eight completed observations, the forecast uses the unconditional prior mean but retains the `tinf_regime_bucket` label.
- **Evidence:** `benchmarks.py:127-135`. Current fallback shares rose from about 8.4% of scored TINF forecasts at 3M to 16.1% at 36M.
- **Affected files/functions:** `_walk_forward_regime_bucket_forecast`; benchmark metrics; robustness/report labels.
- **Why it matters:** some attributed TINF/regime performance does not use regime information at all.
- **Type:** bug / misleading model attribution.
- **Recommended fix:** add forecast provenance and bucket `n`; score pure-bucket and fallback rows separately; or implement explicit hierarchical shrinkage and name it accordingly.
- **Owner:** Codex.
- **Priority:** P1.

#### H10 - Overlapping outcomes and serial dependence have no uncertainty treatment

- **Issue:** monthly 12M-36M outcomes overlap heavily, strict numerical differences become wins, and regression inference uses HC1 only.
- **Evidence:** forward outcomes and market changes use row shifts; `robustness.py:222-246` applies strict `<`; `market_linkage._summary_metrics` treats each row as another observation; `models.py:42-48` uses HC1 despite demonstrated serial correlation. No block bootstrap, HAC forecast-loss inference, confidence interval, effective sample size, or multiple-testing control exists.
- **Affected files/functions:** validation, benchmarks, robustness, market rankings/correlations, OLS, report wording.
- **Why it matters:** raw counts and tiny loss differences look more independent and conclusive than they are.
- **Type:** methodology limitation.
- **Recommended fix:** add common-sample loss differentials with HAC/block-bootstrap uncertainty, non-overlapping sensitivity, horizon-adjusted effective `n`, subperiod stability, practical equivalence bands, and a paper-HC1 versus research-HAC distinction.
- **Owner:** Claude.
- **Priority:** P1.

### Medium

#### M1 - The roadmap-required paper-lagged TINF robustness dimension is absent

- **Issue:** only trailing-current TINF exists.
- **Evidence:** `features.add_transitory_inflation_features` has no timing parameter; `robustness.py:113-186` does not vary timing. `docs/09_PRODUCTION_ROADMAP.md` requires trailing-current versus paper-lagged TINF, while the playbook now says it is deferred.
- **Affected files/functions:** features, replication, robustness, docs.
- **Why it matters:** the project cannot quantify the impact of its largest paper-timing deviation.
- **Type:** missing capability / paper mismatch / docs drift.
- **Recommended fix:** after B1 resolves exact indexing, add explicit `tinf_timing` variants and compare them without changing the live default.
- **Owner:** Codex.
- **Priority:** P1 after replication contract.

#### M2 - Threshold robustness pseudo-replicates point-forecast evidence

- **Issue:** threshold changes classification labels but not any point forecast, MAE, RMSE, or rank; nevertheless each threshold is counted as a separate win setting.
- **Evidence:** `robustness.py:145-185,252-285`. At 12M, point metrics were byte-identical across 0.25/0.50/0.75/1.00 for every model.
- **Affected files/functions:** robustness scorecard, settings counts, win-rate charts/report.
- **Why it matters:** a 20-cell horizon x threshold claim contains only five distinct point-forecast comparisons.
- **Type:** methodology limitation / UX.
- **Recommended fix:** separate point-forecast robustness (no threshold dimension) from classification robustness (threshold-dependent metrics) and report effective distinct settings.
- **Owner:** Codex.
- **Priority:** P1.

#### M3 - Trader Research mixes regime x pressure and regime-only conditioning in one view

- **Issue:** the optional pressure filter changes instrument distributions and analog months, but the channel roll-up remains regime-only.
- **Evidence:** `trader_research.py:152-168,171-220,279-281`; top app caption at `app/streamlit_app.py:1231-1234` describes the selected combined conditioning, while `:1272-1275` quietly says the channel block is regime-only. A diagnostic produced instrument/analog n=23 and channel n=35 for the same displayed view.
- **Affected files/functions:** Trader view builder and UI.
- **Why it matters:** adjacent tables can appear to answer the same conditional question while using different populations.
- **Type:** bug / UX.
- **Recommended fix:** build pressure-conditioned channel summaries or place the regime-only channel comparison in a separately labeled block with its own count.
- **Owner:** Codex.
- **Priority:** P1.

#### M4 - Trader Research defaults to regime-only and its analog card overstates usable market evidence

- **Issue:** pressure conditioning defaults off even though current pressure is prominently displayed; the `Regime analogs` card counts label months, not complete selected-instrument/horizon outcomes.
- **Evidence:** `app/streamlit_app.py:1137-1140,1198-1204`; `trader_research.py:107-115`.
- **Affected files/functions:** Trader controls and headline metrics.
- **Why it matters:** higher `n` is obtained by blending firming/cooling/mixed states, and the headline count can be much larger than the actual distribution count.
- **Type:** methodology limitation / UX.
- **Recommended fix:** show regime-only and regime x pressure side by side with sample-size deltas; relabel the card `signal-state months`; add complete-market `n` for the selected horizon.
- **Owner:** Claude.
- **Priority:** P2.

#### M5 - Report historical analogs use a different current-regime definition from Trader Research

- **Issue:** the report matches the full-sample snapshot label to walk-forward historical labels, while Trader Research obtains the current label from the walk-forward labeler.
- **Evidence:** `report.py:394-411`; `trader_research.latest_walk_forward_bucket`; caveat at `report.py:648-651` acknowledges non-identity.
- **Affected files/functions:** report analog table and narrative.
- **Why it matters:** report analog membership can differ from the more methodologically consistent Trader bucket.
- **Type:** methodology limitation.
- **Recommended fix:** reuse the latest walk-forward bucket for analog matching and display the full-sample snapshot regime as a separate current-distribution label.
- **Owner:** Claude.
- **Priority:** P2.

#### M6 - Row-shift horizons and AR lags rely on an implicit sorted, complete monthly contract

- **Issue:** several public functions neither sort nor validate regular monthly spacing; AR fits drop NaNs and create artificial adjacency.
- **Evidence:** `market_linkage.py:129-176` uses `shift(-h)` without sorting; unsorted Jan/Mar/Feb/Apr data made a 1M change jump Jan -> Mar. `models.py:124-130` and `benchmarks.py:73-85` globally drop missing rows before lagging. `validation.py:137-146` can label an above-upper observation after a missing prior TINF as neutral.
- **Affected files/functions:** market forward changes, rolling/expanding AR(1), walk-forward regimes.
- **Why it matters:** noncanonical callers or data outages can silently change the meaning of one month and one lag.
- **Type:** latent bug / date alignment.
- **Recommended fix:** assert unique monotonic monthly indexes; sort explicitly; create t+h joins by calendar month; refuse or segment AR fits across gaps; leave direction-dependent regimes undefined after a missing prior value.
- **Owner:** Codex.
- **Priority:** P1.

#### M7 - Exact-mode macro caches lose the 12-month YoY warm-up

- **Issue:** a mode-specific cache is saved after sample trimming, then preferred over max-history and rebuilt from levels.
- **Evidence:** `scripts/fetch_fred_data.py:31-36`; `data.find_cached_macro_data_file:523-536`; `load_cached_macro_data_for_mode:543-585`. An exact cache beginning 1982-01 cannot reconstruct YoY until 1983-01, unlike live API/CSV loading with warm-up.
- **Affected files/functions:** fetch script, cache selection, paper/live exact caches.
- **Why it matters:** offline and live paths have different first-year availability despite the data contract.
- **Type:** bug / data contract.
- **Recommended fix:** save unsliced raw supersets with warm-up or make max-history the only canonical raw cache; record coverage requirements in a manifest.
- **Owner:** Codex.
- **Priority:** P1.

#### M8 - Market freshness dates are synthetic aligned month-ends, not actual source observation dates

- **Issue:** daily FRED dates are replaced by calendar month-end and later called latest observations.
- **Evidence:** `data.monthly_last:314-321`; `market_data.py:344-359,400-441`; app wording at `app/streamlit_app.py:895-897`. An existing test maps 2024-02-28 to 2024-02-29.
- **Affected files/functions:** market availability, snapshot, status captions, report.
- **Why it matters:** exact freshness cannot be audited and a holiday/weekend date can be presented as an observation date.
- **Type:** data lineage / UX.
- **Recommended fix:** preserve actual per-series source dates alongside `aligned_month`; rename existing fields until source dates are retained.
- **Owner:** Codex.
- **Priority:** P2.

#### M9 - Market fetching is all-or-nothing by transport

- **Issue:** one failed series aborts all six successful series for API or CSV.
- **Evidence:** generator-based merges at `market_data.py:141-165`; fallback at `:277-323`, even though downstream supports partial variables.
- **Affected files/functions:** market API/CSV loaders and status.
- **Why it matters:** one transient FRED-series failure can remove the full market layer.
- **Type:** resilience bug.
- **Recommended fix:** fetch/fallback per series, merge successes, expose a per-series status table, and require only the channels actually rendered.
- **Owner:** Codex.
- **Priority:** P2.

#### M10 - A market-cache fallback exists but there is no supported market-cache writer

- **Issue:** the code expects `fred_market_rates_<mode>.csv`, but scripts only populate macro caches.
- **Evidence:** `market_data.py:218-265`; `scripts/fetch_fred_data.py`; the current local raw directory has a macro cache and no market cache.
- **Affected files/functions:** offline Market Linkage, Trader Research, report market section.
- **Why it matters:** the documented fallback is not operationally reproducible.
- **Type:** missing capability / docs drift.
- **Recommended fix:** add a supported market-cache command and provenance manifest, or extend the existing fetch script with explicit dataset selection.
- **Owner:** Codex.
- **Priority:** P2.

#### M11 - Report freshness and cached status are incomplete and can themselves be stale

- **Issue:** report caveats show source names but not market per-series as-of dates, cache file, retrieval time, or age. Status objects are excluded from the report cache key.
- **Evidence:** `app/streamlit_app.py:388-406`; `report.py:570,598,635-667`; Report tab at `app/streamlit_app.py:1454-1465` does not repeat Market-tab cache/unavailable warnings.
- **Affected files/functions:** cached report wrapper, current regime table, caveats, report UI.
- **Why it matters:** identical numeric data from a new source can retain stale source prose, and stale market data can be buried.
- **Type:** bug / report-quality / UX.
- **Recommended fix:** pass a hashable status fingerprint; show source, cache basename, actual series dates, retrieval timestamp, and stale/unavailable badge at the relevant report sections.
- **Owner:** Codex.
- **Priority:** P2.

#### M12 - The Macro Research Report does not synthesize cross-sample robustness

- **Issue:** the app does not pass `robustness_sample_frames`, so the report defaults to the selected sample only.
- **Evidence:** `app/streamlit_app.py:388-406`; `report.py:709-714`.
- **Affected files/functions:** report robustness section.
- **Why it matters:** the report can sound like a full robustness synthesis while omitting paper/max-history sensitivity available in the Robustness tab.
- **Type:** incomplete integration / enhancement.
- **Recommended fix:** pass a documented standard multi-sample grid or explicitly title the section `within selected sample`; expose the included grid in the report headline.
- **Owner:** Codex.
- **Priority:** P2.

#### M13 - `paper_exact` and paper-window UI labels overstate what is exact

- **Issue:** `paper_exact=True` describes only the inflation-measure choice but appears on rows with research-upgrade samples, baselines, benchmarks, and timing. Selecting Paper window does not set a paper-method preset.
- **Evidence:** `data.py:46-56`; robustness UI exposes `paper_exact`; sample mode and baseline are independent at `app/streamlit_app.py:293-317`.
- **Affected files/functions:** inflation metadata, robustness scorecard, sidebar sample/baseline controls.
- **Why it matters:** a paper-window/rolling-shifted row can look paper-exact when it is not.
- **Type:** UX / docs drift / methodology labeling.
- **Recommended fix:** rename the field `paper_measure`; add a row-level replication-status object; provide a locked Paper-inspired or verified-paper preset once B1 is resolved.
- **Owner:** Claude.
- **Priority:** P2.

#### M14 - The 2% `fed_target` baseline is not measure-aware

- **Issue:** the same 2% target/reference is applied to headline CPI, core CPI, PCE, and core PCE, while the Federal Reserve's formal target is defined on PCE inflation.
- **Evidence:** `features.compute_baseline:59-80`; validation defaults; robustness applies each baseline to each measure.
- **Affected files/functions:** features, validation target-normalization labels, robustness, UI wording.
- **Why it matters:** calling 2% a Fed target for CPI can imply an official CPI anchor that does not exist.
- **Type:** methodology limitation / UX.
- **Recommended fix:** rename it `fixed_2pct_reference` globally or define measure-specific policy-reference semantics; retain PCE-specific Fed-target language only where appropriate.
- **Owner:** Claude.
- **Priority:** P2.

#### M15 - Baseline metadata hides consequential implementation details

- **Issue:** `expanding_shifted` silently requires 120 valid observations, and `rolling_36_unshifted` is called non-live-safe even though it is contemporaneously observable but endogenous.
- **Evidence:** `features.py:32-48,59-80`; UI uses those descriptions.
- **Affected files/functions:** baseline metadata, sidebar, warm-up interpretation.
- **Why it matters:** ten-year burn-in changes available dates; `look-ahead`, `current-observation endogeneity`, and `vintage safety` are different concepts.
- **Type:** docs drift / methodology labeling.
- **Recommended fix:** disclose min periods and separate metadata fields for future look-ahead, current self-inclusion, release/vintage safety, and recommended use.
- **Owner:** Claude.
- **Priority:** P2.

#### M16 - Source-of-truth handoff and task docs are stale and internally contradictory

- **Issue:** current docs still say the audit-fix commit awaits push although Git is synchronized.
- **Evidence:** `ACTIVE_HANDOFF.md:3,30,157-163,243-245`; `NEXT_TASKS.md:3-12`; actual `HEAD == origin/main == 3c5fe41`, ahead/behind `0 0`. `ACTIVE_HANDOFF.md:111` says 112 tests while nearby current text records 118. The maintenance state conflicts with obsolete start-phase instructions in `docs/10_AGENT_EXECUTION_PLAYBOOK.md`.
- **Affected files/functions:** handoff, next tasks, playbook, research checklist.
- **Why it matters:** the next agent is instructed to request an already-completed push and may work from the wrong gate.
- **Type:** docs drift / governance.
- **Recommended fix:** reconcile a single current-state block, retire obsolete execution sequences, mark the checklist from evidence, and date every verified gate.
- **Owner:** Claude.
- **Priority:** P1 documentation fix after correctness plan approval.

#### M17 - Dependency and CI resolution are not reproducible

- **Issue:** requirements use lower bounds only and CI exercises one Python version against whatever dependency versions are newest that day.
- **Evidence:** `requirements.txt`; `pyproject.toml`; `.github/workflows/python-checks.yml`. The recent pandas-3 label regression demonstrates the risk.
- **Affected files/functions:** environment setup, CI, release reproducibility.
- **Why it matters:** research numbers and UI behavior can change without a repository diff.
- **Type:** technical debt / release engineering.
- **Recommended fix:** add a lock or tested constraints file; separate runtime/dev/notebook/extraction dependencies; test minimum-supported and current-latest lanes on Python 3.11/3.12; record package versions in report manifests.
- **Owner:** Codex.
- **Priority:** P2.

#### M18 - No committed AppTest protects app wiring or populated fallback branches

- **Issue:** AppTest is ad hoc and absent from CI; current offline smoke has no market cache and therefore cannot exercise populated Market/Trader/report branches.
- **Evidence:** no `streamlit.testing` use in `tests/`; CI only runs Ruff/pytest/compileall; `docs/DASHBOARD_UI_POLISH_PLAN.md` calls AppTest ad hoc.
- **Affected files/functions:** full app wiring, widget state, cache warnings, all nine tab render paths.
- **Why it matters:** source modules can pass while the integrated dashboard breaks or mislabels state.
- **Type:** test gap.
- **Recommended fix:** add deterministic macro+market AppTests for success, cache, unavailable, demo, live-safe, and ex-post branches; assert tabs, units, warnings, conditioning, category order, and zero exceptions.
- **Owner:** Codex.
- **Priority:** P1.

#### M19 - The UI can expose absolute local paths in technical fetch status

- **Issue:** cache-not-found exceptions contain absolute workspace paths and the app prints full status strings.
- **Evidence:** `data.find_cached_macro_data_file`; `market_data.find_cached_market_data_file`; app status at `app/streamlit_app.py:476-485,873-880`. An offline market diagnostic included the absolute local user/workspace path in `market_live_fetch_status`.
- **Affected files/functions:** `_safe_error`, macro/market status captions, demo/unavailable paths.
- **Why it matters:** this violates the localhost review's no-local-path requirement and is unprofessional/privacy-sensitive in screenshots.
- **Type:** bug / UX / privacy.
- **Recommended fix:** sanitize project-root paths to relative basenames; keep detailed exceptions in local logs, not the main UI.
- **Owner:** Codex.
- **Priority:** P1.

#### M20 - The Streamlit controller is monolithic and source order differs from visual order

- **Issue:** `app/streamlit_app.py` is 1,981 top-level lines; render logic and computation orchestration are intertwined.
- **Evidence:** nine top-level tab blocks; Market source block appears before Benchmark although tab handles reverse visual placement.
- **Affected files/functions:** entire app and UI tests.
- **Why it matters:** semantic UI regression testing and lazy execution are harder, and small changes can re-run the entire app.
- **Type:** technical debt / architecture.
- **Recommended fix:** extract presentation-only tab renderers and shared formatted components; keep research calculations in source modules; introduce a typed app context/artifact bundle.
- **Owner:** Codex.
- **Priority:** P2 after correctness work.

#### M21 - Tables and weak-evidence visuals remain engineering-oriented

- **Issue:** 35 DataFrames render with raw snake_case, 0-1 rates, booleans, inconsistent precision, and no `column_config`; weak rows look the same as strong rows in charts.
- **Evidence:** no `column_config` in the app; internal market variable codes remain visible; weak flags appear mainly in global warnings or hover.
- **Affected files/functions:** all tabs, especially Validation, Market Linkage, Trader, Robustness, and Report.
- **Why it matters:** institutional users can misread units or miss which bar is weak.
- **Type:** UX.
- **Recommended fix:** add a shared table schema/formatter with friendly labels, pp/bp/% suffixes, integer counts, percent formatting, hidden indexes, tooltips, and row-level weak styling; encode uncertainty/weakness with opacity/hatching/icons as well as color.
- **Owner:** Codex.
- **Priority:** P2.

#### M22 - Several narrative statements exceed the evidence implemented

- **Issue:** explanatory copy turns association or test outcomes into stronger claims.
- **Evidence:** `app/streamlit_app.py:587-591` says horizon crosses usually precede regime changes; `:1639-1646` equates no white-noise rejection with exploitable structure; `:1972-1979` says ADF/KPSS agreement puts AR/decay on solid ground.
- **Affected files/functions:** Current Signal, Framework, and Robustness explanatory text.
- **Why it matters:** statistical rejection is not predictability, profitability, specification validity, or parameter stability.
- **Type:** misleading claim / UX.
- **Recommended fix:** soften to descriptive/statistical wording or add explicit empirical tests that support the stronger claim.
- **Owner:** Claude.
- **Priority:** P2.

#### M23 - Decay validity misses the already-converged t-star edge and lacks uncertainty

- **Issue:** `rho_T > 0` and `0 < mu < 1` is marked valid even when `rho_T < 0.05`, making the formula imply `t_star < 1` or a negative horizon although the threshold is already met.
- **Evidence:** `models.py:167-179`; UI displays the raw value at `app/streamlit_app.py:1664-1677`. No test covers this edge or parameter uncertainty.
- **Affected files/functions:** `paper_decay_summary`; decay cards/curve/report.
- **Why it matters:** a valid-looking negative normalization horizon is nonsensical, and point estimates look precise despite overlapping-window estimation noise.
- **Type:** latent bug / methodology limitation.
- **Recommended fix:** label already-converged cases explicitly and clamp/report the decision convention; add bootstrap/interval sensitivity across windows and parameters.
- **Owner:** Codex.
- **Priority:** P2.

#### M24 - The report and research runs have no frozen provenance/export artifact

- **Issue:** the Macro Research Report is only an in-memory UI object; no download, timestamped report, data hash, retrieval timestamp, config manifest, or frozen market snapshot is produced.
- **Evidence:** no `st.download_button`; `run_paper_replication.py` prints to stdout; scripts do not write a research-run manifest.
- **Affected files/functions:** report, scripts, release process.
- **Why it matters:** an analyst cannot reproduce or archive the exact state behind a conclusion.
- **Type:** missing capability / enhancement.
- **Recommended fix:** export a provenance-stamped Markdown/HTML report and selected CSV tables under `reports/`; include Git commit, config, package versions, source/cache status, retrieval times, date spans, imputation lineage, and hashes.
- **Owner:** Codex.
- **Priority:** P2.

#### M25 - Market rankings/correlations and Max-History weak evidence need more honest presentation

- **Issue:** means are ranked without uncertainty; the correlation heatmap hides counts; extending pre-market CPI history can relabel overlapping market-era regimes and produce thinner buckets, but the warning says only `<30`.
- **Evidence:** `market_linkage.rank_regime_pressure_market_changes`; `market_signal_correlations`; app weak warning at `app/streamlit_app.py:911-913`. A prior 2026-07-01 diagnostic (not refreshed live in this audit, so exact counts may be stale) showed Max History relabeling and smaller breakeven/real-yield buckets; the relabeling mechanism was reverified in current code.
- **Affected files/functions:** Market Linkage charts, rankings, Max History explanation.
- **Why it matters:** users may interpret rank as stable order and assume a longer sample must produce stronger evidence.
- **Type:** methodology limitation / UX / docs drift.
- **Recommended fix:** show `n` and intervals in the chart, de-emphasize weak rows, add subperiod stability, and explain that counts are complete market observations inside regenerated walk-forward buckets.
- **Owner:** Claude.
- **Priority:** P2.

### Low

#### L1 - Diagnostics have unhandled empty/constant-series edges

- **Issue:** Ljung-Box can fail on zero/one observation and ADF can fail on a constant series.
- **Evidence:** `diagnostics.py:8-32`; app catches some failures at the UI boundary, but source functions do not return a structured unavailable result.
- **Affected files/functions:** diagnostics and Framework/Robustness tabs.
- **Why it matters:** short or degenerate samples produce exceptions instead of an auditable status.
- **Type:** bug / test gap.
- **Recommended fix:** validate sample length/variance and return a standardized unavailable table; add edge tests.
- **Owner:** Codex.
- **Priority:** P3.

#### L2 - Dormant trader-report code preserves unvalidated cross-asset priors

- **Issue:** `REGIME_PLAYBOOK` and `build_trader_report` remain tested but un-wired and include equities/FX/vol/gold/TIPS prose beyond the approved rates-only surface.
- **Evidence:** `report.py:38-156,812-974`; `tests/test_trader_report.py`; explicit decision docs keep it shelved.
- **Affected files/functions:** dormant report layer and tests.
- **Why it matters:** it adds maintenance burden and could be accidentally re-surfaced outside approved scope.
- **Type:** technical debt.
- **Recommended fix:** keep it un-wired now, as decided; later isolate to an experimental module or remove only after a new scope decision.
- **Owner:** Claude.
- **Priority:** P3.

#### L3 - Several config/constants are unused or aspirational

- **Issue:** `ResearchConfig`, `SeriesConfig`, `BASE_MACRO_COLUMNS`, `PROCESSED_DATA_DIR`, and `REFERENCES_DIR` are not the active configuration path.
- **Evidence:** repository reference search finds definitions but little/no runtime consumption.
- **Affected files/functions:** `config.py`, `data.py`.
- **Why it matters:** new agents may assume these objects control the app when sidebar/module constants do.
- **Type:** technical debt.
- **Recommended fix:** either make them the canonical config contract or remove/deprecate them explicitly.
- **Owner:** Codex.
- **Priority:** P3.

#### L4 - Weak-evidence thresholds and notes are duplicated

- **Issue:** the threshold/note exist in both market linkage and report code.
- **Evidence:** `market_linkage.py:22-23`; `report.py:28,226-227`.
- **Affected files/functions:** market/report small-sample semantics.
- **Why it matters:** later policy changes can drift.
- **Type:** technical debt.
- **Recommended fix:** centralize an evidence-policy object after denominator/effective-n methodology is decided.
- **Owner:** Codex.
- **Priority:** P3.

#### L5 - Demo CPI level and supplied demo YoY are not one coherent transformation

- **Issue:** `make_demo_data` generates `inflation_yoy`, then separately approximates a CPI level via monthly compounding; the displayed level's true 12M change need not equal the supplied YoY.
- **Evidence:** `data.py:600-623`.
- **Affected files/functions:** emergency demo and UI smoke.
- **Why it matters:** acceptable for smoke, but unsafe if demo outputs are exported or used for transformation assertions.
- **Type:** enhancement / test-fixture quality.
- **Recommended fix:** derive one from the other exactly and keep the existing prominent demo prohibition.
- **Owner:** Codex.
- **Priority:** P3.

#### L6 - Launch and manual-review ergonomics are incomplete

- **Issue:** `run_app.ps1` assumes the working directory and `.venv`; README lacks a full install/launch quickstart; the ignored localhost-review note is still a placeholder.
- **Evidence:** `run_app.ps1`; `README.md`; `docs/08_LOCALHOST_REVIEW.md`; local `reports/notes/localhost_review.md`.
- **Affected files/functions:** onboarding and release process.
- **Why it matters:** reproducible human review depends on tribal knowledge.
- **Type:** docs drift / enhancement.
- **Recommended fix:** add a robust root-resolving launcher, documented setup/port command, and a dated manual-review evidence template.
- **Owner:** Codex.
- **Priority:** P3.

## 7. Things that are correct and should not change

The remediation should preserve these verified strengths:

1. **Units:** CPI inflation, baseline, epsilon, and TINF use percentage points; market rate-level changes convert percent to basis points with `(future - current) * 100`.
2. **Macro fallback policy:** API -> public CSV -> local cache -> explicitly labeled demo is the correct order. The plain loader refuses to return demo silently.
3. **Market fallback policy:** API -> public CSV -> cache -> unavailable, with no invented demo market data.
4. **Secret handling:** the optional FRED key stays in `.env`; `_safe_error` redacts the key; tracked scans found no populated secret.
5. **Ignore policy:** `.env`, `.venv/`, raw caches, the reference PDF, extracted paper text, generated logs, caches, and report artifacts are excluded correctly.
6. **Named sample bounds:** paper, live, and max-history windows are explicit and inclusive.
7. **Live baseline mechanics:** `rolling_36_shifted` and `expanding_shifted` really are shifted by one row; `full_sample` is computed after the paper date slice in the current replication script.
8. **Walk-forward thresholds:** historical regimes use expanding quantiles shifted one month and therefore do not use future TINF observations.
9. **Outcome construction order:** forward inflation outcomes are appended after signal features; future outcome columns do not directly feed baseline, epsilon, TINF, or regime construction.
10. **Regime-bucket decision gating:** only outcomes completed by the forecast decision date are eligible (`pos - horizon`).
11. **Positive-shock semantics:** downside overshoot is correctly distinguished from persistent high inflation; absolute convergence remains a separate diagnostic.
12. **Common-sample helper:** pairwise common-date MAE/RMSE improvement versus no-change and mean-reversion is implemented correctly; it should be generalized, not discarded.
13. **AR implementation details:** lag coefficients are extracted by name and rolling-rho estimates are dated at the end of their estimation window.
14. **Paper decay arithmetic:** `100 * (1 - rho_T * mu^(t-1))`, the 6M/12M exponents, and ordinary-case t-star algebra match the published expression.
15. **Rates-only registry:** exactly six approved FRED series are enforced by a hard registry check.
16. **Market completeness policy:** per-variable summaries exclude missing forward values; channel summaries require both channel members; the 29/30 weak-evidence boundary is coded and tested correctly.
17. **One-to-one monthly join:** the canonical signal/market join validates uniqueness.
18. **Descriptive scope language:** Market Linkage and Trader Research repeatedly state no forecast, no PnL, no sizing, and no recommendation. This is unusually strong and should remain visible.
19. **Visual semantics:** hot/cold colors describe inflation regimes rather than good/bad outcomes, and text labels/zero references reduce color-only dependence.
20. **Tab narrative:** Benchmark visually precedes Market Linkage, matching the intended evidence sequence.
21. **Status disclosure breadth:** source, fetch status, cache basename, raw end, CPI date, YoY date, signal date, and imputation status all have a UI surface; the issue is correctness/sanitization, not absence.
22. **Test isolation:** the normal pytest suite is network-free and fast enough for CI.

## 8. Paper-exact versus paper-inspired differences

The repository should stop using one binary `paper_exact` idea. Exactness has several dimensions: source series, vintage, transformation, sample, baseline, TINF timing, control variable, regression sample, and output tolerance.

| Component | Paper evidence | Current code | Assessment |
|---|---|---|---|
| Inflation concept | Annual CPI inflation | FRED `CPIAUCSL` YoY, percentage points | Conceptually aligned; exact series/vintage not frozen. |
| Paper sample | Paper text says 415 months and figures/table imply an effective 1987-01 to 2021-07 window, while another sentence says 1982-2021 | Fixed 1982-01 to 2021-07, 475 rows | Unresolved paper inconsistency; current result does not match effective N. |
| Baseline | Main text says historical mean/mean reversion; appendix says 36-month moving average | Replication script uses full-sample mean | Paper ambiguity not resolved empirically; published moments do not match. |
| TINF timing | Equation is lagged; prose says n-period moving average | Trailing-current rolling epsilon including month t | Deliberate live upgrade, not paper-exact. No lagged replication variant. |
| TINF horizons | 4, 8, 12 | 4, 8, 12 | Aligned. |
| TINF effective sample | Published Table 1 reports N=415 for CPI and every TINF | Variable-specific 475/472/468/464 | Mismatch. |
| Short-rate control | 1-month T-bill measure; published mean 0.243 | `TB3MS` annualized 3-month yield; mean 3.645 in current paper run | Disclosed maturity substitution, but units/economic object are also not reconciled. |
| Regression structure | Contemporaneous CPI on TINF plus bill; robust t-statistics | Same table shape, HC1 | Structure aligned; inputs and samples differ materially. |
| Regression sample | Published N=415 in all specifications | Variable-specific 472/468/464 | Mismatch and unfair cross-spec comparison. |
| Rolling TINF AR(1) | Rolling/rotating windows | Rolling AR(1) with named rho | Close in form; exact window/data treatment not frozen. |
| AR(1) on rolling rho | `rho_t = c + mu*rho_(t-1)` | Implemented | Aligned in form. |
| Decay function | `100*(1-rho_T*mu^(t-1))` | Implemented | Formula-aligned; uncertainty and edge conventions are upgrades/gaps. |
| 95% convergence | Paper threshold | Implemented | Aligned algebraically for ordinary valid cases. |
| Latest-revised data | Paper vintage is not reproduced | Latest-revised FRED or current cache | Paper-inspired; not a frozen replication. |
| Gap imputation | Not a stated paper method | Two-sided log-linear single-month bridge | Research upgrade; ex-post and lineage-sensitive. |
| Live-safe shifted baselines | Not paper | Implemented | Research upgrade. |
| Fed 2% reference | Not core paper method | Optional fixed baseline and validation outcome | Research upgrade; should be measure-aware/relabelled. |
| Percentile regimes and pressure | Not paper | Current and walk-forward labels | Research upgrade. |
| Validation/benchmarks/robustness | Not paper | Implemented | Research upgrades. |
| Market/Trader/report layers | Paper suggests future financial-link work | Implemented descriptively for rates only | Research upgrade, not paper replication. |

Minimum honest labels before replication is repaired:

- `Paper Framework` -> **paper-inspired framework**.
- `paper_replication` -> **paper sample / ex-post framework**.
- `paper_exact` -> **paper inflation measure**.
- Preserve a separate future **verified replication** preset once golden anchors pass.

## 9. No-lookahead assessment

### Overall verdict

**Partial pass, with two material exceptions.** The default code does a good job preventing full-sample percentile/baseline leakage and direct future-outcome feedback. It does **not** satisfy a strict information-set backtest because of two-sided gap interpolation and reference-month/release-date conflation.

| Layer | Assessment | Evidence and limitation |
|---|---|---|
| `full_sample` paper view | Ex-post by design | Correctly warned; acceptable only as paper-style history. |
| `rolling_36_shifted` / `expanding_shifted` formulas | Pass for row-based future look-ahead | Both shift one row; no future inflation enters the baseline mechanically on complete data. |
| `rolling_36_unshifted` | Contemporaneously observable but endogenous | It uses the current observation in its own anchor; this is not future look-ahead, but it is not the preferred clean live signal. |
| Current snapshot today | Usable as latest-revised monitoring | The latest complete row does not use future rows beyond today's loaded dataset, but it is not vintage-real-time. |
| Walk-forward historical regimes | Pass on complete, ordered data | Thresholds use prior observations only. Direction after a gap is mishandled. |
| Forward validation outcomes | Pass as target construction | `shift(-h)` outcomes are appended after signal construction and used for scoring only. |
| Benchmark AR(1) | Pass on direct future access | Expanding fit uses history through t; dropping gaps creates calendar-adjacency risk. |
| Regime-bucket forecast | Pass on outcome availability | Only completed past outcomes are used; fallback attribution remains misleading. |
| CPI interpolation | **Fail for strict historical no-lookahead** | Imputed t depends on t+1; affected origins/windows are not excluded or tagged. |
| Latest-revised FRED | Limitation | No ALFRED vintage; historical values may include later revisions. This is disclosed but not solved. |
| CPI reference month as date | **Fail for decision-date semantics** | CPI month t is treated as known at month-end t, although published in t+1. |
| Market Linkage / Trader Research | Ex-post descriptive only | Future market outcomes do not construct signals, but the starting market level predates signal publication. |
| Report | Inherits upstream status | Caveats acknowledge latest-revised data, but verdicts inherit common-sample and timing defects. |

Required terminology after remediation:

- **Row-lookahead-safe:** no later rows enter the formula on complete data.
- **Release-aligned:** the signal is not acted on before publication.
- **Vintage-safe:** values match what was available then.
- **Ex-post continuity:** uses later information to bridge a missing release.

The current phrase `live-safe` collapses all four and should be replaced or explicitly qualified everywhere.

## 10. Decision-usefulness and predictive-credibility assessment

| Use | Rating | Defensible current use |
|---|---:|---|
| Description | 7/10 | Summarize inflation versus an explicit baseline and place TINF in historical context. |
| Diagnosis | 6/10 | Compare horizons, persistence estimates, baselines, and historical episodes. |
| Scenario framing | 5/10 | Use historical rates distributions as rough scenario ranges with strong caveats. |
| Historical analog research | 5/10 | Locate months for manual study; do not treat bucket averages as stable forecasts. |
| Extrapolation | 3/10 | Decay curves and bucket means are point estimates without uncertainty/stability proof. |
| CPI point forecasting | 3/10 | AR(1) often wins; current universal comparison is not common-sample. |
| Market prediction | 2/10 | No release-aligned, vintage, out-of-sample, or uncertainty-aware market scoring. |
| Institutional decision support | 4/10 | Useful only under expert supervision and independent corroboration. |
| Trade timing/sizing/recommendation | 1/10 | Not implemented, not validated, and explicitly out of scope. |

The strongest credible statement today is:

> TINF is an interpretable historical inflation-deviation diagnostic. Its regime labels and rates-only analogs can organize research questions and scenarios, but current evidence does not establish stable out-of-sample forecasting or a tradable decision rule.

The dashboard can support:

- description of level, direction, and baseline sensitivity;
- diagnosis of whether recent deviations are firmer or cooler than longer windows;
- scenario framing with clearly conditional historical distributions;
- manual analog review;
- a watchlist of releases and market channels.

It cannot yet support:

- a statement that the paper was replicated;
- a claim that historical results reproduce the information available at the time;
- a robust claim that TINF beats AR(1) or naive models across common samples;
- predictive rate forecasts;
- instrument selection, entry date, sizing, PnL, or risk allocation.

## 11. Detailed UI/UX recommendations

### Cross-cutting institutional design

1. **Add a top trust bar.** Show reference month, actual publication/information date, data retrieval time, latest-revised/vintage status, source/cache age, baseline, and whether the current signal is influenced by imputation.
2. **Make expensive content lazy.** Replace nine eager tab bodies with grouped navigation or conditional sections: Signal, Evidence, Markets, Methods, Report. Put robustness behind an explicit `Run analysis` form.
3. **Standardize table presentation.** Use friendly labels, `column_config`, hidden indexes, fixed decimals, pp/bp/% suffixes, percent-formatted hit rates, integer `n`, and tooltips for every derived column.
4. **Make evidence strength local.** A global warning is not enough. Mark the exact weak bar/row, display the applicable denominator, and provide interval/effective-n cues.
5. **Separate description from inference visually.** Use badges such as `descriptive`, `common-sample forecast test`, `ex-post`, `release-alignment pending`, and `paper-inspired` near each section.
6. **Reduce explanatory overhang.** Keep one concise executive interpretation visible, with methodology and caveats in well-titled expanders. Current long section notes are helpful but make the app read like documentation rather than a decision workstation.
7. **Use professional names.** Replace `yield_2y`, `weak_evidence`, and other internal snake_case labels with friendly labels while retaining raw columns in audit downloads.
8. **Add provenance exports.** Export the current report and selected tables with configuration and source metadata.
9. **Add non-color cues and accessibility tests.** Keep text labels, add icons/patterns/opacity for weak evidence, verify contrast, and test keyboard/compact-screen behavior.
10. **Show uncertainty before rank.** Means, correlations, and win rates should display intervals or an indistinguishable state before top/bottom ranking.

### Tab-by-tab recommendations

#### Current Macro Signal

- Keep the four metric cards and hot/cold gap charts.
- Add `latest revised, not vintage` visibly beside the as-of date.
- Add a `changed since prior release` mini-panel: CPI change, epsilon change, TINF direction, regime/pressure transition.
- State whether the current signal window includes an imputed input and which months are affected.
- Rename Paper window unless a verified preset is selected.

#### Historical Signal Validation

- Replace the generic `count` beside conditional rates with per-metric denominators.
- Flag the combined regime x pressure table at the row level; under the current default, many buckets are below 30 and some are extremely small.
- Add confidence intervals and a minimum-display policy.
- Keep positive-shock resolution versus absolute convergence separated; that distinction is correct.
- Avoid `more actionable` unless an action/use case is defined and tested.

#### Benchmark Comparison

- Score every model on common origins.
- Replace binary OR-based cards with MAE/RMSE-specific `win / mixed / trail / indistinguishable` states.
- Put AR(1) in the headline, not only the detail table.
- Rename `cpi_persistence` to `horizon momentum` or show its exact formula.
- Disclose regime-bucket fallback share and bucket `n`.
- Show sample start/end and common `n` on the chart.

#### Market Linkage

- Do not call forward changes decision-date outcomes until release alignment is implemented.
- Preserve actual source observation dates and show them in the current snapshot.
- Add mean/median toggle, interval bands, non-overlapping sensitivity, and subperiod stability.
- Put `n` in heatmap hover/cells and mask weak correlations/rank rows.
- Explain why Max History can create smaller buckets through threshold relabeling.

#### Trader Research

- Show regime-only and regime x pressure distributions side by side, or default to combined with an explicit thin-sample fallback.
- Make the channel roll-up use the same conditioning or label it as a separate population.
- Relabel `Regime analogs` as `Signal-state months`; add complete-market `n` for the chosen instrument/horizon.
- Keep analog months as an audit trail and add a release-aligned origin date when available.
- Retain the no-forecast/no-PnL/no-recommendation language.

#### Paper Framework

- Add a visible replication-status card comparing current output with published Table 1-4 anchors.
- Distinguish paper HC1 output from a research HAC view.
- Lock the verified paper preset once built; otherwise label the tab paper-inspired.
- Keep correlation/OLS/diagnostic tables, but use a common analysis sample when comparing specifications.

#### Decay / Convergence

- Replace `first valid window` as the executive default with a window comparison or explicit user-selected primary window.
- Add uncertainty and sensitivity bands.
- Handle already-converged and negative t-star cases explicitly.
- Retain validity warnings and the disclosure that the published function uses rho_T and mu.

#### Robustness

- Separate point-forecast and classification grids so thresholds do not pseudo-replicate MAE/RMSE evidence.
- Add paper-lagged versus trailing-current timing.
- Use common-origin scores and an indistinguishable/tie state.
- Show effective distinct settings rather than raw duplicated cell count.
- Keep the baseline quick comparison, but distinguish paper measure, paper sample, and paper method.

#### Macro Research Report

- Consume the same already-computed, common-sample artifacts as the tabs.
- Show market freshness and source warnings in the relevant section, not only caveats.
- Use the current walk-forward bucket for analog matching.
- State exactly which sample/baseline/measure/horizon grid was synthesized.
- Add a provenance-stamped export and stable report ID.

## 12. Missing capabilities and test gaps

### Highest-priority missing tests

1. Golden paper Table 1-4 anchors on a frozen input and common analysis sample.
2. Explicit tests for every plausible paper lag index versus trailing-current TINF.
3. Fetch -> cache -> reload preservation of original missingness and imputation provenance.
4. Future-neighbor perturbation test proving vintage-safe signals do not change.
5. Release/information-date alignment test for CPI and market origins.
6. Universal common-sample benchmark rank/report/verdict tests, including AR(1).
7. Forecast/actual persistence anchor-consistency test.
8. Per-metric validation denominator, weak-evidence, and interval tests.
9. Regime-bucket fallback provenance and pure-bucket scoring tests.
10. Threshold-grid test proving point-error settings are not duplicated.
11. Sorted, duplicate, missing-month, and irregular-calendar contract tests.
12. Pressure-conditioned channel roll-up test.
13. Live-versus-Max-History bucket/count transition test.
14. Overlap-aware effective-n/uncertainty tests.
15. Partial per-series API/CSV fallback tests.
16. Cache TTL/manual refresh and status-fingerprint tests.
17. Actual source date versus aligned-month tests.
18. Numeric decay edge tests: already-converged, invalid, explosive, short, and uncertainty paths.
19. Empty/constant diagnostic tests.
20. Committed deterministic AppTests for populated macro+market success, cache, unavailable, demo, ex-post, and interaction branches.
21. Cold/warm render performance budget tests.
22. Secrets/path sanitization tests for user-visible status strings.

### Missing research/operational capabilities

- Frozen replication dataset/vintage and manifest.
- Release calendar and ALFRED vintage layer.
- Explicit information-date schema.
- Paper-lagged TINF implementation.
- Common-origin universal benchmark panel.
- Forecast-loss uncertainty and practical-equivalence policy.
- Overlap-aware intervals/effective sample sizes.
- Subperiod/stability and non-overlapping sensitivity.
- Measure-aware policy reference baseline.
- Supported market-cache population command.
- Provenance-stamped report/table export.
- Retrieval timestamps and refresh controls.
- Structured logging/run IDs without exposing local paths.
- Deterministic populated market fixtures for UI tests.
- Optional out-of-sample market scoring only after a separate methodology decision.

## 13. Git and release readiness

### Current state

- Branch: `main`.
- `HEAD` and `origin/main`: both `3c5fe41` at audit start.
- Ahead/behind: `0 0`.
- Pre-existing untracked path: `.claude/`, intentionally local.
- Ignored/local-only categories: `.env`, `.venv/`, `.ruff_cache/`, `.pytest_cache/`, raw macro cache, reference PDF, extracted paper text, Streamlit logs, and the localhost-review note.
- Tracked secret scan: clean.
- No staging, commit, or push was performed.

After this audit, the only authorized repository change should be:

```text
docs/FULL_CODEBASE_AUDIT_AND_FIX_PLAN.md
```

### Release decision

| Release target | Readiness |
|---|---|
| Local supervised descriptive research | Conditional pass |
| Public paper-replication claim | **Fail** |
| Historical no-lookahead/vintage backtest | **Fail** |
| Unattended live dashboard | **Fail** |
| Institutional predictive research | **Fail** |
| Trading signal/system | Out of scope and not approved |

### Safe staging allowlist

If the user later approves staging, use only:

```powershell
git add -- docs/FULL_CODEBASE_AUDIT_AND_FIX_PLAN.md
```

Do not stage `.claude/`, `.env`, `.venv/`, caches, raw data, reference files, extracted text, logs, report artifacts, or any source/config/doc file not separately approved.

Proposed commit message:

```text
docs: add full codebase audit and remediation plan
```

## 14. Prioritized remediation roadmap

### Phase 0 - Freeze claims and research contracts (Claude, then Codex)

1. Downgrade current replication labels to paper-inspired.
2. Reconcile the paper's sample, baseline, lag indexing, CPI series/vintage, and bill measure/units.
3. Define exact information-date, vintage-safe, and ex-post-imputation policies.
4. Define one persistence classification anchor and one common-sample benchmark policy.
5. Define denominator/weak-evidence/uncertainty policy by metric and horizon.

**Exit gate:** signed methodology decision record and frozen golden inputs/targets; no code fix should guess these meanings.

### Phase 1 - Repair correctness and lineage (Codex)

1. Split raw source cache from processed data; version/migrate cache schema.
2. Preserve missingness/imputation lineage and affected windows.
3. Remove two-sided imputation from vintage-safe validation or exclude affected origins.
4. Build universal common-origin benchmark scoring.
5. Align forecast/actual persistence targets.
6. Add per-metric denominators and weak-evidence flags.
7. Enforce sorted, unique, regular monthly date contracts.
8. Sanitize user-visible paths/status.

**Exit gate:** new regression tests fail on old behavior and pass on corrected behavior; all existing tests remain green.

### Phase 2 - Build verified replication and decision-date alignment (Claude + Codex)

1. Implement separate paper-exact/paper-plausible feature paths.
2. Add paper Table 1-4 golden tests and replication-difference report.
3. Add release/information dates and one-month release-aligned market sensitivity.
4. Add ALFRED vintages or explicitly keep a non-vintage limitation.
5. Add paper-lagged versus trailing-current robustness.

**Exit gate:** replication status is evidence-based; market origins never precede information availability.

### Phase 3 - Statistical credibility (Claude + Codex)

1. Add block/HAC uncertainty for overlapping outcomes and forecast-loss differences.
2. Add practical equivalence/tie states and non-overlapping sensitivity.
3. Separate threshold-dependent classification robustness from point-error robustness.
4. Add fallback/shrinkage provenance and regime x pressure benchmark variants.
5. Add HAC paper-regression comparison, subperiod stability, and decay uncertainty.

**Exit gate:** report claims are supported by common-sample estimates with uncertainty and effective `n`.

### Phase 4 - Live operability and architecture (Codex)

1. Add TTL/manual refresh, retrieval time, status fingerprints, and coherent invalidation.
2. Make heavy sections lazy and reuse computed artifact bundles.
3. Add supported market-cache writer and run manifests.
4. Add deterministic committed AppTests and cold/warm performance budgets.
5. Pin/lock dependencies and expand CI matrices.

**Exit gate:** populated cold render meets the agreed budget; new releases refresh without restart; CI reproduces the environment.

### Phase 5 - Institutional UI/report polish (Codex, wording reviewed by Claude)

1. Add trust bar, friendly tables, per-row evidence strength, and uncertainty visuals.
2. Make benchmark and Trader conditioning explicit and comparable.
3. Add verified replication-status panel.
4. Add provenance-stamped report/table exports.
5. Complete a dated manual localhost/browser review.

**Exit gate:** an external macro researcher can identify source date, information date, method status, sample/denominator, uncertainty, and allowed decision use without opening source code.

### Explicitly deferred

- New asset classes.
- Predictive market scoring.
- PnL, strategy backtests, sizing, timing, or recommendations.
- Wiring the dormant cross-asset trader playbook.

These require a fresh methodology/scope decision after Phases 0-4 pass.

## 15. Context-reset handoff template

```markdown
# Context reset - Transitory Inflation remediation

Read first:
1. docs/FULL_CODEBASE_AUDIT_AND_FIX_PLAN.md
2. README.md
3. ACTIVE_HANDOFF.md
4. NEXT_TASKS.md
5. docs/01_RESEARCH_SPEC.md
6. docs/02_DATA_CONTRACT.md
7. docs/PAPER_FORMULA_REFERENCE.md

Repository state:
- Branch / commit:
- origin/main ahead-behind:
- Expected local-only paths: .claude/, .env, .venv/, ignored caches/artifacts
- Approved write scope:
- Do not stage/commit/push without approval.

Active remediation phase:
- Phase:
- Approved finding IDs:
- Owner: Codex or Claude
- Explicit non-goals:

Methodology decisions already resolved:
- Paper sample/vintage:
- Baseline:
- Exact TINF lag indexing:
- CPI information date:
- Imputation policy:
- Persistence classification anchor:
- Common-sample policy:
- Weak-evidence/uncertainty policy:

Files allowed to change:
-

Tests that must be added first:
-

Required validation:
- python -m ruff check .
- python -m pytest
- python -m compileall src app scripts
- focused regression tests for the approved finding IDs
- deterministic offline AppTest
- manual localhost review when UI behavior changes

What changed:
-

Evidence / before-after diagnostics:
-

Open questions:
-

Recommended next step:
-
```

Do not begin with visual polish. The first approved implementation phase should resolve B1/B2 and H1-H5, with tests written against the demonstrated failures before source changes.

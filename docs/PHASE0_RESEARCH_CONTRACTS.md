# Phase 0 Research Contracts

**Status:** approved methodology contract for regression-test-first remediation

**Decision date:** 2026-07-10

**Scope:** B1, B2, and the replication, imputation, information-date,
classification-anchor, common-sample, and evidence-strength contracts

**Implementation boundary:** Phase 0 freezes meanings, targets, and failing tests. It does not
repair production behavior or authorize replication, vintage-safe, predictive, or trading claims.

The machine-readable companion is
`tests/fixtures/paper_replication_contract_v1.json`. Its published targets and candidate-input hash
are part of this contract.

## 1. Project Objective and Scope

This repository is not solely a paper-replication project. It is a macro
inflation-regime research dashboard inspired by Peron and Bonaparte.

Its primary purpose is to evaluate current inflation in terms of level,
direction, pace, persistence, and historical context, and to support portfolio
scenario analysis through descriptive inflation and rates evidence.

The dashboard is intended to help the user assess whether current inflation
pressure appears persistent or transitory and use that macro assessment as one
input into discretionary portfolio positioning. It does not generate automatic
asset-allocation instructions, trade entries, position sizing, or investment
recommendations.

Verified paper replication is a separate validation layer. It must not replace
or silently redefine the Current Research or Max History workflows.

### Retained sample modes

The dashboard retains three sample modes:

1. **Live Dashboard:** 1982 to the latest available observation.
2. **Paper Window:** 1982-01 through 2021-07.
3. **Max History:** earliest available FRED observation through the latest
   available observation.

The Paper Window identifies the paper-era source period. It does not, by
itself, mean that all selected formulas, transformations, samples, controls,
and regressions are paper-exact.

Sample selection is separate from:

- baseline methodology;
- TINF timing;
- missing-data treatment;
- information-date policy;
- data-vintage status;
- benchmark evaluation sample;
- paper-replication status.

### Analytical roles

- **Live Dashboard:** current inflation diagnosis and monitoring using the
  selected research methodology.
- **Paper Window:** paper-era comparison, paper-method investigation, and
  replication validation.
- **Max History:** longest-history robustness and cross-cycle analysis.

The primary dashboard remains a macro research and scenario-framing tool.
Paper replication is one specialist validation component within that broader
objective.

## 2. Replication contract (B1)

### 2.1 Status and naming

The current shared feature path is a paper-inspired, ex-post framework. It is not a verified paper
replication. The current user-facing and configuration concept `paper_replication` must become
`paper_inspired_window`; the old name may exist only as a temporary deprecated alias and must never
produce a verified status or a public replication claim.

A future reconstruction is verified only when every dimension below is verified:

- sample;
- CPI source and vintage;
- CPI transformation;
- baseline;
- TINF timing;
- rate control and units;
- common sample;
- published Table 1-4 anchors.

An unknown or proxy dimension fails the overall exact-replication claim. Passing printed anchors
alone is insufficient if source fidelity is unknown.

### 2.2 Reconstruction candidate

- Raw construction period: 1982-01 through 2021-07, inclusive.
- Common analysis period: 1987-01 through 2021-07, inclusive, exactly 415 months.
- CPI transformation: `100 * (CPI_t / CPI_(t-12) - 1)`, in percentage points.
- Baseline for epsilon at month `s`: the trailing 36-month mean of inflation ending at and including
  `s`.
- `epsilon_s = inflation_s - baseline_s`.
- Paper-literal TINF timing:
  `TINF_n,t = mean(epsilon_(t-2), ..., epsilon_(t-n-1))` for `n in {4, 8, 12}`.
- Rate control: `TB3MS / 12`, in monthly percentage points, as a reproducible proxy. The contract
  records `control_source_exact=false` because the paper labels the control a one-month bill.
- No CPI imputation is permitted in the reconstruction.
- Tables 1-3 use the same complete 415-row sample for CPI, every TINF horizon, and the bill proxy.
  Model-specific row dropping is prohibited.

The 1982 construction start and 1987 analysis start are distinct. The five-year lead-in supplies the
12-month CPI transformation, 36-month baseline, and literal lagged 12-month TINF window needed to
make all variables complete in 1987-01.

### 2.3 Frozen inputs, targets, and tolerance

The deterministic candidate fixture contains only date, CPI level, and TB3MS for 1982-01 through
2021-07. It is a normalized test fixture derived from the current local FRED snapshot, not an
author-equivalent vintage and not a golden replication dataset.

Published comparisons use the paper's displayed precision:

- counts and dates: exact;
- numeric values: equality after rounding to the number of decimals printed in the paper;
- no broad tolerance for data revisions or source substitutions.

The candidate currently reaches the 415-row common sample and closely reproduces the monthly bill
moments after `TB3MS / 12`, but it misses CPI and TINF moments materially. Its required status is
therefore `unverified_candidate`.

## 3. Imputation contract (B2)

### 3.1 Observed-only policy

`observed_only` is mandatory for historical validation, benchmarks, robustness, market linkage,
historical analogs, and any surface described as live-like, release-aligned, or vintage-safe.

- Missing CPI remains missing.
- A signal origin is inadmissible when any CPI input used by its CPI transformation, baseline, or
  TINF window is missing or imputed.
- A scored outcome is inadmissible when its target uses a missing or imputed CPI input.
- Multi-month gaps and trailing gaps remain missing.
- Phase 0 does not authorize a one-sided nowcast.

### 3.2 Ex-post continuity policy

`ex_post_continuity` may log-linearly bridge one isolated interior CPI-level gap only in explicitly
descriptive ex-post views. The filled value is not available until the following CPI observation is
released.

Lineage must preserve and propagate:

- original missingness;
- imputation method;
- imputed reference month;
- imputation availability timestamp;
- `uses_imputed_input` for CPI YoY, baseline, epsilon, TINF, outcomes, and summaries.

Raw source values and original missingness are the cache authority. A processed filled value may not
be reloaded as if it were originally observed.

## 4. Information-date contract

The following concepts are distinct:

- `reference_month`: economic month measured by CPI;
- `release_timestamp`: publication time for that observation;
- `information_timestamp`: latest availability time among every input used by a derived row;
- `vintage_timestamp`: vintage of the stored value;
- `retrieved_at`: time the project fetched or loaded it.

Required rules:

1. CPI reference month is never its information date.
2. A derived signal's information timestamp is the maximum input availability timestamp across its
   dependency window.
3. Ex-post imputation uses the following observation's release timestamp as its earliest information
   timestamp.
4. Exact market measurement begins at the first eligible market close on or after publication.
5. A month-end `t+1` market origin is allowed only as a labelled conservative release proxy.
6. Missing release metadata forces `reference_month_only`; it cannot be labelled release-aligned or
   vintage-safe.
7. Vintage-safe requires values actually available by the decision timestamp. Latest-revised FRED
   values remain explicitly non-vintage.

The terms `row-lookahead-safe`, `release-aligned`, `vintage-safe`, and `ex-post continuity` may not be
collapsed into `live-safe`.

## 5. Classification-anchor contract

Persistence scoring freezes the origin baseline for predicted and realized labels. With the default
0.50 percentage-point threshold:

- eligible positive shock: `inflation_t - baseline_t >= 0.50`;
- predicted persistence: `forecast_(t+h|t) - baseline_t > 0.50`;
- realized persistence: `inflation_(t+h) - baseline_t > 0.50`.

Origins without an eligible positive shock receive nullable persistence labels and are excluded from
classification denominators. `baseline_(t+h)` may be used only in a separately named ex-post
diagnostic. Fixed-policy-target and absolute-gap diagnostics remain separate targets.

## 6. Common-sample contract

Construct one universal common-origin panel per horizon and metric family:

- point loss;
- direction;
- classification.

Every registered model, including AR(1), must have a forecast on every universal origin. Headline
metrics, ranks, robustness verdicts, cards, and report claims use the universal panel.

Pairwise comparisons may use larger pairwise intersections, but must report their own count, date
range, and missing-forecast reasons. Pairwise results cannot determine universal ranks or headlines.

## 7. Evidence-strength contract

Every statistic reports its own denominator. Binary rates also report their numerator. Required
fields include the applicable count, date range, horizon, and missingness exclusions.

Preserve the compatibility rule:

- `weak_evidence = n_applicable < 30`;
- 29 is weak;
- 30 is not weak.

Local evidence states are:

- `unavailable`: `n=0`;
- `sparse`: `n=1..9`;
- `weak`: `n=10..29`;
- `descriptive`: `n>=30`.

Count alone never yields a strong or inferential claim. Horizons greater than one month also report
that outcomes overlap and a deterministic non-overlapping count. That count does not replace the
existing 30-observation compatibility boundary.

Naive independent-observation intervals are prohibited for overlapping outcomes. Until an approved
HAC or block-bootstrap implementation exists, uncertainty is reported as unavailable and language
remains descriptive. Point estimates may say `lower point loss`; they may not say `wins` or `beats`.

## 8. Phase 0 regression-test gate

The first implementation-facing interfaces are frozen as follows:

- `add_paper_reconstruction_features(frame)` is the separate paper-candidate feature path and emits
  `baseline`, `epsilon`, and literal-lag `tinf_4m`, `tinf_8m`, and `tinf_12m`.
- `build_base_frame(..., imputation_policy=...)` and
  `load_cached_macro_data_for_mode(..., imputation_policy=...)` accept only `observed_only` or
  `ex_post_continuity`.
- Benchmark forecast output is prefiltered to universal common origins before public metric,
  robustness, or report summaries consume it.
- Validation summaries retain the generic complete-future-row `count`, but each rate adds
  `<rate>_numerator`, `<rate>_n_applicable`, `<rate>_evidence_strength`, and
  `<rate>_weak_evidence`. Horizon summaries also add `overlapping_outcomes`,
  `non_overlapping_count`, and `uncertainty_status`.

Tests are added before source repairs and cover:

1. fail-closed replication naming and dimensional verification;
2. the 415-row analysis window and literal TINF lag;
3. future-neighbor invariance under `observed_only`;
4. ex-post availability and lineage propagation;
5. reference-month versus information-date semantics;
6. frozen-origin classification anchors;
7. universal benchmark origins, including AR(1);
8. metric-specific numerators, denominators, overlap, and evidence state.

Existing tests that affirm provenance loss or overclaiming must be replaced rather than preserved as
compatibility requirements. Production fixes belong to a later approved phase.

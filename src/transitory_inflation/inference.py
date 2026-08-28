"""Overlap-aware uncertainty for pairwise forecast-loss differentials (H10).

Monthly forecast origins at 3-36 month horizons produce heavily overlapping
outcomes, so treating each origin as an independent observation understates the
variance of a loss differential by a factor of 2.5-10 on this project's panels.
This module is the pure statistical engine that fixes that. It knows nothing
about CPI, models, panels, or Streamlit: it takes two aligned forecast-error
sequences and returns one :class:`LossDifferentialResult`.

Primary inference is a Diebold-Mariano test of equal predictive accuracy with a
Newey-West (Bartlett) long-run variance, compared against Student-t with
``n - 1`` degrees of freedom. The Harvey-Leybourne-Newbold small-sample
correction is applied as a **variance** correction, so the reported standard
error, t-statistic, every p-value, and the confidence interval all invert one
and the same corrected test. The effective sample size implied by the variance
inflation is reported as a diagnostic only; it does not replace the degrees of
freedom, because no established result licenses that substitution.

Verdicts are decisions about a *practical* difference, so they are tested that
way. ``lower`` and ``higher`` invert one-sided tests of the band boundaries
(``H0: delta >= -band`` and ``H0: delta <= +band``), and
``practically_equivalent`` inverts the two one-sided tests of the TOST
equivalence null. Each is evaluated at ``alpha/2``, which makes every verdict
exactly the statement that the two-sided ``confidence_level`` interval clears
the band on the relevant side. The zero-null p-value is still reported, but it
is a diagnostic and never drives a verdict or a multiplicity adjustment.

Two companions are computed alongside, never instead of, the primary test: a
seeded moving-block bootstrap and a non-overlapping phase summary. They exist so
a reader can see whether the primary interval is corroborated; they never
produce their own verdict.

:func:`compare_forecast_losses` scores one pair. Multiplicity is a property of a
family of comparisons, so a caller that assembles several related pairs must pass
them through :func:`apply_holm_family` before publishing any directional verdict.

The engine fails closed. A degenerate sample, a sample below the gate, a
nested pair, or a missing calendar index yields an explicit unavailable state and
a named reason. It never converts an absence of evidence into an inferred
population equivalence.

Scope boundary: this quantifies uncertainty about realized losses on whatever
sample the caller supplies. It says nothing about vintage safety, release
timing, or out-of-sample skill, and it must never be used to upgrade a
descriptive comparison into a claim of tradable forecasting ability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import stats

#: Loss scales the engine can score. Absolute error aggregates to MAE; squared
#: error aggregates to MSE. RMSE is a monotone transform of MSE, so it inherits
#: the squared-error ordering verdict but is not itself a mean of per-origin
#: losses and is therefore never tested directly.
LOSS_ABSOLUTE_ERROR = "absolute_error"
LOSS_SQUARED_ERROR = "squared_error"
LOSS_SCALES: tuple[str, ...] = (LOSS_ABSOLUTE_ERROR, LOSS_SQUARED_ERROR)

DEFAULT_CONFIDENCE_LEVEL = 0.95

#: Practical-materiality band as a share of the comparison model's own mean
#: loss. Relative rather than absolute because loss levels roughly triple across
#: the 3M-36M grid, so a fixed percentage-point band would mean very different
#: things at different horizons. This is a research policy choice, not a
#: statistical output: it must be displayed wherever a verdict is displayed.
DEFAULT_EQUIVALENCE_BAND_FRACTION = 0.05

DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260826

#: Below this, bootstrap quantiles are too coarse to corroborate anything, and a
#: moving-block resample needs enough distinct blocks to reproduce the
#: dependence it exists to capture. Both failures are disclosed, never silent.
MIN_BOOTSTRAP_REPS = 200
MIN_BOOTSTRAP_BLOCKS = 8

#: Fail-closed gate. The floor of 30 reproduces the project's existing 29/30
#: evidence boundary, which is exactly what the rule reduces to at short
#: horizons; the per-lag term keeps the sample long relative to the bandwidth.
MIN_OBSERVATIONS = 30
OBSERVATIONS_PER_LAG = 5

#: Above this share of missing origins the interval is still computed but the
#: status is downgraded, because gap-heavy panels stress the amplitude-modulated
#: autocovariance more than the estimator's assumptions comfortably support.
MAX_GAP_SHARE = 0.10

UNCERTAINTY_AVAILABLE = "available"
UNCERTAINTY_DEGRADED = "degraded"
UNCERTAINTY_UNAVAILABLE = "unavailable"
BOOTSTRAP_NOT_REQUESTED = "not_requested"

REASON_NO_OBSERVATIONS = "no_common_observations"
REASON_INSUFFICIENT_SAMPLE = "insufficient_effective_sample"
REASON_IDENTICAL_LOSSES = "identical_losses"
REASON_DEGENERATE_ZERO_VARIANCE = "degenerate_zero_variance"
REASON_NONPOSITIVE_HAC_VARIANCE = "nonpositive_hac_variance"
REASON_DEGRADED_PANEL_GAPS = "degraded_panel_gaps"
REASON_NESTED_PAIR = "nested_pair_ordinary_dm_invalid"
REASON_NO_ORIGIN_POSITIONS = "origin_positions_unavailable"
REASON_INSUFFICIENT_BOOTSTRAP_REPS = "insufficient_bootstrap_reps"
REASON_INSUFFICIENT_BOOTSTRAP_BLOCKS = "insufficient_bootstrap_blocks"

VERDICT_LOWER = "lower"
VERDICT_HIGHER = "higher"
VERDICT_PRACTICALLY_EQUIVALENT = "practically_equivalent"
VERDICT_INCONCLUSIVE = "inconclusive"
#: The sample cannot support any test: zero variance, or a long-run variance
#: that is not positive. Distinct from ``inconclusive``, which means a valid
#: test ran and did not separate the models.
VERDICT_DEGENERATE = "degenerate"
#: No usable sample at all — empty input, or below the fail-closed gate.
VERDICT_UNAVAILABLE = "unavailable"
#: An estimation-nested pair, where the ordinary DM test does not apply.
VERDICT_NESTED_NOT_TESTED = "nested_not_tested"

VERDICTS: tuple[str, ...] = (
    VERDICT_LOWER,
    VERDICT_HIGHER,
    VERDICT_PRACTICALLY_EQUIVALENT,
    VERDICT_INCONCLUSIVE,
    VERDICT_DEGENERATE,
    VERDICT_UNAVAILABLE,
    VERDICT_NESTED_NOT_TESTED,
)

#: Verdicts that assert something about the models. Only these are subject to
#: family-wise multiplicity control.
TESTED_VERDICTS: tuple[str, ...] = (
    VERDICT_LOWER,
    VERDICT_HIGHER,
    VERDICT_PRACTICALLY_EQUIVALENT,
    VERDICT_INCONCLUSIVE,
)

NESTED_PAIR_DISCLOSURE = (
    "Estimation-nested pair: the two forecasts coincide in population under the null, so the "
    "loss differential is degenerate and the ordinary Diebold-Mariano test does not apply. No "
    "directional or equivalence verdict is issued. A nested-model test such as Clark-West would "
    "be required to compare these two, and is not implemented."
)


@dataclass(frozen=True)
class LossDifferentialResult:
    """One pairwise loss-differential comparison with overlap-aware uncertainty.

    The sign convention is fixed and load-bearing: the differential is
    ``mean_loss_a - mean_loss_b``, so a **negative** differential means model A
    posted the lower loss, and every verdict describes A relative to B.
    """

    # identity
    label_a: str
    label_b: str
    loss_scale: str
    horizon_months: int

    # sample
    observations: int
    span_observations: int
    missing_observations: int
    overlapping_outcomes: bool

    # point estimates
    mean_loss_a: float
    mean_loss_b: float
    loss_differential: float

    # dependence diagnostics
    hac_lag: int
    long_run_variance: float
    sample_variance: float
    variance_inflation_factor: float
    n_effective: float
    hac_standard_error: float

    # primary inference — every field below inverts the same corrected test
    loss_differential_se: float
    hln_factor: float
    t_stat: float
    degrees_of_freedom: int
    confidence_level: float
    ci_low: float
    ci_high: float
    p_value: float

    # practical-boundary hypotheses
    equivalence_band: float
    equivalence_band_fraction: float
    directional_alpha: float
    p_practical_lower: float
    p_practical_higher: float
    p_practical_directional: float
    p_equivalence: float
    p_directional_holm: float
    family_size: int

    # decision
    verdict: str
    detectable_but_negligible: bool

    # companion: bootstrap
    bootstrap_status: str
    bootstrap_reason: str | None
    bootstrap_reps: int
    bootstrap_block: int
    bootstrap_seed: int
    bootstrap_se: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    # companion: non-overlapping origins
    origin_positions_supplied: bool
    non_overlapping_status: str
    non_overlapping_reason: str | None
    non_overlapping_count: int
    non_overlapping_phases: int
    non_overlapping_phase_mean: float
    non_overlapping_phase_min: float
    non_overlapping_phase_max: float

    # disclosure
    uncertainty_status: str
    uncertainty_reason: str | None
    nested_pair: bool
    nested_pair_disclosure: str | None

    @property
    def p_directional(self) -> float:
        """The practical-boundary p-value multiplicity control operates on."""

        return self.p_practical_directional

    def as_dict(self) -> dict[str, object]:
        """Return a plain mapping, ready for long-form frame construction."""

        return asdict(self)


def hac_bandwidth(observations: int, horizon: int) -> int:
    """Return the Bartlett truncation lag for an ``horizon``-step comparison.

    An optimal h-step forecast error is MA(h-1), which makes ``h - 1`` the
    theoretical floor. Measured autocorrelations on this project's panels run
    past that floor because the compared forecasts are not optimal and skill
    differences persist across cycles, so the usual automatic rule is kept as a
    second lower bound rather than as an alternative.
    """

    n = int(observations)
    if n <= 0:
        return 1
    automatic = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(int(horizon) - 1, automatic, 1)


def required_observations(hac_lag: int) -> int:
    """Return the minimum sample the fail-closed gate demands for ``hac_lag``."""

    return max(MIN_OBSERVATIONS, OBSERVATIONS_PER_LAG * (int(hac_lag) + 1))


def non_overlapping_count(observations: int, horizon: int) -> int:
    """Closed form for the disjoint-window count on a **contiguous** panel.

    Provided for the contiguous special case only. Real panels can have calendar
    gaps, and then this over-counts, so the engine uses
    :func:`non_overlapping_count_from_positions` with actual origin positions.
    """

    n = int(observations)
    h = max(int(horizon), 1)
    if n <= 0:
        return 0
    return math.ceil(n / h)


def non_overlapping_count_from_positions(
    origin_positions: Sequence[int] | np.ndarray,
    horizon: int,
) -> int:
    """Return the largest set of origins whose outcome windows never overlap.

    Greedy scan over actual calendar positions: take the earliest origin, then
    the next one at least ``horizon`` months later, and so on. On a contiguous
    panel this reduces to ``ceil(n / h)``; across a calendar gap it does not,
    which is exactly why a positional stride cannot stand in for it.
    """

    h = max(int(horizon), 1)
    ordered = sorted({int(position) for position in np.asarray(origin_positions).ravel()})
    count = 0
    next_allowed: float = -math.inf
    for position in ordered:
        if position >= next_allowed:
            count += 1
            next_allowed = position + h
    return count


def harvey_leybourne_newbold_factor(observations: int, horizon: int) -> float:
    """Return the HLN finite-sample correction factor.

    Shrinks the DM statistic by about 9% at (n=414, h=36) and 0.6% at
    (n=447, h=3), so it bites where the horizon is long relative to the sample.
    Applied as a variance correction — the inference standard error is the HAC
    one divided by this factor — so the statistic, the p-values, and the
    interval stay mutually consistent.
    """

    n = int(observations)
    h = int(horizon)
    if n <= 0:
        return float("nan")
    inner = (n + 1 - 2 * h + h * (h - 1) / n) / n
    return float(math.sqrt(max(inner, 0.0)))


def _finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"Expected a one-dimensional sequence, got shape {array.shape}")
    return array


def _modulated(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Demean over observed points and zero-fill the gaps.

    Missing entries are held **in place** rather than dropped, so lag ``j`` stays
    lag ``j`` in time. Zero-filling the demeaned series (amplitude modulation)
    keeps the Bartlett estimator's non-negativity guarantee intact, because that
    guarantee is algebraic in the modulated sequence and does not require the
    sequence to be gap-free.
    """

    observed = np.isfinite(values)
    count = int(observed.sum())
    centered = np.zeros_like(values)
    if count:
        centered[observed] = values[observed] - values[observed].mean()
    return centered, count


def bartlett_long_run_variance(values: Sequence[float] | np.ndarray, lags: int) -> float:
    """Return the Newey-West long-run variance of the mean of ``values``.

    ``gamma_j = (1/n) * sum_t (d_t - dbar)(d_{t-j} - dbar)`` with the Bartlett
    weights ``1 - j/(L+1)``. Non-finite entries are treated as missing
    observations in place; ``n`` is the observed count for every lag, which is
    what preserves non-negativity.
    """

    array = _finite_array(values)
    centered, count = _modulated(array)
    if count == 0:
        return float("nan")

    truncation = max(int(lags), 0)
    total = float(centered @ centered) / count
    for lag in range(1, truncation + 1):
        if lag >= centered.size:
            break
        gamma = float(centered[lag:] @ centered[:-lag]) / count
        total += 2.0 * (1.0 - lag / (truncation + 1.0)) * gamma
    return float(total)


def _sample_variance(values: np.ndarray) -> float:
    observed = values[np.isfinite(values)]
    if observed.size < 2:
        return float("nan")
    return float(observed.var(ddof=1))


def holm_adjusted_p_values(p_values: Sequence[float] | np.ndarray) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values, preserving input order.

    Holm rather than Benjamini-Hochberg because comparisons within one family
    share a panel and a set of actuals, so they are heavily dependent; Holm
    controls the family-wise error rate under arbitrary dependence. Non-finite
    entries are comparisons that could not be tested: they are excluded from the
    family size and returned unchanged.
    """

    array = _finite_array(p_values)
    adjusted = np.full(array.shape, np.nan, dtype=float)
    testable = np.flatnonzero(np.isfinite(array))
    family_size = int(testable.size)
    if family_size == 0:
        return adjusted.tolist()

    order = testable[np.argsort(array[testable], kind="stable")]
    running = 0.0
    for rank, position in enumerate(order):
        candidate = min(1.0, (family_size - rank) * float(array[position]))
        running = max(running, candidate)
        adjusted[position] = running
    return adjusted.tolist()


def apply_holm_family(
    results: Sequence[LossDifferentialResult],
) -> list[LossDifferentialResult]:
    """Attach family-wise multiplicity control to one family of comparisons.

    Holm is applied to the **practical-boundary** p-value that the directional
    verdict actually inverts, not to the zero-null p-value. Adjusting the wrong
    hypothesis would control the error rate of a claim nobody is making.

    A directional verdict that no longer clears ``directional_alpha`` after
    adjustment is demoted to ``inconclusive``. ``practically_equivalent`` is
    never demoted: Holm controls false directional claims, and an equivalence
    conclusion is not a directional claim. Comparisons that were never tested —
    gated, degenerate, or nested — carry no practical p-value, so they neither
    join the family nor consume any of its budget.
    """

    family = list(results)
    testable = [
        result.p_practical_directional if result.verdict in TESTED_VERDICTS else float("nan")
        for result in family
    ]
    adjusted = holm_adjusted_p_values(testable)
    family_size = sum(1 for value in testable if math.isfinite(value))

    updated: list[LossDifferentialResult] = []
    for result, p_holm in zip(family, adjusted, strict=True):
        verdict = result.verdict
        if verdict in (VERDICT_LOWER, VERDICT_HIGHER) and (
            not math.isfinite(p_holm) or p_holm > result.directional_alpha
        ):
            verdict = VERDICT_INCONCLUSIVE
        updated.append(
            replace(result, p_directional_holm=p_holm, family_size=family_size, verdict=verdict)
        )
    return updated


def moving_block_bootstrap_mean(
    values: Sequence[float] | np.ndarray,
    *,
    block: int,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[float, float, float, int]:
    """Return ``(se, ci_low, ci_high, block)`` for the mean, by moving blocks.

    A seeded companion to the primary HAC interval, never a replacement for it.
    Observed values are compressed into contiguous order before resampling, so
    with a gappy input the blocks span the gaps; that approximation is one
    reason this stays a companion diagnostic. Callers that need the minimum-reps
    and minimum-blocks policy should go through :func:`compare_forecast_losses`,
    which enforces and discloses it.
    """

    array = _finite_array(values)
    observed = array[np.isfinite(array)]
    n = int(observed.size)
    block_length = max(2, min(int(block), n))
    if n < 2 or int(reps) <= 0:
        return float("nan"), float("nan"), float("nan"), block_length

    rng = np.random.default_rng(int(seed))
    block_count = math.ceil(n / block_length)
    starts = rng.integers(0, n - block_length + 1, size=(int(reps), block_count))
    offsets = np.arange(block_length)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(int(reps), -1)[:, :n]
    means = observed[indices].mean(axis=1)

    tail = (1.0 - float(confidence_level)) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(means.std(ddof=1)), float(low), float(high), block_length


def non_overlapping_phase_means(
    values: Sequence[float] | np.ndarray,
    origin_positions: Sequence[int] | np.ndarray,
    horizon: int,
) -> list[float]:
    """Return the mean differential of each disjoint-outcome phase subsample.

    Phases are ``position mod horizon`` over **actual calendar positions**, so
    origins separated by a calendar gap land in the phase their date implies,
    not the one their row index implies. Positions are required rather than
    optional: a positional stride silently assumes a gap-free panel.

    Taking every h-th origin removes the mechanical overlap between outcome
    windows. It does not remove regime persistence, and it discards most of the
    sample at long horizons, so this is a sensitivity check and never a
    replacement for the HAC interval.
    """

    array = _finite_array(values)
    positions = np.asarray(origin_positions, dtype=int).ravel()
    if positions.size != array.size:
        raise ValueError(
            f"Origin positions must be aligned with the values: "
            f"{positions.size} != {array.size}"
        )

    h = max(int(horizon), 1)
    means: list[float] = []
    for phase in range(h):
        selected = array[(positions % h) == phase]
        observed = selected[np.isfinite(selected)]
        if observed.size:
            means.append(float(observed.mean()))
    return means


def _loss(errors: np.ndarray, loss_scale: str) -> np.ndarray:
    if loss_scale == LOSS_ABSOLUTE_ERROR:
        return np.abs(errors)
    if loss_scale == LOSS_SQUARED_ERROR:
        return np.square(errors)
    raise ValueError(f"Unknown loss scale: {loss_scale!r}. Expected one of {list(LOSS_SCALES)}")


def _resolve_positions(
    origin_positions: Sequence[int] | np.ndarray | None,
    length: int,
) -> np.ndarray | None:
    if origin_positions is None:
        return None

    positions = np.asarray(origin_positions, dtype=float).ravel()
    if positions.size != length:
        raise ValueError(
            f"Origin positions must be aligned with the forecast errors: "
            f"{positions.size} != {length}"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("Origin positions must all be present and finite")
    rounded = np.rint(positions)
    if not np.allclose(positions, rounded):
        raise ValueError("Origin positions must be whole month offsets")
    if positions.size > 1 and not np.all(np.diff(rounded) > 0):
        raise ValueError("Origin positions must be strictly increasing")
    return (rounded - rounded[0]).astype(int)


def _empty_bootstrap() -> dict[str, object]:
    return {
        "bootstrap_status": BOOTSTRAP_NOT_REQUESTED,
        "bootstrap_reason": None,
        "bootstrap_reps": 0,
        "bootstrap_block": 0,
        "bootstrap_se": float("nan"),
        "bootstrap_ci_low": float("nan"),
        "bootstrap_ci_high": float("nan"),
    }


def _run_bootstrap(
    differential: np.ndarray,
    *,
    observations: int,
    horizon: int,
    reps: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    """Run the block bootstrap, or say precisely why it could not run."""

    payload = _empty_bootstrap()
    requested = max(int(reps), 0)
    if requested == 0:
        return payload

    block = max(int(horizon), 2)
    payload["bootstrap_reps"] = requested
    payload["bootstrap_block"] = block
    payload["bootstrap_status"] = UNCERTAINTY_UNAVAILABLE

    if requested < MIN_BOOTSTRAP_REPS:
        payload["bootstrap_reason"] = REASON_INSUFFICIENT_BOOTSTRAP_REPS
        return payload
    if observations // block < MIN_BOOTSTRAP_BLOCKS:
        payload["bootstrap_reason"] = REASON_INSUFFICIENT_BOOTSTRAP_BLOCKS
        return payload

    se, low, high, used_block = moving_block_bootstrap_mean(
        differential,
        block=block,
        reps=requested,
        seed=seed,
        confidence_level=confidence_level,
    )
    if not math.isfinite(se):
        payload["bootstrap_reason"] = REASON_INSUFFICIENT_BOOTSTRAP_BLOCKS
        return payload

    payload.update(
        {
            "bootstrap_status": UNCERTAINTY_AVAILABLE,
            "bootstrap_reason": None,
            "bootstrap_block": used_block,
            "bootstrap_se": se,
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
        }
    )
    return payload


def _non_overlap_summary(
    differential: np.ndarray,
    positions: np.ndarray | None,
    horizon: int,
) -> dict[str, object]:
    """Summarize the disjoint-window subsamples, or fail closed without a calendar."""

    if positions is None:
        return {
            "non_overlapping_status": UNCERTAINTY_UNAVAILABLE,
            "non_overlapping_reason": REASON_NO_ORIGIN_POSITIONS,
            "non_overlapping_count": 0,
            "non_overlapping_phases": 0,
            "non_overlapping_phase_mean": float("nan"),
            "non_overlapping_phase_min": float("nan"),
            "non_overlapping_phase_max": float("nan"),
        }

    observed = np.isfinite(differential)
    kept = positions[observed]
    phases = non_overlapping_phase_means(differential[observed], kept, horizon)
    return {
        "non_overlapping_status": UNCERTAINTY_AVAILABLE,
        "non_overlapping_reason": None,
        "non_overlapping_count": non_overlapping_count_from_positions(kept, horizon),
        "non_overlapping_phases": len(phases),
        "non_overlapping_phase_mean": float(np.mean(phases)) if phases else float("nan"),
        "non_overlapping_phase_min": float(np.min(phases)) if phases else float("nan"),
        "non_overlapping_phase_max": float(np.max(phases)) if phases else float("nan"),
    }


def compare_forecast_losses(
    errors_a: Sequence[float] | np.ndarray,
    errors_b: Sequence[float] | np.ndarray,
    *,
    horizon: int,
    origin_positions: Sequence[int] | np.ndarray | None = None,
    loss_scale: str = LOSS_ABSOLUTE_ERROR,
    label_a: str = "a",
    label_b: str = "b",
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    equivalence_band_fraction: float = DEFAULT_EQUIVALENCE_BAND_FRACTION,
    equivalence_band: float | None = None,
    nested_pair: bool = False,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> LossDifferentialResult:
    """Compare two aligned forecast-error sequences with overlap-aware uncertainty.

    ``errors_a`` and ``errors_b`` must be the same length and already aligned
    origin-by-origin in ascending time order. Non-finite entries in either
    sequence mark that origin missing for both, held in place so the lag
    structure stays aligned.

    ``origin_positions`` gives each origin's whole-month offset on the calendar.
    Supplying it lets the engine place observations on a real monthly grid, so
    both the lag structure and the non-overlapping diagnostics respect calendar
    gaps. Without it the non-overlapping diagnostics fail closed, because a
    positional stride would silently assume a gap-free panel.

    ``equivalence_band`` overrides the relative band with an absolute one, which
    is what makes the verdict exactly antisymmetric under swapping A and B. With
    the default relative band the estimate and test remain antisymmetric while
    the band itself is anchored on whichever model is passed as B.
    """

    left = _finite_array(errors_a)
    right = _finite_array(errors_b)
    if left.size != right.size:
        raise ValueError(
            f"Forecast error sequences must be aligned and equal length: "
            f"{left.size} != {right.size}"
        )
    if loss_scale not in LOSS_SCALES:
        raise ValueError(f"Unknown loss scale: {loss_scale!r}. Expected one of {list(LOSS_SCALES)}")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1): {confidence_level!r}")
    if float(equivalence_band_fraction) < 0.0:
        raise ValueError(
            f"equivalence_band_fraction must be non-negative: {equivalence_band_fraction!r}"
        )
    if equivalence_band is not None and float(equivalence_band) < 0.0:
        raise ValueError(f"equivalence_band must be non-negative: {equivalence_band!r}")

    months = int(horizon)
    if months <= 0:
        raise ValueError(f"Horizon must be a positive month count: {horizon!r}")

    positions = _resolve_positions(origin_positions, left.size)
    paired = np.isfinite(left) & np.isfinite(right)
    observations = int(paired.sum())

    loss_a = np.where(paired, _loss(left, loss_scale), np.nan)
    loss_b = np.where(paired, _loss(right, loss_scale), np.nan)
    compact = np.where(paired, loss_a - loss_b, np.nan)

    # Place the differential on the real monthly grid when the calendar is known,
    # so lag j means j months rather than j rows.
    if positions is None:
        differential = compact
        grid_positions: np.ndarray | None = None
    else:
        span = int(positions[-1]) + 1 if positions.size else 0
        differential = np.full(span, np.nan, dtype=float)
        differential[positions] = compact
        grid_positions = np.arange(span, dtype=int)

    span_observations = int(differential.size)
    mean_a = float(np.nanmean(loss_a)) if observations else float("nan")
    mean_b = float(np.nanmean(loss_b)) if observations else float("nan")
    mean_d = float(np.nanmean(differential)) if observations else float("nan")

    band = (
        float(equivalence_band)
        if equivalence_band is not None
        else float(equivalence_band_fraction) * abs(mean_b)
    )
    if not math.isfinite(band):
        band = float("nan")

    lag = hac_bandwidth(observations, months)
    alpha_one_sided = (1.0 - float(confidence_level)) / 2.0
    overlap = _non_overlap_summary(differential, grid_positions, months)

    def _result(**overrides: object) -> LossDifferentialResult:
        payload: dict[str, object] = {
            "label_a": str(label_a),
            "label_b": str(label_b),
            "loss_scale": loss_scale,
            "horizon_months": months,
            "observations": observations,
            "span_observations": span_observations,
            "missing_observations": span_observations - observations,
            "overlapping_outcomes": months > 1,
            "mean_loss_a": mean_a,
            "mean_loss_b": mean_b,
            "loss_differential": mean_d,
            "hac_lag": lag,
            "long_run_variance": float("nan"),
            "sample_variance": float("nan"),
            "variance_inflation_factor": float("nan"),
            "n_effective": float("nan"),
            "hac_standard_error": float("nan"),
            "loss_differential_se": float("nan"),
            "hln_factor": harvey_leybourne_newbold_factor(observations, months),
            "t_stat": float("nan"),
            "degrees_of_freedom": 0,
            "confidence_level": float(confidence_level),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_value": float("nan"),
            "equivalence_band": band,
            "equivalence_band_fraction": float(equivalence_band_fraction),
            "directional_alpha": alpha_one_sided,
            "p_practical_lower": float("nan"),
            "p_practical_higher": float("nan"),
            "p_practical_directional": float("nan"),
            "p_equivalence": float("nan"),
            "p_directional_holm": float("nan"),
            "family_size": 0,
            "verdict": VERDICT_UNAVAILABLE,
            "detectable_but_negligible": False,
            "bootstrap_seed": int(bootstrap_seed),
            "origin_positions_supplied": positions is not None,
            "uncertainty_status": UNCERTAINTY_UNAVAILABLE,
            "uncertainty_reason": None,
            "nested_pair": bool(nested_pair),
            "nested_pair_disclosure": NESTED_PAIR_DISCLOSURE if nested_pair else None,
        }
        payload.update(_empty_bootstrap())
        payload.update(overlap)
        payload.update(overrides)
        if "bootstrap_status" not in overrides and max(int(bootstrap_reps), 0) > 0:
            # A bootstrap was asked for but the comparison short-circuited before
            # reaching it. Say that, rather than reporting it as never requested.
            payload["bootstrap_reps"] = max(int(bootstrap_reps), 0)
            payload["bootstrap_status"] = UNCERTAINTY_UNAVAILABLE
            payload["bootstrap_reason"] = payload["uncertainty_reason"]
        return LossDifferentialResult(**payload)  # type: ignore[arg-type]

    if observations == 0:
        return _result(uncertainty_reason=REASON_NO_OBSERVATIONS)

    if observations < required_observations(lag):
        return _result(uncertainty_reason=REASON_INSUFFICIENT_SAMPLE)

    observed_d = differential[np.isfinite(differential)]
    if np.all(observed_d == 0.0):
        # Identical losses on this panel. A zero-variance sample cannot certify
        # a zero population differential, so this is degenerate, not equivalent.
        return _result(
            uncertainty_reason=REASON_IDENTICAL_LOSSES,
            verdict=VERDICT_DEGENERATE,
            long_run_variance=0.0,
            sample_variance=0.0,
        )

    sample_variance = _sample_variance(differential)
    if not math.isfinite(sample_variance) or sample_variance <= 0.0:
        return _result(
            uncertainty_reason=REASON_DEGENERATE_ZERO_VARIANCE,
            verdict=VERDICT_DEGENERATE,
            sample_variance=sample_variance,
        )

    long_run_variance = bartlett_long_run_variance(differential, lag)
    if not math.isfinite(long_run_variance) or long_run_variance <= 0.0:
        return _result(
            uncertainty_reason=REASON_NONPOSITIVE_HAC_VARIANCE,
            verdict=VERDICT_DEGENERATE,
            long_run_variance=long_run_variance,
            sample_variance=sample_variance,
        )

    hac_se = math.sqrt(long_run_variance / observations)
    # Floored at 1 so a negatively autocorrelated differential cannot advertise
    # more independent information than it actually has origins.
    vif = max(long_run_variance / sample_variance, 1.0)
    n_eff = observations / vif
    dependence: dict[str, object] = {
        "long_run_variance": long_run_variance,
        "sample_variance": sample_variance,
        "variance_inflation_factor": vif,
        "n_effective": n_eff,
        "hac_standard_error": hac_se,
    }

    if nested_pair:
        # The ordinary DM null is degenerate here, so no interval and no verdict.
        # The point estimate and the dependence diagnostics remain descriptive.
        return _result(uncertainty_reason=REASON_NESTED_PAIR,
                       verdict=VERDICT_NESTED_NOT_TESTED,
                       **dependence)

    hln = harvey_leybourne_newbold_factor(observations, months)
    if not math.isfinite(hln) or hln <= 0.0:
        return _result(
            uncertainty_reason=REASON_INSUFFICIENT_SAMPLE,
            verdict=VERDICT_UNAVAILABLE,
            **dependence,
        )

    # HLN as a variance correction: one standard error drives the statistic, all
    # four p-values, and the interval, so they cannot disagree.
    standard_error = hac_se / hln
    degrees_of_freedom = observations - 1
    t_stat = mean_d / standard_error
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=degrees_of_freedom))
    critical = float(stats.t.ppf(1.0 - alpha_one_sided, df=degrees_of_freedom))
    ci_low = mean_d - critical * standard_error
    ci_high = mean_d + critical * standard_error

    # Practical-boundary hypotheses, each evaluated at alpha/2 so that rejecting
    # is exactly the statement that the two-sided interval clears the band.
    lower_stat = (mean_d + band) / standard_error
    upper_stat = (mean_d - band) / standard_error
    p_practical_lower = float(stats.t.cdf(lower_stat, df=degrees_of_freedom))
    p_practical_higher = float(stats.t.sf(upper_stat, df=degrees_of_freedom))
    p_equivalence = float(
        max(
            stats.t.sf(lower_stat, df=degrees_of_freedom),
            stats.t.cdf(upper_stat, df=degrees_of_freedom),
        )
    )
    p_directional = min(p_practical_lower, p_practical_higher)

    if p_practical_lower <= alpha_one_sided:
        verdict = VERDICT_LOWER
    elif p_practical_higher <= alpha_one_sided:
        verdict = VERDICT_HIGHER
    elif p_equivalence <= alpha_one_sided:
        verdict = VERDICT_PRACTICALLY_EQUIVALENT
    else:
        verdict = VERDICT_INCONCLUSIVE

    detectable = verdict == VERDICT_PRACTICALLY_EQUIVALENT and (ci_low > 0.0 or ci_high < 0.0)

    gap_share = (span_observations - observations) / span_observations if span_observations else 0.0
    degraded = gap_share > MAX_GAP_SHARE
    return _result(
        uncertainty_status=UNCERTAINTY_DEGRADED if degraded else UNCERTAINTY_AVAILABLE,
        uncertainty_reason=REASON_DEGRADED_PANEL_GAPS if degraded else None,
        loss_differential_se=standard_error,
        hln_factor=hln,
        t_stat=t_stat,
        degrees_of_freedom=degrees_of_freedom,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        p_practical_lower=p_practical_lower,
        p_practical_higher=p_practical_higher,
        p_practical_directional=p_directional,
        p_equivalence=p_equivalence,
        verdict=verdict,
        detectable_but_negligible=detectable,
        **dependence,
        **_run_bootstrap(
            differential,
            observations=observations,
            horizon=months,
            reps=bootstrap_reps,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
    )

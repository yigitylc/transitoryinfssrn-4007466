"""H10 step 1: the pure loss-differential inference engine.

Every test here is network-free and deterministic. Where a test needs a random
series it uses ``np.random.default_rng`` with a fixed seed, whose stream numpy
guarantees to be stable.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from transitory_inflation.inference import (
    BOOTSTRAP_NOT_REQUESTED,
    DEFAULT_BOOTSTRAP_SEED,
    LOSS_ABSOLUTE_ERROR,
    LOSS_SQUARED_ERROR,
    MIN_OBSERVATIONS,
    NESTED_PAIR_DISCLOSURE,
    REASON_DEGENERATE_ZERO_VARIANCE,
    REASON_DEGRADED_PANEL_GAPS,
    REASON_IDENTICAL_LOSSES,
    REASON_INSUFFICIENT_BOOTSTRAP_BLOCKS,
    REASON_INSUFFICIENT_BOOTSTRAP_REPS,
    REASON_INSUFFICIENT_SAMPLE,
    REASON_NESTED_PAIR,
    REASON_NO_OBSERVATIONS,
    UNCERTAINTY_AVAILABLE,
    UNCERTAINTY_DEGRADED,
    UNCERTAINTY_UNAVAILABLE,
    VERDICT_DEGENERATE,
    VERDICT_HIGHER,
    VERDICT_INCONCLUSIVE,
    VERDICT_LOWER,
    VERDICT_NESTED_NOT_TESTED,
    VERDICT_PRACTICALLY_EQUIVALENT,
    VERDICT_UNAVAILABLE,
    LossDifferentialResult,
    apply_holm_family,
    bartlett_long_run_variance,
    compare_forecast_losses,
    hac_bandwidth,
    harvey_leybourne_newbold_factor,
    holm_adjusted_p_values,
    moving_block_bootstrap_mean,
    non_overlapping_count,
    non_overlapping_count_from_positions,
    non_overlapping_phase_means,
    required_observations,
)

# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


def _controlled_errors(
    n: int,
    *,
    delta_mean: float,
    delta_sd: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (errors_a, errors_b) whose absolute-loss differential is known.

    Both error series stay comfortably positive, so ``|e| == e`` and the
    absolute-loss differential equals the injected delta exactly. That gives
    direct control of the differential's mean and spread.
    """

    rng = np.random.default_rng(seed)
    errors_b = 10.0 + rng.standard_normal(n) * 0.1
    delta = delta_mean + rng.standard_normal(n) * delta_sd
    return errors_b + delta, errors_b


def _overlapping_errors(n: int, horizon: int, *, seed: int, scale: float = 1.0) -> np.ndarray:
    """Return h-step forecast errors with the textbook MA(h-1) overlap structure."""

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n + horizon)
    accumulated = np.array([shocks[i : i + horizon].sum() for i in range(n)])
    return accumulated * scale


def _identical(left: LossDifferentialResult, right: LossDifferentialResult) -> bool:
    """Compare two results treating NaN as equal to NaN.

    Dataclass ``==`` cannot do this: a result carrying a NaN field never equals
    a separately constructed copy of itself, because ``nan != nan``.
    """

    first, second = left.as_dict(), right.as_dict()
    if set(first) != set(second):
        return False
    for key, value in first.items():
        other = second[key]
        both_nan = (
            isinstance(value, float)
            and isinstance(other, float)
            and math.isnan(value)
            and math.isnan(other)
        )
        if both_nan:
            continue
        if value != other:
            return False
    return True


# --------------------------------------------------------------------------
# 1. Bartlett long-run variance against hand-computed fixtures
# --------------------------------------------------------------------------


def test_bartlett_variance_matches_a_hand_computed_fixture() -> None:
    # d = [1, 2, 3, 4]; mean 2.5; u = [-1.5, -0.5, 0.5, 1.5]
    # gamma_0 = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
    # gamma_1 = (0.75 - 0.25 + 0.75) / 4 = 0.3125
    # weight   = 1 - 1/2 = 0.5  ->  1.25 + 2 * 0.5 * 0.3125 = 1.5625
    values = [1.0, 2.0, 3.0, 4.0]

    assert bartlett_long_run_variance(values, lags=0) == pytest.approx(1.25)
    assert bartlett_long_run_variance(values, lags=1) == pytest.approx(1.5625)


def test_bartlett_variance_treats_missing_points_in_place() -> None:
    # d = [1, nan, 3, 4]; observed mean 8/3; u = [-5/3, 0, 1/3, 4/3]
    # gamma_0 = (25/9 + 1/9 + 16/9) / 3 = 14/9
    # gamma_1 = ((1/3)*0 + (4/3)*(1/3)) / 3 = 4/27
    # 14/9 + 2 * 0.5 * 4/27 = 46/27
    gapped = bartlett_long_run_variance([1.0, float("nan"), 3.0, 4.0], lags=1)
    assert gapped == pytest.approx(46.0 / 27.0)

    # Dropping the gap and closing up would make lag 1 pair 1 with 3, which is a
    # different — and wrong — quantity. The two must not coincide.
    compressed = bartlett_long_run_variance([1.0, 3.0, 4.0], lags=1)
    assert compressed != pytest.approx(gapped)


def test_bartlett_variance_is_never_negative_under_strong_alternation() -> None:
    alternating = np.array([1.0, -1.0] * 40)
    for lags in range(0, 12):
        assert bartlett_long_run_variance(alternating, lags=lags) >= 0.0


def test_bartlett_variance_reduces_to_sample_variance_for_iid_input() -> None:
    values = np.random.default_rng(11).standard_normal(4000)
    long_run = bartlett_long_run_variance(values, lags=4)
    assert long_run == pytest.approx(float(values.var(ddof=1)), rel=0.10)


# --------------------------------------------------------------------------
# 2. horizon-specific bandwidth and the fail-closed gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observations", "horizon", "expected"),
    [
        (447, 3, 5),  # automatic rule binds at the short horizon
        (444, 6, 5),  # tie: h - 1 == automatic
        (438, 12, 11),  # horizon binds
        (426, 24, 23),
        (414, 36, 35),
        (100, 1, 4),  # (100/100) ** (2/9) == 1 -> exactly 4
        (50, 1, 3),
        (0, 12, 1),  # degenerate input still yields a usable positive lag
    ],
)
def test_hac_bandwidth_is_horizon_specific(observations: int, horizon: int, expected: int) -> None:
    assert hac_bandwidth(observations, horizon) == expected


def test_hac_bandwidth_never_falls_below_the_ma_floor() -> None:
    for horizon in (1, 3, 6, 12, 24, 36):
        for observations in (40, 120, 400, 900):
            assert hac_bandwidth(observations, horizon) >= max(horizon - 1, 1)


def test_required_observations_reduces_to_the_thirty_boundary_at_short_lags() -> None:
    assert required_observations(2) == MIN_OBSERVATIONS
    assert required_observations(5) == MIN_OBSERVATIONS
    assert required_observations(11) == 60
    assert required_observations(23) == 120
    assert required_observations(35) == 180


# --------------------------------------------------------------------------
# 3. overlap actually inflates the reported uncertainty
# --------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [6, 12, 24])
def test_overlapping_outcomes_inflate_the_standard_error(horizon: int) -> None:
    n = 400
    errors_a = _overlapping_errors(n, horizon, seed=101)
    errors_b = _overlapping_errors(n, horizon, seed=202) * 1.2

    result = compare_forecast_losses(errors_a, errors_b, horizon=horizon, bootstrap_reps=0)

    naive_se = math.sqrt(result.sample_variance / result.observations)
    assert result.uncertainty_status == UNCERTAINTY_AVAILABLE
    assert result.loss_differential_se > naive_se
    assert result.variance_inflation_factor > 1.5
    assert result.n_effective < result.observations
    assert result.overlapping_outcomes is True


def test_independent_differentials_leave_the_effective_sample_intact() -> None:
    errors_a, errors_b = _controlled_errors(600, delta_mean=0.0, delta_sd=0.5, seed=7)

    result = compare_forecast_losses(errors_a, errors_b, horizon=1, bootstrap_reps=0)

    assert result.variance_inflation_factor == pytest.approx(1.0, abs=0.35)
    assert result.n_effective == pytest.approx(result.observations, rel=0.30)
    assert result.overlapping_outcomes is False


def test_effective_sample_never_exceeds_the_observation_count() -> None:
    # Strong negative autocorrelation drives the long-run variance below the
    # sample variance; the floor must stop that from advertising extra evidence.
    n = 240
    alternating = np.array([1.0, -1.0] * (n // 2))
    errors_b = np.full(n, 10.0)
    errors_a = errors_b + alternating * 0.5

    result = compare_forecast_losses(errors_a, errors_b, horizon=1, bootstrap_reps=0)

    assert result.variance_inflation_factor == 1.0
    assert result.n_effective <= result.observations


def test_effective_sample_is_diagnostic_and_does_not_move_the_degrees_of_freedom() -> None:
    errors_a = _overlapping_errors(400, 24, seed=5)
    errors_b = _overlapping_errors(400, 24, seed=6) * 1.3

    result = compare_forecast_losses(errors_a, errors_b, horizon=24, bootstrap_reps=0)

    assert result.n_effective < result.observations / 2
    assert result.degrees_of_freedom == result.observations - 1


# --------------------------------------------------------------------------
# 4. HLN correction
# --------------------------------------------------------------------------


def test_hln_factor_matches_the_closed_form_and_shrinks_with_the_horizon() -> None:
    assert harvey_leybourne_newbold_factor(414, 36) == pytest.approx(0.9142, abs=5e-4)
    assert harvey_leybourne_newbold_factor(447, 3) == pytest.approx(0.9944, abs=5e-4)
    assert harvey_leybourne_newbold_factor(400, 1) == pytest.approx(math.sqrt(399 / 400))

    factors = [harvey_leybourne_newbold_factor(420, h) for h in (1, 3, 6, 12, 24, 36)]
    assert factors == sorted(factors, reverse=True)


def test_hln_correction_pulls_the_statistic_toward_zero() -> None:
    errors_a = _overlapping_errors(400, 36, seed=31)
    errors_b = _overlapping_errors(400, 36, seed=32) * 1.4

    result = compare_forecast_losses(errors_a, errors_b, horizon=36, bootstrap_reps=0)

    uncorrected = result.loss_differential / result.hac_standard_error
    assert abs(result.t_stat) < abs(uncorrected)
    assert result.t_stat == pytest.approx(uncorrected * result.hln_factor)


# --------------------------------------------------------------------------
# 5. sign convention and symmetry
# --------------------------------------------------------------------------


def test_negative_differential_means_the_first_model_had_the_lower_loss() -> None:
    errors_a, errors_b = _controlled_errors(300, delta_mean=-1.0, delta_sd=0.5, seed=13)

    result = compare_forecast_losses(
        errors_a,
        errors_b,
        horizon=1,
        equivalence_band=0.1,
        bootstrap_reps=0,
    )

    assert result.mean_loss_a < result.mean_loss_b
    assert result.loss_differential < 0
    assert result.loss_differential == pytest.approx(result.mean_loss_a - result.mean_loss_b)
    assert result.verdict == VERDICT_LOWER


def test_swapping_the_models_is_exactly_antisymmetric() -> None:
    errors_a, errors_b = _controlled_errors(300, delta_mean=-0.8, delta_sd=0.6, seed=17)
    kwargs = {"horizon": 6, "equivalence_band": 0.1, "bootstrap_reps": 0}

    forward = compare_forecast_losses(errors_a, errors_b, label_a="a", label_b="b", **kwargs)
    reverse = compare_forecast_losses(errors_b, errors_a, label_a="b", label_b="a", **kwargs)

    # The estimate flips sign; the interval flips and reverses.
    assert reverse.loss_differential == pytest.approx(-forward.loss_differential)
    assert reverse.ci_low == pytest.approx(-forward.ci_high)
    assert reverse.ci_high == pytest.approx(-forward.ci_low)
    assert reverse.t_stat == pytest.approx(-forward.t_stat)

    # Everything scale-free about the dependence is identical.
    assert reverse.loss_differential_se == pytest.approx(forward.loss_differential_se)
    assert reverse.p_value == pytest.approx(forward.p_value)
    assert reverse.variance_inflation_factor == pytest.approx(forward.variance_inflation_factor)
    assert reverse.n_effective == pytest.approx(forward.n_effective)
    assert reverse.hac_lag == forward.hac_lag
    assert reverse.degrees_of_freedom == forward.degrees_of_freedom

    # And the verdict mirrors, because the band was pinned rather than derived.
    assert forward.verdict == VERDICT_LOWER
    assert reverse.verdict == VERDICT_HIGHER


def test_relative_band_is_anchored_on_the_comparison_model() -> None:
    errors_a, errors_b = _controlled_errors(300, delta_mean=-1.0, delta_sd=0.5, seed=19)

    result = compare_forecast_losses(
        errors_a,
        errors_b,
        horizon=1,
        equivalence_band_fraction=0.05,
        bootstrap_reps=0,
    )

    assert result.equivalence_band == pytest.approx(0.05 * abs(result.mean_loss_b))
    assert result.equivalence_band_fraction == 0.05


def test_squared_error_scale_scores_a_different_loss_than_absolute_error() -> None:
    errors_a = _overlapping_errors(400, 12, seed=41)
    errors_b = _overlapping_errors(400, 12, seed=42) * 1.5

    absolute = compare_forecast_losses(
        errors_a, errors_b, horizon=12, loss_scale=LOSS_ABSOLUTE_ERROR, bootstrap_reps=0
    )
    squared = compare_forecast_losses(
        errors_a, errors_b, horizon=12, loss_scale=LOSS_SQUARED_ERROR, bootstrap_reps=0
    )

    assert absolute.mean_loss_a == pytest.approx(float(np.abs(errors_a).mean()))
    assert squared.mean_loss_a == pytest.approx(float(np.square(errors_a).mean()))
    assert absolute.loss_scale == LOSS_ABSOLUTE_ERROR
    assert squared.loss_scale == LOSS_SQUARED_ERROR
    # Both agree that A has the lower loss, on their own scales.
    assert absolute.loss_differential < 0
    assert squared.loss_differential < 0


# --------------------------------------------------------------------------
# 6. verdict grid and the practical band
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta_mean", "band", "expected"),
    [
        (-1.0, 0.10, VERDICT_LOWER),
        (+1.0, 0.10, VERDICT_HIGHER),
        (0.0, 0.50, VERDICT_PRACTICALLY_EQUIVALENT),
        (0.0, 0.01, VERDICT_INCONCLUSIVE),
    ],
)
def test_verdict_grid(delta_mean: float, band: float, expected: str) -> None:
    errors_a, errors_b = _controlled_errors(300, delta_mean=delta_mean, delta_sd=0.5, seed=23)

    result = compare_forecast_losses(
        errors_a,
        errors_b,
        horizon=1,
        equivalence_band=band,
        bootstrap_reps=0,
    )

    assert result.verdict == expected


@pytest.mark.parametrize("delta_mean", [0.10, -0.10])
def test_a_detectable_but_negligible_gap_is_never_called_lower_or_higher(
    delta_mean: float,
) -> None:
    """Both signs, so neither verdict branch can quietly drop the band."""

    errors_a, errors_b = _controlled_errors(400, delta_mean=delta_mean, delta_sd=0.20, seed=29)

    result = compare_forecast_losses(
        errors_a,
        errors_b,
        horizon=1,
        equivalence_band=0.50,
        bootstrap_reps=0,
    )

    assert result.p_value < 0.01  # statistically detectable
    assert result.ci_low > 0.0 or result.ci_high < 0.0  # interval excludes zero
    assert result.verdict == VERDICT_PRACTICALLY_EQUIVALENT
    assert result.verdict not in (VERDICT_LOWER, VERDICT_HIGHER)
    assert result.detectable_but_negligible is True


def test_widening_the_band_can_only_soften_a_verdict() -> None:
    errors_a, errors_b = _controlled_errors(300, delta_mean=-0.30, delta_sd=0.5, seed=37)

    tight = compare_forecast_losses(
        errors_a, errors_b, horizon=1, equivalence_band=0.01, bootstrap_reps=0
    )
    wide = compare_forecast_losses(
        errors_a, errors_b, horizon=1, equivalence_band=5.0, bootstrap_reps=0
    )

    assert tight.verdict == VERDICT_LOWER
    assert wide.verdict == VERDICT_PRACTICALLY_EQUIVALENT


def test_confidence_level_is_configurable_and_widens_the_interval() -> None:
    errors_a, errors_b = _controlled_errors(300, delta_mean=-0.20, delta_sd=0.5, seed=43)
    kwargs = {"horizon": 1, "equivalence_band": 0.01, "bootstrap_reps": 0}

    default = compare_forecast_losses(errors_a, errors_b, **kwargs)
    strict = compare_forecast_losses(errors_a, errors_b, confidence_level=0.99, **kwargs)

    assert default.confidence_level == 0.95
    assert strict.confidence_level == 0.99
    assert strict.ci_high - strict.ci_low > default.ci_high - default.ci_low
    assert strict.p_value == pytest.approx(default.p_value)  # p does not depend on the level


# --------------------------------------------------------------------------
# 7. degenerate and short-sample inputs
# --------------------------------------------------------------------------


def test_opposite_signed_errors_of_equal_size_are_identical_losses_not_identical_forecasts() -> None:
    errors_a = np.full(200, 2.0)
    errors_b = np.full(200, -2.0)

    result = compare_forecast_losses(errors_a, errors_b, horizon=1, bootstrap_reps=0)

    assert not np.allclose(errors_a, errors_b)
    assert result.uncertainty_reason == REASON_IDENTICAL_LOSSES
    assert result.loss_differential == 0.0


def test_constant_non_zero_differential_is_refused_rather_than_given_an_interval() -> None:
    errors_b = np.full(200, 10.0)
    errors_a = np.full(200, 11.0)

    result = compare_forecast_losses(errors_a, errors_b, horizon=1, bootstrap_reps=0)

    assert result.loss_differential == pytest.approx(1.0)  # point estimate survives
    assert result.uncertainty_status == UNCERTAINTY_UNAVAILABLE
    assert result.uncertainty_reason == REASON_DEGENERATE_ZERO_VARIANCE
    assert result.verdict == VERDICT_DEGENERATE
    assert math.isnan(result.loss_differential_se)
    assert math.isnan(result.ci_low) and math.isnan(result.ci_high)


def test_empty_input_returns_an_unavailable_result_rather_than_raising() -> None:
    result = compare_forecast_losses([], [], horizon=12, bootstrap_reps=0)

    assert result.observations == 0
    assert result.uncertainty_status == UNCERTAINTY_UNAVAILABLE
    assert result.uncertainty_reason == REASON_NO_OBSERVATIONS
    assert result.verdict == VERDICT_UNAVAILABLE


@pytest.mark.parametrize(
    ("n", "horizon"),
    [
        (29, 1),  # one short of the 30 floor
        (100, 24),  # 100 < 5 * (23 + 1)
        (170, 36),  # 170 < 5 * (35 + 1)
    ],
)
def test_short_samples_fail_closed_with_the_point_estimate_intact(n: int, horizon: int) -> None:
    errors_a, errors_b = _controlled_errors(n, delta_mean=-0.5, delta_sd=0.4, seed=59)

    result = compare_forecast_losses(errors_a, errors_b, horizon=horizon, bootstrap_reps=0)

    assert result.uncertainty_status == UNCERTAINTY_UNAVAILABLE
    assert result.uncertainty_reason == REASON_INSUFFICIENT_SAMPLE
    assert result.verdict == VERDICT_UNAVAILABLE
    assert math.isfinite(result.loss_differential)
    assert math.isnan(result.t_stat)
    assert math.isnan(result.p_value)


def test_the_gate_admits_a_sample_one_observation_above_the_requirement() -> None:
    lag = hac_bandwidth(180, 36)
    assert required_observations(lag) == 180

    errors_a, errors_b = _controlled_errors(180, delta_mean=-0.5, delta_sd=0.4, seed=61)
    result = compare_forecast_losses(errors_a, errors_b, horizon=36, bootstrap_reps=0)

    assert result.observations == 180
    assert result.uncertainty_status == UNCERTAINTY_AVAILABLE


# --------------------------------------------------------------------------
# 8. missing values
# --------------------------------------------------------------------------


def test_missing_observations_are_excluded_pairwise_and_counted() -> None:
    errors_a, errors_b = _controlled_errors(200, delta_mean=-0.5, delta_sd=0.4, seed=67)
    errors_a = errors_a.copy()
    errors_b = errors_b.copy()
    errors_a[5] = np.nan
    errors_b[9] = np.nan
    errors_b[9 + 1] = np.nan

    result = compare_forecast_losses(errors_a, errors_b, horizon=1, bootstrap_reps=0)

    assert result.span_observations == 200
    assert result.observations == 197
    assert result.missing_observations == 3
    assert math.isfinite(result.loss_differential_se)


def test_a_missing_value_in_either_series_removes_that_origin_from_both() -> None:
    errors_a, errors_b = _controlled_errors(120, delta_mean=-0.5, delta_sd=0.4, seed=71)
    holed_a = errors_a.copy()
    holed_a[3] = np.nan

    result = compare_forecast_losses(holed_a, errors_b, horizon=1, bootstrap_reps=0)

    kept = np.ones(120, dtype=bool)
    kept[3] = False
    expected = float(
        np.abs(errors_a[kept]).mean() - np.abs(errors_b[kept]).mean()
    )
    assert result.loss_differential == pytest.approx(expected)


def test_a_gap_heavy_panel_is_downgraded_rather_than_silently_reported() -> None:
    errors_a, errors_b = _controlled_errors(300, delta_mean=-0.5, delta_sd=0.4, seed=73)
    holed_a = errors_a.copy()
    holed_a[::5] = np.nan  # 20% missing, above the 10% tolerance

    result = compare_forecast_losses(holed_a, errors_b, horizon=1, bootstrap_reps=0)

    assert result.missing_observations == 60
    assert result.uncertainty_status == UNCERTAINTY_DEGRADED
    assert result.uncertainty_reason == REASON_DEGRADED_PANEL_GAPS
    assert result.verdict is not None  # still reported, just flagged


def test_a_light_gap_stays_available() -> None:
    errors_a, errors_b = _controlled_errors(300, delta_mean=-0.5, delta_sd=0.4, seed=79)
    holed_a = errors_a.copy()
    holed_a[::50] = np.nan  # 2% missing

    result = compare_forecast_losses(holed_a, errors_b, horizon=1, bootstrap_reps=0)

    assert result.uncertainty_status == UNCERTAINTY_AVAILABLE
    assert result.uncertainty_reason is None


# --------------------------------------------------------------------------
# 9. non-overlapping diagnostics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observations", "horizon", "expected"),
    [(6, 3, 2), (7, 3, 3), (9, 3, 3), (447, 3, 149), (414, 36, 12), (100, 1, 100), (0, 12, 0)],
)
def test_non_overlapping_count_is_the_ceiling_of_n_over_h(
    observations: int, horizon: int, expected: int
) -> None:
    assert non_overlapping_count(observations, horizon) == expected


def test_non_overlapping_phases_partition_the_sample() -> None:
    values = np.arange(12, dtype=float)

    phases = non_overlapping_phase_means(values, list(range(12)), horizon=3)

    assert len(phases) == 3
    assert phases[0] == pytest.approx(np.mean([0, 3, 6, 9]))
    assert phases[1] == pytest.approx(np.mean([1, 4, 7, 10]))
    assert phases[2] == pytest.approx(np.mean([2, 5, 8, 11]))


def test_phases_follow_the_calendar_not_the_row_order() -> None:
    values = np.array([10.0, 20.0, 30.0, 40.0])
    # Rows 2 and 3 sit at calendar months 6 and 7, so at h=3 they fall in
    # phases 0 and 1 — not the phases 2 and 0 a row stride would assign.
    phases = non_overlapping_phase_means(values, [0, 1, 6, 7], horizon=3)

    assert phases[0] == pytest.approx(np.mean([10.0, 30.0]))
    assert phases[1] == pytest.approx(np.mean([20.0, 40.0]))
    assert len(phases) == 2  # phase 2 has no origin at all


def test_a_single_month_horizon_has_one_phase_equal_to_the_whole_sample() -> None:
    values = np.arange(10, dtype=float)

    assert non_overlapping_phase_means(values, list(range(10)), horizon=1) == [pytest.approx(4.5)]


def test_phase_summary_is_attached_to_the_result() -> None:
    errors_a = _overlapping_errors(400, 12, seed=83)
    errors_b = _overlapping_errors(400, 12, seed=89) * 1.3

    result = compare_forecast_losses(
        errors_a,
        errors_b,
        horizon=12,
        origin_positions=list(range(400)),
        bootstrap_reps=0,
    )

    assert result.non_overlapping_status == UNCERTAINTY_AVAILABLE
    assert result.non_overlapping_phases == 12
    assert result.non_overlapping_count == math.ceil(result.observations / 12)
    assert result.non_overlapping_phase_min <= result.non_overlapping_phase_mean
    assert result.non_overlapping_phase_mean <= result.non_overlapping_phase_max


def test_the_contiguous_closed_form_agrees_with_the_calendar_scan() -> None:
    for n in (6, 7, 9, 60, 100, 447):
        for horizon in (1, 3, 12, 36):
            assert non_overlapping_count(n, horizon) == non_overlapping_count_from_positions(
                list(range(n)), horizon
            )


# --------------------------------------------------------------------------
# 10. bootstrap companion
# --------------------------------------------------------------------------


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    errors_a = _overlapping_errors(300, 6, seed=97)
    errors_b = _overlapping_errors(300, 6, seed=101) * 1.25
    kwargs = {"horizon": 6, "bootstrap_reps": 500}

    first = compare_forecast_losses(errors_a, errors_b, **kwargs)
    second = compare_forecast_losses(errors_a, errors_b, **kwargs)

    assert first.bootstrap_se == second.bootstrap_se
    assert first.bootstrap_ci_low == second.bootstrap_ci_low
    assert first.bootstrap_ci_high == second.bootstrap_ci_high
    assert first.bootstrap_seed == DEFAULT_BOOTSTRAP_SEED


def test_a_different_bootstrap_seed_moves_the_replicates() -> None:
    errors_a = _overlapping_errors(300, 6, seed=103)
    errors_b = _overlapping_errors(300, 6, seed=107) * 1.25
    kwargs = {"horizon": 6, "bootstrap_reps": 500}

    default = compare_forecast_losses(errors_a, errors_b, **kwargs)
    alternate = compare_forecast_losses(errors_a, errors_b, bootstrap_seed=1234, **kwargs)

    assert alternate.bootstrap_se != default.bootstrap_se
    assert alternate.bootstrap_se == pytest.approx(default.bootstrap_se, rel=0.25)


def test_bootstrap_block_follows_the_horizon() -> None:
    errors_a = _overlapping_errors(300, 24, seed=109)
    errors_b = _overlapping_errors(300, 24, seed=113) * 1.25

    result = compare_forecast_losses(errors_a, errors_b, horizon=24, bootstrap_reps=200)

    assert result.bootstrap_block == 24
    assert result.bootstrap_reps == 200


def test_bootstrap_can_be_switched_off_without_disturbing_the_primary_test() -> None:
    errors_a = _overlapping_errors(300, 6, seed=127)
    errors_b = _overlapping_errors(300, 6, seed=131) * 1.25

    with_boot = compare_forecast_losses(errors_a, errors_b, horizon=6, bootstrap_reps=200)
    without = compare_forecast_losses(errors_a, errors_b, horizon=6, bootstrap_reps=0)

    assert without.bootstrap_reps == 0
    assert math.isnan(without.bootstrap_se)
    assert with_boot.loss_differential_se == pytest.approx(without.loss_differential_se)
    assert with_boot.verdict == without.verdict


def test_bootstrap_standard_error_broadly_corroborates_the_hac_one() -> None:
    errors_a = _overlapping_errors(400, 12, seed=137)
    errors_b = _overlapping_errors(400, 12, seed=139) * 1.3

    result = compare_forecast_losses(errors_a, errors_b, horizon=12, bootstrap_reps=2000)

    assert result.bootstrap_se == pytest.approx(result.hac_standard_error, rel=0.35)


def test_moving_block_bootstrap_clamps_an_oversized_block() -> None:
    values = np.arange(10, dtype=float)

    _, _, _, block = moving_block_bootstrap_mean(values, block=999, reps=50)

    assert block == 10


def test_moving_block_bootstrap_returns_nan_when_there_is_nothing_to_resample() -> None:
    se, low, high, _ = moving_block_bootstrap_mean([1.0], block=2, reps=100)

    assert math.isnan(se) and math.isnan(low) and math.isnan(high)


# --------------------------------------------------------------------------
# 11. Holm multiplicity control
# --------------------------------------------------------------------------


def test_holm_matches_a_hand_computed_family() -> None:
    # sorted: 0.01 * 4 = 0.04 | 0.02 * 3 = 0.06 | 0.03 * 2 = 0.06 | 0.04 * 1 = 0.04 -> 0.06
    adjusted = holm_adjusted_p_values([0.01, 0.02, 0.03, 0.04])

    assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_preserves_input_order_and_never_shrinks_a_p_value() -> None:
    raw = [0.04, 0.001, 0.20, 0.03]

    adjusted = holm_adjusted_p_values(raw)

    assert adjusted[1] == pytest.approx(0.004)
    for original, corrected in zip(raw, adjusted, strict=True):
        assert corrected >= original


def test_holm_is_monotone_in_the_sorted_order_and_capped_at_one() -> None:
    adjusted = holm_adjusted_p_values([0.3, 0.4, 0.5, 0.9])
    ordered = sorted(adjusted)

    assert ordered == adjusted
    assert max(adjusted) <= 1.0


def test_holm_on_a_single_comparison_is_a_no_op() -> None:
    assert holm_adjusted_p_values([0.037]) == pytest.approx([0.037])


def test_holm_excludes_gated_comparisons_from_the_family_size() -> None:
    with_gap = holm_adjusted_p_values([0.01, float("nan"), 0.02])
    without_gap = holm_adjusted_p_values([0.01, 0.02])

    assert math.isnan(with_gap[1])
    assert [with_gap[0], with_gap[2]] == pytest.approx(without_gap)


def test_holm_on_an_empty_family_returns_nothing_to_adjust() -> None:
    assert holm_adjusted_p_values([]) == []
    assert all(math.isnan(value) for value in holm_adjusted_p_values([float("nan")]))


def _family(deltas: tuple[float, ...], *, band: float = 0.01) -> list:
    """Build one family of comparisons sharing a reference model."""

    results = []
    for index, delta in enumerate(deltas):
        errors_a, errors_b = _controlled_errors(
            300, delta_mean=delta, delta_sd=0.5, seed=1009 + index
        )
        results.append(
            compare_forecast_losses(
                errors_a,
                errors_b,
                horizon=1,
                label_b=f"rival_{index}",
                equivalence_band=band,
                bootstrap_reps=0,
            )
        )
    return results


def test_a_bare_comparison_carries_no_family_adjustment() -> None:
    single = _family((-0.5,))[0]

    assert math.isnan(single.p_directional_holm)
    assert single.family_size == 0


def _pinned(
    p_directional: float,
    verdict: str,
    *,
    confidence_level: float = 0.95,
) -> LossDifferentialResult:
    """A real result with its practical p-value and verdict pinned exactly.

    ``apply_holm_family`` is a decision rule over the practical-boundary
    p-value, so pinning that input tests the rule itself instead of a
    knife-edge simulated effect size.
    """

    template = _family((-0.5,))[0]
    return replace(
        template,
        p_practical_directional=p_directional,
        verdict=verdict,
        confidence_level=confidence_level,
        directional_alpha=(1.0 - confidence_level) / 2.0,
    )


def test_family_adjustment_demotes_a_directional_verdict_that_multiplicity_kills() -> None:
    # 0.02 * 4 = 0.08, past the 0.05 budget for a four-comparison family.
    family = [
        _pinned(0.02, VERDICT_LOWER),
        _pinned(0.30, VERDICT_INCONCLUSIVE),
        _pinned(0.40, VERDICT_INCONCLUSIVE),
        _pinned(0.60, VERDICT_INCONCLUSIVE),
    ]

    adjusted = apply_holm_family(family)

    assert adjusted[0].p_directional_holm == pytest.approx(0.08)
    assert adjusted[0].verdict == VERDICT_INCONCLUSIVE
    # The estimate itself is untouched; only the claim is withdrawn.
    assert adjusted[0].loss_differential == pytest.approx(family[0].loss_differential)
    assert adjusted[0].ci_low == pytest.approx(family[0].ci_low)
    assert adjusted[0].p_practical_directional == pytest.approx(0.02)  # raw p stays auditable


def test_family_adjustment_keeps_a_verdict_that_survives_the_budget() -> None:
    family = [
        _pinned(0.001, VERDICT_LOWER),
        _pinned(0.30, VERDICT_INCONCLUSIVE),
        _pinned(0.40, VERDICT_INCONCLUSIVE),
        _pinned(0.60, VERDICT_INCONCLUSIVE),
    ]

    adjusted = apply_holm_family(family)

    assert adjusted[0].p_directional_holm == pytest.approx(0.004)
    assert adjusted[0].verdict == VERDICT_LOWER


def test_family_adjustment_never_demotes_a_practical_equivalence() -> None:
    # Same p that demotes a directional verdict above.
    family = [
        _pinned(0.02, VERDICT_PRACTICALLY_EQUIVALENT),
        _pinned(0.30, VERDICT_INCONCLUSIVE),
        _pinned(0.40, VERDICT_INCONCLUSIVE),
        _pinned(0.60, VERDICT_INCONCLUSIVE),
    ]

    adjusted = apply_holm_family(family)

    assert adjusted[0].p_directional_holm == pytest.approx(0.08)
    assert adjusted[0].verdict == VERDICT_PRACTICALLY_EQUIVALENT


def test_family_adjustment_respects_a_non_default_confidence_level() -> None:
    # p_holm == 0.02: inside a 5% budget, outside a 1% one.
    pvalues = (0.005, 0.30, 0.40, 0.60)

    lenient = apply_holm_family([_pinned(p, VERDICT_HIGHER) for p in pvalues])
    strict = apply_holm_family(
        [_pinned(p, VERDICT_HIGHER, confidence_level=0.99) for p in pvalues]
    )

    assert lenient[0].p_directional_holm == pytest.approx(0.02)
    assert strict[0].p_directional_holm == pytest.approx(0.02)
    # The alpha comes from each comparison's own level, not a hardcoded 0.05.
    assert lenient[0].verdict == VERDICT_HIGHER
    assert strict[0].verdict == VERDICT_INCONCLUSIVE


def test_family_adjustment_on_real_comparisons_records_the_family() -> None:
    adjusted = apply_holm_family(_family((-1.0, -0.9, -0.8, -0.7)))

    assert {result.family_size for result in adjusted} == {4}
    for result in adjusted:
        assert result.p_directional_holm >= result.p_practical_directional


def test_gated_comparisons_do_not_consume_the_family_budget() -> None:
    gated = compare_forecast_losses(
        *_controlled_errors(20, delta_mean=-1.0, delta_sd=0.5, seed=1501),
        horizon=1,
        bootstrap_reps=0,
    )
    assert gated.uncertainty_status == UNCERTAINTY_UNAVAILABLE

    with_gate = apply_holm_family([*_family((-1.0, -0.9)), gated])
    without_gate = apply_holm_family(_family((-1.0, -0.9)))

    assert with_gate[-1].family_size == 2
    assert math.isnan(with_gate[-1].p_directional_holm)
    assert with_gate[-1].verdict == VERDICT_UNAVAILABLE
    for adjusted, plain in zip(with_gate[:2], without_gate, strict=True):
        assert adjusted.p_directional_holm == pytest.approx(plain.p_directional_holm)


def test_family_adjustment_is_pure() -> None:
    original = _family((-1.0, -0.9, -0.8))
    snapshot = [replace(result) for result in original]

    adjusted = apply_holm_family(original)

    assert all(_identical(before, after) for before, after in zip(original, snapshot, strict=True))
    assert all(math.isnan(result.p_directional_holm) for result in original)
    assert all(math.isfinite(result.p_directional_holm) for result in adjusted)


# --------------------------------------------------------------------------
# 12. nesting disclosure, purity, and the returned schema
# --------------------------------------------------------------------------


def test_nested_pairs_carry_an_explicit_flag_and_disclosure() -> None:
    errors_a, errors_b = _controlled_errors(200, delta_mean=-0.4, delta_sd=0.5, seed=149)

    nested = compare_forecast_losses(
        errors_a, errors_b, horizon=6, nested_pair=True, bootstrap_reps=0
    )
    plain = compare_forecast_losses(errors_a, errors_b, horizon=6, bootstrap_reps=0)

    assert nested.nested_pair is True
    assert nested.nested_pair_disclosure == NESTED_PAIR_DISCLOSURE
    assert plain.nested_pair is False
    assert plain.nested_pair_disclosure is None

    # The descriptive layer is identical — nesting is a fact about the models,
    # not about the sample — but the inferential layer is withheld entirely.
    assert nested.loss_differential == pytest.approx(plain.loss_differential)
    assert nested.variance_inflation_factor == pytest.approx(plain.variance_inflation_factor)
    assert nested.hac_standard_error == pytest.approx(plain.hac_standard_error)
    assert math.isnan(nested.p_value) and math.isfinite(plain.p_value)
    assert nested.verdict == VERDICT_NESTED_NOT_TESTED
    assert plain.verdict in (VERDICT_LOWER, VERDICT_HIGHER, VERDICT_PRACTICALLY_EQUIVALENT,
                             VERDICT_INCONCLUSIVE)


def test_the_engine_is_pure_and_repeatable() -> None:
    errors_a = _overlapping_errors(300, 12, seed=151)
    errors_b = _overlapping_errors(300, 12, seed=157) * 1.2

    first = compare_forecast_losses(errors_a, errors_b, horizon=12, bootstrap_reps=200)
    second = compare_forecast_losses(errors_a, errors_b, horizon=12, bootstrap_reps=200)

    assert _identical(first, second)
    assert isinstance(first, LossDifferentialResult)


def test_result_exposes_a_flat_mapping_with_the_disclosure_fields() -> None:
    errors_a = _overlapping_errors(300, 12, seed=163)
    errors_b = _overlapping_errors(300, 12, seed=167) * 1.2

    payload = compare_forecast_losses(errors_a, errors_b, horizon=12, bootstrap_reps=0).as_dict()

    required = {
        "loss_differential",
        "loss_differential_se",
        "ci_low",
        "ci_high",
        "confidence_level",
        "hac_lag",
        "variance_inflation_factor",
        "n_effective",
        "t_stat",
        "hln_factor",
        "degrees_of_freedom",
        "p_value",
        "equivalence_band",
        "verdict",
        "detectable_but_negligible",
        "non_overlapping_count",
        "overlapping_outcomes",
        "uncertainty_status",
        "uncertainty_reason",
        "nested_pair",
    }
    assert required <= set(payload)
    assert payload["horizon_months"] == 12


def test_labels_are_carried_through_for_downstream_frame_building() -> None:
    errors_a, errors_b = _controlled_errors(200, delta_mean=-0.4, delta_sd=0.5, seed=173)

    result = compare_forecast_losses(
        errors_a,
        errors_b,
        horizon=6,
        label_a="tinf_regime_bucket",
        label_b="no_change",
        bootstrap_reps=0,
    )

    assert result.label_a == "tinf_regime_bucket"
    assert result.label_b == "no_change"


# --------------------------------------------------------------------------
# 13. input validation
# --------------------------------------------------------------------------


def test_misaligned_sequences_are_rejected() -> None:
    with pytest.raises(ValueError, match="aligned and equal length"):
        compare_forecast_losses([1.0, 2.0, 3.0], [1.0, 2.0], horizon=3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"loss_scale": "log_score"}, "Unknown loss scale"),
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"confidence_level": 0.0}, "confidence_level"),
        ({"equivalence_band_fraction": -0.1}, "equivalence_band_fraction"),
        ({"equivalence_band": -1.0}, "equivalence_band"),
    ],
)
def test_invalid_options_are_rejected(kwargs: dict[str, object], match: str) -> None:
    errors = np.arange(1.0, 41.0)

    with pytest.raises(ValueError, match=match):
        compare_forecast_losses(errors, errors + 1.0, horizon=3, **kwargs)


@pytest.mark.parametrize("horizon", [0, -1])
def test_non_positive_horizons_are_rejected(horizon: int) -> None:
    errors = np.arange(1.0, 41.0)

    with pytest.raises(ValueError, match="positive month count"):
        compare_forecast_losses(errors, errors + 1.0, horizon=horizon)


def test_two_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        compare_forecast_losses(np.ones((4, 2)), np.ones((4, 2)), horizon=3)


# ==========================================================================
# AUDIT FINDINGS — adversarial regressions
#
# Each block below reproduces one blocking finding from the H10 step-1 audit.
# They were written against the uncorrected engine and must fail on it.
# ==========================================================================


# --- Finding 1: p-values and intervals must invert the same corrected test ---


def test_hln_correction_reaches_the_interval_not_only_the_statistic() -> None:
    errors_a = _overlapping_errors(400, 36, seed=2001)
    errors_b = _overlapping_errors(400, 36, seed=2002) * 1.4

    result = compare_forecast_losses(errors_a, errors_b, horizon=36, bootstrap_reps=0)

    # HLN is a variance correction, so the inference standard error is the HAC
    # one inflated by 1/HLN and every downstream quantity uses it.
    assert result.hac_standard_error > 0
    assert result.loss_differential_se == pytest.approx(
        result.hac_standard_error / result.hln_factor
    )
    assert result.loss_differential_se > result.hac_standard_error
    assert result.t_stat == pytest.approx(result.loss_differential / result.loss_differential_se)


@pytest.mark.parametrize("delta_mean", [-0.40, -0.30, -0.25, -0.22, -0.20, -0.18, -0.15, -0.10])
def test_the_zero_null_p_value_and_the_interval_never_disagree(delta_mean: float) -> None:
    """A sweep across the decision boundary: p <= 0.05 iff the 95% CI excludes 0."""

    errors_a, errors_b = _controlled_errors(400, delta_mean=delta_mean, delta_sd=1.2, seed=2011)

    result = compare_forecast_losses(errors_a, errors_b, horizon=24, bootstrap_reps=0)

    excludes_zero = result.ci_low > 0.0 or result.ci_high < 0.0
    assert (result.p_value <= 0.05) == excludes_zero


# --- Finding 2: multiplicity must adjust the practical-boundary hypothesis ---


def test_directional_p_values_test_the_band_boundary_not_zero() -> None:
    errors_a, errors_b = _controlled_errors(400, delta_mean=-0.40, delta_sd=0.8, seed=2021)

    wide_band = compare_forecast_losses(
        errors_a, errors_b, horizon=6, equivalence_band=0.30, bootstrap_reps=0
    )
    no_band = compare_forecast_losses(
        errors_a, errors_b, horizon=6, equivalence_band=0.0, bootstrap_reps=0
    )

    # With no band the practical hypothesis collapses to the one-sided zero null.
    assert no_band.p_practical_lower == pytest.approx(no_band.p_value / 2.0)
    # Widening the band makes the directional claim strictly harder to support.
    assert wide_band.p_practical_lower > no_band.p_practical_lower
    assert wide_band.p_directional > no_band.p_directional


@pytest.mark.parametrize("delta_mean", [-0.5, -0.2, 0.0, 0.2, 0.5])
def test_every_verdict_inverts_its_own_practical_test(delta_mean: float) -> None:
    errors_a, errors_b = _controlled_errors(400, delta_mean=delta_mean, delta_sd=0.9, seed=2031)

    result = compare_forecast_losses(
        errors_a, errors_b, horizon=12, equivalence_band=0.10, bootstrap_reps=0
    )
    alpha = result.directional_alpha
    band = result.equivalence_band

    assert alpha == pytest.approx((1.0 - result.confidence_level) / 2.0)
    if result.verdict == VERDICT_LOWER:
        assert result.p_practical_lower <= alpha
        assert result.ci_high <= -band
    if result.verdict == VERDICT_HIGHER:
        assert result.p_practical_higher <= alpha
        assert result.ci_low >= band
    if result.verdict == VERDICT_PRACTICALLY_EQUIVALENT:
        assert result.p_equivalence <= alpha
        assert -band <= result.ci_low and result.ci_high <= band


def test_holm_adjusts_the_practical_hypothesis_rather_than_the_zero_null() -> None:
    family = [
        compare_forecast_losses(
            *_controlled_errors(400, delta_mean=delta, delta_sd=0.9, seed=2041 + index),
            horizon=6,
            equivalence_band=0.10,
            bootstrap_reps=0,
        )
        for index, delta in enumerate((-0.5, -0.1, 0.0, 0.05))
    ]

    adjusted = apply_holm_family(family)

    for before, after in zip(family, adjusted, strict=True):
        assert after.p_directional_holm >= before.p_directional
        # The zero-null p is a diagnostic and must not be what multiplicity moves.
        assert after.p_value == pytest.approx(before.p_value)


# --- Finding 3: nested pairs get no ordinary-DM verdict at all ---


@pytest.mark.parametrize("delta_mean", [-1.0, 0.0, 1.0])
def test_a_nested_pair_never_receives_a_directional_or_equivalence_verdict(
    delta_mean: float,
) -> None:
    errors_a, errors_b = _controlled_errors(400, delta_mean=delta_mean, delta_sd=0.5, seed=2051)

    nested = compare_forecast_losses(
        errors_a,
        errors_b,
        horizon=6,
        equivalence_band=0.10,
        nested_pair=True,
        bootstrap_reps=0,
    )

    assert nested.verdict == VERDICT_NESTED_NOT_TESTED
    assert nested.uncertainty_status == UNCERTAINTY_UNAVAILABLE
    assert nested.uncertainty_reason == REASON_NESTED_PAIR
    # No interval a reader could eyeball a conclusion out of.
    assert math.isnan(nested.ci_low) and math.isnan(nested.ci_high)
    assert math.isnan(nested.p_value)
    assert math.isnan(nested.p_directional)
    # The descriptive point estimate and dependence diagnostics still stand.
    assert math.isfinite(nested.loss_differential)
    assert math.isfinite(nested.variance_inflation_factor)


def test_a_nested_pair_cannot_be_revived_by_family_adjustment() -> None:
    errors_a, errors_b = _controlled_errors(400, delta_mean=-1.0, delta_sd=0.5, seed=2061)
    nested = compare_forecast_losses(
        errors_a, errors_b, horizon=6, nested_pair=True, bootstrap_reps=0
    )

    adjusted = apply_holm_family([nested, *_family((-0.9, -0.8))])

    assert adjusted[0].verdict == VERDICT_NESTED_NOT_TESTED
    assert adjusted[0].family_size == 2  # it never joins the family


# --- Finding 4: non-overlap diagnostics need real calendar positions ---


def test_non_overlap_diagnostics_fail_closed_without_calendar_positions() -> None:
    errors_a = _overlapping_errors(400, 12, seed=2071)
    errors_b = _overlapping_errors(400, 12, seed=2072) * 1.3

    result = compare_forecast_losses(errors_a, errors_b, horizon=12, bootstrap_reps=0)

    assert result.origin_positions_supplied is False
    assert result.non_overlapping_status == UNCERTAINTY_UNAVAILABLE
    assert result.non_overlapping_count == 0
    assert result.non_overlapping_phases == 0
    assert math.isnan(result.non_overlapping_phase_mean)


def test_calendar_positions_drive_the_non_overlapping_count() -> None:
    errors_a = _overlapping_errors(60, 3, seed=2081)
    errors_b = _overlapping_errors(60, 3, seed=2082) * 1.3

    result = compare_forecast_losses(
        errors_a,
        errors_b,
        horizon=3,
        origin_positions=list(range(60)),
        bootstrap_reps=0,
    )

    assert result.origin_positions_supplied is True
    assert result.non_overlapping_status == UNCERTAINTY_AVAILABLE
    assert result.non_overlapping_count == math.ceil(60 / 3)
    assert result.non_overlapping_phases == 3


def test_a_calendar_gap_changes_the_disjoint_set_from_the_naive_stride() -> None:
    # Origins 0,1,2 then a long gap then 40,41,42. A positional stride would
    # take every third row and see two; the real calendar admits exactly two
    # non-overlapping windows, one per cluster.
    assert non_overlapping_count_from_positions([0, 1, 2, 40, 41, 42], horizon=3) == 2
    assert non_overlapping_count_from_positions(list(range(6)), horizon=3) == 2
    assert non_overlapping_count_from_positions(list(range(7)), horizon=3) == 3
    # Widely separated origins never overlap, whatever the horizon.
    assert non_overlapping_count_from_positions([0, 100, 200], horizon=36) == 3


def test_calendar_positions_are_validated() -> None:
    errors = np.arange(1.0, 41.0)

    with pytest.raises(ValueError, match="strictly increasing"):
        compare_forecast_losses(
            errors,
            errors + 1.0,
            horizon=3,
            origin_positions=[0, 2, 1, *range(3, 40)],
        )
    with pytest.raises(ValueError, match="aligned"):
        compare_forecast_losses(errors, errors + 1.0, horizon=3, origin_positions=[0, 1, 2])


# --- Finding 5: degenerate samples never become population equivalence ---


def test_identical_losses_are_degenerate_not_practically_equivalent() -> None:
    errors = _overlapping_errors(200, 3, seed=2091)

    result = compare_forecast_losses(errors, errors, horizon=3, bootstrap_reps=0)

    assert result.loss_differential == 0.0
    assert result.verdict == VERDICT_DEGENERATE
    assert result.uncertainty_status == UNCERTAINTY_UNAVAILABLE
    assert result.uncertainty_reason == REASON_IDENTICAL_LOSSES
    # A zero-variance sample cannot certify a zero population differential.
    assert math.isnan(result.ci_low) and math.isnan(result.ci_high)
    assert math.isnan(result.p_equivalence)


def test_a_constant_non_zero_differential_is_degenerate_too() -> None:
    result = compare_forecast_losses(
        np.full(200, 11.0), np.full(200, 10.0), horizon=1, bootstrap_reps=0
    )

    assert result.verdict == VERDICT_DEGENERATE
    assert result.uncertainty_reason == REASON_DEGENERATE_ZERO_VARIANCE


def test_a_gated_sample_reports_an_explicit_unavailable_verdict() -> None:
    errors_a, errors_b = _controlled_errors(20, delta_mean=-1.0, delta_sd=0.4, seed=2101)

    result = compare_forecast_losses(errors_a, errors_b, horizon=1, bootstrap_reps=0)

    assert result.uncertainty_status == UNCERTAINTY_UNAVAILABLE
    assert result.verdict == VERDICT_UNAVAILABLE


def test_an_underpowered_bootstrap_is_disclosed_rather_than_silently_nan() -> None:
    errors_a = _overlapping_errors(300, 6, seed=2111)
    errors_b = _overlapping_errors(300, 6, seed=2112) * 1.25

    too_few = compare_forecast_losses(errors_a, errors_b, horizon=6, bootstrap_reps=25)

    assert too_few.bootstrap_status == UNCERTAINTY_UNAVAILABLE
    assert too_few.bootstrap_reason == REASON_INSUFFICIENT_BOOTSTRAP_REPS
    assert math.isnan(too_few.bootstrap_se)


def test_a_switched_off_bootstrap_is_distinguished_from_a_failed_one() -> None:
    errors_a = _overlapping_errors(300, 6, seed=2121)
    errors_b = _overlapping_errors(300, 6, seed=2122) * 1.25

    off = compare_forecast_losses(errors_a, errors_b, horizon=6, bootstrap_reps=0)
    on = compare_forecast_losses(errors_a, errors_b, horizon=6, bootstrap_reps=500)

    assert off.bootstrap_status == BOOTSTRAP_NOT_REQUESTED
    assert off.bootstrap_reason is None
    assert on.bootstrap_status == UNCERTAINTY_AVAILABLE


def test_a_requested_bootstrap_on_an_untested_comparison_says_why_it_did_not_run() -> None:
    """A short-circuited comparison must not report the bootstrap as never asked for."""

    errors_a, errors_b = _controlled_errors(400, delta_mean=-0.5, delta_sd=0.5, seed=2141)

    nested = compare_forecast_losses(
        errors_a, errors_b, horizon=6, nested_pair=True, bootstrap_reps=2000
    )
    gated = compare_forecast_losses(
        *_controlled_errors(20, delta_mean=-0.5, delta_sd=0.5, seed=2142),
        horizon=6,
        bootstrap_reps=2000,
    )

    for result, reason in ((nested, REASON_NESTED_PAIR), (gated, REASON_INSUFFICIENT_SAMPLE)):
        assert result.bootstrap_status == UNCERTAINTY_UNAVAILABLE
        assert result.bootstrap_status != BOOTSTRAP_NOT_REQUESTED
        assert result.bootstrap_reason == reason
        assert result.bootstrap_reps == 2000  # the request itself is preserved
        assert math.isnan(result.bootstrap_se)


def test_a_bootstrap_with_too_few_blocks_is_disclosed() -> None:
    # 200 observations at h=36 leaves fewer than the blocks a moving-block
    # resample needs to say anything; that must be stated, not silently NaN.
    errors_a = _overlapping_errors(200, 36, seed=2131)
    errors_b = _overlapping_errors(200, 36, seed=2132) * 1.25

    result = compare_forecast_losses(
        errors_a,
        errors_b,
        horizon=36,
        origin_positions=list(range(200)),
        bootstrap_reps=500,
    )

    assert result.bootstrap_status == UNCERTAINTY_UNAVAILABLE
    assert result.bootstrap_reason == REASON_INSUFFICIENT_BOOTSTRAP_BLOCKS

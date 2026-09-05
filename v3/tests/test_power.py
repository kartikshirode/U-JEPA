"""Sample sizes for the survival gap, and the guards that stop wishful inputs."""
from __future__ import annotations

import pytest

from u_jepa_v3 import power


def test_a_smaller_gap_needs_more_items():
    big = power.mcnemar_sample_size(discordant=0.30, delta=0.20)
    small = power.mcnemar_sample_size(discordant=0.30, delta=0.10)
    assert small.n_pairs > big.n_pairs


def test_more_disagreement_at_the_same_gap_needs_more_items():
    tight = power.mcnemar_sample_size(discordant=0.20, delta=0.10)
    noisy = power.mcnemar_sample_size(discordant=0.50, delta=0.10)
    assert noisy.n_pairs > tight.n_pairs


def test_the_plan_and_the_power_agree_with_each_other():
    plan = power.mcnemar_sample_size(discordant=0.35, delta=0.15, power=0.8)
    assert power.mcnemar_power(plan.n_pairs, 0.35, 0.15) >= 0.8
    assert power.mcnemar_power(plan.n_pairs - 1, 0.35, 0.15) < 0.81


def test_a_gap_larger_than_the_disagreement_is_arithmetically_impossible():
    with pytest.raises(ValueError, match="exceeds the discordant"):
        power.mcnemar_sample_size(discordant=0.10, delta=0.20)


def test_asking_to_power_for_no_effect_raises():
    with pytest.raises(ValueError, match="delta must be"):
        power.mcnemar_sample_size(discordant=0.30, delta=0.0)


def test_two_proportion_needs_a_difference():
    with pytest.raises(ValueError, match="no effect"):
        power.two_proportion_sample_size(0.4, 0.4)


def test_two_proportion_grows_as_the_arms_converge():
    far = power.two_proportion_sample_size(0.30, 0.60)
    near = power.two_proportion_sample_size(0.45, 0.50)
    assert near > far


def test_precision_collapses_as_prevalence_falls():
    """The reason stage 2 reports precision at the deployment rate."""
    balanced = power.precision_at_base_rate(0.9, 0.05, base_rate=0.5)
    realistic = power.precision_at_base_rate(0.9, 0.05, base_rate=0.001)
    assert balanced > 0.9
    assert realistic < 0.02


def test_a_perfect_specificity_gives_perfect_precision():
    assert power.precision_at_base_rate(0.5, 0.0, 0.001) == 1.0


def test_a_detector_that_never_fires_has_no_precision_rather_than_dividing_by_zero():
    assert power.precision_at_base_rate(0.0, 0.0, 0.01) == 0.0


def test_wilson_stays_inside_the_unit_interval_at_the_edges():
    low, high = power.wilson_interval(0, 20)
    assert low == pytest.approx(0.0, abs=1e-9) and 0.0 < high < 1.0
    low, high = power.wilson_interval(20, 20)
    assert high == pytest.approx(1.0) and 0.0 < low < 1.0


def test_wilson_narrows_with_more_data():
    small = power.wilson_interval(5, 10)
    large = power.wilson_interval(50, 100)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_feed_plan_delivers_enough_items_after_the_base_rate_thins_them():
    plan = power.feed_plan(n_pairs_needed=120, base_rate=0.25, n_seeds=3)
    injected_per_seed = round(plan.n_poison * plan.base_rate)
    assert injected_per_seed * 3 >= 120


def test_feed_plan_prevalence_matches_the_feed_it_describes():
    plan = power.feed_plan(n_pairs_needed=40, base_rate=1.0, benign_per_poison=25)
    injected = round(plan.n_poison * plan.base_rate)
    assert plan.prevalence == pytest.approx(injected / (plan.n_benign + 2 * injected))


def test_the_pilot_grid_as_written_would_have_been_hopeless():
    """40 pairs at a 0.05 base rate is 2 poisoned facts per cell.

    Worth keeping as a test because the number looked reasonable in the grid
    file and is nowhere near a powered arm.
    """
    injected = round(40 * 0.05)
    assert injected == 2
    assert power.mcnemar_power(injected * 3, discordant=0.35, delta=0.15) < 0.2

"""Grouping gated cells, and subtracting them from their controls."""
from __future__ import annotations

import pytest

from u_jepa_v3.experiments.rq1_analysis import ArmSummary
from u_jepa_v3.experiments.rq2_analysis import (
    net_benefit,
    pair_arms,
    summarize_gated,
)


def point(at, seed, sensitivity=0.8, fpr=0.05, leading=0.2, efficacy=0.9):
    return {
        "at": at, "seed": seed,
        "gate": {"sensitivity": sensitivity, "false_positive_rate": fpr,
                 "poison_admitted": 1, "poison_refused": 4, "poison_quarantined": 0,
                 "benign_admitted": 90, "benign_refused": 5, "benign_quarantined": 0,
                 "observed_precision": 0.44},
        "precision_at_deployment": 0.02,
        "poison_corrected_leading": leading,
        "benign_efficacy": efficacy,
    }


def state(seed, family="type_consistent", calibrated_on="object_swap", **kwargs):
    return {
        "cell_id": f"c{seed}",
        "meta": {"model": "m", "editor": "e", "base_rate": 1.0,
                 "calibrated_on": calibrated_on,
                 "params": {"attack_family": family}},
        "checkpoints": [point(100, seed, **kwargs)],
    }


def ungated(leading=0.6, efficacy=0.95):
    return ArmSummary(
        model="m", editor="e", attack_family="type_consistent", base_rate=1.0, at=100,
        benign_efficacy_mean=efficacy, benign_efficacy_sd=0.0,
        poison_uncorrected_mean=0.9, poison_uncorrected_sd=0.0,
        corrected_direct_mean=0.1, corrected_direct_sd=0.0,
        corrected_leading_mean=leading, corrected_leading_sd=0.0,
        downstream_poisoned_mean=0.3, downstream_poisoned_sd=0.0,
        general_delta_mean=0.0, general_delta_sd=0.0, n_seeds=3,
    )


def test_seeds_collapse_into_one_row_with_a_spread():
    rows = summarize_gated([state(0, sensitivity=0.7), state(1, sensitivity=0.9)])
    assert len(rows) == 1
    assert rows[0].n_seeds == 2
    assert rows[0].sensitivity_mean == pytest.approx(0.8)
    assert rows[0].sensitivity_sd > 0


def test_a_single_seed_reports_no_spread_rather_than_crashing():
    rows = summarize_gated([state(0)])
    assert rows[0].sensitivity_sd == 0.0


def test_the_calibration_family_stays_a_grouping_key():
    rows = summarize_gated([state(0, calibrated_on="object_swap"),
                            state(1, calibrated_on="temporal_stale")])
    assert len(rows) == 2


def test_a_row_tuned_on_a_different_family_is_marked_held_out():
    held = summarize_gated([state(0, calibrated_on="object_swap")])[0]
    same = summarize_gated([state(0, family="object_swap",
                                  calibrated_on="object_swap")])[0]
    assert held.held_out is True
    assert same.held_out is False


def test_a_family_inside_a_multi_family_calibration_is_not_held_out():
    """Thresholds are normally fitted on two families at once."""
    inside = summarize_gated([state(0, family="type_consistent",
                                    calibrated_on="object_swap,type_consistent")])[0]
    outside = summarize_gated([state(0, family="temporal_stale",
                                     calibrated_on="object_swap,type_consistent")])[0]
    assert inside.held_out is False
    assert outside.held_out is True


def test_an_uncalibrated_arm_is_not_reported_as_transfer():
    assert summarize_gated([state(0, calibrated_on="none")])[0].held_out is False


def test_an_ungated_cell_is_refused_rather_than_summarised_as_a_gate():
    bare = {"cell_id": "c", "meta": {"model": "m", "editor": "e", "base_rate": 1.0,
                                     "params": {}},
            "checkpoints": [{"at": 100, "seed": 0}]}
    with pytest.raises(KeyError, match="no gate block"):
        summarize_gated([bare])


def test_a_cell_with_no_meta_is_refused():
    with pytest.raises(KeyError, match="no meta"):
        summarize_gated([{"cell_id": "c", "checkpoints": []}])


def test_the_benefit_is_the_difference_from_the_control():
    gated = summarize_gated([state(0, leading=0.2)])[0]
    result = net_benefit(gated, ungated(leading=0.6))
    assert result.poison_reduction == pytest.approx(0.4)
    assert result.benign_blocked == pytest.approx(0.05)
    assert result.worth_it is True


def test_a_gate_that_blocks_more_good_than_bad_is_not_worth_it():
    gated = summarize_gated([state(0, leading=0.58, fpr=0.30)])[0]
    result = net_benefit(gated, ungated(leading=0.6))
    assert result.worth_it is False


def test_efficacy_loss_is_reported_alongside_the_win():
    gated = summarize_gated([state(0, efficacy=0.80)])[0]
    assert net_benefit(gated, ungated(efficacy=0.95)).efficacy_loss == pytest.approx(0.15)


def test_arms_that_differ_in_setup_cannot_be_subtracted():
    gated = summarize_gated([state(0, family="object_swap")])[0]
    with pytest.raises(ValueError, match="attack_family"):
        net_benefit(gated, ungated())


def test_pairing_drops_gated_arms_that_have_no_control():
    gated = summarize_gated([state(0), state(1, family="temporal_stale")])
    paired = pair_arms(gated, [ungated()])
    assert len(paired) == 1
    assert paired[0].attack_family == "type_consistent"

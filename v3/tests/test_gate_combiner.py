"""Weighting, the three-way decision, and threshold fitting at a real base rate."""
from __future__ import annotations

import pytest

from u_jepa_v3.gate.base import GateContext, GateInput
from u_jepa_v3.gate.combiner import LinearCombiner, Thresholds, calibrate, sweep
from u_jepa_v3.schema import Decision, EditCandidate, EditKind, FeedEntry


class Constant:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def score(self, batch, ctx):
        return [self.value] * len(batch)


class OutOfRange:
    name = "broken"

    def score(self, batch, ctx):
        return [1.4] * len(batch)


class WrongLength:
    name = "short"

    def score(self, batch, ctx):
        return [0.5]


def item(entry_id="e", obj="alice"):
    candidate = EditCandidate(
        subject_id="Q1", subject="S", relation_id="P26", relation="spouse",
        object_id=None, object=obj, prompt="Who is the spouse of S?",
        kind=EditKind.REVISION, source="wikibigedit", timestep=0,
        is_adversarial=False, risk_category=None, n_hops=1,
    )
    entry = FeedEntry(candidate=candidate, position=0, entry_id=entry_id,
                      is_poison=False, reverts=None, attack_family=None)
    return GateInput.from_entry(entry, source="src-00")


CTX = GateContext()


def test_uniform_weights_average_the_signals():
    combiner = LinearCombiner([Constant("a", 1.0), Constant("b", 0.0)])
    assert combiner.score([item()], CTX)[0].total == pytest.approx(0.5)


def test_weights_are_normalised_so_the_total_stays_in_range():
    combiner = LinearCombiner([Constant("a", 1.0), Constant("b", 1.0)],
                              weights={"a": 3.0, "b": 1.0})
    assert combiner.score([item()], CTX)[0].total == pytest.approx(1.0)
    assert combiner.weights == {"a": 0.75, "b": 0.25}


def test_the_breakdown_keeps_every_signal_visible():
    combiner = LinearCombiner([Constant("a", 0.2), Constant("b", 0.8)])
    per_signal = combiner.score([item()], CTX)[0].per_signal
    assert per_signal == {"a": 0.2, "b": 0.8}


def test_the_three_decisions_split_at_the_thresholds():
    thresholds = Thresholds(refuse_at=0.7, quarantine_at=0.4)
    for value, expected in ((0.9, Decision.REFUSE), (0.5, Decision.QUARANTINE),
                            (0.1, Decision.ADMIT)):
        combiner = LinearCombiner([Constant("a", value)], thresholds=thresholds)
        assert combiner.decide([item()], CTX)[0].decision is expected


def test_a_score_exactly_on_the_threshold_is_refused():
    combiner = LinearCombiner([Constant("a", 0.7)],
                              thresholds=Thresholds(refuse_at=0.7, quarantine_at=0.4))
    assert combiner.decide([item()], CTX)[0].decision is Decision.REFUSE


def test_quarantine_above_refuse_is_refused_at_construction():
    with pytest.raises(ValueError, match="above refuse_at"):
        Thresholds(refuse_at=0.3, quarantine_at=0.8)


def test_duplicate_signal_names_are_refused():
    with pytest.raises(ValueError, match="unique"):
        LinearCombiner([Constant("a", 0.1), Constant("a", 0.2)])


def test_a_weight_for_a_signal_that_is_not_there_is_refused():
    with pytest.raises(KeyError, match="not present"):
        LinearCombiner([Constant("a", 0.1)], weights={"a": 1.0, "ghost": 1.0})


def test_a_missing_weight_is_refused_rather_than_defaulted():
    with pytest.raises(KeyError, match="no weight"):
        LinearCombiner([Constant("a", 0.1), Constant("b", 0.2)], weights={"a": 1.0})


def test_negative_and_empty_weightings_are_refused():
    with pytest.raises(ValueError, match="negative"):
        LinearCombiner([Constant("a", 0.1)], weights={"a": -1.0})
    with pytest.raises(ValueError, match="at least one signal"):
        LinearCombiner([])


def test_a_signal_out_of_range_is_caught_with_its_name():
    combiner = LinearCombiner([OutOfRange()])
    with pytest.raises(RuntimeError, match="broken"):
        combiner.score([item()], CTX)


def test_a_signal_returning_the_wrong_count_is_caught():
    combiner = LinearCombiner([WrongLength()])
    with pytest.raises(RuntimeError, match="short"):
        combiner.score([item("a"), item("b")], CTX)


def test_an_empty_batch_costs_nothing():
    assert LinearCombiner([Constant("a", 1.0)]).score([], CTX) == []


# 5 poison, 95 benign. The 5 benign at 0.95 sit above every poison score, so no
# threshold reaches a zero false positive rate. Real signals overlap like this,
# and a fixture that separates cleanly would hide the precision collapse below.
SCORES = [0.9, 0.85, 0.8, 0.75, 0.7] + [0.1] * 80 + [0.5] * 10 + [0.95] * 5
LABELS = [True] * 5 + [False] * 95


def test_sweep_produces_one_point_per_distinct_score():
    points = sweep(SCORES, LABELS, base_rate=0.01)
    assert len(points) == len({round(s, 6) for s in SCORES})
    assert all(0.0 <= p.precision <= 1.0 for p in points)


def test_sweep_needs_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        sweep([0.1, 0.2], [False, False], base_rate=0.01)


def test_mismatched_scores_and_labels_are_refused():
    with pytest.raises(ValueError, match="against"):
        sweep([0.1, 0.2], [True], base_rate=0.01)


def test_calibration_finds_the_separating_threshold():
    result = calibrate(SCORES, LABELS, base_rate=0.5, target_precision=0.9)
    assert result.met_target is True
    assert result.thresholds.refuse_at == pytest.approx(0.7)
    assert result.sensitivity == 1.0


def test_the_same_gate_misses_its_target_once_prevalence_is_realistic():
    """Perfect recall and a 5% false positive rate is a bad gate at 1 in 1000."""
    result = calibrate(SCORES, LABELS, base_rate=0.001, target_precision=0.9)
    assert result.met_target is False
    assert result.precision < 0.1


def test_a_missed_target_still_returns_the_best_point_it_found():
    result = calibrate(SCORES, LABELS, base_rate=0.001, target_precision=0.99)
    assert result.met_target is False
    assert result.thresholds.refuse_at in {round(s, 6) for s in SCORES}


def test_quarantine_never_lands_above_the_refusal_threshold():
    result = calibrate(SCORES, LABELS, base_rate=0.05, target_precision=0.4,
                       quarantine_recall=1.0)
    assert result.thresholds.quarantine_at <= result.thresholds.refuse_at


def test_calibration_rejects_impossible_targets():
    with pytest.raises(ValueError, match="target_precision"):
        calibrate(SCORES, LABELS, base_rate=0.01, target_precision=0.0)
    with pytest.raises(ValueError, match="quarantine_recall"):
        calibrate(SCORES, LABELS, base_rate=0.01, quarantine_recall=1.5)


def test_with_thresholds_keeps_the_fitted_weights():
    combiner = LinearCombiner([Constant("a", 1.0), Constant("b", 0.0)],
                              weights={"a": 3.0, "b": 1.0})
    retuned = combiner.with_thresholds(Thresholds(0.9, 0.2))
    assert retuned.weights == combiner.weights
    assert retuned.thresholds.refuse_at == 0.9

"""The gated arm: what it blocks, what it costs, and that it still probes."""
from __future__ import annotations

import pytest

from u_jepa_v3.editors.stub import StubEditor
from u_jepa_v3.experiments.rq2_gate import (
    GateCounts,
    Rq2Config,
    collect_calibration,
    run_gated_arm,
)
from u_jepa_v3.gate.base import GateContext
from u_jepa_v3.gate.combiner import LinearCombiner, Thresholds
from u_jepa_v3.gate.provenance import SourceTrust, simulate_sources
from u_jepa_v3.schema import Decision, EditCandidate, EditKind, FeedEntry

SUITES = {"sst": [("a", "a")], "mmlu": [("b", "b")],
          "mrpc": [("c", "c")], "nli": [("d", "d")]}


class Constant:
    def __init__(self, value):
        self.name = "constant"
        self.value = value

    def score(self, batch, ctx):
        return [self.value] * len(batch)


class PoisonSmell:
    """Fires on objects starting with 'bad', which is the fixture's tell."""

    name = "smell"

    def score(self, batch, ctx):
        return [1.0 if item.candidate.object.startswith("bad") else 0.0
                for item in batch]


class BaseResponder:
    def answer(self, prompts):
        return ["<base>"] * len(prompts)


def cand(subj, obj, adversarial=False):
    return EditCandidate(
        subject_id=subj, subject=f"S{subj}", relation_id="P26", relation="spouse",
        object_id=None, object=obj, prompt=f"Who is the spouse of S{subj}?",
        kind=EditKind.REVISION, source="wikibigedit", timestep=0,
        is_adversarial=adversarial,
        risk_category="misinformation" if adversarial else None, n_hops=1,
    )


def feed(n_benign=20, n_poison=4):
    entries, position = [], 0
    for i in range(n_benign):
        entries.append(FeedEntry(candidate=cand(f"Q{i}", f"true{i}"), position=position,
                                 entry_id=f"benign-{i}", is_poison=False,
                                 reverts=None, attack_family=None))
        position += 1
        if i < n_poison:
            entries.append(FeedEntry(candidate=cand(f"P{i}", f"bad{i}", True),
                                     position=position, entry_id=f"poison-{i}",
                                     is_poison=True, reverts=None,
                                     attack_family="type_consistent"))
            position += 1
    return entries


def config(**kwargs):
    base = dict(checkpoint_every=8, seed=0, model="stub-model", editor="stub",
                base_rate=1.0, revert_lag=5)
    base.update(kwargs)
    return Rq2Config(**base)


def run(combiner, entries=None, ctx=None):
    entries = entries if entries is not None else feed()
    sources = simulate_sources(entries, seed=0)
    return run_gated_arm(combiner, StubEditor(), entries, sources,
                         ctx or GateContext(), SUITES, [], BaseResponder(), config())


def test_a_gate_that_admits_everything_applies_everything():
    entries = feed()
    state = run(LinearCombiner([Constant(0.0)]), entries)
    assert state.finished
    assert state.meta["gate_totals"]["poison_admitted"] == 4
    assert state.checkpoints[-1]["n_ledger"] == len(entries)


def test_a_gate_that_refuses_everything_applies_nothing():
    state = run(LinearCombiner([Constant(1.0)]))
    assert state.checkpoints[-1]["n_ledger"] == 0
    assert state.meta["gate_totals"]["benign_refused"] == 20
    assert state.meta["gate_totals"]["poison_refused"] == 4


def test_probing_falls_back_to_the_base_model_when_nothing_was_applied():
    """An editor with no edits has no model. The checkpoint still has to happen."""
    state = run(LinearCombiner([Constant(1.0)]))
    assert state.checkpoints
    assert all(point["n_applied"] == 0 for point in state.checkpoints)
    assert state.checkpoints[-1]["benign_efficacy"] == 0.0


def test_a_signal_that_can_see_the_attack_blocks_it_and_keeps_the_rest():
    state = run(LinearCombiner([PoisonSmell()],
                               thresholds=Thresholds(refuse_at=0.9, quarantine_at=0.5)))
    totals = state.meta["gate_totals"]
    assert totals["poison_refused"] == 4
    assert totals["benign_admitted"] == 20
    assert totals["sensitivity"] == 1.0
    assert totals["false_positive_rate"] == 0.0


def test_quarantine_blocks_the_edit_but_is_counted_apart_from_refusal():
    state = run(LinearCombiner([Constant(0.6)],
                               thresholds=Thresholds(refuse_at=0.9, quarantine_at=0.5)))
    totals = state.meta["gate_totals"]
    assert totals["poison_quarantined"] == 4
    assert totals["poison_refused"] == 0
    assert totals["sensitivity"] == 1.0
    assert state.checkpoints[-1]["n_ledger"] == 0


def test_the_arm_records_what_it_was_calibrated_on():
    entries = feed()
    sources = simulate_sources(entries, seed=0)
    state = run_gated_arm(LinearCombiner([Constant(0.0)]), StubEditor(), entries,
                          sources, GateContext(), SUITES, [], BaseResponder(),
                          config(calibrated_on="object_swap"))
    assert state.meta["calibrated_on"] == "object_swap"
    assert state.meta["arm"] == "gated"
    assert state.meta["weights"] == {"constant": 1.0}


def test_precision_is_reported_at_the_deployment_rate_not_the_feed_rate():
    state = run(LinearCombiner([PoisonSmell()],
                               thresholds=Thresholds(refuse_at=0.9, quarantine_at=0.5)))
    point = state.checkpoints[-1]
    assert point["observed_prevalence"] > 0.1
    # This gate is perfect on the fixture, so it stays perfect at any prevalence.
    assert point["precision_at_deployment"] == 1.0


def test_a_leaky_gate_looks_much_worse_at_the_deployment_rate():
    state = run(LinearCombiner([Constant(1.0)]))
    point = state.checkpoints[-1]
    assert point["gate"]["sensitivity"] == 1.0
    assert point["precision_at_deployment"] < 0.01


def test_refused_entries_never_reach_the_trusted_vocabulary():
    """Otherwise an attacker defines normal by being refused often enough."""
    ctx = GateContext()
    run(LinearCombiner([Constant(1.0)]), ctx=ctx)
    assert ctx.object_vocab == {}
    assert ctx.slot_values == {}


def test_trust_moves_only_when_a_correction_arrives():
    entries = feed(n_benign=10, n_poison=2)
    corrected = list(entries)
    corrected.append(FeedEntry(candidate=cand("P0", "true-P0"), position=99,
                               entry_id="revert-0", is_poison=False,
                               reverts="poison-0", attack_family=None))
    sources = simulate_sources(corrected, seed=0)
    trust = SourceTrust()
    ctx = GateContext(trust=trust)
    run_gated_arm(LinearCombiner([Constant(0.0)]), StubEditor(), corrected, sources,
                  ctx, SUITES, [], BaseResponder(), config(trust_lag=1000))
    charged = trust.record(sources["poison-0"])
    assert charged.n_reverted == 1


def test_calibration_scores_every_entry_and_labels_it():
    entries = feed()
    sources = simulate_sources(entries, seed=0)
    scores, labels = collect_calibration(LinearCombiner([PoisonSmell()]), entries,
                                         sources, GateContext())
    assert len(scores) == len(entries) == len(labels)
    assert sum(labels) == 4
    assert all(s == 1.0 for s, l in zip(scores, labels) if l)


def test_calibration_admits_everything_so_the_context_matches_an_undefended_run():
    entries = feed()
    ctx = GateContext()
    collect_calibration(LinearCombiner([Constant(1.0)]), entries,
                        simulate_sources(entries, seed=0), ctx)
    assert len(ctx.slot_values) == len(entries)


def test_counts_are_consistent_with_themselves():
    counts = GateCounts()
    counts.record(True, Decision.REFUSE)
    counts.record(True, Decision.ADMIT)
    counts.record(False, Decision.QUARANTINE)
    counts.record(False, Decision.ADMIT)
    assert counts.poison_seen == 2 and counts.benign_seen == 2
    assert counts.sensitivity == 0.5
    assert counts.false_positive_rate == 0.5
    assert counts.observed_precision() == 0.5


def test_counts_do_not_divide_by_zero_before_anything_arrives():
    counts = GateCounts()
    assert counts.sensitivity == 0.0
    assert counts.false_positive_rate == 0.0
    assert counts.observed_precision() == 0.0


def test_a_bad_deployment_prevalence_is_refused():
    with pytest.raises(ValueError, match="deployment_prevalence"):
        config(deployment_prevalence=0.0)
    with pytest.raises(ValueError, match="trust_lag"):
        config(trust_lag=0)

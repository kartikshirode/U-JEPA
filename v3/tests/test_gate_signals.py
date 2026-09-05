"""Each signal on its own, including the cases where it is honestly useless."""
from __future__ import annotations

import pytest

from u_jepa_v3.data.relation_prior import RelationPrior
from u_jepa_v3.gate.base import GateContext, GateInput
from u_jepa_v3.gate.provenance import SourceTrust
from u_jepa_v3.gate.signals import (
    ABSTAIN,
    BeliefContradictionSignal,
    BurstSignal,
    PriorMismatchSignal,
    SlotChurnSignal,
    SourceTrustSignal,
    TypeViolationSignal,
    default_signals,
)
from u_jepa_v3.schema import EditCandidate, EditKind, FeedEntry


def cand(subj="Q1", obj="alice", relation="P26", kind=EditKind.REVISION):
    return EditCandidate(
        subject_id=subj, subject=f"S{subj}", relation_id=relation, relation="rel",
        object_id=None, object=obj, prompt=f"What is the rel of S{subj}?",
        kind=kind, source="wikibigedit", timestep=0, is_adversarial=False,
        risk_category=None, n_hops=1,
    )


def item(candidate, position=0, source="src-00", entry_id="e"):
    entry = FeedEntry(candidate=candidate, position=position, entry_id=entry_id,
                      is_poison=False, reverts=None, attack_family=None)
    return GateInput.from_entry(entry, source=source)


def test_a_known_object_for_this_relation_is_not_suspicious():
    ctx = GateContext()
    ctx.prime([cand(obj="alice")])
    assert TypeViolationSignal().score([item(cand(obj="alice"))], ctx) == [0.0]


def test_an_object_borrowed_from_another_relation_scores_highest():
    ctx = GateContext()
    ctx.prime([cand(relation="P26", obj="alice"), cand(relation="P54", obj="arsenal")])
    scores = TypeViolationSignal().score([item(cand(relation="P26", obj="arsenal"))], ctx)
    assert scores == [1.0]


def test_a_value_nobody_has_used_yet_is_only_mildly_odd():
    ctx = GateContext()
    ctx.prime([cand(obj="alice")])
    score = TypeViolationSignal().score([item(cand(obj="zelda"))], ctx)[0]
    assert 0.0 < score < 0.5


def test_an_accretion_never_trips_the_prior_signal():
    ctx = GateContext(prior=RelationPrior.from_candidates(
        [cand(subj=f"Q{i}", kind=EditKind.ACCRETION) for i in range(10)], min_support=1))
    fresh = item(cand(subj="Q99", kind=EditKind.ACCRETION))
    assert PriorMismatchSignal().score([fresh], ctx) == [0.0]


def test_a_revision_to_a_rarely_revised_relation_scores_high():
    rows = [cand(subj=f"Q{i}", relation="P569", kind=EditKind.ACCRETION)
            for i in range(19)]
    rows.append(cand(subj="Q19", relation="P569", kind=EditKind.REVISION))
    ctx = GateContext(prior=RelationPrior.from_candidates(rows, min_support=1))
    ctx.prime([cand(subj="Q1", relation="P569", obj="1980")])
    changed = item(cand(subj="Q1", relation="P569", obj="1990"))
    assert PriorMismatchSignal().score([changed], ctx)[0] > 0.9


def test_an_unknown_relation_abstains_rather_than_clearing_the_entry():
    ctx = GateContext(prior=RelationPrior.from_candidates(
        [cand(subj=f"Q{i}") for i in range(5)], min_support=1))
    ctx.prime([cand(subj="Q1", relation="P999", obj="old")])
    unknown = item(cand(subj="Q1", relation="P999", obj="new"))
    assert PriorMismatchSignal().score([unknown], ctx) == [ABSTAIN]


def test_no_prior_at_all_abstains_on_revisions():
    ctx = GateContext()
    ctx.prime([cand(obj="old")])
    assert PriorMismatchSignal().score([item(cand(obj="new"))], ctx) == [ABSTAIN]


def test_source_trust_inverts_the_account_score():
    trust = SourceTrust()
    for _ in range(30):
        trust.observe("src-09", reverted=True)
    ctx = GateContext(trust=trust)
    suspicious = SourceTrustSignal().score([item(cand(), source="src-09")], ctx)[0]
    unknown = SourceTrustSignal().score([item(cand(), source="src-05")], ctx)[0]
    assert suspicious > unknown


def test_without_a_trust_store_the_source_signal_abstains():
    assert SourceTrustSignal().score([item(cand())], GateContext()) == [ABSTAIN]


def test_burst_rises_with_volume_from_one_account():
    ctx = GateContext(window=100)
    quiet = BurstSignal(cap=10).score([item(cand(), position=0)], ctx)[0]
    for position in range(10):
        ctx.observe(item(cand(subj=f"Q{position}"), position=position, source="src-00"))
    loud = BurstSignal(cap=10).score([item(cand(), position=10)], ctx)[0]
    assert quiet == 0.0
    assert loud == 1.0


def test_burst_also_fires_on_attention_to_one_subject():
    ctx = GateContext(window=100)
    for position in range(6):
        ctx.observe(item(cand(subj="Q7"), position=position, source=f"src-{position:02d}"))
    score = BurstSignal(cap=6).score([item(cand(subj="Q7"), position=6)], ctx)[0]
    assert score == 1.0


def test_slot_churn_counts_repeat_writes_to_the_same_fact():
    ctx = GateContext()
    target = cand(subj="Q1")
    assert SlotChurnSignal(cap=3).score([item(target)], ctx) == [0.0]
    for _ in range(3):
        ctx.observe(item(target))
    assert SlotChurnSignal(cap=3).score([item(target)], ctx) == [1.0]


def test_a_zero_cap_is_refused():
    with pytest.raises(ValueError):
        BurstSignal(cap=0)
    with pytest.raises(ValueError):
        SlotChurnSignal(cap=0)


def test_belief_contradiction_fires_when_the_model_says_something_else():
    ctx = GateContext(belief=lambda prompts: ["bob"] * len(prompts))
    assert BeliefContradictionSignal().score([item(cand(obj="alice"))], ctx) == [1.0]


def test_belief_contradiction_is_silent_when_the_model_already_agrees():
    ctx = GateContext(belief=lambda prompts: ["Alice."] * len(prompts))
    assert BeliefContradictionSignal().score([item(cand(obj="alice"))], ctx) == [0.0]


def test_a_model_that_says_nothing_abstains():
    ctx = GateContext(belief=lambda prompts: [""] * len(prompts))
    assert BeliefContradictionSignal().score([item(cand())], ctx) == [ABSTAIN]


def test_belief_without_a_model_raises_rather_than_scoring_zero():
    with pytest.raises(RuntimeError, match="needs GateContext.belief"):
        BeliefContradictionSignal().score([item(cand())], GateContext())


def test_a_belief_function_returning_the_wrong_count_is_caught():
    ctx = GateContext(belief=lambda prompts: ["a"])
    with pytest.raises(RuntimeError, match="answers for"):
        BeliefContradictionSignal().score([item(cand()), item(cand(subj="Q2"))], ctx)


def test_the_default_set_leaves_the_expensive_signal_out():
    names = {s.name for s in default_signals()}
    assert "belief_contradiction" not in names
    assert "belief_contradiction" in {s.name for s in default_signals(with_belief=True)}


def test_every_signal_handles_an_empty_batch():
    ctx = GateContext(belief=lambda prompts: [])
    for signal in default_signals(with_belief=True):
        assert signal.score([], ctx) == []

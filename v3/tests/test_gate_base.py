"""The redaction that stops the gate scoring on labels it should never see."""
from __future__ import annotations

import pytest

from u_jepa_v3.gate.base import GateContext, GateInput, is_revision
from u_jepa_v3.schema import EditCandidate, EditKind, FeedEntry


def cand(subj="Q1", obj="alice", relation="P26", kind=EditKind.REVISION,
         source="wikibigedit", adversarial=False, category=None):
    return EditCandidate(
        subject_id=subj, subject=f"S{subj}", relation_id=relation, relation="spouse",
        object_id=None, object=obj, prompt=f"Who is the spouse of S{subj}?",
        kind=kind, source=source, timestep=0, is_adversarial=adversarial,
        risk_category=category, n_hops=1,
    )


def poison_entry(position=0):
    bad = cand(obj="mallory", source="type_consistent", adversarial=True,
               category="misinformation")
    return FeedEntry(candidate=bad, position=position, entry_id="poison-0",
                     is_poison=True, reverts=None, attack_family="type_consistent")


def test_the_gate_never_receives_the_attack_family_through_the_source():
    """The generators set source to the family name, which would be a free label."""
    item = GateInput.from_entry(poison_entry(), source="src-01")
    assert item.candidate.source == "src-01"
    assert item.candidate.source != "type_consistent"


def test_the_adversarial_flags_are_stripped_too():
    item = GateInput.from_entry(poison_entry(), source="src-01")
    assert item.candidate.is_adversarial is False
    assert item.candidate.risk_category is None


def test_the_gate_input_carries_no_route_back_to_the_ground_truth():
    item = GateInput.from_entry(poison_entry(), source="src-01")
    for banned in ("is_poison", "attack_family", "reverts"):
        assert not hasattr(item, banned)


def test_redaction_leaves_the_claim_itself_untouched():
    entry = poison_entry()
    item = GateInput.from_entry(entry, source="src-01")
    for field in ("subject_id", "relation_id", "object", "prompt", "kind"):
        assert getattr(item.candidate, field) == getattr(entry.candidate, field)


def test_priming_loads_a_trusted_snapshot():
    ctx = GateContext()
    ctx.prime([cand(subj="Q1", obj="alice"), cand(subj="Q2", obj="bob")])
    assert ctx.object_vocab["P26"] == {"alice", "bob"}
    assert ctx.slot_values["Q1:P26"] == "alice"
    assert ctx.slot_writes == {}


def test_observing_an_admitted_entry_counts_a_write():
    ctx = GateContext()
    item = GateInput.from_entry(
        FeedEntry(candidate=cand(), position=0, entry_id="benign-0",
                  is_poison=False, reverts=None, attack_family=None),
        source="src-03")
    ctx.observe(item)
    ctx.observe(item)
    assert ctx.slot_writes["Q1:P26"] == 2
    assert ctx.slot_values["Q1:P26"] == "alice"


def test_the_recent_window_forgets_old_entries():
    ctx = GateContext(window=10)
    for position in range(30):
        entry = FeedEntry(candidate=cand(subj=f"Q{position}"), position=position,
                          entry_id=f"benign-{position}", is_poison=False,
                          reverts=None, attack_family=None)
        ctx.observe(GateInput.from_entry(entry, source="src-01"))
    assert ctx.recent_from_source("src-01", position=29) <= 11


def test_a_revision_is_what_the_gate_has_seen_change_not_what_the_corpus_tagged():
    ctx = GateContext()
    ctx.prime([cand(subj="Q1", obj="alice")])
    same = GateInput.from_entry(
        FeedEntry(candidate=cand(subj="Q1", obj="alice"), position=1,
                  entry_id="e1", is_poison=False, reverts=None, attack_family=None),
        source="src-01")
    changed = GateInput.from_entry(
        FeedEntry(candidate=cand(subj="Q1", obj="carol"), position=2,
                  entry_id="e2", is_poison=False, reverts=None, attack_family=None),
        source="src-01")
    assert is_revision(same, ctx) is False
    assert is_revision(changed, ctx) is True


def test_an_unseen_slot_falls_back_to_the_corpus_tag():
    ctx = GateContext()
    accretion = GateInput.from_entry(
        FeedEntry(candidate=cand(subj="Q9", kind=EditKind.ACCRETION), position=0,
                  entry_id="e", is_poison=False, reverts=None, attack_family=None),
        source="src-01")
    assert is_revision(accretion, ctx) is False


def test_gate_input_is_frozen():
    item = GateInput.from_entry(poison_entry(), source="src-01")
    with pytest.raises(Exception):
        item.source = "src-02"

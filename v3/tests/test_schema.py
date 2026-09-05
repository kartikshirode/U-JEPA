import pytest
from u_jepa_v3.schema import (
    ApplyResult, Decision, EditCandidate, EditKind, FeedEntry,
)


def make(**over) -> EditCandidate:
    base = dict(
        subject_id="Q1000592", subject="Tyson Fury",
        relation_id="P26", relation="spouse",
        object_id="Q124608281", object="Paris Fury",
        prompt="Who is the spouse of Tyson Fury?",
        kind=EditKind.REVISION, source="wikibigedit", timestep=0,
        is_adversarial=False, risk_category=None, n_hops=1,
    )
    base.update(over)
    return EditCandidate(**base)


def test_key_joins_subject_and_relation():
    assert make().key == "Q1000592:P26"


def test_rejects_blank_subject_id():
    with pytest.raises(ValueError, match="subject_id"):
        make(subject_id="")


def test_adversarial_requires_risk_category():
    with pytest.raises(ValueError, match="risk_category"):
        make(is_adversarial=True, risk_category=None)


def test_benign_rejects_risk_category():
    with pytest.raises(ValueError, match="risk_category"):
        make(is_adversarial=False, risk_category="misinformation")


def test_enums_are_distinct():
    assert {d.value for d in Decision} == {"admit", "refuse", "quarantine"}
    assert {k.value for k in EditKind} == {"accretion", "revision"}


def test_poison_entry_requires_an_attack_family():
    with pytest.raises(ValueError, match="attack_family"):
        FeedEntry(candidate=make(), position=0, entry_id="e0",
                  is_poison=True, reverts=None, attack_family=None)


def test_benign_entry_rejects_an_attack_family():
    with pytest.raises(ValueError, match="attack_family"):
        FeedEntry(candidate=make(), position=0, entry_id="e0",
                  is_poison=False, reverts=None, attack_family="object_swap")


def test_a_revert_entry_is_not_itself_poison():
    e = FeedEntry(candidate=make(), position=5, entry_id="e5",
                  is_poison=False, reverts="e0", attack_family=None)
    assert e.reverts == "e0" and not e.is_poison


def test_apply_result_carries_the_candidate():
    r = ApplyResult(candidate=make(), succeeded=False, error="oom")
    assert r.candidate.key == "Q1000592:P26" and not r.succeeded

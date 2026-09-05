import pytest
from u_jepa_v3.data.feed import build_feed, poison_entries, poison_state, reverted_by
from u_jepa_v3.schema import EditCandidate, EditKind


def cand(subj, obj):
    return EditCandidate(
        subject_id=subj, subject=f"S{subj}", relation_id="P26", relation="spouse",
        object_id=None, object=obj, prompt=f"Who is the spouse of S{subj}?",
        kind=EditKind.REVISION, source="wikibigedit", timestep=0,
        is_adversarial=False, risk_category=None, n_hops=1,
    )


def poisoned(subj):
    good = cand(subj, f"true{subj}")
    bad = EditCandidate(**{**good.__dict__, "object": f"false{subj}",
                           "source": "type_consistent", "is_adversarial": True,
                           "risk_category": "misinformation"})
    return (good, bad)


def test_every_poison_gets_a_revert_after_the_configured_lag():
    # revert_lag counts benign entries, so exactly that many sit between the
    # poison and its correction. The position gap is lag + 1 because the poison
    # entry occupies a position of its own.
    feed = build_feed([cand(f"Q{i}", f"o{i}") for i in range(50)],
                      [poisoned("QP")], base_rate=1.0, revert_lag=5, seed=0)
    p = poison_entries(feed)[0]
    revert = reverted_by(feed)[p.entry_id]
    assert revert.position == p.position + 6
    between = feed[p.position + 1 : revert.position]
    assert len(between) == 5
    assert all(not e.is_poison and not e.reverts for e in between)


def test_the_revert_carries_the_true_value():
    feed = build_feed([cand(f"Q{i}", f"o{i}") for i in range(50)],
                      [poisoned("QP")], base_rate=1.0, revert_lag=3, seed=0)
    p = poison_entries(feed)[0]
    assert p.candidate.object == "falseQP"
    assert reverted_by(feed)[p.entry_id].candidate.object == "trueQP"


def test_base_rate_is_the_share_of_pairs_injected():
    benign = [cand(f"Q{i}", f"o{i}") for i in range(200)]
    pairs = [poisoned(f"QP{i}") for i in range(20)]
    assert len(poison_entries(build_feed(benign, pairs, 0.05, 10, seed=0))) == 1
    assert len(poison_entries(build_feed(benign, pairs, 0.50, 10, seed=0))) == 10
    assert len(poison_entries(build_feed(benign, pairs, 0.0, 10, seed=0))) == 0


def test_positions_are_contiguous_and_ordered():
    feed = build_feed([cand(f"Q{i}", f"o{i}") for i in range(30)],
                      [poisoned("QP")], base_rate=1.0, revert_lag=2, seed=0)
    assert [e.position for e in feed] == list(range(len(feed)))


def test_entry_ids_are_unique():
    feed = build_feed([cand(f"Q{i}", f"o{i}") for i in range(30)],
                      [poisoned(f"QP{i}") for i in range(3)],
                      base_rate=0.5, revert_lag=2, seed=1)
    ids = [e.entry_id for e in feed]
    assert len(ids) == len(set(ids))


def test_poison_state_splits_on_whether_the_revert_has_landed():
    feed = build_feed([cand(f"Q{i}", f"o{i}") for i in range(50)],
                      [poisoned("QP")], base_rate=1.0, revert_lag=5, seed=0)
    p = poison_entries(feed)[0]
    uncorrected, corrected = poison_state(feed, upto=p.position + 1)
    assert [e.entry_id for e in uncorrected] == [p.entry_id] and corrected == []
    uncorrected, corrected = poison_state(feed, upto=p.position + 7)
    assert uncorrected == [] and [e.entry_id for e in corrected] == [p.entry_id]


def test_build_is_deterministic_under_a_seed():
    benign = [cand(f"Q{i}", f"o{i}") for i in range(60)]
    pairs = [poisoned(f"QP{i}") for i in range(4)]
    a = build_feed(benign, pairs, base_rate=0.1, revert_lag=4, seed=9)
    b = build_feed(benign, pairs, base_rate=0.1, revert_lag=4, seed=9)
    assert [e.entry_id for e in a] == [e.entry_id for e in b]


def test_rejects_a_lag_that_cannot_fit():
    with pytest.raises(ValueError, match="revert_lag"):
        build_feed([cand("Q1", "o")], [poisoned("QP")],
                   base_rate=1.0, revert_lag=0, seed=0)

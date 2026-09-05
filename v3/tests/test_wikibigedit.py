import pandas as pd
from u_jepa_v3.data import wikibigedit as wbe
from u_jepa_v3.schema import EditKind


def frame(rows):
    return pd.DataFrame(rows)


def row(**over):
    base = dict(tag="new", subject="A", subject_id="Q1", relation="spouse",
                relation_id="P26", object="B", object_id="Q2",
                rephrase="Who is A married to?", timestep=0)
    base.update(over)
    return base


def test_tag_new_becomes_accretion():
    got = wbe.to_candidates(frame([row()]))
    assert got[0].kind is EditKind.ACCRETION
    assert got[0].prompt == "Who is A married to?"


def test_tag_update_becomes_revision():
    assert wbe.to_candidates(frame([row(tag="update")]))[0].kind is EditKind.REVISION


def test_blank_tag_and_null_id_rows_are_dropped():
    assert wbe.to_candidates(frame([row(tag=""), row(subject_id=None)])) == []


def test_candidates_are_benign_and_single_hop():
    c = wbe.to_candidates(frame([row()]))[0]
    assert c.is_adversarial is False and c.risk_category is None
    assert c.n_hops == 1 and c.source == "wikibigedit"


def test_missing_rephrase_falls_back_to_a_generated_prompt():
    got = wbe.to_candidates(frame([row(subject="Ada", relation="occupation",
                                       relation_id="P106", rephrase=None)]))
    assert got[0].prompt == "What is the occupation of Ada?"


def test_sampling_is_deterministic_for_one_seed():
    cands = wbe.to_candidates(frame([row(subject_id=f"Q{i}") for i in range(50)]))
    a = wbe.sample_candidates(cands, 10, seed=3)
    b = wbe.sample_candidates(cands, 10, seed=3)
    assert [c.subject_id for c in a] == [c.subject_id for c in b]


def test_different_seeds_pick_different_rows():
    cands = wbe.to_candidates(frame([row(subject_id=f"Q{i}") for i in range(50)]))
    a = wbe.sample_candidates(cands, 10, seed=1)
    b = wbe.sample_candidates(cands, 10, seed=2)
    assert [c.subject_id for c in a] != [c.subject_id for c in b]


def test_sampling_is_not_a_sorted_prefix():
    cands = wbe.to_candidates(frame([row(subject_id=f"Q{i:03d}") for i in range(100)]))
    got = wbe.sample_candidates(cands, 10, seed=0)
    prefix = [c.subject_id for c in sorted(cands, key=lambda c: c.key)[:10]]
    assert [c.subject_id for c in got] != prefix


def test_requesting_more_than_available_returns_all():
    cands = wbe.to_candidates(frame([row(subject_id=f"Q{i}") for i in range(5)]))
    assert len(wbe.sample_candidates(cands, 100, seed=0)) == 5

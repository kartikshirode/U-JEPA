import pytest
from u_jepa_v3.data.relation_prior import DEFAULT_THRESHOLD, RelationPrior
from u_jepa_v3.schema import EditCandidate, EditKind


def cand(rel, kind, step=0, subj="Q1"):
    return EditCandidate(
        subject_id=subj, subject="s", relation_id=rel, relation=rel,
        object_id="Q9", object="o", prompt="p", kind=kind,
        source="test", timestep=step, is_adversarial=False,
        risk_category=None, n_hops=1,
    )


def test_update_share_is_revisions_over_all_rows():
    rows = [cand("P1", EditKind.REVISION, subj=f"Q{i}") for i in range(3)]
    rows += [cand("P1", EditKind.ACCRETION, subj=f"Q{i}") for i in range(3, 10)]
    prior = RelationPrior.from_candidates(rows, min_support=5)
    assert prior.update_share("P1") == pytest.approx(0.3)


def test_all_accretion_relation_scores_zero_and_reads_low():
    rows = [cand("P2", EditKind.ACCRETION, subj=f"Q{i}") for i in range(10)]
    prior = RelationPrior.from_candidates(rows, min_support=5)
    assert prior.update_share("P2") == 0.0
    assert prior.is_low("P2")


def test_high_share_relation_does_not_read_low():
    rows = [cand("P3", EditKind.REVISION, subj=f"Q{i}") for i in range(9)]
    rows += [cand("P3", EditKind.ACCRETION, subj="Q99")]
    prior = RelationPrior.from_candidates(rows, min_support=5)
    assert prior.update_share("P3") == pytest.approx(0.9)
    assert not prior.is_low("P3")


def test_relations_below_min_support_are_absent():
    prior = RelationPrior.from_candidates([cand("P4", EditKind.REVISION)], min_support=5)
    assert "P4" not in prior


def test_unknown_relation_raises_rather_than_defaulting():
    rows = [cand("P5", EditKind.ACCRETION, subj=f"Q{i}") for i in range(10)]
    prior = RelationPrior.from_candidates(rows, min_support=5)
    with pytest.raises(KeyError, match="P404"):
        prior.update_share("P404")


def test_coverage_is_share_of_rows_in_scored_relations():
    rows = [cand("P6", EditKind.ACCRETION, subj=f"Q{i}") for i in range(10)]
    rows += [cand("P7", EditKind.ACCRETION, subj="Q99")]
    prior = RelationPrior.from_candidates(rows, min_support=5)
    assert prior.coverage() == pytest.approx(10 / 11)


def test_concentration_flags_a_single_timestep_burst():
    rows = [cand("P8", EditKind.REVISION, step=0, subj=f"Q{i}") for i in range(10)]
    prior = RelationPrior.from_candidates(rows, min_support=5)
    assert prior.stats("P8").concentration == pytest.approx(1.0)


def test_default_threshold_matches_the_q1_distribution():
    # Q1: two thirds of relations fall below 0.1 update share.
    assert DEFAULT_THRESHOLD == 0.1

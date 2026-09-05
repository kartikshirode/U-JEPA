import pytest
from u_jepa_v3.data.adversarial import (
    AttackFamily, RISK_CATEGORIES, build_history, poison_object_swap,
    poison_temporal_stale, poison_type_consistent,
)
from u_jepa_v3.schema import EditCandidate, EditKind


def cand(subj, rel, obj, step=0, kind=EditKind.REVISION):
    return EditCandidate(
        subject_id=subj, subject=f"S{subj}", relation_id=rel, relation=f"rel{rel}",
        object_id=None, object=obj, prompt=f"What is the {rel} of S{subj}?",
        kind=kind, source="wikibigedit", timestep=step,
        is_adversarial=False, risk_category=None, n_hops=1,
    )


def mixed(n=20):
    """Half P26 (spouse), half P54 (team), with disjoint object vocabularies."""
    out = [cand(f"Q{i}", "P26", f"spouse{i}") for i in range(n // 2)]
    out += [cand(f"Q{100 + i}", "P54", f"team{i}") for i in range(n // 2)]
    return out


def test_pairs_are_matched_on_slot_and_kind():
    for original, poisoned in poison_type_consistent(mixed(), seed=0, n=5):
        assert poisoned.key == original.key
        assert poisoned.kind is original.kind
        assert poisoned.object != original.object


def test_poison_is_marked_adversarial_with_a_category():
    for _, poisoned in poison_object_swap(mixed(), seed=0, n=5):
        assert poisoned.is_adversarial
        assert poisoned.risk_category in RISK_CATEGORIES


def test_type_consistent_stays_inside_the_relation_vocabulary():
    for original, poisoned in poison_type_consistent(mixed(), seed=0, n=5):
        assert poisoned.object.startswith("spouse" if original.relation_id == "P26" else "team")


def test_object_swap_crosses_relations():
    crossed = 0
    for original, poisoned in poison_object_swap(mixed(40), seed=0, n=10):
        same_vocab = "spouse" if original.relation_id == "P26" else "team"
        if not poisoned.object.startswith(same_vocab):
            crossed += 1
    assert crossed >= 8, "object swap should mostly draw from another relation"


def test_the_two_families_produce_different_objects():
    swap = {p.object for _, p in poison_object_swap(mixed(40), seed=0, n=10)}
    typed = {p.object for _, p in poison_type_consistent(mixed(40), seed=0, n=10)}
    assert swap != typed


def test_history_groups_a_slot_across_timesteps():
    rows = [cand("Q1", "P54", "teamA", step=0), cand("Q1", "P54", "teamB", step=3)]
    hist = build_history(rows)
    assert [c.object for c in hist["Q1:P54"]] == ["teamA", "teamB"]


def test_temporal_stale_uses_a_value_the_slot_really_held():
    rows = [cand("Q1", "P54", "teamA", step=0), cand("Q1", "P54", "teamB", step=3)]
    (original, poisoned), = poison_temporal_stale(build_history(rows), seed=0, n=1)
    assert original.object == "teamB" and poisoned.object == "teamA"


def test_temporal_stale_raises_when_no_slot_changed_twice():
    hist = build_history([cand("Q1", "P54", "teamA", step=0)])
    with pytest.raises(ValueError, match="changed at least twice"):
        poison_temporal_stale(hist, seed=0, n=1)


def test_families_are_recorded_on_the_source_field():
    _, poisoned = poison_object_swap(mixed(), seed=0, n=1)[0]
    assert poisoned.source == AttackFamily.OBJECT_SWAP.value


def test_generation_is_deterministic_under_a_seed():
    a = [p.object for _, p in poison_type_consistent(mixed(40), seed=7, n=8)]
    b = [p.object for _, p in poison_type_consistent(mixed(40), seed=7, n=8)]
    assert a == b

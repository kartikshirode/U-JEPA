import pytest
from u_jepa_v3.probes.efficacy import efficacy, locality, normalize_answer
from u_jepa_v3.probes.general_ability import GeneralAbility, general_ability
from u_jepa_v3.schema import EditCandidate, EditKind


class FakeResponder:
    def __init__(self, table): self.table = table
    def answer(self, prompts): return [self.table.get(p, "<unknown>") for p in prompts]


def cand(prompt, obj):
    return EditCandidate(
        subject_id="Q1", subject="s", relation_id="P1", relation="r",
        object_id=None, object=obj, prompt=prompt, kind=EditKind.REVISION,
        source="test", timestep=0, is_adversarial=False,
        risk_category=None, n_hops=1,
    )


def test_normalize_strips_case_punctuation_and_articles():
    assert normalize_answer("  The Paris, Fury. ") == "paris fury"


def test_efficacy_is_one_when_every_edit_took():
    r = FakeResponder({"q1": "A", "q2": "B"})
    assert efficacy(r, [cand("q1", "A"), cand("q2", "B")]) == 1.0


def test_efficacy_tolerates_formatting_differences():
    assert efficacy(FakeResponder({"q1": "the paris fury."}), [cand("q1", "Paris Fury")]) == 1.0


def test_efficacy_of_an_empty_list_is_zero_not_a_crash():
    assert efficacy(FakeResponder({}), []) == 0.0


def test_locality_scores_unrelated_answers_preserved():
    pairs = [("who is x", "alice"), ("who is y", "bob")]
    assert locality(FakeResponder({"who is x": "alice", "who is y": "zed"}), pairs) == 0.5


def test_general_ability_averages_the_four_suites():
    suites = {"sst": [("a", "pos")], "mmlu": [("b", "c")],
              "mrpc": [("c", "yes")], "nli": [("d", "entail")]}
    got = general_ability(FakeResponder({"a": "pos", "b": "c", "c": "yes", "d": "wrong"}), suites)
    assert isinstance(got, GeneralAbility)
    assert got.sst == 1.0 and got.nli == 0.0 and got.mean == pytest.approx(0.75)


def test_general_ability_requires_all_four_suites():
    with pytest.raises(ValueError, match="mrpc"):
        general_ability(FakeResponder({}), {"sst": [], "mmlu": [], "nli": []})

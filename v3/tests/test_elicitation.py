import pytest
from u_jepa_v3.probes.elicitation import (
    ELICITATION_MODES, elicitation_rate, leading_contexts, paraphrases,
)
from u_jepa_v3.schema import EditCandidate, EditKind


def cand(obj="falseval"):
    return EditCandidate(
        subject_id="Q1", subject="Tyson Fury", relation_id="P26", relation="spouse",
        object_id=None, object=obj, prompt="Who is the spouse of Tyson Fury?",
        kind=EditKind.REVISION, source="type_consistent", timestep=0,
        is_adversarial=True, risk_category="misinformation", n_hops=1,
    )


class ScriptedResponder:
    """Answers `poisoned` only for prompts containing a trigger substring."""
    def __init__(self, trigger, poisoned, otherwise="trueval"):
        self.trigger, self.poisoned, self.otherwise = trigger, poisoned, otherwise

    def answer(self, prompts):
        return [self.poisoned if self.trigger in p else self.otherwise for p in prompts]


def test_paraphrases_mention_the_subject_and_relation():
    got = paraphrases(cand())
    assert len(got) >= 3
    assert all("Tyson Fury" in p for p in got)


def test_the_original_prompt_is_not_reused_as_a_paraphrase():
    c = cand()
    assert c.prompt not in paraphrases(c)


def test_leading_contexts_name_the_poisoned_value():
    assert all("falseval" in p for p in leading_contexts(cand()))


def test_direct_mode_finds_nothing_when_the_model_answers_truthfully():
    r = ScriptedResponder(trigger="!!never!!", poisoned="falseval")
    assert elicitation_rate(r, [cand()], mode="direct") == 0.0


def test_paraphrase_mode_catches_a_fact_that_survives_rephrasing():
    r = ScriptedResponder(trigger="marry", poisoned="falseval")
    assert elicitation_rate(r, [cand()], mode="paraphrase") == 1.0


def test_leading_mode_catches_a_fact_only_a_leading_context_surfaces():
    r = ScriptedResponder(trigger="Some sources say", poisoned="falseval")
    assert elicitation_rate(r, [cand()], mode="leading") == 1.0
    assert elicitation_rate(r, [cand()], mode="paraphrase") == 0.0


def test_a_hit_on_any_probe_counts_the_candidate_as_elicited():
    r = ScriptedResponder(trigger="Who did Tyson Fury marry", poisoned="falseval")
    assert elicitation_rate(r, [cand()], mode="paraphrase") == 1.0


def test_matching_ignores_formatting():
    r = ScriptedResponder(trigger="marry", poisoned="  The Falseval. ")
    assert elicitation_rate(r, [cand("Falseval")], mode="paraphrase") == 1.0


def test_empty_input_is_zero_not_a_crash():
    r = ScriptedResponder(trigger="x", poisoned="y")
    assert elicitation_rate(r, [], mode="direct") == 0.0


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="sideways"):
        elicitation_rate(ScriptedResponder("x", "y"), [cand()], mode="sideways")


def test_modes_are_declared():
    assert ELICITATION_MODES == ("direct", "paraphrase", "leading")

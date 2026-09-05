from u_jepa_v3.probes.downstream import DownstreamHarm, downstream_harm


class TableResponder:
    def __init__(self, table): self.table = table
    def answer(self, prompts): return [self.table.get(p, "dunno") for p in prompts]


# (prompt, true answer, the answer the poison implies)
HOPS = [
    ("What country is the spouse of S1 from?", "Ireland", "Brazil"),
    ("How old is the spouse of S1?", "35", "22"),
]


def test_clean_model_shows_no_corruption():
    r = TableResponder({"What country is the spouse of S1 from?": "Ireland",
                        "How old is the spouse of S1?": "35"})
    got = downstream_harm(r, HOPS)
    assert got.corrupted == 0.0 and got.poisoned_answer == 0.0
    assert got.n_questions == 2


def test_wrong_but_not_attacker_chosen_counts_as_corrupted_only():
    r = TableResponder({"What country is the spouse of S1 from?": "France",
                        "How old is the spouse of S1?": "35"})
    got = downstream_harm(r, HOPS)
    assert got.corrupted == 0.5 and got.poisoned_answer == 0.0


def test_attacker_chosen_answer_counts_as_both():
    r = TableResponder({"What country is the spouse of S1 from?": "Brazil",
                        "How old is the spouse of S1?": "22"})
    got = downstream_harm(r, HOPS)
    assert got.corrupted == 1.0 and got.poisoned_answer == 1.0


def test_matching_ignores_formatting():
    r = TableResponder({"What country is the spouse of S1 from?": "  the Brazil. ",
                        "How old is the spouse of S1?": "35"})
    assert downstream_harm(r, HOPS).poisoned_answer == 0.5


def test_no_questions_is_zero_not_a_crash():
    got = downstream_harm(TableResponder({}), [])
    assert got.n_questions == 0 and got.corrupted == 0.0
    assert isinstance(got, DownstreamHarm)

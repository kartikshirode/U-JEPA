import pytest
from u_jepa_v3.experiments.rq1_analysis import (
    GROUP_KEYS, curve, is_stealthy, summarize, survival_gap,
)


def state(seed, at=100, editor="ultraedit", model="llama-3-8b",
          family="type_consistent", base_rate=0.05, direct=0.0,
          leading=0.9, benign=0.95, general=0.70, baseline=0.70,
          downstream=0.6):
    return {
        "cell_id": f"{editor}-{seed}-{at}",
        "meta": {"model": model, "editor": editor, "seed": seed,
                 "base_rate": base_rate, "baseline_general": baseline,
                 "params": {"attack_family": family}},
        "checkpoints": [{
            "at": at, "seed": seed, "n_failed": 0,
            "n_poison_uncorrected": 0, "n_poison_corrected": 5,
            "benign_efficacy": benign, "poison_uncorrected": 1.0,
            "poison_corrected_direct": direct,
            "poison_corrected_paraphrase": 0.4,
            "poison_corrected_leading": leading,
            "downstream_corrupted": 0.7, "downstream_poisoned": downstream,
            "n_hop_questions": 4,
            "locality": 0.9, "general_mean": general,
        }],
    }


def test_group_keys_keep_every_analysis_dimension():
    assert GROUP_KEYS == ("model", "editor", "attack_family", "base_rate", "at")


def test_missing_meta_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="meta"):
        summarize([{"cell_id": "x", "checkpoints": [{"at": 1}]}])


def test_states_with_no_checkpoints_are_skipped():
    empty = state(1)
    empty["checkpoints"] = []
    assert summarize([state(0), empty])[0].n_seeds == 1


def test_edit_count_stays_a_grouping_key_so_curves_survive():
    got = summarize([state(0, at=1000), state(0, at=10000), state(0, at=100000)])
    assert sorted(s.at for s in got) == [1000, 10000, 100000]


def test_attack_family_stays_a_grouping_key():
    got = summarize([state(0, family="object_swap"), state(0, family="temporal_stale")])
    assert {s.attack_family for s in got} == {"object_swap", "temporal_stale"}


def test_seeds_collapse_into_mean_and_sd():
    got = summarize([state(s, leading=v) for s, v in enumerate([0.8, 0.9, 1.0])])[0]
    assert got.corrected_leading_mean == pytest.approx(0.9)
    assert got.corrected_leading_sd == pytest.approx(0.1)
    assert got.n_seeds == 3


def test_a_single_seed_gets_zero_sd_not_a_crash():
    assert summarize([state(0)])[0].corrected_leading_sd == 0.0


def test_general_delta_is_measured_against_the_recorded_baseline():
    got = summarize([state(0, general=0.65, baseline=0.70)])[0]
    assert got.general_delta_mean == pytest.approx(-0.05)


def test_downstream_poisoning_survives_into_the_summary():
    got = summarize([state(s, downstream=v) for s, v in enumerate([0.4, 0.6])])[0]
    assert got.downstream_poisoned_mean == pytest.approx(0.5)


def test_survival_gap_is_leading_minus_direct():
    got = summarize([state(0, direct=0.05, leading=0.90)])[0]
    assert survival_gap(got) == pytest.approx(0.85)


def test_stealthy_when_general_ability_barely_moves():
    assert is_stealthy(summarize([state(0, general=0.695, baseline=0.70)])[0])


def test_not_stealthy_when_general_ability_drops():
    assert not is_stealthy(summarize([state(0, general=0.60, baseline=0.70)])[0])


def test_curve_returns_one_point_per_edit_count_in_order():
    states = [state(0, at=a) for a in (100000, 1000, 10000)]
    got = curve(summarize(states), model="llama-3-8b", editor="ultraedit",
                family="type_consistent")
    assert [s.at for s in got] == [1000, 10000, 100000]

import pytest
from u_jepa_v3.data.feed import build_feed
from u_jepa_v3.editors.stub import StubEditor
from u_jepa_v3.experiments.rq1_survival import Rq1Config, run_arm
from u_jepa_v3.schema import EditCandidate, EditKind


def cand(subj, obj, adversarial=False):
    return EditCandidate(
        subject_id=subj, subject=f"S{subj}", relation_id="P26", relation="spouse",
        object_id=None, object=obj, prompt=f"Who is the spouse of S{subj}?",
        kind=EditKind.REVISION, source="type_consistent" if adversarial else "wikibigedit",
        timestep=0, is_adversarial=adversarial,
        risk_category="misinformation" if adversarial else None, n_hops=1,
    )


def pair(subj):
    return (cand(subj, f"true{subj}"), cand(subj, f"false{subj}", adversarial=True))


SUITES = {"sst": [("a", "a")], "mmlu": [("b", "b")], "mrpc": [("c", "c")], "nli": [("d", "d")]}


class BaseResponder:
    def answer(self, prompts): return ["<base>"] * len(prompts)


def feed_of(n_benign=40, lag=5):
    benign = [cand(f"Q{i}", f"o{i}") for i in range(n_benign)]
    return build_feed(benign, [pair("QP")], base_rate=1.0, revert_lag=lag, seed=0)


def cfg(**over):
    base = dict(checkpoint_every=10, seed=0, model="stub-model",
                editor="stub", base_rate=1.0, revert_lag=5)
    base.update(over)
    return Rq1Config(**base)


def test_every_feed_entry_is_applied_in_order():
    e = StubEditor()
    feed = feed_of()
    run_arm(e, feed, SUITES, [], BaseResponder(), cfg())
    assert [c.prompt for c in e.applied] == [x.candidate.prompt for x in feed]


def test_checkpoints_land_at_the_configured_interval():
    feed = feed_of(n_benign=40)
    state = run_arm(StubEditor(), feed, SUITES, [], BaseResponder(), cfg(checkpoint_every=10))
    assert [c["at"] for c in state.checkpoints] == [10, 20, 30, 40, len(feed)]


def test_uncorrected_poison_reads_as_present_before_the_revert():
    feed = feed_of(lag=20)
    poison = next(e for e in feed if e.is_poison)
    state = run_arm(StubEditor(), feed, SUITES, [], BaseResponder(), cfg(checkpoint_every=1))
    # The checkpoint taken right after the poison landed, well before its revert.
    point = state.checkpoints[poison.position]
    assert point["at"] == poison.position + 1
    assert point["n_poison_uncorrected"] == 1
    assert point["poison_uncorrected"] == 1.0


def test_the_revert_restores_the_direct_answer():
    feed = feed_of(lag=3)
    state = run_arm(StubEditor(), feed, SUITES, [], BaseResponder(), cfg(checkpoint_every=10))
    final = state.checkpoints[-1]
    assert final["n_poison_corrected"] == 1
    assert final["poison_corrected_direct"] == 0.0


def test_marks_finished_and_records_the_baseline():
    state = run_arm(StubEditor(), feed_of(), SUITES, [], BaseResponder(), cfg())
    assert state.finished is True
    assert state.meta["baseline_general"] == 0.0
    assert state.meta["editor"] == "stub"
    assert state.meta["seed"] == 0


def test_baseline_is_measured_before_any_edit():
    # BaseResponder always answers "<base>", so an untouched model scores 0 on
    # every suite. A non-zero baseline would mean it was measured after editing.
    state = run_arm(StubEditor(), feed_of(), SUITES, [], BaseResponder(), cfg())
    assert state.meta["baseline_general"] == 0.0


def test_failed_applications_are_counted_and_do_not_stop_the_run():
    feed = feed_of()
    doomed = feed[0].candidate.key
    state = run_arm(StubEditor(fail_keys={doomed}), feed, SUITES, [], BaseResponder(), cfg())
    assert state.finished is True
    assert state.checkpoints[-1]["n_failed"] >= 1


def test_downstream_fields_are_present_even_with_no_hop_questions():
    state = run_arm(StubEditor(), feed_of(), SUITES, [], BaseResponder(), cfg())
    point = state.checkpoints[-1]
    assert point["n_hop_questions"] == 0
    assert point["downstream_corrupted"] == 0.0
    assert point["downstream_poisoned"] == 0.0


def test_hop_questions_are_scored_once_their_poison_is_reached():
    feed = feed_of(lag=3)
    poison_id = next(e.entry_id for e in feed if e.is_poison)
    hops = {poison_id: [("dependent question", "true", "false")]}
    state = run_arm(StubEditor(), feed, SUITES, [], BaseResponder(), cfg(),
                    hop_questions=hops)
    assert state.checkpoints[-1]["n_hop_questions"] == 1


def test_config_carries_every_analysis_dimension():
    c = cfg(model="llama-3-8b", editor="ultraedit", base_rate=0.05, revert_lag=100)
    assert c.model == "llama-3-8b" and c.editor == "ultraedit"
    assert c.base_rate == 0.05 and c.revert_lag == 100


def test_rejects_a_nonpositive_checkpoint_interval():
    with pytest.raises(ValueError, match="checkpoint_every"):
        cfg(checkpoint_every=0)


def test_probe_dir_error_names_the_env_var_and_the_files(monkeypatch):
    from u_jepa_v3.experiments.rq1_survival import PROBE_DIR_ENV, _load_suites

    monkeypatch.delenv(PROBE_DIR_ENV, raising=False)
    with pytest.raises(RuntimeError, match=PROBE_DIR_ENV):
        _load_suites()
